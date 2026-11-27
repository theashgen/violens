"""Selectively download unedited continuous UCF-Crime surveillance videos via HTTP Range."""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main"
OUT = Path("data/external/ucf-crime")
ANN_PATH = OUT / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"


class HTTPRangeFile:
    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.final_url = r.geturl()
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n == 0:
            return b""
        end = self.size - 1 if n < 0 else min(self.size - 1, self.pos + n - 1)
        if self.pos > end:
            return b""
        req = urllib.request.Request(self.final_url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data


def extract_members(zip_name: str, member_infos: list[zipfile.ZipInfo], zf: zipfile.ZipFile, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for info in member_infos:
        fname = Path(info.filename).name
        target = dest_dir / fname
        if target.exists() and target.stat().st_size == info.file_size:
            saved.append(str(target))
            continue
        print(f"Downloading {fname} ({info.file_size/1e6:.2f} MB)...", flush=True)
        data = zf.read(info)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
        saved.append(str(target))
    return saved


def main():
    test_ann = {}
    for line in ANN_PATH.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            test_ann[parts[0]] = {
                "category": parts[1],
                "intervals_frames": [
                    [int(parts[2]), int(parts[3])],
                    *([[int(parts[4]), int(parts[5])]] if int(parts[4]) >= 0 and int(parts[5]) >= 0 else []),
                ],
            }

    # 1. All 10 official physical-violence test videos (5 Fighting, 3 Assault, 2 Abuse)
    target_pos_test = {
        k for k, v in test_ann.items() if v["category"] in ("Fighting", "Assault", "Abuse")
    }
    test_manifest = {"positives": [], "normals": []}

    for zname in ("Anomaly-Videos-Part-1.zip", "Anomaly-Videos-Part-2.zip"):
        with zipfile.ZipFile(HTTPRangeFile(f"{BASE}/{zname}")) as zf:
            members = [
                i for i in zf.infolist()
                if i.filename.endswith(".mp4") and Path(i.filename).name in target_pos_test
            ]
            paths = extract_members(zname, members, zf, OUT / "test_positive")
            for p in paths:
                fn = Path(p).name
                test_manifest["positives"].append({"file": p, "name": fn, **test_ann[fn]})

    # 2. Deterministic seeded sample of 24 official Testing_Normal_Videos (1-15 MB)
    with zipfile.ZipFile(HTTPRangeFile(f"{BASE}/Testing_Normal_Videos.zip")) as zf:
        infos = sorted(
            [i for i in zf.infolist() if i.filename.endswith(".mp4") and 1_000_000 <= i.file_size <= 15_000_000],
            key=lambda i: i.filename,
        )
        rng = random.Random(42)
        chosen = sorted(rng.sample(infos, min(24, len(infos))), key=lambda i: i.filename)
        paths = extract_members("Testing_Normal_Videos.zip", chosen, zf, OUT / "test_normal")
        for p in paths:
            fn = Path(p).name
            test_manifest["normals"].append({"file": p, "name": fn, "category": "Normal", "intervals_frames": []})

    (OUT / "test_manifest.json").write_text(json.dumps(test_manifest, indent=2))
    print(f"Saved test manifest: {len(test_manifest['positives'])} positives, {len(test_manifest['normals'])} normals", flush=True)


if __name__ == "__main__":
    main()
