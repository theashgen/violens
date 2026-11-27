"""Download disjoint UCF-Crime continuous validation videos (not in official UCF-Crime test)."""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main"
OUT = Path("data/external/ucf-crime")
ANN_PATH = OUT / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

# Independently annotated violent intervals from UCA (CVPR 2024) UCFCrime_Val.json
# Strictly disjoint from Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
VAL_POSITIVE_SPECS = {
    "Assault042_x264.mp4": {"category": "Assault", "intervals_sec": [[50.9, 100.4]]},
    "Assault047_x264.mp4": {"category": "Assault", "intervals_sec": [[1.6, 10.8]]},
    "Assault048_x264.mp4": {"category": "Assault", "intervals_sec": [[5.3, 10.7]]},
    "Assault049_x264.mp4": {"category": "Assault", "intervals_sec": [[12.5, 18.1]]},
    "Assault050_x264.mp4": {"category": "Assault", "intervals_sec": [[1.9, 19.6]]},
    "Assault051_x264.mp4": {"category": "Assault", "intervals_sec": [[9.3, 18.8]]},
    "Assault052_x264.mp4": {"category": "Assault", "intervals_sec": [[1.8, 17.2]]},
    "Fighting045_x264.mp4": {"category": "Fighting", "intervals_sec": [[11.4, 21.2], [26.9, 34.2], [43.7, 68.2]]},
    "Fighting049_x264.mp4": {"category": "Fighting", "intervals_sec": [[0.7, 28.2], [43.3, 50.9]]},
    "Fighting051_x264.mp4": {"category": "Fighting", "intervals_sec": [[39.1, 108.1], [148.8, 209.6]]},
}


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


def extract_members(member_infos, zf, dest_dir: Path):
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
    test_names = {l.split()[0] for l in ANN_PATH.read_text().splitlines() if l.strip()}
    assert not (set(VAL_POSITIVE_SPECS) & test_names), "Validation/test overlap detected!"

    val_manifest = {"positives": [], "normals": []}

    for zname in ("Anomaly-Videos-Part-1.zip", "Anomaly-Videos-Part-2.zip"):
        with zipfile.ZipFile(HTTPRangeFile(f"{BASE}/{zname}")) as zf:
            members = [
                i for i in zf.infolist()
                if i.filename.endswith(".mp4") and Path(i.filename).name in VAL_POSITIVE_SPECS
            ]
            paths = extract_members(members, zf, OUT / "val_positive")
            for p in paths:
                fn = Path(p).name
                val_manifest["positives"].append({"file": p, "name": fn, **VAL_POSITIVE_SPECS[fn]})

    with zipfile.ZipFile(HTTPRangeFile(f"{BASE}/Training-Normal-Videos-Part-1.zip")) as zf:
        infos = sorted(
            [
                i for i in zf.infolist()
                if i.filename.endswith(".mp4")
                and Path(i.filename).name not in test_names
                and 1_500_000 <= i.file_size <= 12_000_000
            ],
            key=lambda i: i.filename,
        )
        rng = random.Random(101)
        chosen = sorted(rng.sample(infos, min(14, len(infos))), key=lambda i: i.filename)
        paths = extract_members(chosen, zf, OUT / "val_normal")
        for p in paths:
            fn = Path(p).name
            val_manifest["normals"].append({"file": p, "name": fn, "category": "Normal", "intervals_sec": []})

    (OUT / "val_manifest.json").write_text(json.dumps(val_manifest, indent=2))
    print(f"Saved val manifest: {len(val_manifest['positives'])} positives, {len(val_manifest['normals'])} normals", flush=True)


if __name__ == "__main__":
    main()
