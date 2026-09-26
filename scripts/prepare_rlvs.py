"""Prepare the Kaggle RLVS folder for the independent CCTV experiment.

Expected layout after extraction:
  <root>/Violence Fight Detection dataset/RLVS/{train,test}/{Fight,NonFight}/*.mp4

The script creates a CSV manifest and uses video hashes as groups, preventing
identical clips from crossing splits. It does not download or modify source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

LABELS = {"fight": 1, "nonfight": 0, "non- fight": 0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/rlvs_splits.csv"))
    args = parser.parse_args()
    videos = sorted(args.root.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found under {args.root}")
    rows = []
    for path in videos:
        class_name = path.parent.name.lower().replace("_", " ")
        normalized = class_name.replace(" ", "")
        if normalized not in {"fight", "nonfight"}:
            raise SystemExit(f"Unknown class folder: {path.parent}")
        split_name = next((p.lower() for p in path.parts if p.lower() in {"train", "test"}), None)
        if split_name is None:
            raise SystemExit(f"Expected train/test folder: {path}")
        digest = sha256(path)
        rows.append({"path": path.relative_to(args.root).as_posix(), "label": LABELS[normalized], "split": split_name, "sha256": digest, "group": digest})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["path", "label", "split", "sha256", "group"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} RLVS records to {args.output}")


if __name__ == "__main__":
    main()
