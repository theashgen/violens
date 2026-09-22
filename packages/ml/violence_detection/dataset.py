"""Read-only dataset inspection and auditable, group-aware split manifests."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from violence_detection.video import EXTENSIONS, VideoError, inspect_video, load_video

LABELS = {
    "fight": 1,
    "violence": 1,
    "violent": 1,
    "nonfight": 0,
    "nonviolence": 0,
    "nonviolent": 0,
    "normal": 0,
}
SPLITS = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def label_from_path(path: Path) -> int:
    labels = {
        LABELS[name]
        for part in path.parts[:-1]
        if (name := part.lower().replace("-", "").replace("_", "")) in LABELS
    }
    if len(labels) != 1:
        raise ValueError(f"Expected one class folder (Fight/NonFight etc.): {path}")
    return labels.pop()


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inspect_dataset(root: Path) -> dict:
    root = root.resolve()
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported videos found under {root}.")
    records, errors = [], []
    discovered_classes: Counter[str] = Counter()
    for index, path in enumerate(paths):
        relative = path.relative_to(root)
        try:
            label = label_from_path(relative)
            discovered_classes[str(label)] += 1
            splits = {
                SPLITS[part.lower()] for part in relative.parts[:-1] if part.lower() in SPLITS
            }
            if len(splits) > 1:
                raise ValueError(f"Ambiguous split folders: {relative}")
            info = inspect_video(path)
            records.append(
                {
                    "path": relative.as_posix(),
                    "label": label,
                    "split": next(iter(splits), ""),
                    "sha256": file_hash(path),
                    **info,
                }
            )
        except (VideoError, ValueError, OSError) as exc:
            errors.append({"path": relative.as_posix(), "error": str(exc)})
        if (index + 1) % 100 == 0:
            print(f"Inspected {index + 1}/{len(paths)} videos", flush=True)
    durations = [row["duration_seconds"] for row in records if row["duration_seconds"] is not None]
    hashes = Counter(row["sha256"] for row in records)
    summary = {
        "total_videos": len(paths),
        "usable_videos": len(records),
        "excluded_videos": len(errors),
        "discovered_class_counts": dict(discovered_classes),
        "class_counts": dict(Counter(str(row["label"]) for row in records)),
        "extensions": dict(Counter(path.suffix.lower() for path in paths)),
        "split_counts": dict(Counter(row["split"] or "unsplit" for row in records)),
        "duplicate_copies": sum(count - 1 for count in hashes.values()),
        "duration_seconds": {
            "min": min(durations, default=0),
            "max": max(durations, default=0),
            "mean": sum(durations) / len(durations) if durations else 0,
        },
        "resolutions": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
        "fps_counts": dict(Counter(str(round(r["fps"], 2)) for r in records)),
    }
    duplicates = [
        [row["path"] for row in records if row["sha256"] == digest]
        for digest, count in hashes.items()
        if count > 1
    ]
    return {
        "root": str(root),
        "summary": summary,
        "errors": errors,
        "duplicate_groups": duplicates,
        "videos": records,
    }


def validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("The manifest is empty.")
    for key in ("path", "sha256", "group"):
        assignments: dict[str, str] = {}
        labels: dict[str, int] = {}
        for row in records:
            value = row[key]
            if not value:
                raise ValueError(f"Missing {key} in manifest.")
            if value in assignments and assignments[value] != row["split"]:
                raise ValueError(f"Data leakage: {key} {value} crosses splits.")
            if key == "sha256" and value in labels and labels[value] != int(row["label"]):
                raise ValueError("Identical videos have conflicting labels.")
            assignments[value] = row["split"]
            labels[value] = int(row["label"])
    for split in ("train", "val", "test"):
        if {int(row["label"]) for row in records if row["split"] == split} != {0, 1}:
            raise ValueError(f"Split {split} must contain both classes.")


def create_splits(report: dict, seed: int, groups: dict[str, str] | None = None) -> list[dict]:
    records = []
    seen: dict[str, dict] = {}
    for source in report["videos"]:
        row = dict(source)
        if groups is not None and not groups.get(row["path"]):
            raise ValueError(f"Missing source group for {row['path']}.")
        row["group"] = groups[row["path"]] if groups is not None else row["sha256"]
        if row["sha256"] in seen:
            previous = seen[row["sha256"]]
            if previous["label"] != row["label"] or previous["split"] != row["split"]:
                raise ValueError("Duplicate video has conflicting labels or supplied splits.")
            if previous["group"] != row["group"]:
                raise ValueError("Duplicate video has conflicting source groups.")
            continue
        seen[row["sha256"]] = row
        records.append(row)
    present = {r["split"] for r in records}
    if "" in present and len(present) > 1:
        raise ValueError("Mixed split/unsplit inputs; inspect each dataset collection separately.")
    if present == {""}:
        for row in records:
            row["split"] = "train"
    elif "train" not in present:
        raise ValueError("Supplied splits must include train.")
    # Reserve 15% for test, then 15% of the original pool for validation.
    for missing, fraction in (("test", 0.15), ("val", 0.15 / 0.85)):
        if any(row["split"] == missing for row in records):
            continue
        pool = [row for row in records if row["split"] == "train"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        _, held_out = next(splitter.split(pool, groups=[row["group"] for row in pool]))
        for index in held_out:
            pool[index]["split"] = missing
    validate_records(records)
    return records


def read_manifest(path: Path, root: Path) -> list[dict]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    validate_records(rows)
    resolved_root = root.resolve()
    for row in rows:
        resolved = (resolved_root / row["path"]).resolve()
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise ValueError(f"Missing video or path outside dataset root: {row['path']}")
        row["label"] = int(row["label"])
    return rows


class VideoDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], num_frames: int):
        self.root, self.rows, self.num_frames = root, rows, num_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return load_video(self.root / row["path"], self.num_frames), int(row["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--root", required=True, type=Path)
    inspect.add_argument("--output", type=Path, default=Path("artifacts/metrics/dataset.json"))
    split = commands.add_parser("split")
    split.add_argument("--report", type=Path, required=True)
    split.add_argument("--output", type=Path, default=Path("data/splits.csv"))
    split.add_argument("--groups", type=Path, help="CSV with path,group for related source clips")
    split.add_argument("--exclude", type=Path, help="Reviewed CSV with path,reason exclusions")
    split.add_argument("--acknowledge-unknown-groups", action="store_true")
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "inspect":
        report = inspect_dataset(args.root)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
    else:
        if not args.groups and not args.acknowledge_unknown_groups:
            parser.error("Provide --groups or --acknowledge-unknown-groups after source review.")
        groups = None
        if args.groups:
            with args.groups.open(newline="") as source:
                groups = {row["path"]: row["group"] for row in csv.DictReader(source)}
        report = json.loads(args.report.read_text())
        if args.exclude:
            with args.exclude.open(newline="") as source:
                exclusions = list(csv.DictReader(source))
            excluded = {row["path"] for row in exclusions if row["reason"].strip()}
            known = {row["path"] for row in report["videos"]}
            if len(excluded) != len(exclusions) or not excluded <= known:
                parser.error("Exclusions must have unique known paths and a nonempty reason.")
            report["videos"] = [row for row in report["videos"] if row["path"] not in excluded]
            print(f"Applied {len(excluded)} reviewed exclusions; source videos are unchanged.")
        records = create_splits(report, args.seed, groups)
        with args.output.open("w", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=["path", "label", "split", "sha256", "group"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps(dict(Counter(f"{r['split']}/{r['label']}" for r in records)), indent=2))
        if groups is None:
            print(
                "Source groups unknown: exact duplicates checked; related-clip leakage unverified."
            )


if __name__ == "__main__":
    main()
