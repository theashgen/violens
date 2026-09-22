import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from violence_detection.dataset import VideoDataset, file_hash, read_manifest
from violence_detection.inference import load_checkpoint
from violence_detection.model import CLASS_NAMES


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader: DataLoader) -> dict:
    model.eval()
    device = next(model.parameters()).device
    truths, predictions = [], []
    loss_sum = 0.0
    for videos, labels in loader:
        logits = model(videos.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model returned non-finite logits during evaluation.")
        loss_sum += torch.nn.functional.cross_entropy(
            logits, labels.to(device), reduction="sum"
        ).item()
        truths.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not truths:
        raise ValueError("Cannot evaluate an empty split.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        truths, predictions, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truths, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(truths, predictions, labels=[0, 1]).tolist(),
        "class_names": CLASS_NAMES,
        "positive_class": "violence",
        "loss": loss_sum / len(truths),
        "count": len(truths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an untouched test split.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/test_metrics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if checkpoint["manifest_sha256"] != file_hash(args.manifest):
        raise ValueError("Manifest differs from training; cannot verify test isolation.")
    rows = read_manifest(args.manifest, args.root)
    test_rows = [row for row in rows if row["split"] == "test"]
    for row in test_rows:
        if file_hash(args.root / row["path"]) != row["sha256"]:
            raise ValueError(f"Test video changed since inspection: {row['path']}")
    loader = DataLoader(
        VideoDataset(args.root, test_rows, checkpoint["num_frames"]), batch_size=args.batch_size
    )
    metrics = evaluate_model(model, loader)
    metrics.update(
        {
            "split": "test",
            "checkpoint_sha256": file_hash(args.checkpoint),
            "manifest_sha256": file_hash(args.manifest),
            "confusion_matrix_axes": "rows=actual, columns=predicted",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
