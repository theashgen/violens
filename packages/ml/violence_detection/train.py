"""Offline supervised transfer learning. The test split is never loaded here."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from violence_detection.dataset import VideoDataset, file_hash, read_manifest
from violence_detection.evaluate import evaluate_model
from violence_detection.model import ARCHITECTURE, CLASS_NAMES, build_model, select_device
from violence_detection.video import PREPROCESSING


def train(config: dict) -> None:
    settings, data = config["training"], config["data"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["learning_rate"] <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive.")
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = select_device(settings["device"])
    root, manifest = Path(data["root"]), Path(data["manifest"])
    rows = read_manifest(manifest, root)
    # Verify the inspected bytes, not just filenames, before fitting anything.
    for row in rows:
        if row["split"] != "test" and file_hash(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Video changed since inspection: {row['path']}")
    loaders = {
        split: DataLoader(
            VideoDataset(root, [r for r in rows if r["split"] == split], data["num_frames"]),
            batch_size=settings["batch_size"],
            shuffle=split == "train",
            num_workers=settings["num_workers"],
            generator=torch.Generator().manual_seed(seed),
        )
        for split in ("train", "val")
    }
    model = build_model(pretrained=True).to(device)
    if settings["freeze_backbone"]:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"],
    )
    checkpoint_path = Path(config["output"]["checkpoint"])
    history_path = Path(config["output"]["history"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, history = float("inf"), []
    print(
        f"Device: {device}; train={sum(r['split'] == 'train' for r in rows)}; "
        f"val={sum(r['split'] == 'val' for r in rows)}",
        flush=True,
    )
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        if settings["freeze_backbone"]:
            model.eval()  # Keep pretrained batch-normalization statistics frozen too.
        loss_sum = 0.0
        count = 0
        for step, (videos, labels) in enumerate(loaders["train"], start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(videos.to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; no checkpoint saved.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            if step % 25 == 0:
                print(f"Epoch {epoch}: {count} training videos processed", flush=True)
        print(f"Epoch {epoch}: validating held-out validation videos", flush=True)
        validation = evaluate_model(model, loaders["val"])
        entry = {"epoch": epoch, "train_loss": loss_sum / count, "validation": validation}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "format_version": 1,
                    "architecture": ARCHITECTURE,
                    "class_names": CLASS_NAMES,
                    "preprocessing": PREPROCESSING,
                    "num_frames": data["num_frames"],
                    "pretrained_weights": "MC3_18_Weights.KINETICS400_V1",
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config,
                    "epoch": epoch,
                    "validation": validation,
                    "manifest_sha256": file_hash(manifest),
                    "torch_version": str(torch.__version__),
                    "torchvision_version": str(torchvision.__version__),
                },
                checkpoint_path,
            )
        history_path.write_text(json.dumps(history, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    train(yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
