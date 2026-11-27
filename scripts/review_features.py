"""Extract frame fingerprints and frozen MC3-18 features for the reviewer study."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
from scipy.fft import dctn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import MC3_18_Weights
from violence_detection.model import build_model, select_device

SAMPLES = 12


def decode(root, record, architecture):
    path = root / record["path"]
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != record["sha256"]:
            raise ValueError(f"Unexpected file hash: {path}")
    frames = record["frames"]
    image_indices = np.linspace(0, frames - 1, 16).astype(int)
    hash_indices = np.linspace(0, frames - 1, SAMPLES).astype(int)
    images, hashes = {}, {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in hash_indices:
                grey = frame.reformat(width=32, height=32, format="gray").to_ndarray()
                low = dctn(grey.astype(float), norm="ortho")[:8, :8].ravel()[1:]
                hashes[i] = low > np.median(low)
            if i in image_indices:
                images[i] = frame.to_ndarray(format="rgb24")
    if i + 1 != frames:
        raise ValueError(f"Frame count changed: {path}")
    raw = torch.from_numpy(np.stack([images[j] for j in image_indices])).permute(0, 3, 1, 2)
    weights = (
        MC3_18_Weights.KINETICS400_V1 if architecture == "mc3" else ResNet18_Weights.IMAGENET1K_V1
    )
    transform = weights.transforms()
    return transform(raw), np.stack([hashes[j] for j in hash_indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/review"))
    parser.add_argument("--architecture", choices=["mc3", "resnet"], default="mc3")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path("artifacts/metrics/rlvs-inspection.json").read_text())
    records = audit["videos"]
    root = Path("data/raw/Violence Fight Detection dataset/RLVS")
    torch.set_num_threads(2)
    torch.hub.set_dir(str(Path("artifacts/torch/hub").resolve()))
    device = select_device()
    model = (
        build_model() if args.architecture == "mc3" else resnet18(weights=ResNet18_Weights.DEFAULT)
    )
    model = model.to(device).eval()
    model.fc = torch.nn.Identity()
    print(f"Extracting {len(records)} clips on {device}", flush=True)
    # Resume independently saved chunks after an interrupted run.
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for start in range(0, len(records), 24):
            prefix = "features" if args.architecture == "mc3" else "resnet"
            batch_path = output / f"{prefix}-{start:04}.npz"
            if batch_path.exists():
                continue
            inputs = list(
                pool.map(
                    lambda row: decode(root, row, args.architecture), records[start : start + 24]
                )
            )
            features = []
            for begin in range(0, len(inputs), 4):
                batch = torch.stack([item[0] for item in inputs[begin : begin + 4]])
                batch = batch.to(device)
                if args.architecture == "mc3":
                    pooled = model(batch)
                else:
                    pooled = model(batch.flatten(0, 1)).reshape(len(batch), 16, 512).mean(dim=1)
                features.append(pooled.cpu().numpy())
            temp_path = batch_path.with_suffix(".tmp.npz")
            np.savez(
                temp_path,
                features=np.concatenate(features),
                hashes=np.stack([x[1] for x in inputs]),
            )
            os.replace(temp_path, batch_path)
            if start % 120 == 0:
                print(f"{min(start + 24, len(records))}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
