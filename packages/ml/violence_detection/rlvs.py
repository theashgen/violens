"""RLVS-specific training and inference helpers.

This module intentionally leaves the existing baseline model and pipeline unchanged.
It provides an independent checkpoint format for the Real Life Violence Situations
(RLVS) dataset and a CCTV-friendly sliding-window predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from violence_detection.model import select_device
from violence_detection.video import load_video

RLVS_CLASS_NAMES = ["non-violence", "violence"]
RLVS_ARCHITECTURE = "r3d_18_rlvs"
RLVS_PREPROCESSING = "R3D_18_Weights.KINETICS400_V1.transforms;uniform-index-v1"


def build_rlvs_model(pretrained: bool = True) -> nn.Module:
    """Build an RLVS-only 3D ResNet checkpoint model."""
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(RLVS_CLASS_NAMES))
    return model


def load_rlvs_checkpoint(path: Path, device: str = "auto") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": 1,
        "architecture": RLVS_ARCHITECTURE,
        "class_names": RLVS_CLASS_NAMES,
        "preprocessing": RLVS_PREPROCESSING,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint is not an RLVS checkpoint or uses incompatible preprocessing.")
    model = build_rlvs_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(select_device(device)).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_window(video: Path, model: nn.Module, num_frames: int) -> dict:
    tensor = load_video(video, num_frames).unsqueeze(0)
    probabilities = model(tensor.to(next(model.parameters()).device)).softmax(dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    return {
        "prediction": RLVS_CLASS_NAMES[index.item()],
        "confidence": float(confidence),
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(RLVS_CLASS_NAMES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an RLVS-trained CCTV video window.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, metadata = load_rlvs_checkpoint(args.checkpoint, args.device)
    print(json.dumps(predict_window(args.video, model, metadata["num_frames"]), indent=2))


if __name__ == "__main__":
    main()
