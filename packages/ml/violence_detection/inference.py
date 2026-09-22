import argparse
import json
from pathlib import Path

import torch

from violence_detection.model import ARCHITECTURE, CLASS_NAMES, build_model, select_device
from violence_detection.video import PREPROCESSING, load_video


def load_checkpoint(path: Path, device: str = "auto") -> tuple:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("architecture") != ARCHITECTURE
        or checkpoint.get("class_names") != CLASS_NAMES
        or checkpoint.get("preprocessing") != PREPROCESSING
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Checkpoint architecture, classes, or preprocessing is incompatible.")
    if not 2 <= checkpoint["num_frames"] <= 64:
        raise ValueError("Invalid checkpoint frame count.")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(select_device(device)).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_video(video: Path, model: torch.nn.Module, num_frames: int) -> dict:
    tensor = load_video(video, num_frames).unsqueeze(0)
    device = next(model.parameters()).device
    probabilities = model(tensor.to(device)).softmax(dim=1)[0]
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Model returned non-finite probabilities.")
    confidence, index = probabilities.max(dim=0)
    return {"prediction": CLASS_NAMES[index.item()], "confidence": float(confidence.item())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained checkpoint on one video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    result = predict_video(args.video, model, metadata["num_frames"])
    print(f"Video: {args.video}\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
