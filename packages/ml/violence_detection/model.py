import torch
from torch import nn
from torchvision.models.video import MC3_18_Weights, mc3_18

CLASS_NAMES = ["non-violence", "violence"]
ARCHITECTURE = "mc3_18"


def build_model(*, pretrained: bool = True) -> nn.Module:
    model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps.")
    return torch.device(requested)
