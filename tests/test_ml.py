from pathlib import Path

import pytest
import torch
from violence_detection.dataset import (
    VideoDataset,
    create_splits,
    label_from_path,
    validate_records,
)
from violence_detection.inference import load_checkpoint, predict_video
from violence_detection.model import ARCHITECTURE, CLASS_NAMES, build_model
from violence_detection.video import PREPROCESSING, VideoError, inspect_video, load_video


@pytest.mark.parametrize(
    "folder,label", [("Fight", 1), ("NonFight", 0), ("Violence", 1), ("Non-Violence", 0)]
)
def test_labels(folder, label):
    assert label_from_path(Path("collection") / "train" / folder / "clip.mp4") == label


def test_unknown_label():
    with pytest.raises(ValueError, match="class folder"):
        label_from_path(Path("unknown/clip.mp4"))


def test_preprocessing_and_short_video(video):
    tensor = load_video(video)
    assert tensor.shape == (3, 16, 112, 112)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert torch.equal(tensor[:, 0], tensor[:, 1])
    assert not torch.equal(tensor[:, 0], tensor[:, -1])
    dataset = VideoDataset(video.parent, [{"path": video.name, "label": 1}], 16)
    training_tensor, label = dataset[0]
    assert label == 1 and torch.equal(training_tensor, tensor)
    assert inspect_video(video)["frames"] == 3


def test_corrupt_video(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(VideoError, match="Cannot decode"):
        load_video(path)


def test_model_and_checkpoint_roundtrip(video, tmp_path):
    model = build_model(pretrained=False).eval()
    with torch.inference_mode():
        assert model(load_video(video).unsqueeze(0)).shape == (1, 2)
    checkpoint = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "class_names": CLASS_NAMES,
        "preprocessing": PREPROCESSING,
        "num_frames": 16,
        "model_state": model.state_dict(),
    }
    path = tmp_path / "test-only-random-weights.pt"
    torch.save(checkpoint, path)
    loaded, metadata = load_checkpoint(path, "cpu")
    assert predict_video(video, loaded, metadata["num_frames"]) == predict_video(video, model, 16)
    checkpoint["class_names"] = list(reversed(CLASS_NAMES))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path)


def records():
    return [{"path": f"{i}.mp4", "label": i % 2, "split": "", "sha256": str(i)} for i in range(100)]


def test_deterministic_splits_and_official_validation():
    report = {"videos": records()}
    first = create_splits(report, 42)
    assert first == create_splits(report, 42)
    assert {r["split"] for r in first} == {"train", "val", "test"}
    for row in report["videos"]:
        row["split"] = "val" if int(row["sha256"]) >= 80 else "train"
    result = create_splits(report, 42)
    assert all(r["split"] == "val" for r in result if int(r["sha256"]) >= 80)


def test_related_groups_stay_together():
    rows = records()
    groups = {r["path"]: str(int(r["sha256"]) // 2) for r in rows}
    result = create_splits({"videos": rows}, 42, groups)
    validate_records(result)
    for group in groups.values():
        assert len({r["split"] for r in result if r["group"] == group}) == 1


def test_cross_split_duplicates_rejected():
    rows = records()
    rows[0]["split"] = "train"
    rows[1].update(sha256="0", split="val", label=0)
    with pytest.raises(ValueError, match="Duplicate"):
        create_splits({"videos": rows}, 42)
