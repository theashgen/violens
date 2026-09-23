import threading
import time

import av
import numpy as np
import pytest
import torch
from torch import nn
from violence_api.surveillance import Camera, SurveillanceManager


class FixedModel(nn.Module):
    def __init__(self, violence_score: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.violence_score = violence_score

    def forward(self, clip):
        return torch.tensor([[1 - self.violence_score, self.violence_score]], device=clip.device)


def manager(score: float) -> SurveillanceManager:
    return SurveillanceManager(FixedModel(score), 2, threading.Lock())


def test_hysteresis_opens_and_closes_event():
    service = manager(0.9)
    camera = Camera(
        "one", "Camera one", "rtsp://localhost/stream", positive_windows=2, negative_windows=2
    )
    clip = torch.zeros(1, 3, 2, 112, 112)
    service._infer(camera, clip)
    assert camera.state == "NORMAL"
    service._infer(camera, clip)
    assert camera.state == "VIOLENCE"
    assert camera.event and camera.event["status"] == "ACTIVE"

    service.model = FixedModel(0.1)
    service._infer(camera, clip)
    assert camera.state == "VIOLENCE"
    service._infer(camera, clip)
    assert camera.state == "NORMAL"
    assert camera.event is None


def test_old_prediction_cannot_overwrite_disconnected_state():
    service = manager(0.9)
    camera = Camera("one", "One", "rtsp://localhost/1")
    camera.state = "OFFLINE"
    camera.generation = 1
    service._infer(camera, torch.zeros(1, 3, 2, 112, 112), generation=0)
    assert camera.state == "OFFLINE"
    assert camera.score is None


@pytest.mark.parametrize("source_fps", [2, 10])
def test_local_capture_is_paced_and_predictions_are_continuous(tmp_path, source_fps):
    path = tmp_path / "camera.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=source_fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for _ in range(30):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    service = manager(0.9)
    try:
        service.add(
            {
                "camera_id": "test",
                "url": path.as_uri(),
                "window_seconds": 0.5,
                "sample_fps": 10,
                "stride_seconds": 0.1,
            }
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if service.snapshot()[0]["last_prediction_time"]:
                break
            time.sleep(0.02)
        status = service.snapshot()[0]
        assert status["connection_status"] == "ONLINE"
        assert status["last_prediction_time"] is not None
        assert status["buffered_frames"] <= 7
        assert service.cameras["test"].snapshot.startswith(b"\xff\xd8")
    finally:
        service.stop_all()


def test_camera_ids_have_independent_states():
    service = manager(0.9)
    first = Camera("one", "One", "rtsp://localhost/1")
    second = Camera("two", "Two", "rtsp://localhost/2")
    clip = torch.zeros(1, 3, 2, 112, 112)
    first.positives = first.positive_windows - 1
    service._infer(first, clip)
    assert first.state == "VIOLENCE"
    assert second.state == "CONNECTING"
