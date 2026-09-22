from pathlib import Path

import av
import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def limit_threads():
    torch.set_num_threads(2)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "short.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for value in (30, 100, 180):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), value, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
