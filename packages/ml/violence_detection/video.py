"""Decode bounded clips and apply the pretrained weights' spatial transform."""

import argparse
from pathlib import Path

import av
import numpy as np
import torch
from torchvision.models.video import MC3_18_Weights

EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_SECONDS = 120
MAX_FRAMES = 12_000
MAX_PIXELS = 3840 * 2160
PREPROCESSING = "MC3_18_Weights.KINETICS400_V1.transforms;uniform-index-v1"


class VideoError(ValueError):
    """The video cannot be decoded or exceeds the baseline's limits."""


def inspect_video(path: str | Path) -> dict:
    """Fully decode without retaining frames, detecting errors beyond the first frame."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError("No video stream found.")
            stream = container.streams.video[0]
            if stream.codec_context.width * stream.codec_context.height > MAX_PIXELS:
                raise VideoError("Video exceeds the resolution limit.")
            fps = float(stream.average_rate or 0)
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else None
            )
            if duration and duration > MAX_SECONDS:
                raise VideoError(f"Video exceeds {MAX_SECONDS} seconds.")
            count = 0
            width = height = 0
            first_time = None
            last_time = None
            for frame in container.decode(stream):
                count += 1
                width, height = frame.width, frame.height
                if width * height > MAX_PIXELS or count > MAX_FRAMES:
                    raise VideoError("Video exceeds the resolution or decoded-frame limit.")
                if frame.is_corrupt:
                    raise VideoError("Video contains a corrupted frame.")
                if frame.time is not None:
                    first_time = frame.time if first_time is None else first_time
                    last_time = frame.time
                    if last_time - first_time > MAX_SECONDS:
                        raise VideoError(f"Video exceeds {MAX_SECONDS} seconds.")
            if not count:
                raise VideoError("Video contains no decodable frames.")
            if duration is None:
                duration = count / fps if fps else None
            return {
                "frames": count,
                "duration_seconds": duration,
                "fps": fps,
                "width": width,
                "height": height,
            }
    except (av.FFmpegError, OSError) as exc:
        raise VideoError(f"Cannot decode {Path(path).name}: {exc}") from exc


def load_video(path: str | Path, num_frames: int = 16) -> torch.Tensor:
    """Return float32 [C,T,H,W]; duplicate sampled indices for short clips.

    A count pass avoids trusting unreliable container frame counts. Only sampled
    frames are converted to RGB and retained; sequential decode handles interframes.
    """
    if not 2 <= num_frames <= 64:
        raise ValueError("num_frames must be between 2 and 64.")
    count = inspect_video(path)["frames"]
    indices = np.linspace(0, count - 1, num_frames).astype(int).tolist()
    selected: dict[int, torch.Tensor] = {}
    wanted = set(indices)
    transform = MC3_18_Weights.KINETICS400_V1.transforms()
    try:
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                    selected[index] = transform(rgb.unsqueeze(0)).squeeze(1)
                if index == indices[-1]:
                    break
    except (av.FFmpegError, OSError) as exc:
        raise VideoError(f"Cannot sample {Path(path).name}: {exc}") from exc
    if len(selected) != len(wanted):
        raise VideoError("Frame count changed between decoding passes.")
    return torch.stack([selected[index] for index in indices], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one video → float32 [C,T,H,W].")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()
    tensor = load_video(args.video, args.num_frames)
    print(f"Video: {args.video}\nShape: {list(tensor.shape)}\nDtype: {tensor.dtype}")


if __name__ == "__main__":
    main()
