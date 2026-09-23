"""Streaming session state: bounded ring buffer of recent preprocessed frames.

Sources push frames continuously:
- webcam sessions receive RGB frame arrays decoded from browser JPEG uploads,
- RTSP/IP sessions are decoded server-side by :class:`RtspWorker`.

Analysis consumes the oldest ``num_frames`` buffered frames as one clip,
exactly matching the whole-video sampling semantics of the offline baseline.
"""

import threading
import time
from collections import deque
from io import BytesIO
from urllib.parse import urlsplit

import av
import numpy as np
import torch
from PIL import Image
from torchvision.models.video import MC3_18_Weights
from violence_detection.video import MAX_PIXELS, VideoError

MAX_FPS = 30
JPEG_SNAPSHOT_QUALITY = 70
ALLOWED_CAMERA_SCHEMES = {"rtsp", "rtmp", "http", "https", "udp"}
TRANSFORM = MC3_18_Weights.KINETICS400_V1.transforms()


def tensor_from_frame(frame: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWx3 array -> preprocessed 3x112x112, like video.load_video."""
    rgb = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True)).permute(2, 0, 1)
    return TRANSFORM(rgb.unsqueeze(0)).squeeze(1)


def validate_camera_url(url: str) -> str:
    """Reject anything that is not a direct camera stream URL."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_CAMERA_SCHEMES or not parts.hostname:
        raise ValueError("Provide an rtsp://, rtmp://, http://, https://, or udp:// camera URL.")
    return url


def frame_from_jpeg(body: bytes) -> np.ndarray:
    """Decode an uploaded JPEG frame; the browser remains the encoder."""
    try:
        with Image.open(BytesIO(body)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise VideoError("Frame exceeds the resolution limit.")
            return np.asarray(image.convert("RGB"))
    except (VideoError, OSError, ValueError) as exc:
        raise ValueError("The uploaded frame is not a decodable image.") from exc


def encode_snapshot_jpeg(frame: np.ndarray) -> bytes:
    with Image.fromarray(frame) as image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_SNAPSHOT_QUALITY)
        return buffer.getvalue()


class StreamSession:
    """Thread-safe ring buffer of preprocessed frames plus the latest snapshot."""

    def __init__(self, source: str, max_frames: int):
        self.source = source
        self.buffer: deque[torch.Tensor] = deque(maxlen=max_frames)
        self.lock = threading.Lock()
        self.dropped_frames = 0
        self.last_frame_at: float | None = None
        self.last_push_at = 0.0
        self.created_at = time.time()
        self.snapshot: bytes | None = None
        self.snapshot_at: float | None = None

    def push(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self.lock:
            if now - self.last_push_at < 1 / MAX_FPS:
                self.dropped_frames += 1
                return
            self.last_push_at = now
            self.buffer.append(tensor_from_frame(frame))
            self.last_frame_at = time.time()

    def take_clip(self, num_frames: int) -> torch.Tensor | None:
        """Return the oldest [1,C,T,H,W] clip, or None while the buffer is short.

        Frames are consumed: the next analysis covers only newer footage, so
        consecutive analyses never re-report the same incident.
        """
        with self.lock:
            if len(self.buffer) < num_frames:
                return None
            frames = [self.buffer.popleft() for _ in range(num_frames)]
        return torch.stack(frames, dim=1).unsqueeze(0)

    def set_snapshot(self, frame: np.ndarray) -> None:
        data = encode_snapshot_jpeg(frame)
        with self.lock:
            self.snapshot, self.snapshot_at = data, time.time()

    def get_snapshot(self) -> bytes | None:
        with self.lock:
            return self.snapshot

    @property
    def buffered(self) -> int:
        with self.lock:
            return len(self.buffer)


class RtspWorker:
    """Decodes a camera URL in a daemon thread, reconnecting with backoff."""

    def __init__(self, url: str, session: StreamSession):
        self.url = url
        self.session = session
        self.stop_event = threading.Event()
        self.last_error: str | None = None
        self.connected = False

    def run(self) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                # Never log or embed the URL: it can contain camera credentials.
                with av.open(
                    self.url,
                    options={"rtsp_transport": "tcp", "stimeout": "5000000"},
                ) as container:
                    if not container.streams.video:
                        raise VideoError("No video stream found.")
                    stream = container.streams.video[0]
                    if stream.codec_context.width * stream.codec_context.height > MAX_PIXELS:
                        raise VideoError("Camera exceeds the resolution limit.")
                    self.connected, self.last_error, backoff = True, None, 1.0
                    for frame in container.decode(stream):
                        if self.stop_event.is_set():
                            break
                        frame_array = frame.to_ndarray(format="rgb24")
                        self.session.push(frame_array)
                        self.session.set_snapshot(frame_array)
                    if not self.stop_event.is_set():
                        self.last_error = "Camera stream ended."
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                self.connected = False
                self.last_error = f"Camera error: {type(exc).__name__}."
            if self.stop_event.wait(backoff):
                break
            backoff = min(backoff * 2, 10.0)
        self.connected = False


class StreamManager:
    """Owns active sessions and their RTSP workers; the process is the unit of isolation."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions: dict[str, StreamSession] = {}
        self.workers: dict[str, RtspWorker] = {}

    def create(self, source: str, max_frames: int, url: str | None = None) -> StreamSession:
        with self.lock:
            self._evict_stale_locked()
            if self.sessions:
                raise RuntimeError("A streaming session is already active.")
            session = StreamSession(source, max_frames)
            if url is not None:
                worker = RtspWorker(url, session)
                thread = threading.Thread(target=worker.run, name=f"rtsp-{source}", daemon=True)
                self.workers[source] = worker
                thread.start()
            self.sessions[source] = session
            return session

    def get(self, source: str) -> StreamSession | None:
        with self.lock:
            return self.sessions.get(source)

    def active_source(self) -> str | None:
        with self.lock:
            return next(iter(self.sessions), None)

    def stop(self, source: str, join_timeout: float = 2.0) -> bool:
        with self.lock:
            worker = self.workers.pop(source, None)
            stopped = self.sessions.pop(source, None) is not None
        if worker is not None:
            worker.stop_event.set()
            worker.stop_event.wait(join_timeout)
        return stopped

    def stop_all(self) -> None:
        with self.lock:
            sources = list(self.sessions)
        for source in sources:
            self.stop(source)

    def _evict_stale_locked(self, timeout: float = 15.0) -> None:
        now = time.time()
        for source, session in list(self.sessions.items()):
            last = session.last_frame_at or session.created_at
            if now - last > timeout:
                worker = self.workers.pop(source, None)
                if worker is not None:
                    worker.stop_event.set()
                del self.sessions[source]


manager = StreamManager()
