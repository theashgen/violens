"""Continuous, bounded-memory camera monitoring around the existing MC3-18 model."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import av
import numpy as np
import torch

from violence_api.stream import encode_snapshot_jpeg, tensor_from_frame, validate_camera_url


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Camera:
    camera_id: str
    name: str
    url: str
    window_seconds: float = 4.0
    sample_fps: float = 4.0
    stride_seconds: float = 1.0
    positive_threshold: float = 0.65
    negative_threshold: float = 0.4
    positive_windows: int = 3
    negative_windows: int = 4
    frames: deque[tuple[float, torch.Tensor]] = field(default_factory=deque)
    state: str = "CONNECTING"
    connection_status: str = "CONNECTING"
    score: float | None = None
    last_prediction_time: str | None = None
    event: dict[str, Any] | None = None
    last_positive: float | None = None
    positives: int = 0
    negatives: int = 0
    dropped_windows: int = 0
    processed_fps: float = 0.0
    inference_latency_ms: float | None = None
    error: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_inference: float = 0.0
    pending: tuple[float, torch.Tensor] | None = None
    generation: int = 0
    snapshot: bytes | None = None
    first_frame_at: float | None = None
    last_frame_at: float | None = None

    def public(self) -> dict[str, Any]:
        with self.lock:
            return {
                "camera_id": self.camera_id,
                "camera_name": self.name,
                "connection_status": self.connection_status,
                "state": self.state,
                "violence_score": self.score,
                "last_prediction_time": self.last_prediction_time,
                "violence_event": dict(self.event) if self.event else None,
                "error": self.error,
                "processed_fps": round(self.processed_fps, 2),
                "inference_latency_ms": self.inference_latency_ms,
                "dropped_windows": self.dropped_windows,
                "buffered_frames": len(self.frames),
                "seconds_since_frame": (
                    time.monotonic() - self.last_frame_at
                    if self.last_frame_at is not None
                    else None
                ),
            }


class SurveillanceManager:
    """One capture thread per camera; shared serialized model inference; bounded queues."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_frames: int,
        infer_lock: threading.Lock,
        publish: Callable[[], None] | None = None,
    ):
        self.model, self.num_frames, self.infer_lock = model, num_frames, infer_lock
        self.publish = publish or (lambda: None)
        self.cameras: dict[str, Camera] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.RLock()

    def add(self, config: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(config.get("camera_id", "")).strip()
        url = str(config.get("url", "")).strip()
        if not camera_id or len(camera_id) > 80:
            raise ValueError("camera_id is required (max 80 characters).")
        if camera_id in self.cameras:
            raise ValueError(f"Camera {camera_id} already exists.")
        if url.startswith("file://"):
            if not os.path.isfile(url[7:]):
                raise ValueError("Local test-video path does not exist.")
        else:
            validate_camera_url(url)
        camera = Camera(
            camera_id=camera_id,
            name=str(config.get("name") or camera_id),
            url=url,
            window_seconds=float(config.get("window_seconds", 4)),
            sample_fps=float(config.get("sample_fps", 4)),
            stride_seconds=float(config.get("stride_seconds", 1)),
            positive_threshold=float(config.get("positive_threshold", 0.65)),
            negative_threshold=float(config.get("negative_threshold", 0.4)),
            positive_windows=int(config.get("positive_windows", 3)),
            negative_windows=int(config.get("negative_windows", 4)),
        )
        if not (
            0 < camera.sample_fps <= 30
            and 0 < camera.window_seconds <= 60
            and 0 < camera.stride_seconds <= 60
            and 0 <= camera.negative_threshold < camera.positive_threshold <= 1
            and camera.positive_windows >= 1
            and camera.negative_windows >= 1
        ):
            raise ValueError("Invalid camera window, sampling, threshold, or hysteresis settings.")
        with self.lock:
            if camera_id in self.cameras:
                raise ValueError("Camera ID already exists.")
            self.cameras[camera_id] = camera
            thread = threading.Thread(
                target=self._run, args=(camera,), name=f"camera-{camera_id}", daemon=True
            )
            self.threads[camera_id] = thread
            thread.start()
        self.publish()
        return camera.public()

    def _consume(self, camera: Camera) -> None:
        while not camera.stop.wait(0.02):
            # Acquire before taking the newest window; never queue stale clips.
            if not self.infer_lock.acquire(blocking=False):
                continue
            try:
                with camera.lock:
                    pending, camera.pending = camera.pending, None
                    generation = camera.generation
                if (
                    pending is not None
                    and time.monotonic() - pending[0] <= 2 * camera.stride_seconds
                ):
                    self._infer(camera, pending[1], locked=True, generation=generation)
            finally:
                self.infer_lock.release()

    def _run(self, camera: Camera) -> None:
        consumer = threading.Thread(target=self._consume, args=(camera,), daemon=True)
        consumer.start()
        backoff = 1.0
        while not camera.stop.is_set():
            try:
                options = (
                    {"rtsp_transport": "tcp", "stimeout": "5000000"}
                    if camera.url.startswith("rtsp")
                    else {}
                )
                local = camera.url.startswith("file://")
                source = camera.url[7:] if local else camera.url
                with av.open(source, options=options, timeout=5) as container:
                    if not container.streams.video:
                        raise ValueError("No video stream found")
                    stream = container.streams.video[0]
                    with camera.lock:
                        camera.connection_status, camera.error = "ONLINE", None
                        if camera.state in {"CONNECTING", "OFFLINE", "ERROR"}:
                            camera.state = "CONNECTING"
                    self.publish()
                    backoff = 1.0
                    last_sample = 0.0
                    playback_start = time.monotonic()
                    first_pts = None
                    for index, frame in enumerate(container.decode(stream)):
                        if local:
                            pts = frame.time
                            if pts is None:
                                pts = index / float(stream.average_rate or 25)
                            if first_pts is None:
                                first_pts = pts
                            if camera.stop.wait(
                                max(0, playback_start + pts - first_pts - time.monotonic())
                            ):
                                break
                        if camera.stop.is_set():
                            break
                        current = time.monotonic()
                        if frame.is_corrupt:
                            raise ValueError("Corrupted camera frame")
                        if frame.width * frame.height > 3840 * 2160:
                            raise ValueError("Camera resolution exceeds limit")
                        if current - last_sample < 1.0 / camera.sample_fps:
                            continue
                        last_sample = current
                        try:
                            rgb = frame.to_ndarray(format="rgb24")
                            tensor = tensor_from_frame(rgb)
                            preview = encode_snapshot_jpeg(rgb)
                        except Exception as exc:
                            raise ValueError("Frame preprocessing failed") from exc
                        with camera.lock:
                            camera.snapshot = preview
                            camera.last_frame_at = current
                            if camera.first_frame_at is None:
                                camera.first_frame_at = current
                            if camera.state == "CONNECTING":
                                camera.state = "BUFFERING"
                            camera.frames.append((current, tensor))
                            cutoff = current - camera.window_seconds
                            # Retain one sample preceding the boundary, even at low FPS.
                            while len(camera.frames) > 1 and camera.frames[1][0] <= cutoff:
                                camera.frames.popleft()
                            ready = (
                                len(camera.frames) >= 2
                                and current - camera.first_frame_at >= camera.window_seconds
                            )
                            due = current - camera.last_inference >= camera.stride_seconds
                            if ready and due:
                                # Latest frames only, no growing inference queue.
                                frames = list(camera.frames)
                                timestamps = np.array([item[0] for item in frames])
                                targets = np.linspace(cutoff, current, self.num_frames)
                                indices = np.abs(timestamps[:, None] - targets).argmin(axis=0)
                                chosen = [frames[i] for i in indices]
                                clip = torch.stack([item[1] for item in chosen], dim=1).unsqueeze(0)
                                camera.last_inference = current
                                if camera.pending is not None:
                                    camera.dropped_windows += 1
                                camera.pending = (current, clip)
                    if not camera.stop.is_set():
                        raise EOFError("Stream ended")
            except Exception as exc:  # stream failures are camera state, not process failures
                with camera.lock:
                    camera.connection_status = "OFFLINE"
                    camera.state = "OFFLINE"
                    camera.error = type(exc).__name__
                    camera.frames.clear()
                    camera.snapshot = None
                    camera.first_frame_at = camera.last_frame_at = None
                    camera.pending = None
                    camera.generation += 1
                    camera.score = None
                    camera.event = None
                    camera.positives = camera.negatives = 0
                self.publish()
            if camera.stop.wait(backoff):
                break
            backoff = min(backoff * 2, 15.0)

    def _infer(
        self,
        camera: Camera,
        clip: torch.Tensor,
        *,
        locked: bool = False,
        generation: int | None = None,
    ) -> None:
        started = time.perf_counter()
        if not locked and not self.infer_lock.acquire(timeout=2):
            with camera.lock:
                camera.dropped_windows += 1
            return
        try:
            with torch.inference_mode():
                output = self.model(clip.to(next(self.model.parameters()).device))
                score = float(output.softmax(dim=1)[0, 1].item())
            if not np.isfinite(score):
                raise RuntimeError("Non-finite model score")
        except Exception as exc:
            with camera.lock:
                camera.error = f"Inference error: {type(exc).__name__}"
                camera.state = "ERROR"
                camera.connection_status = "ERROR"
            self.publish()
            return
        finally:
            if not locked:
                self.infer_lock.release()
        stamp, elapsed = now_iso(), (time.perf_counter() - started) * 1000
        with camera.lock:
            if camera.stop.is_set() or (generation is not None and generation != camera.generation):
                return
            camera.connection_status = "ONLINE"
            previous = camera.state
            camera.score, camera.last_prediction_time = score, stamp
            camera.inference_latency_ms = round(elapsed, 1)
            camera.processed_fps = 1000 / elapsed if elapsed else 0
            camera.error = None
            if score >= camera.positive_threshold:
                camera.positives += 1
                camera.negatives = 0
                camera.last_positive = time.time()
            elif score <= camera.negative_threshold:
                camera.negatives += 1
                camera.positives = 0
            else:
                camera.positives = camera.negatives = 0
            if previous != "VIOLENCE" and camera.positives >= camera.positive_windows:
                camera.state = "VIOLENCE"
                camera.event = {
                    "camera_id": camera.camera_id,
                    "event_start": stamp,
                    "latest_detection": stamp,
                    "highest_score": score,
                    "status": "ACTIVE",
                }
            elif previous == "VIOLENCE":
                if camera.event:
                    camera.event["latest_detection"] = stamp
                    camera.event["highest_score"] = max(camera.event["highest_score"], score)
                if camera.negatives >= camera.negative_windows:
                    camera.state = "NORMAL"
                    if camera.event:
                        camera.event.update(status="CLOSED", event_end=stamp)
                    camera.event = None
            else:
                camera.state = "NORMAL"
        self.publish()

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            cameras = list(self.cameras.values())
        return [camera.public() for camera in cameras]

    def remove(self, camera_id: str) -> bool:
        with self.lock:
            camera = self.cameras.pop(camera_id, None)
            thread = self.threads.pop(camera_id, None)
        if camera is None:
            return False
        camera.stop.set()
        if thread:
            thread.join(timeout=6)
        self.publish()
        return True

    def stop_all(self) -> None:
        with self.lock:
            ids = list(self.cameras)
        for camera_id in ids:
            self.remove(camera_id)

    @classmethod
    def configured(
        cls,
        model: torch.nn.Module,
        num_frames: int,
        infer_lock: threading.Lock,
        publish: Callable[[], None] | None = None,
    ) -> SurveillanceManager:
        manager = cls(model, num_frames, infer_lock, publish)
        raw = os.getenv("CAMERAS_JSON", "[]")
        for item in json.loads(raw):
            manager.add(item)
        return manager
