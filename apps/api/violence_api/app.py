import asyncio
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from violence_detection.inference import load_checkpoint, predict_video
from violence_detection.video import EXTENSIONS, VideoError

from violence_api.stream import (
    MAX_FPS,
    StreamSession,
    frame_from_jpeg,
    manager,
    validate_camera_url,
)
from violence_api.surveillance import SurveillanceManager

STREAM_MAX_FRAMES = 64
STREAM_STATUS_INTERVAL = 5.0

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
logger = logging.getLogger(__name__)


class BodyLimit:
    """Limit streamed multipart bodies before the parser writes them to disk."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/api/v1/predict":
            await self.app(scope, receive, send)
            return
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise HTTPException(413, "Upload exceeds 100 MiB.")
            return message

        await self.app(scope, bounded_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
    app.state.model = None
    app.state.metadata = None
    app.state.inference_lock = threading.Lock()
    app.state.stream_predictions = {}
    app.state.stream_errors = {}
    checkpoint = Path(os.getenv("CHECKPOINT_PATH", "artifacts/checkpoints/best.pt"))
    if checkpoint.is_file():
        # Invalid existing checkpoints fail startup visibly instead of serving random weights.
        app.state.model, app.state.metadata = load_checkpoint(
            checkpoint, os.getenv("MODEL_DEVICE", "auto")
        )
    app.state.surveillance = (
        SurveillanceManager.configured(
            app.state.model, app.state.metadata["num_frames"], app.state.inference_lock
        )
        if app.state.model is not None
        else None
    )
    yield
    if app.state.surveillance is not None:
        app.state.surveillance.stop_all()
    app.state.model = None
    manager.stop_all()


app = FastAPI(title="Violence Detection · Phase 1", version="0.1.0", lifespan=lifespan)
app.add_middleware(BodyLimit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WEB_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class Prediction(BaseModel):
    prediction: Literal["violence", "non-violence"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class CameraConnect(BaseModel):
    url: str


class StreamStatus(BaseModel):
    source: Literal["webcam", "rtsp"]
    active: bool
    buffered_frames: int
    frames_needed: int
    dropped_frames: int
    latest_prediction: Prediction | None
    analyzing: bool
    last_analysis_error: str | None
    connected: bool | None = None
    last_error: str | None = None
    seconds_since_frame: float | None = Field(default=None, ge=0)
    interval_seconds: float = STREAM_STATUS_INTERVAL


def require_session(source: str) -> StreamSession:
    session = manager.get(source)
    if session is None:
        raise HTTPException(404, "No active stream for this source. Connect first.")
    return session


def analyze_stream_clip(source: str) -> dict:
    """One analysis cycle: pop a buffered clip and classify it."""
    session = require_session(source)
    if app.state.model is None:
        raise HTTPException(503, "No trained checkpoint loaded.")
    if not app.state.inference_lock.acquire(blocking=False):
        raise HTTPException(429, "The model is busy. Retry shortly.")
    try:
        clip = session.take_clip(app.state.metadata["num_frames"])
        if clip is None:
            buffered, needed = session.buffered, app.state.metadata["num_frames"]
            raise HTTPException(
                409,
                f"Buffering {buffered}/{needed} frames. Keep the camera on.",
            )
        device = next(app.state.model.parameters()).device
        logits = app.state.model(clip.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model returned non-finite logits during streaming analysis.")
        probabilities = logits.softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        return {
            "prediction": ("non-violence", "violence")[index.item()],
            "confidence": float(confidence.item()),
        }
    except VideoError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Streaming model inference failed")
        raise HTTPException(500, "Model inference failed. Check the API logs.") from exc
    finally:
        app.state.inference_lock.release()


@app.get("/api/v1/cameras")
def list_cameras() -> list[dict]:
    if app.state.surveillance is None:
        return []
    return app.state.surveillance.snapshot()


@app.post("/api/v1/cameras")
def add_camera(camera: dict) -> dict:
    if app.state.surveillance is None:
        raise HTTPException(503, "No trained checkpoint loaded.")
    try:
        return app.state.surveillance.add(camera)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/api/v1/cameras/{camera_id}")
def remove_camera(camera_id: str) -> dict:
    if app.state.surveillance is None:
        raise HTTPException(503, "No trained checkpoint loaded.")
    if not app.state.surveillance.remove(camera_id):
        raise HTTPException(404, "Camera not found.")
    return {"removed": camera_id}


@app.get("/api/v1/cameras/{camera_id}/preview")
async def camera_preview(camera_id: str) -> StreamingResponse:
    service = app.state.surveillance
    if service is None:
        raise HTTPException(503, "No trained checkpoint loaded.")
    with service.lock:
        camera = service.cameras.get(camera_id)
    if camera is None:
        raise HTTPException(404, "Camera not found.")

    async def frames():
        previous = None
        while not camera.stop.is_set():
            with camera.lock:
                data = camera.snapshot
            if data is not None and data is not previous:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(data)).encode()
                    + b"\r\n\r\n"
                    + data
                    + b"\r\n"
                )
                previous = data
            await asyncio.sleep(0.1)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.websocket("/api/v1/cameras/ws")
async def camera_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            cameras = app.state.surveillance.snapshot() if app.state.surveillance else []
            await websocket.send_json({"cameras": cameras})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@app.get("/health")
def health() -> dict:
    ready = app.state.model is not None
    return {
        "status": "ok",
        "model_ready": ready,
        "model": "mc3_18" if ready else None,
        "detail": "Ready" if ready else "Train a baseline and configure CHECKPOINT_PATH.",
    }


@app.post("/api/v1/stream/webcam/connect")
def connect_webcam() -> dict:
    """Open a webcam session; the browser pushes JPEG keyframes to it."""
    if app.state.model is None:
        raise HTTPException(503, "No trained checkpoint loaded.")
    try:
        manager.create("webcam", STREAM_MAX_FRAMES)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "source": "webcam",
        "max_fps": MAX_FPS,
        "frames_needed": app.state.metadata["num_frames"] if app.state.metadata else 16,
    }


@app.post("/api/v1/stream/rtsp/connect")
def connect_rtsp(camera: CameraConnect) -> dict:
    """Open an RTSP/IP-camera session decoded server-side."""
    if app.state.model is None:
        raise HTTPException(503, "No trained checkpoint loaded.")
    try:
        validate_camera_url(camera.url)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        manager.create("rtsp", STREAM_MAX_FRAMES, url=camera.url)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    frames_needed = app.state.metadata["num_frames"] if app.state.metadata else 16
    return {"source": "rtsp", "frames_needed": frames_needed}


@app.get("/api/v1/stream/active")
def active_stream() -> dict:
    source = manager.active_source()
    return {"source": source}


@app.post("/api/v1/stream/{source}/frames")
def push_frames(
    source: str,
    frames: Annotated[list[UploadFile] | None, File()] = None,
) -> dict:
    """Accept periodic JPEG keyframes from the browser capturer."""
    if not frames:
        raise HTTPException(422, "Attach at least one frame.")
    session = require_session(source)
    accepted = 0
    try:
        for frame in frames:
            decoded = frame_from_jpeg(frame.file.read())
            session.push(decoded)
            session.set_snapshot(decoded)  # feeds the snapshot/debug stream too
            accepted += 1
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        for frame in frames:
            frame.file.close()
    return {"accepted": accepted, "buffered": session.buffered}


@app.get("/api/v1/stream/{source}/snapshot")
def snapshot(source: str) -> Response:
    session = require_session(source)
    data = session.get_snapshot()
    if data is None:
        raise HTTPException(404, "No frame captured yet.")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/v1/stream/{source}/analyze", response_model=Prediction)
def analyze_stream(source: str) -> dict:
    try:
        result = analyze_stream_clip(source)
    except HTTPException as exc:
        app.state.stream_errors[source] = str(exc.detail)
        raise
    app.state.stream_errors[source] = None
    app.state.stream_predictions[source] = result
    return result


@app.get("/api/v1/stream/{source}/status", response_model=StreamStatus)
def stream_status(source: str) -> dict:
    session = require_session(source)
    worker = manager.workers.get(source)
    seconds_since_frame = (
        time.time() - session.last_frame_at if session.last_frame_at is not None else None
    )
    return {
        "source": source,
        "active": True,
        "buffered_frames": session.buffered,
        "frames_needed": app.state.metadata["num_frames"] if app.state.metadata else 16,
        "dropped_frames": session.dropped_frames,
        "latest_prediction": getattr(app.state, "stream_predictions", {}).get(source),
        "analyzing": app.state.inference_lock.locked(),
        "last_analysis_error": getattr(app.state, "stream_errors", {}).get(source),
        "connected": worker.connected if worker is not None else None,
        "last_error": worker.last_error if worker is not None else None,
        "seconds_since_frame": seconds_since_frame,
    }


@app.post("/api/v1/stream/{source}/stop")
def stop_stream(source: str) -> dict:
    if not manager.stop(source):
        raise HTTPException(404, "No active stream for this source.")
    app.state.stream_predictions.pop(source, None)
    app.state.stream_errors.pop(source, None)
    return {"stopped": source}


@app.get("/api/v1/stream/mjpeg")
def mjpeg(source: str) -> StreamingResponse:
    """Optional debug view of the exact frames being analyzed."""
    session = require_session(source)
    boundary = "frame"

    def generate():
        while manager.get(source) is session:
            data = session.get_snapshot()
            if data is not None:
                header = b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                yield header + data + b"\r\n"
            time.sleep(0.2)

    return StreamingResponse(
        generate(), media_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )


@app.post("/api/v1/predict", response_model=Prediction)
def predict(video: UploadFile) -> dict:
    if app.state.model is None:
        raise HTTPException(
            503, "No trained checkpoint loaded. Train offline, then restart the API."
        )
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in EXTENSIONS:
        raise HTTPException(415, "Supported formats: MP4, AVI, MOV, MKV, WebM.")
    if video.size is not None and video.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds 100 MiB.")
    if not app.state.inference_lock.acquire(blocking=False):
        raise HTTPException(429, "The model is analyzing another video. Try again shortly.")
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            total = 0
            while chunk := video.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Upload exceeds 100 MiB.")
                temporary.write(chunk)
            if not total:
                raise HTTPException(422, "The uploaded video is empty.")
            temporary.flush()
            return predict_video(
                Path(temporary.name), app.state.model, app.state.metadata["num_frames"]
            )
    except VideoError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Model inference failed")
        raise HTTPException(500, "Model inference failed. Check the API logs.") from exc
    finally:
        video.file.close()
        app.state.inference_lock.release()
