import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send
from violence_detection.inference import load_checkpoint, predict_video
from violence_detection.video import EXTENSIONS, VideoError

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
    checkpoint = Path(os.getenv("CHECKPOINT_PATH", "artifacts/checkpoints/best.pt"))
    if checkpoint.is_file():
        # Invalid existing checkpoints fail startup visibly instead of serving random weights.
        app.state.model, app.state.metadata = load_checkpoint(
            checkpoint, os.getenv("MODEL_DEVICE", "auto")
        )
    yield
    app.state.model = None


app = FastAPI(title="Violence Detection · Phase 1", version="0.1.0", lifespan=lifespan)
app.add_middleware(BodyLimit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WEB_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Prediction(BaseModel):
    prediction: Literal["violence", "non-violence"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


@app.get("/health")
def health() -> dict:
    ready = app.state.model is not None
    return {
        "status": "ok",
        "model_ready": ready,
        "model": "mc3_18" if ready else None,
        "detail": "Ready" if ready else "Train a baseline and configure CHECKPOINT_PATH.",
    }


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
