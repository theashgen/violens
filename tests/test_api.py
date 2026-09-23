import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from violence_api.app import app
from violence_api.stream import manager
from violence_detection.model import build_model


def jpeg_bytes(value: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), (value, value, value)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    with TestClient(app) as client:
        yield client


def test_camera_websocket_updates_without_checkpoint(client):
    with client.websocket_connect("/api/v1/cameras/ws") as websocket:
        assert websocket.receive_json() == {"cameras": []}
        assert websocket.receive_json() == {"cameras": []}


def test_camera_delete_cors(client):
    response = client.options(
        "/api/v1/cameras/test",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_unavailable_model(client, video):
    assert client.get("/health").json()["model_ready"] is False
    response = client.post("/api/v1/predict", files={"video": (video.name, video.read_bytes())})
    assert response.status_code == 503


def test_real_model_response_contract(client, video):
    # Random weights only test the HTTP/ML boundary; never a research checkpoint.
    app.state.model = build_model(pretrained=False).eval()
    app.state.metadata = {"num_frames": 16}
    response = client.post("/api/v1/predict", files={"video": (video.name, video.read_bytes())})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "confidence"}
    assert body["prediction"] in {"violence", "non-violence"}
    assert 0 <= body["confidence"] <= 1
    assert client.post("/api/v1/predict", files={"video": ("bad.txt", b"abc")}).status_code == 415
    assert client.post("/api/v1/predict", files={"video": ("bad.mp4", b"abc")}).status_code == 422
    assert client.post("/api/v1/predict", files={"video": ("empty.mp4", b"")}).status_code == 422


def test_busy_model(client, video):
    app.state.model = object()
    app.state.inference_lock.acquire()
    try:
        assert (
            client.post(
                "/api/v1/predict", files={"video": (video.name, video.read_bytes())}
            ).status_code
            == 429
        )
    finally:
        app.state.inference_lock.release()


def test_streamed_upload_limit(client, monkeypatch):
    monkeypatch.setattr("violence_api.app.MAX_UPLOAD_BYTES", 100)
    response = client.post("/api/v1/predict", files={"video": ("large.mp4", b"a" * 1_100_000)})
    assert response.status_code == 413


def test_stream_requires_model(client):
    assert client.post("/api/v1/stream/webcam/connect").status_code == 503


def test_unknown_source_returns_404(client):
    assert client.get("/api/v1/stream/webcam/status").status_code == 404
    assert client.post("/api/v1/stream/webcam/analyze").status_code == 404


def test_webcam_session_lifecycle(client):
    app.state.model = build_model(pretrained=False).eval()
    app.state.metadata = {"num_frames": 4}
    try:
        assert client.post("/api/v1/stream/webcam/connect").status_code == 200
        assert client.post("/api/v1/stream/webcam/connect").status_code == 409
        # Not enough buffered frames yet.
        assert client.post("/api/v1/stream/webcam/analyze").status_code == 409
        uploads = [
            ("frames", ("f.jpg", jpeg_bytes(value), "image/jpeg")) for value in (10, 60, 120, 200)
        ]
        for form in uploads:
            response = client.post("/api/v1/stream/webcam/frames", files=[form])
            assert response.status_code == 200
            time.sleep(0.05)  # Respect the 30 fps ingestion throttle.
        response = client.post("/api/v1/stream/webcam/analyze")
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in {"violence", "non-violence"}
        assert 0 <= body["confidence"] <= 1
        status = client.get("/api/v1/stream/webcam/status").json()
        assert status["source"] == "webcam"
        assert status["latest_prediction"]["prediction"] == body["prediction"]
        assert status["last_analysis_error"] is None
        snapshot = client.get("/api/v1/stream/webcam/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        # Frames are consumed, so the next analysis must buffer again and
        # records that condition as the latest analysis outcome.
        assert client.post("/api/v1/stream/webcam/analyze").status_code == 409
        status = client.get("/api/v1/stream/webcam/status").json()
        assert "Buffering" in status["last_analysis_error"]
        assert client.post("/api/v1/stream/webcam/stop").status_code == 200
        assert client.get("/api/v1/stream/webcam/status").status_code == 404
    finally:
        manager.stop_all()


def test_undecodable_frame_is_rejected(client):
    app.state.model = build_model(pretrained=False).eval()
    app.state.metadata = {"num_frames": 4}
    try:
        client.post("/api/v1/stream/webcam/connect")
        response = client.post(
            "/api/v1/stream/webcam/frames",
            files=[("frames", ("f.jpg", b"not jpeg", "image/jpeg"))],
        )
        assert response.status_code == 422
    finally:
        manager.stop_all()


def test_invalid_camera_urls_are_rejected(client):
    app.state.model = build_model(pretrained=False).eval()
    for url in ("file:///etc/passwd", "not a url", "rtsp://"):
        response = client.post("/api/v1/stream/rtsp/connect", json={"url": url})
        assert response.status_code == 422
    assert manager.get("rtsp") is None
