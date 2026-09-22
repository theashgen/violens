import pytest
from fastapi.testclient import TestClient
from violence_api.app import app
from violence_detection.model import build_model


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    with TestClient(app) as client:
        yield client


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
