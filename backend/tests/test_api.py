from pathlib import Path

from fastapi.testclient import TestClient

import main


def _mock_status_connected() -> dict:
    return {
        "model_connected": True,
        "model_path": "mock_model.pth",
        "message": "Model connected.",
        "charset_source": "mock_charset.json",
        "charset_size": 30,
    }


def test_health_endpoint_returns_status_ok(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", _mock_status_connected)
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_connected"] is True
    assert "message" in payload


def test_predict_success_response_shape(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", _mock_status_connected)
    monkeypatch.setattr(main, "_get_runtime", lambda _path: (object(), None))

    def fake_predict_text(_runtime, _content: bytes):
        return "නම", 0.91, {"width": 320, "height": 64, "detected_lines": 1}

    monkeypatch.setattr(main, "_predict_text", fake_predict_text)
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files=[("files", ("sample.png", b"dummy-bytes", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_connected"] is True
    assert isinstance(payload.get("lines"), list)
    assert len(payload["lines"]) == 1

    item = payload["lines"][0]
    assert item["line_id"] == "sample.png"
    assert item["text"] == "නම"
    assert 0 <= item["confidence"] <= 1
    assert "meta" in item


def test_predict_missing_files_validation(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", _mock_status_connected)
    client = TestClient(main.app)

    response = client.post("/predict")

    assert response.status_code == 422


def test_predict_empty_file_gets_line_level_error(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", _mock_status_connected)
    monkeypatch.setattr(main, "_get_runtime", lambda _path: (object(), None))
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files=[("files", ("empty.png", b"", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["lines"]) == 1
    item = payload["lines"][0]
    assert item["line_id"] == "empty.png"
    assert item["text"] == ""
    assert item["confidence"] == 0.0
    assert item["error"] == "Empty file"


def test_predict_invalid_image_gets_line_level_error(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", _mock_status_connected)
    monkeypatch.setattr(main, "_get_runtime", lambda _path: (object(), None))

    def fake_predict_text(_runtime, _content: bytes):
        raise ValueError("Invalid image file")

    monkeypatch.setattr(main, "_predict_text", fake_predict_text)
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files=[("files", ("invalid.png", b"not-an-image", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["lines"]) == 1
    item = payload["lines"][0]
    assert item["line_id"] == "invalid.png"
    assert item["text"] == ""
    assert item["confidence"] == 0.0
    assert item["error"] == "Invalid image file"


def test_predict_returns_503_when_model_not_connected(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_status",
        lambda: {
            "model_connected": False,
            "model_path": None,
            "message": "Please connect with model.",
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files=[("files", ("sample.png", b"dummy-bytes", "image/png"))],
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Please connect with model."


def test_predict_returns_503_when_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", _mock_status_connected)
    monkeypatch.setattr(main, "_get_runtime", lambda _path: (None, "Model runtime is unavailable."))
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files=[("files", ("sample.png", b"dummy-bytes", "image/png"))],
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model runtime is unavailable."
