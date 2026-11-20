from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main


class FakeInput:
    name = "images"


class FakeONNXSession:
    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, input_feed):
        # Simulate YOLO output: 4 box values + 12 class scores
        output = np.zeros((1, 16, 18900), dtype=np.float32)
        return [output]


@pytest.fixture(autouse=True)
def mock_onnx_model(monkeypatch):
    # The CI tests use a fake session; local production inference stays unchanged.
    monkeypatch.setattr(main, "session", FakeONNXSession())
    monkeypatch.setattr(main, "model_status", "loaded")


client = TestClient(main.app)


def create_test_image():
    image = Image.new("RGB", (640, 480), color=(120, 140, 160))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_status"] == "loaded"


def test_model_metadata():
    response = client.get("/model/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "YOLO11n-ONNX"
    assert data["model_status"] == "loaded"
    assert data["input_size"] == 960
    assert data["number_of_classes"] == 12
    assert data["runtime"] == "ONNX Runtime"
    assert data["provider"] == "CPUExecutionProvider"
    assert data["class_thresholds"]["no_goggle"] == 0.15


def test_predict():
    image = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image, "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["model"] == "YOLO11n-ONNX"
    assert "detections" in data
    assert data["total_detections"] == 0


def test_predict_visualize():
    image = create_test_image()

    response = client.post(
        "/predict/visualize",
        files={"file": ("test.jpg", image, "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "annotated_image" in data
    assert len(data["annotated_image"]) > 0


def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"this is not an image", "text/plain")}
    )

    assert response.status_code == 400