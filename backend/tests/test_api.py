
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_metadata():
    response = client.get("/model/metadata")
    assert response.status_code == 200
    assert response.json()["model_name"] == "YOLO"
    assert response.json()["status"] == "Not loaded"
