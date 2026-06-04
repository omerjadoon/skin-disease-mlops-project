"""
Tests for the FastAPI inference API.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_image(width: int = 224, height: int = 224) -> bytes:
    """Create a minimal JPEG image for testing."""
    img = Image.new("RGB", (width, height), color=(180, 120, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─── Mock model loader ────────────────────────────────────────


@pytest.fixture
def mock_model_loader():
    """Patch the model loader to return a dummy model."""
    mock_loader = MagicMock()
    mock_loader.is_loaded = True
    mock_loader.model_name = "skin-severity-classifier"
    mock_loader.model_version = "test-v1"
    mock_loader.model_info = {
        "model_name": "skin-severity-classifier",
        "model_version": "test-v1",
        "backbone": "resnet18",
        "num_classes": 3,
        "class_names": ["mild", "moderate", "severe"],
        "input_size": [224, 224],
        "mlflow_tracking_uri": None,
    }

    # Mock the model's forward pass to return dummy logits
    import torch

    dummy_logits = torch.tensor([[0.1, 0.7, 0.2]])
    mock_model = MagicMock()
    mock_model.return_value = dummy_logits
    mock_model.eval = MagicMock(return_value=None)
    mock_loader.get_model.return_value = mock_model

    return mock_loader


@pytest.fixture
def client(mock_model_loader):
    """Create TestClient with mocked model loader."""
    with (
        patch("api.model_loader._model_loader", mock_model_loader),
        patch("api.main.get_model_loader", return_value=mock_model_loader),
        patch("api.main.log_prediction_to_db"),
    ):
        from api.main import app

        with TestClient(app) as c:
            yield c


# ─── Tests ────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "model_version" in data

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["model_name"] == "skin-severity-classifier"


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_disclaimer(self, client):
        data = client.get("/").json()
        assert "disclaimer" in data
        disclaimer = data["disclaimer"].lower()
        assert "not a medical diagnosis" in disclaimer or "ai-assisted" in disclaimer


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_schema(self, client):
        data = client.get("/model-info").json()
        assert "model_name" in data
        assert "backbone" in data
        assert "num_classes" in data
        assert "class_names" in data
        assert "disclaimer" in data
        assert data["num_classes"] == 3
        assert data["class_names"] == ["mild", "moderate", "severe"]


class TestPredictEndpoint:
    def test_predict_with_valid_image(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        image_bytes = create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        ).json()
        assert "severity" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "model_name" in data
        assert "model_version" in data
        assert "disclaimer" in data

    def test_predict_severity_is_valid_class(self, client):
        image_bytes = create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        ).json()
        assert data["severity"] in ["mild", "moderate", "severe"]

    def test_predict_confidence_in_range(self, client):
        image_bytes = create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, client):
        image_bytes = create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        ).json()
        probs = data["class_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_disclaimer_present(self, client):
        image_bytes = create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        ).json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 10

    def test_predict_empty_file_returns_400(self, client):
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_predict_no_model_returns_503(self):
        """When no model is loaded, /predict should return 503."""
        mock_loader = MagicMock()
        mock_loader.is_loaded = False
        mock_loader.model_name = "skin-severity-classifier"
        mock_loader.model_version = "none"

        with (
            patch("api.model_loader._model_loader", mock_loader),
            patch("api.main.get_model_loader", return_value=mock_loader),
        ):
            from api.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                image_bytes = create_test_image()
                response = c.post(
                    "/predict",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                )
                assert response.status_code == 503
