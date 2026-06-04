"""
Tests for inference pipeline and drift detection.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_image_bytes(width: int = 224, height: int = 224) -> bytes:
    img = Image.new("RGB", (width, height), color=(180, 120, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestInferencePipeline:
    def test_preprocess_image_shape(self):
        from api.inference import preprocess_image
        image_bytes = create_test_image_bytes()
        tensor, image_hash = preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(image_hash, str)
        assert len(image_hash) == 32

    def test_preprocess_same_image_same_hash(self):
        from api.inference import preprocess_image
        image_bytes = create_test_image_bytes()
        _, hash1 = preprocess_image(image_bytes)
        _, hash2 = preprocess_image(image_bytes)
        assert hash1 == hash2

    def test_run_inference_output_format(self):
        from unittest.mock import MagicMock
        from api.inference import run_inference

        dummy_logits = torch.tensor([[2.0, 1.0, 0.5]])
        mock_model = MagicMock(return_value=dummy_logits)
        mock_model.eval = MagicMock()

        tensor = torch.randn(1, 3, 224, 224)
        result = run_inference(
            model=mock_model,
            image_tensor=tensor,
            class_names=["mild", "moderate", "severe"],
            model_name="test-model",
            model_version="1",
        )
        assert "severity" in result
        assert "confidence" in result
        assert "class_probabilities" in result
        assert result["severity"] in ["mild", "moderate", "severe"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert abs(sum(result["class_probabilities"].values()) - 1.0) < 0.01

    def test_run_inference_disclaimer_present(self):
        from unittest.mock import MagicMock
        from api.inference import run_inference

        mock_model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 0.5]]))
        tensor = torch.randn(1, 3, 224, 224)
        result = run_inference(mock_model, tensor)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10


class TestDriftDetection:
    def _make_preds(self, class_name: str, n: int) -> list[dict]:
        return [{"predicted_class": class_name, "confidence": 0.9} for _ in range(n)]

    def test_no_drift_same_distribution(self):
        from monitoring.drift import detect_drift
        baseline = self._make_preds("mild", 50) + self._make_preds("moderate", 30) + self._make_preds("severe", 20)
        current = self._make_preds("mild", 50) + self._make_preds("moderate", 30) + self._make_preds("severe", 20)
        report = detect_drift(baseline, current)
        assert report["drift_detected"] is False
        assert report["drift_score"] < 0.1

    def test_drift_detected_shifted_distribution(self):
        from monitoring.drift import detect_drift
        baseline = self._make_preds("mild", 80) + self._make_preds("moderate", 10) + self._make_preds("severe", 10)
        current = self._make_preds("severe", 80) + self._make_preds("moderate", 10) + self._make_preds("mild", 10)
        report = detect_drift(baseline, current)
        assert report["drift_detected"] is True
        assert report["drift_score"] > 0.2

    def test_empty_baseline_returns_no_drift(self):
        from monitoring.drift import detect_drift
        current = self._make_preds("mild", 50)
        report = detect_drift([], current)
        assert report["drift_detected"] is False
        assert "error" in report

    def test_drift_report_has_required_keys(self):
        from monitoring.drift import detect_drift
        baseline = self._make_preds("mild", 30)
        current = self._make_preds("mild", 25) + self._make_preds("moderate", 5)
        report = detect_drift(baseline, current)
        for key in ["drift_score", "drift_level", "drift_detected", "baseline", "current"]:
            assert key in report

    def test_kl_divergence_method(self):
        from monitoring.drift import detect_drift
        baseline = self._make_preds("mild", 50) + self._make_preds("moderate", 30) + self._make_preds("severe", 20)
        current = self._make_preds("mild", 20) + self._make_preds("moderate", 60) + self._make_preds("severe", 20)
        report = detect_drift(baseline, current, method="kl")
        assert "drift_score" in report
        assert report["method"] == "kl"

    def test_psi_function(self):
        from training.metrics import psi
        baseline = np.array([50.0, 30.0, 20.0])
        current = np.array([50.0, 30.0, 20.0])
        score = psi(baseline, current)
        assert abs(score) < 0.01

    def test_kl_divergence_function(self):
        from training.metrics import kl_divergence
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])
        score = kl_divergence(p, q)
        assert abs(score) < 0.01
