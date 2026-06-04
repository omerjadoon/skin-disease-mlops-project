"""
Model loader for FastAPI inference service.

Loads model from MLflow registry (preferred) or falls back to local checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ModelLoader:
    """Loads and caches the skin severity classifier for inference."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name: str = os.environ.get("MODEL_NAME", "skin-severity-classifier")
        self.model_stage: str = os.environ.get("MODEL_STAGE", "Production")
        self.model_version: str = "unknown"
        self.model_info: dict[str, Any] = {}
        self._loaded: bool = False

    def load(self, config: dict[str, Any] | None = None) -> bool:
        """
        Load model from MLflow registry or local checkpoint.

        Returns True if successful, False otherwise.
        """
        if config is None:
            config = self._load_api_config()

        self.model_name = config.get("model_name", self.model_name)
        self.model_stage = config.get("model_stage", self.model_stage)
        local_path = config.get("model_local_path", "artifacts/model.ckpt")

        # Try MLflow first
        if self._try_load_from_mlflow(config):
            self._loaded = True
            return True

        # Fall back to local checkpoint
        if self._try_load_from_local(local_path, config):
            self._loaded = True
            return True

        print("⚠ WARNING: No model loaded. API will return errors on /predict.")
        return False

    def _try_load_from_mlflow(self, config: dict[str, Any]) -> bool:
        """Attempt to load model from MLflow Model Registry."""
        try:
            import mlflow

            tracking_uri = os.environ.get(
                "MLFLOW_TRACKING_URI",
                config.get("mlflow_tracking_uri", "http://mlflow:5000"),
            )
            mlflow.set_tracking_uri(tracking_uri)

            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            print(f"Loading model from MLflow: {model_uri}")

            self.model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            self.model.eval()

            # Try to get model version
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
            if versions:
                self.model_version = versions[0].version
                self.model_info = {
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "backbone": getattr(self.model, "hparams", {}).get("backbone", "resnet18"),
                    "num_classes": getattr(self.model, "num_classes", 3),
                    "class_names": getattr(
                        self.model, "class_names", ["mild", "moderate", "severe"]
                    ),
                    "input_size": [224, 224],
                    "mlflow_tracking_uri": tracking_uri,
                }

            print(f"✓ Model loaded from MLflow: {self.model_name} v{self.model_version}")
            return True

        except Exception as e:
            print(f"MLflow load failed: {e}")
            return False

    def _try_load_from_local(self, ckpt_path: str, config: dict[str, Any]) -> bool:
        """Load model from local checkpoint file."""
        try:
            from training.model import SkinSeverityClassifier

            path = Path(ckpt_path)
            if not path.exists():
                print(f"Local checkpoint not found: {ckpt_path}")
                return False

            self.model = SkinSeverityClassifier.load_from_checkpoint(
                str(path), map_location="cpu"
            )
            self.model.eval()
            self.model_version = "local"
            self.model_info = {
                "model_name": self.model_name,
                "model_version": "local",
                "backbone": getattr(self.model, "hparams", {}).get("backbone", "resnet18"),
                "num_classes": getattr(self.model, "num_classes", 3),
                "class_names": getattr(
                    self.model, "class_names", config.get("class_names", ["mild", "moderate", "severe"])
                ),
                "input_size": [224, 224],
                "mlflow_tracking_uri": None,
            }
            print(f"✓ Model loaded from local checkpoint: {ckpt_path}")
            return True

        except Exception as e:
            print(f"Local checkpoint load failed: {e}")
            return False

    def _load_api_config(self) -> dict[str, Any]:
        """Load API config from YAML file."""
        config_path = Path("configs/api.yaml")
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def get_model(self) -> Any:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model


# Global singleton instance
_model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Return the global ModelLoader instance."""
    return _model_loader
