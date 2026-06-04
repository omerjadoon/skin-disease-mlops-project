"""
Pydantic schemas for the skin severity prediction API.

DISCLAIMER: This API provides AI-assisted severity estimates only.
It is NOT a medical diagnosis tool. For educational/research purposes only.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

DISCLAIMER = (
    "AI-assisted severity estimate only. Not a medical diagnosis. "
    "For educational and research purposes only."
)

SEVERITY_CLASSES = ["mild", "moderate", "severe"]


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    severity: str = Field(
        ...,
        description="Predicted severity class: mild | moderate | severe",
        examples=["moderate"],
    )
    confidence: float = Field(
        ...,
        description="Confidence of the top prediction (0.0 - 1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.87],
    )
    class_probabilities: dict[str, float] = Field(
        ...,
        description="Softmax probabilities for each severity class",
        examples=[{"mild": 0.08, "moderate": 0.87, "severe": 0.05}],
    )
    model_name: str = Field(
        ...,
        description="Name of the registered model",
        examples=["skin-severity-classifier"],
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for inference",
        examples=["1"],
    )
    prediction_id: str | None = Field(
        default=None,
        description="Unique ID for this prediction (for audit trail)",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Inference latency in milliseconds",
    )
    disclaimer: str = Field(
        default=DISCLAIMER,
        description="Medical disclaimer — always included in every response",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        if v not in SEVERITY_CLASSES:
            raise ValueError(f"severity must be one of {SEVERITY_CLASSES}")
        return v


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(default="ok", examples=["ok"])
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    model_name: str = Field(..., examples=["skin-severity-classifier"])
    model_version: str = Field(..., examples=["1"])
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Response from the /model-info endpoint."""

    model_name: str = Field(..., examples=["skin-severity-classifier"])
    model_version: str = Field(..., examples=["1"])
    backbone: str = Field(..., examples=["resnet18"])
    num_classes: int = Field(..., examples=[3])
    class_names: list[str] = Field(..., examples=[["mild", "moderate", "severe"]])
    input_size: list[int] = Field(..., examples=[[224, 224]])
    mlflow_tracking_uri: str | None = Field(default=None)
    disclaimer: str = Field(default=DISCLAIMER)


class PredictionLog(BaseModel):
    """Internal model for logging predictions to PostgreSQL."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_hash: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    model_name: str
    model_version: str
    latency_ms: float
    prediction_id: str | None = None
    source_ip: str | None = None
