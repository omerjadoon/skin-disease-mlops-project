"""
Inference pipeline: image preprocessing → model forward pass → formatted response.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image

from api.schemas import DISCLAIMER, PredictionResponse

CLASS_NAMES = ["mild", "moderate", "severe"]

# Default inference transforms (ImageNet normalization, no augmentation)
_DEFAULT_TRANSFORM = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(
    image_bytes: bytes,
    image_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> tuple[torch.Tensor, str]:
    """
    Preprocess raw image bytes into a model-ready tensor.

    Returns:
        (tensor of shape [1, 3, H, W], image_hash)
    """
    image_hash = hashlib.md5(image_bytes).hexdigest()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    transform = T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
    return tensor, image_hash


@torch.no_grad()
def run_inference(
    model: Any,
    image_tensor: torch.Tensor,
    class_names: list[str] | None = None,
    model_name: str = "skin-severity-classifier",
    model_version: str = "unknown",
) -> dict[str, Any]:
    """
    Run model forward pass and return formatted prediction dict.

    Args:
        model: Loaded PyTorch model
        image_tensor: Preprocessed image tensor [1, C, H, W]
        class_names: Ordered class name list
        model_name: Model registry name
        model_version: Model version string

    Returns:
        Dict matching PredictionResponse schema
    """
    if class_names is None:
        class_names = CLASS_NAMES

    model.eval()
    logits = model(image_tensor)  # [1, num_classes]
    probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

    confidence, predicted_idx = probs.max(dim=0)
    predicted_class = class_names[predicted_idx.item()]

    class_probabilities = {
        class_names[i]: round(float(probs[i].item()), 4) for i in range(len(class_names))
    }

    return {
        "severity": predicted_class,
        "confidence": round(float(confidence.item()), 4),
        "class_probabilities": class_probabilities,
        "model_name": model_name,
        "model_version": str(model_version),
        "disclaimer": DISCLAIMER,
    }


def build_prediction_response(
    inference_result: dict[str, Any],
    prediction_id: str | None = None,
    latency_ms: float | None = None,
) -> PredictionResponse:
    """Combine inference result with metadata into a PredictionResponse."""
    return PredictionResponse(
        severity=inference_result["severity"],
        confidence=inference_result["confidence"],
        class_probabilities=inference_result["class_probabilities"],
        model_name=inference_result["model_name"],
        model_version=inference_result["model_version"],
        prediction_id=prediction_id,
        latency_ms=latency_ms,
        disclaimer=inference_result["disclaimer"],
    )
