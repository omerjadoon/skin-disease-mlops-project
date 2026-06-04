"""
Metrics helpers for skin severity classification training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def compute_confusion_matrix_dict(
    confusion_matrix: Tensor,
    class_names: list[str],
) -> dict[str, Any]:
    """Convert a torchmetrics confusion matrix tensor to a loggable dict."""
    cm = confusion_matrix.cpu().numpy()
    result: dict[str, Any] = {}
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            result[f"cm/{true_class}_as_{pred_class}"] = int(cm[i, j])
    return result


def compute_per_class_metrics(
    preds: Tensor,
    labels: Tensor,
    class_names: list[str],
) -> dict[str, float]:
    """
    Compute per-class accuracy from predictions and ground truth.

    Args:
        preds: (N,) predicted class indices
        labels: (N,) ground truth class indices
        class_names: list of class name strings

    Returns:
        dict with per-class accuracy
    """
    metrics: dict[str, float] = {}
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            metrics[f"acc_per_class/{class_name}"] = 0.0
        else:
            correct = (preds[mask] == labels[mask]).float().sum().item()
            total = mask.sum().item()
            metrics[f"acc_per_class/{class_name}"] = float(correct / total)
    return metrics


def format_metrics_for_mlflow(metrics_dict: dict[str, Any]) -> dict[str, float]:
    """
    Flatten and convert metrics to MLflow-compatible floats.
    Skips non-numeric values.
    """
    result: dict[str, float] = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif isinstance(v, Tensor):
            result[k] = float(v.item())
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            result[k] = float(v)
    return result


def psi(
    baseline: np.ndarray,
    current: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Population Stability Index (PSI) for distribution drift detection.

    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change (trigger retraining)
    """
    # Normalize to probabilities
    baseline_p = baseline / (baseline.sum() + eps)
    current_p = current / (current.sum() + eps)

    # Add epsilon to avoid log(0)
    baseline_p = np.clip(baseline_p, eps, 1.0)
    current_p = np.clip(current_p, eps, 1.0)

    psi_value = np.sum((current_p - baseline_p) * np.log(current_p / baseline_p))
    return float(psi_value)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """KL Divergence D_KL(P || Q) for distribution comparison."""
    p_norm = np.clip(p / (p.sum() + eps), eps, 1.0)
    q_norm = np.clip(q / (q.sum() + eps), eps, 1.0)
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))
