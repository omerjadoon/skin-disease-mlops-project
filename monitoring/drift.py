"""
Drift detection — compares recent prediction distribution to a baseline.

Methods:
- PSI (Population Stability Index): default
- KL-Divergence: optional

Thresholds:
- PSI < 0.1  → No significant drift
- PSI 0.1-0.2 → Moderate drift (warn)
- PSI > 0.2  → Significant drift (trigger retraining)

Run standalone:
    python monitoring/drift.py --baseline-hours 168 --current-hours 24
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.metrics import kl_divergence, psi

CLASS_NAMES = ["mild", "moderate", "severe"]

# Thresholds
PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.2


def compute_class_distribution(
    predictions: list[dict[str, Any]],
    class_names: list[str],
) -> np.ndarray:
    """Convert list of predictions to a class count array."""
    counts = np.zeros(len(class_names), dtype=float)
    for p in predictions:
        cls = p.get("predicted_class", "")
        if cls in class_names:
            counts[class_names.index(cls)] += 1
    return counts


def detect_drift(
    baseline_predictions: list[dict[str, Any]],
    current_predictions: list[dict[str, Any]],
    class_names: list[str] | None = None,
    method: str = "psi",
    warn_threshold: float = PSI_WARN_THRESHOLD,
    alert_threshold: float = PSI_ALERT_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare baseline vs current prediction distributions for drift.

    Args:
        baseline_predictions: Historical prediction records
        current_predictions:  Recent prediction records
        class_names:          List of class names
        method:               "psi" or "kl"
        warn_threshold:       Drift warning threshold
        alert_threshold:      Drift alert threshold (triggers retraining)

    Returns:
        Drift report dict
    """
    if class_names is None:
        class_names = CLASS_NAMES

    if not baseline_predictions:
        return {
            "error": "No baseline predictions available",
            "drift_detected": False,
            "drift_score": 0.0,
            "method": method,
        }

    if not current_predictions:
        return {
            "error": "No current predictions available",
            "drift_detected": False,
            "drift_score": 0.0,
            "method": method,
        }

    baseline_counts = compute_class_distribution(baseline_predictions, class_names)
    current_counts = compute_class_distribution(current_predictions, class_names)

    # Baseline distribution
    baseline_dist = baseline_counts / (baseline_counts.sum() + 1e-8)
    current_dist = current_counts / (current_counts.sum() + 1e-8)

    # Compute drift score
    if method == "psi":
        drift_score = psi(baseline_counts, current_counts)
    elif method == "kl":
        drift_score = kl_divergence(baseline_dist, current_dist)
    else:
        raise ValueError(f"Unknown drift method: {method}. Use 'psi' or 'kl'.")

    drift_level = "none"
    drift_detected = False
    if drift_score >= alert_threshold:
        drift_level = "high"
        drift_detected = True
    elif drift_score >= warn_threshold:
        drift_level = "moderate"

    return {
        "computed_at": datetime.utcnow().isoformat(),
        "method": method,
        "drift_score": round(float(drift_score), 6),
        "drift_level": drift_level,
        "drift_detected": drift_detected,
        "warn_threshold": warn_threshold,
        "alert_threshold": alert_threshold,
        "baseline": {
            "num_samples": len(baseline_predictions),
            "distribution": {
                class_names[i]: round(float(baseline_dist[i]), 4) for i in range(len(class_names))
            },
        },
        "current": {
            "num_samples": len(current_predictions),
            "distribution": {
                class_names[i]: round(float(current_dist[i]), 4) for i in range(len(class_names))
            },
        },
        "per_class_shift": {
            class_names[i]: round(float(current_dist[i] - baseline_dist[i]), 4)
            for i in range(len(class_names))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect prediction distribution drift")
    parser.add_argument("--baseline-hours", type=int, default=168, help="Baseline window (hours)")
    parser.add_argument("--current-hours", type=int, default=24, help="Current window (hours)")
    parser.add_argument("--method", type=str, default="psi", choices=["psi", "kl"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    args = parser.parse_args()

    try:
        from monitoring.prediction_logger import (
            fetch_recent_predictions,
            get_db_connection,
            log_drift_metric,
        )

        conn = get_db_connection()
        print(
            f"Fetching baseline ({args.baseline_hours}h) and current ({args.current_hours}h) predictions..."
        )
        baseline = fetch_recent_predictions(
            conn, hours=args.baseline_hours, model_version=args.model_version
        )
        current = fetch_recent_predictions(
            conn, hours=args.current_hours, model_version=args.model_version
        )
    except Exception as e:
        print(f"⚠ DB connection failed: {e}. Using empty data for demo.")
        baseline, current = [], []
        conn = None

    report = detect_drift(baseline, current, method=args.method)

    print("\n── Drift Report ────────────────────────────")
    print(json.dumps(report, indent=2))

    if report.get("drift_detected"):
        print("\n🚨 DRIFT DETECTED — Retraining may be needed!")
    elif report.get("drift_level") == "moderate":
        print("\n⚠ Moderate drift detected — monitoring closely.")
    else:
        print("\n✅ No significant drift detected.")

    # Log to database
    if conn is not None and "error" not in report:
        try:
            log_drift_metric(
                conn,
                metric_name=f"drift_{args.method}",
                metric_value=report["drift_score"],
                baseline_window=f"{args.baseline_hours}h",
                current_window=f"{args.current_hours}h",
                drift_detected=report["drift_detected"],
                model_version=args.model_version,
            )
            conn.close()
            print("✓ Drift metric logged to database")
        except Exception as e:
            print(f"⚠ Could not log drift metric: {e}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved: {args.output}")


if __name__ == "__main__":
    main()
