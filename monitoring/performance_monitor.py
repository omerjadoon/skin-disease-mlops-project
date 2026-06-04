"""
Performance monitoring — reads prediction logs and computes summary statistics.

Run standalone:
    python monitoring/performance_monitor.py --hours 24

Or called by Airflow DAG.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_performance_report(
    predictions: list[dict[str, Any]],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute summary statistics from a list of prediction log records.

    Args:
        predictions: List of dicts from prediction_logs table
        class_names: Expected class names

    Returns:
        Report dict with distribution, confidence, latency stats
    """
    if class_names is None:
        class_names = ["mild", "moderate", "severe"]

    if not predictions:
        return {
            "num_requests": 0,
            "error": "No predictions found in the specified window",
        }

    import numpy as np

    n = len(predictions)
    confidences = [float(p["confidence"]) for p in predictions if p.get("confidence") is not None]
    latencies = [float(p["latency_ms"]) for p in predictions if p.get("latency_ms") is not None]
    classes = [p["predicted_class"] for p in predictions]

    # Class distribution
    class_counts = {c: classes.count(c) for c in class_names}
    class_distribution = {c: round(class_counts[c] / n, 4) for c in class_names}

    # Confidence stats
    conf_arr = np.array(confidences) if confidences else np.array([0.0])
    latency_arr = np.array(latencies) if latencies else np.array([0.0])

    return {
        "report_generated_at": datetime.utcnow().isoformat(),
        "num_requests": n,
        "class_distribution": class_distribution,
        "class_counts": class_counts,
        "confidence": {
            "mean": round(float(np.mean(conf_arr)), 4),
            "std": round(float(np.std(conf_arr)), 4),
            "min": round(float(np.min(conf_arr)), 4),
            "max": round(float(np.max(conf_arr)), 4),
            "p50": round(float(np.percentile(conf_arr, 50)), 4),
            "p95": round(float(np.percentile(conf_arr, 95)), 4),
        },
        "latency_ms": {
            "mean": round(float(np.mean(latency_arr)), 2),
            "p50": round(float(np.percentile(latency_arr, 50)), 2),
            "p95": round(float(np.percentile(latency_arr, 95)), 2),
            "p99": round(float(np.percentile(latency_arr, 99)), 2),
            "max": round(float(np.max(latency_arr)), 2),
        },
        "model_versions": list({p.get("model_version", "unknown") for p in predictions}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute prediction performance report")
    parser.add_argument("--hours", type=int, default=24, help="Window in hours")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--model-version", type=str, default=None)
    args = parser.parse_args()

    try:
        from monitoring.prediction_logger import fetch_recent_predictions, get_db_connection

        print(f"Fetching predictions from last {args.hours} hours...")
        conn = get_db_connection()
        predictions = fetch_recent_predictions(conn, hours=args.hours, model_version=args.model_version)
        conn.close()
    except Exception as e:
        print(f"⚠ Could not connect to database: {e}")
        print("Generating empty report.")
        predictions = []

    report = compute_performance_report(predictions)

    print("\n── Performance Report ──────────────────────")
    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved: {output_path}")


if __name__ == "__main__":
    main()
