"""
Seed script: generates prediction_logs.csv and drift_metrics.csv
in data/exports/ so Meltano can ETL them into the meltano_staging schema.

DISCLAIMER: Synthetic data — NOT real medical data.

Run:
    python scripts/seed_meltano_exports.py
"""

from __future__ import annotations

import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

CLASSES = ["mild", "moderate", "severe"]
MODEL_VERSIONS = ["v1.0", "v1.1", "v1.2"]
EXPORT_DIR = Path("data/exports")


def random_probabilities(predicted: str) -> dict[str, float]:
    """Return softmax-like probabilities biased towards predicted class."""
    base = {"mild": 0.1, "moderate": 0.1, "severe": 0.1}
    base[predicted] = random.uniform(0.6, 0.95)
    total = sum(base.values())
    return {k: round(v / total, 4) for k, v in base.items()}


def generate_prediction_logs(n: int = 200) -> list[dict]:
    rows = []
    now = datetime.utcnow()
    for i in range(n):
        predicted = random.choices(CLASSES, weights=[0.5, 0.35, 0.15])[0]
        probs = random_probabilities(predicted)
        confidence = probs[predicted]
        ts = now - timedelta(hours=random.randint(0, 72))
        rows.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "timestamp": ts.isoformat(),
                "image_hash": uuid.uuid4().hex[:16],
                "predicted_class": predicted,
                "confidence": round(confidence, 4),
                "prob_mild": probs["mild"],
                "prob_moderate": probs["moderate"],
                "prob_severe": probs["severe"],
                "model_name": "skin-severity-classifier",
                "model_version": random.choice(MODEL_VERSIONS),
                "latency_ms": round(random.uniform(50, 300), 2),
                "source_ip": f"10.0.0.{random.randint(1, 50)}",
            }
        )
    return rows


def generate_drift_metrics(n: int = 50) -> list[dict]:
    metric_names = [
        "confidence_mean",
        "confidence_std",
        "class_distribution_psi",
        "prediction_entropy",
        "severe_rate",
    ]
    rows = []
    now = datetime.utcnow()
    for i in range(n):
        metric = random.choice(metric_names)
        value = round(random.uniform(0.01, 0.95), 4)
        drift = value > 0.6
        ts = now - timedelta(hours=random.randint(0, 48))
        rows.append(
            {
                "id": i + 1,
                "timestamp": ts.isoformat(),
                "metric_name": metric,
                "metric_value": value,
                "baseline_window": "2024-01-01/2024-01-07",
                "current_window": "2024-01-08/2024-01-14",
                "drift_detected": drift,
                "model_version": random.choice(MODEL_VERSIONS),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Written {len(rows)} rows → {path}")


def main() -> None:
    print("=" * 55)
    print("  Meltano Export CSV Seeder")
    print("  DISCLAIMER: Synthetic data, NOT real medical data.")
    print("=" * 55)

    pred_logs = generate_prediction_logs(200)
    drift_metrics = generate_drift_metrics(50)

    write_csv(EXPORT_DIR / "prediction_logs.csv", pred_logs)
    write_csv(EXPORT_DIR / "drift_metrics.csv", drift_metrics)

    print("\n  ✅ CSVs ready in data/exports/")
    print("  Next: docker exec skin-mlops-meltano meltano run tap-csv target-postgres")


if __name__ == "__main__":
    main()
