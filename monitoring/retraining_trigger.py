"""
Retraining trigger — writes a retraining signal when drift is detected.

This can be polled by an Airflow DAG sensor or a separate service.

Run standalone:
    python monitoring/retraining_trigger.py --check-drift

Or called programmatically from drift.py / Airflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SIGNAL_FILE = Path("artifacts/retraining_signal.json")


def write_retraining_signal(
    reason: str,
    drift_score: float | None = None,
    drift_method: str | None = None,
    model_version: str | None = None,
) -> Path:
    """
    Write a retraining trigger signal to disk.

    The Airflow DAG polls for this file via a FileSensor.
    """
    signal = {
        "triggered_at": datetime.utcnow().isoformat(),
        "reason": reason,
        "drift_score": drift_score,
        "drift_method": drift_method,
        "model_version": model_version,
        "status": "pending",
    }

    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_FILE, "w") as f:
        json.dump(signal, f, indent=2)

    print(f"✓ Retraining signal written: {SIGNAL_FILE}")
    print(f"  Reason: {reason}")
    if drift_score is not None:
        print(f"  Drift score ({drift_method}): {drift_score:.4f}")
    return SIGNAL_FILE


def clear_retraining_signal() -> None:
    """Remove the retraining signal file after retraining is complete."""
    if SIGNAL_FILE.exists():
        SIGNAL_FILE.unlink()
        print(f"✓ Retraining signal cleared: {SIGNAL_FILE}")
    else:
        print("No retraining signal found to clear.")


def check_retraining_signal() -> dict | None:
    """Return signal dict if a pending retraining signal exists, else None."""
    if SIGNAL_FILE.exists():
        with open(SIGNAL_FILE) as f:
            signal = json.load(f)
        if signal.get("status") == "pending":
            return signal
    return None


def trigger_if_drift_detected(
    baseline_hours: int = 168,
    current_hours: int = 24,
    method: str = "psi",
    model_version: str | None = None,
) -> bool:
    """
    Run drift detection and trigger retraining if drift exceeds threshold.

    Returns True if retraining was triggered.
    """
    from monitoring.drift import detect_drift

    try:
        from monitoring.prediction_logger import fetch_recent_predictions, get_db_connection

        conn = get_db_connection()
        baseline = fetch_recent_predictions(conn, hours=baseline_hours, model_version=model_version)
        current = fetch_recent_predictions(conn, hours=current_hours, model_version=model_version)
        conn.close()
    except Exception as e:
        print(f"⚠ DB connection failed: {e}")
        baseline, current = [], []

    report = detect_drift(baseline, current, method=method)

    print(f"Drift report: score={report.get('drift_score', 0):.4f}, "
          f"level={report.get('drift_level', 'unknown')}")

    if report.get("drift_detected", False):
        write_retraining_signal(
            reason=f"Distribution drift detected via {method.upper()}",
            drift_score=report.get("drift_score"),
            drift_method=method,
            model_version=model_version,
        )
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Retraining trigger for drift detection")
    parser.add_argument("--check-drift", action="store_true",
                        help="Run drift detection and trigger if needed")
    parser.add_argument("--baseline-hours", type=int, default=168)
    parser.add_argument("--current-hours", type=int, default=24)
    parser.add_argument("--method", type=str, default="psi", choices=["psi", "kl"])
    parser.add_argument("--clear-signal", action="store_true",
                        help="Clear the retraining signal after retraining")
    parser.add_argument("--check-signal", action="store_true",
                        help="Check if a retraining signal is pending")
    args = parser.parse_args()

    if args.clear_signal:
        clear_retraining_signal()
        return

    if args.check_signal:
        signal = check_retraining_signal()
        if signal:
            print("⚠ Pending retraining signal found:")
            print(json.dumps(signal, indent=2))
        else:
            print("✅ No pending retraining signal.")
        return

    if args.check_drift:
        triggered = trigger_if_drift_detected(
            baseline_hours=args.baseline_hours,
            current_hours=args.current_hours,
            method=args.method,
        )
        sys.exit(0 if not triggered else 1)


if __name__ == "__main__":
    main()
