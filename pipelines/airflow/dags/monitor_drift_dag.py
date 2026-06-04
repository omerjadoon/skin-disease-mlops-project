"""
Airflow DAG: Hourly drift monitoring — compute drift and trigger retraining if needed.
Schedule: Hourly
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

WORKDIR = "/app"
PYTHON = "python"
SIGNAL_FILE = "/app/artifacts/retraining_signal.json"


def compute_drift_and_decide(**context) -> str:
    """
    Run drift detection and decide whether to trigger retraining.

    Returns the branch task_id to execute next.
    """
    import sys

    sys.path.insert(0, WORKDIR)

    try:
        from monitoring.drift import detect_drift
        from monitoring.prediction_logger import fetch_recent_predictions, get_db_connection

        conn = get_db_connection()
        baseline = fetch_recent_predictions(conn, hours=168)   # 7 days
        current = fetch_recent_predictions(conn, hours=24)     # 24 hours
        conn.close()

        report = detect_drift(baseline, current, method="psi")
        drift_score = report.get("drift_score", 0.0)
        drift_detected = report.get("drift_detected", False)

        context["ti"].xcom_push(key="drift_score", value=drift_score)
        context["ti"].xcom_push(key="drift_detected", value=drift_detected)
        context["ti"].xcom_push(key="drift_level", value=report.get("drift_level", "none"))

        print(f"Drift score: {drift_score:.4f} | Detected: {drift_detected}")

        if drift_detected:
            return "write_retraining_signal"
        return "no_drift_detected"

    except Exception as e:
        print(f"⚠ Drift computation failed: {e}")
        return "no_drift_detected"


def write_signal(**context) -> None:
    """Write the retraining signal file."""
    import json
    from datetime import datetime
    from pathlib import Path

    drift_score = context["ti"].xcom_pull(task_ids="compute_drift", key="drift_score")
    signal = {
        "triggered_at": datetime.utcnow().isoformat(),
        "reason": "Distribution drift detected via PSI",
        "drift_score": drift_score,
        "drift_method": "psi",
        "status": "pending",
    }
    signal_path = Path(SIGNAL_FILE)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)
    print(f"✓ Retraining signal written: {signal_path}")


def log_drift_metrics(**context) -> None:
    """Log drift metrics to PostgreSQL."""
    import sys

    sys.path.insert(0, WORKDIR)

    drift_score = context["ti"].xcom_pull(task_ids="compute_drift", key="drift_score") or 0.0
    drift_detected = context["ti"].xcom_pull(task_ids="compute_drift", key="drift_detected") or False

    try:
        from monitoring.prediction_logger import get_db_connection, log_drift_metric

        conn = get_db_connection()
        log_drift_metric(
            conn,
            metric_name="psi_drift",
            metric_value=float(drift_score),
            baseline_window="168h",
            current_window="24h",
            drift_detected=bool(drift_detected),
        )
        conn.close()
        print("✓ Drift metric logged to database")
    except Exception as e:
        print(f"⚠ DB logging failed: {e}")


with DAG(
    dag_id="monitor_drift_dag",
    description="Hourly prediction drift monitoring — triggers retraining if drift detected",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "drift", "mlops", "skin-severity"],
    max_active_runs=1,
) as dag:

    compute_drift = BranchPythonOperator(
        task_id="compute_drift",
        python_callable=compute_drift_and_decide,
    )

    write_retraining_signal = PythonOperator(
        task_id="write_retraining_signal",
        python_callable=write_signal,
    )

    no_drift_detected = BashOperator(
        task_id="no_drift_detected",
        bash_command="echo '✅ No significant drift detected — no action needed'",
    )

    # Log to DB regardless of branch outcome
    log_to_db = PythonOperator(
        task_id="log_drift_to_db",
        python_callable=log_drift_metrics,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # If drift detected, trigger the training DAG via TriggerDagRunOperator
    trigger_training = BashOperator(
        task_id="trigger_training_dag",
        bash_command=(
            "airflow dags trigger train_model_dag "
            "--conf '{\"trigger_reason\": \"drift_detected\"}' || "
            "echo '⚠ Could not trigger train_model_dag — check Airflow connection'"
        ),
    )

    compute_drift >> [write_retraining_signal, no_drift_detected]
    write_retraining_signal >> log_to_db >> trigger_training
    no_drift_detected >> log_to_db
