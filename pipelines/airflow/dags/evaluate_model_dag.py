"""
Airflow DAG: Evaluate the latest registered model on validation/test data.
Schedule: Daily
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

WORKDIR = "/app"
PYTHON = "python"


def get_latest_model_version(**context) -> str:
    """Fetch the latest Production model version from MLflow."""
    import os

    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("skin-severity-classifier", stages=["Production"])
        if versions:
            version = versions[0].version
            print(f"Latest Production model version: {version}")
            context["ti"].xcom_push(key="model_version", value=version)
            return version
    except Exception as e:
        print(f"⚠ Could not fetch model version: {e}")

    return "unknown"


def log_evaluation_to_db(**context) -> None:
    """Read eval_metrics.json and write to model_performance table."""
    import json
    from pathlib import Path

    metrics_path = Path(f"{WORKDIR}/artifacts/eval_metrics.json")
    if not metrics_path.exists():
        print("⚠ No eval_metrics.json found")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    model_version = context["ti"].xcom_pull(
        task_ids="get_latest_model_version", key="model_version"
    ) or "unknown"

    try:
        import sys
        sys.path.insert(0, WORKDIR)
        from monitoring.prediction_logger import get_db_connection, log_model_performance

        conn = get_db_connection()
        log_model_performance(
            conn,
            model_name="skin-severity-classifier",
            model_version=model_version,
            split="test",
            accuracy=metrics.get("test/acc"),
            f1_macro=metrics.get("test/f1_macro"),
            precision_macro=metrics.get("test/precision"),
            recall_macro=metrics.get("test/recall"),
            auroc=metrics.get("test/auroc"),
            loss=metrics.get("test/loss"),
        )
        conn.close()
        print("✓ Evaluation metrics logged to database")
    except Exception as e:
        print(f"⚠ DB logging failed: {e}")


with DAG(
    dag_id="evaluate_model_dag",
    description="Daily evaluation of the latest registered skin severity model",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["evaluation", "mlops", "skin-severity"],
    max_active_runs=1,
) as dag:

    get_model_version = PythonOperator(
        task_id="get_latest_model_version",
        python_callable=get_latest_model_version,
    )

    run_evaluation = BashOperator(
        task_id="run_evaluation",
        bash_command=(
            f"cd {WORKDIR} && {PYTHON} training/evaluate.py "
            "--config configs/train.yaml --split test"
        ),
        execution_timeout=timedelta(hours=1),
    )

    log_to_db = PythonOperator(
        task_id="log_evaluation_to_db",
        python_callable=log_evaluation_to_db,
    )

    get_model_version >> run_evaluation >> log_to_db
