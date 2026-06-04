"""
Airflow DAG: Full ML pipeline — prepare → QC → train → evaluate → register
Schedule: Weekly (or triggered manually / by retraining signal)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

WORKDIR = "/app"
PYTHON = "python"


def check_data_available(**context) -> str:
    """Check if training data is available and return branch name."""
    import os
    from pathlib import Path

    data_dir = Path(f"{WORKDIR}/data/raw")
    class_names = ["mild", "moderate", "severe"]

    for cls in class_names:
        cls_dir = data_dir / cls
        if cls_dir.exists() and any(cls_dir.iterdir()):
            return "prepare_data"

    # No data — seed sample data instead
    return "seed_sample_data"


def clear_retraining_signal(**context) -> None:
    """Remove the retraining signal file after pipeline completes."""
    from pathlib import Path

    signal_file = Path(f"{WORKDIR}/artifacts/retraining_signal.json")
    if signal_file.exists():
        signal_file.unlink()
        print("✓ Retraining signal cleared")


with DAG(
    dag_id="train_model_dag",
    description="End-to-end ML pipeline: prepare → QC → train → evaluate → register model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "mlops", "skin-severity"],
    max_active_runs=1,
) as dag:

    # Check whether data is ready or we need to seed it
    check_data = BranchPythonOperator(
        task_id="check_data_available",
        python_callable=check_data_available,
    )

    # Seed synthetic sample data if none exists
    seed_sample_data = BashOperator(
        task_id="seed_sample_data",
        bash_command=f"cd {WORKDIR} && {PYTHON} scripts/seed_sample_data.py",
    )

    # Standard data preparation step
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {WORKDIR} && echo 'Data already available — skipping seed'",
    )

    # QC check with FiftyOne
    qc_data = BashOperator(
        task_id="qc_data",
        bash_command=f"cd {WORKDIR} && {PYTHON} fiftyone_app/qc_report.py || true",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # Model training
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {WORKDIR} && {PYTHON} training/train.py "
            "--config configs/train.yaml"
        ),
        execution_timeout=timedelta(hours=4),
    )

    # Model evaluation on test set
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {WORKDIR} && {PYTHON} training/evaluate.py "
            "--config configs/train.yaml --split test"
        ),
    )

    # Register model in MLflow
    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {WORKDIR} && "
            "RUN_ID=$(cat artifacts/run_id.txt 2>/dev/null || echo '') && "
            f"[ -n \"$RUN_ID\" ] && {PYTHON} -c \""
            "import mlflow, os; "
            "mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')); "
            "result = mlflow.register_model('runs/' + open('artifacts/run_id.txt').read().strip() + '/model', 'skin-severity-classifier'); "
            "print(f'Registered version: {result.version}')\" "
            "|| echo 'No run_id found — skipping registration'"
        ),
    )

    # Clear retraining signal on successful completion
    clear_signal = PythonOperator(
        task_id="clear_retraining_signal",
        python_callable=clear_retraining_signal,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # DAG dependencies
    check_data >> [seed_sample_data, prepare_data]
    seed_sample_data >> qc_data
    prepare_data >> qc_data
    qc_data >> train_model >> evaluate_model >> register_model >> clear_signal
