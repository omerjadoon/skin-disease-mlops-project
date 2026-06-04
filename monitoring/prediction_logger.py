"""
Prediction logger — writes inference results to PostgreSQL.
Shared by both the API (real-time) and monitoring scripts (batch reads).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras


def get_db_connection(db_config: dict[str, Any] | None = None):
    """Create a PostgreSQL connection from environment variables or config dict."""
    cfg = db_config or {}
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", cfg.get("host", "postgres")),
        port=int(os.environ.get("POSTGRES_PORT", cfg.get("port", 5432))),
        dbname=os.environ.get("POSTGRES_DB", cfg.get("name", "mlops_db")),
        user=os.environ.get("POSTGRES_USER", cfg.get("user", "mlops")),
        password=os.environ.get("POSTGRES_PASSWORD", cfg.get("password", "")),
    )


def ensure_tables(conn) -> None:
    """Ensure all monitoring tables exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id              SERIAL PRIMARY KEY,
        prediction_id   TEXT,
        timestamp       TIMESTAMPTZ DEFAULT NOW(),
        image_hash      TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence      FLOAT NOT NULL,
        prob_mild       FLOAT,
        prob_moderate   FLOAT,
        prob_severe     FLOAT,
        model_name      TEXT,
        model_version   TEXT,
        latency_ms      FLOAT,
        source_ip       TEXT
    );

    CREATE TABLE IF NOT EXISTS drift_metrics (
        id              SERIAL PRIMARY KEY,
        timestamp       TIMESTAMPTZ DEFAULT NOW(),
        metric_name     TEXT NOT NULL,
        metric_value    FLOAT NOT NULL,
        baseline_window TEXT,
        current_window  TEXT,
        drift_detected  BOOLEAN DEFAULT FALSE,
        model_version   TEXT
    );

    CREATE TABLE IF NOT EXISTS model_performance (
        id              SERIAL PRIMARY KEY,
        evaluated_at    TIMESTAMPTZ DEFAULT NOW(),
        model_name      TEXT NOT NULL,
        model_version   TEXT NOT NULL,
        split           TEXT DEFAULT 'test',
        accuracy        FLOAT,
        f1_macro        FLOAT,
        precision_macro FLOAT,
        recall_macro    FLOAT,
        auroc           FLOAT,
        loss            FLOAT,
        num_samples     INT
    );

    CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
        ON prediction_logs (timestamp);
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version
        ON prediction_logs (model_version);
    CREATE INDEX IF NOT EXISTS idx_drift_metrics_timestamp
        ON drift_metrics (timestamp);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def log_prediction(
    conn,
    *,
    prediction_id: str | None = None,
    image_hash: str,
    predicted_class: str,
    confidence: float,
    class_probabilities: dict[str, float],
    model_name: str,
    model_version: str,
    latency_ms: float,
    source_ip: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Insert a single prediction record."""
    ts = timestamp or datetime.utcnow()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_logs
                (prediction_id, timestamp, image_hash, predicted_class, confidence,
                 prob_mild, prob_moderate, prob_severe,
                 model_name, model_version, latency_ms, source_ip)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                prediction_id,
                ts,
                image_hash,
                predicted_class,
                confidence,
                class_probabilities.get("mild"),
                class_probabilities.get("moderate"),
                class_probabilities.get("severe"),
                model_name,
                model_version,
                latency_ms,
                source_ip,
            ),
        )
    conn.commit()


def fetch_recent_predictions(
    conn,
    hours: int = 24,
    model_version: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch prediction logs from the last N hours."""
    query = """
        SELECT *
        FROM prediction_logs
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
    """
    params: list[Any] = [hours]
    if model_version:
        query += " AND model_version = %s"
        params.append(model_version)
    query += " ORDER BY timestamp DESC"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def log_drift_metric(
    conn,
    *,
    metric_name: str,
    metric_value: float,
    baseline_window: str,
    current_window: str,
    drift_detected: bool,
    model_version: str | None = None,
) -> None:
    """Insert a drift metric record."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_metrics
                (metric_name, metric_value, baseline_window, current_window,
                 drift_detected, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                metric_name,
                metric_value,
                baseline_window,
                current_window,
                drift_detected,
                model_version,
            ),
        )
    conn.commit()


def log_model_performance(
    conn,
    *,
    model_name: str,
    model_version: str,
    split: str = "test",
    accuracy: float | None = None,
    f1_macro: float | None = None,
    precision_macro: float | None = None,
    recall_macro: float | None = None,
    auroc: float | None = None,
    loss: float | None = None,
    num_samples: int | None = None,
) -> None:
    """Insert model evaluation metrics."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_performance
                (model_name, model_version, split, accuracy, f1_macro,
                 precision_macro, recall_macro, auroc, loss, num_samples)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                model_name, model_version, split,
                accuracy, f1_macro, precision_macro, recall_macro,
                auroc, loss, num_samples,
            ),
        )
    conn.commit()
