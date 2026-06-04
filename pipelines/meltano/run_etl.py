"""
Direct ETL script — replaces Meltano tap-csv → target-postgres pipeline.
Reads prediction_logs.csv and drift_metrics.csv from data/exports/
and loads them into the meltano_staging schema in PostgreSQL.

Run inside the meltano container:
    python /project/run_etl.py

Or from host:
    docker exec skin-mlops-meltano python /project/run_etl.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

# ── Config ────────────────────────────────────────────────────────────────────
DB = dict(
    host=os.environ.get("POSTGRES_HOST", "postgres"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    dbname=os.environ.get("POSTGRES_DB", "mlops_db"),
    user=os.environ.get("POSTGRES_USER", "mlops"),
    password=os.environ.get("POSTGRES_PASSWORD", "mlops_secret"),
)

EXPORTS_DIR = Path("/app/data/exports")
SCHEMA = "meltano_staging"


# ── Helpers ───────────────────────────────────────────────────────────────────
def connect():
    return psycopg2.connect(**DB)


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.prediction_logs (
                prediction_id   TEXT PRIMARY KEY,
                timestamp       TEXT,
                image_hash      TEXT,
                predicted_class TEXT,
                confidence      FLOAT,
                prob_mild       FLOAT,
                prob_moderate   FLOAT,
                prob_severe     FLOAT,
                model_name      TEXT,
                model_version   TEXT,
                latency_ms      FLOAT,
                source_ip       TEXT,
                _etl_loaded_at  TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS {SCHEMA}.drift_metrics (
                id              INT PRIMARY KEY,
                timestamp       TEXT,
                metric_name     TEXT,
                metric_value    FLOAT,
                baseline_window TEXT,
                current_window  TEXT,
                drift_detected  BOOLEAN,
                model_version   TEXT,
                _etl_loaded_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)
    conn.commit()
    print(f"  ✓ Schema '{SCHEMA}' and tables ready")


def load_csv(conn, csv_path: Path, table: str, pk: str):
    if not csv_path.exists():
        print(f"  ⚠ File not found: {csv_path} — skipping")
        return 0

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  ⚠ No rows in {csv_path.name}")
        return 0

    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    upsert_sql = f"""
        INSERT INTO {SCHEMA}.{table} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT ({pk}) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk)},
            _etl_loaded_at = NOW();
    """

    loaded = 0
    with conn.cursor() as cur:
        for row in rows:
            values = []
            for v in row.values():
                if v.lower() in ("true", "false"):
                    values.append(v.lower() == "true")
                elif v == "":
                    values.append(None)
                else:
                    values.append(v)
            cur.execute(upsert_sql, values)
            loaded += 1
    conn.commit()
    return loaded


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Meltano-style ETL: CSV → PostgreSQL meltano_staging")
    print("=" * 55)

    try:
        conn = connect()
        print(f"  ✓ Connected to PostgreSQL ({DB['host']}:{DB['port']})")
    except Exception as e:
        print(f"  ✗ DB connection failed: {e}")
        sys.exit(1)

    ensure_schema(conn)

    n1 = load_csv(conn, EXPORTS_DIR / "prediction_logs.csv", "prediction_logs", "prediction_id")
    print(f"  ✓ Loaded {n1} rows → {SCHEMA}.prediction_logs")

    n2 = load_csv(conn, EXPORTS_DIR / "drift_metrics.csv", "drift_metrics", "id")
    print(f"  ✓ Loaded {n2} rows → {SCHEMA}.drift_metrics")

    conn.close()
    print(f"\n  ✅ ETL complete! {n1 + n2} total rows loaded into '{SCHEMA}' schema.")
    print(f"     View in Metabase at http://localhost:3000")


if __name__ == "__main__":
    main()
