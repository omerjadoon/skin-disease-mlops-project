# Meltano ELT Pipeline

## Overview

This Meltano project provides a placeholder ELT (Extract, Load, Transform) pipeline for the Skin Severity MLOps system.

## Purpose

Meltano can be used to:
- **Extract** prediction logs from CSV exports or PostgreSQL
- **Load** data into a staging schema for analytics
- **Transform** raw data into dashboard-ready tables

## Current Pipelines

### `export-predictions-to-staging`

Extracts prediction log CSVs from `/app/data/exports/` and loads them into the `meltano_staging` schema in PostgreSQL.

## Running Meltano

```bash
# Install Meltano plugins
meltano install

# Run the ELT job
meltano run tap-csv target-postgres

# Run on schedule
meltano schedule run daily-prediction-export
```

## Future Integrations

| Use Case | Extractor | Loader |
|----------|-----------|--------|
| Pull raw image metadata from S3/MinIO | tap-s3-csv | target-postgres |
| Sync MLflow metrics to data warehouse | tap-mlflow (custom) | target-postgres |
| Ingest external skin condition datasets | tap-csv | target-postgres |
| Export predictions to BI tools | tap-postgres | target-bigquery |

## Architecture

```
MinIO / CSV exports
       │
       ▼ (tap-csv / tap-postgres)
   Meltano ELT
       │
       ▼ (target-postgres)
  PostgreSQL meltano_staging schema
       │
       ▼
  Metabase Dashboards
```

## Disclaimer

> This pipeline processes AI-assisted severity estimates only.
> Data must NOT be used for clinical decisions.
> For educational and research purposes only.
