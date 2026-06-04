# Metabase Dashboard Setup

## Overview

Metabase connects to the PostgreSQL database and visualizes prediction logs, model performance, and drift metrics.

## Access

- **URL:** http://localhost:3000
- **First-time setup:** Follow the Metabase onboarding wizard on first login.

## Connecting to PostgreSQL

When prompted to add a database during setup:

| Field | Value |
|-------|-------|
| Database type | PostgreSQL |
| Host | `postgres` (Docker network) or `localhost` (host) |
| Port | `5432` |
| Database name | `mlops_db` |
| Username | `mlops` |
| Password | `mlops_secret` |

> **Note:** From within Docker Compose, use `postgres` as the host. From your local browser connecting to Metabase at `localhost:3000`, Metabase runs inside Docker and connects to `postgres` on the internal network automatically.

## Creating Dashboards

### Step 1 — Create Questions from SQL

1. Click **New → SQL Query**
2. Select your PostgreSQL connection
3. Paste SQL from `dashboards/sql/`:
   - `prediction_metrics.sql` → "Prediction Metrics"
   - `model_performance.sql` → "Model Performance"
   - `drift_metrics.sql` → "Drift Monitoring"
4. Click **Save** for each

### Step 2 — Build Dashboard

1. Click **New → Dashboard**
2. Name it: "Skin Severity MLOps"
3. Add your saved questions as cards
4. Arrange and resize for best layout

## Recommended Dashboard Cards

### Prediction Metrics Dashboard
| Card | SQL Query | Chart Type |
|------|-----------|------------|
| Predictions per Day | Query #1 from prediction_metrics.sql | Bar chart |
| Class Distribution (7d) | Query #2 | Pie chart |
| Hourly Volume | Query #3 | Line chart |
| Avg Confidence by Class | Query #4 | Table |

### Model Performance Dashboard
| Card | SQL Query | Chart Type |
|------|-----------|------------|
| Latest Model Metrics | Query #1 | Table |
| Accuracy Over Time | Query #2 | Line chart |
| Best Models | Query #3 | Table |

### Drift Monitoring Dashboard
| Card | SQL Query | Chart Type |
|------|-----------|------------|
| Drift Score Over Time | Query #1 | Line chart |
| Recent Drift Alerts | Query #2 | Table |
| Daily Drift Summary | Query #3 | Bar chart |
| Current Drift Status | Query #4 | Table |

## Auto-Refresh

Set dashboard auto-refresh to **1 minute** for real-time monitoring:
- Click the clock icon in the dashboard toolbar
- Select "1 minute"

## Disclaimer

> All data shown is from the AI-assisted severity classification system.
> **This is NOT a medical diagnosis tool.**
> For educational and research purposes only.
