-- Drift metrics dashboard
-- Drift score over time, drift alerts

-- 1. Drift score over time
SELECT
    DATE_TRUNC('hour', timestamp) AS hour,
    metric_name,
    ROUND(AVG(metric_value)::NUMERIC, 6) AS avg_drift_score,
    MAX(metric_value) AS max_drift_score,
    BOOL_OR(drift_detected) AS any_drift_detected,
    model_version
FROM drift_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY 1, 2, 6
ORDER BY 1 DESC;

-- 2. Drift alerts (only rows where drift was detected)
SELECT
    timestamp,
    metric_name,
    ROUND(metric_value::NUMERIC, 6) AS drift_score,
    baseline_window,
    current_window,
    model_version
FROM drift_metrics
WHERE drift_detected = TRUE
ORDER BY timestamp DESC
LIMIT 50;

-- 3. Daily drift summary
SELECT
    DATE_TRUNC('day', timestamp) AS day,
    COUNT(*) AS checks_run,
    COUNT(CASE WHEN drift_detected THEN 1 END) AS drift_alerts,
    ROUND(AVG(metric_value)::NUMERIC, 6) AS avg_psi_score,
    ROUND(MAX(metric_value)::NUMERIC, 6) AS max_psi_score
FROM drift_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1 DESC;

-- 4. Current drift status (latest reading)
SELECT
    metric_name,
    ROUND(metric_value::NUMERIC, 6) AS drift_score,
    drift_detected,
    CASE
        WHEN metric_value < 0.1 THEN 'Stable'
        WHEN metric_value < 0.2 THEN 'Moderate Drift'
        ELSE 'HIGH DRIFT - Retraining Recommended'
    END AS drift_status,
    timestamp
FROM drift_metrics
ORDER BY timestamp DESC
LIMIT 10;
