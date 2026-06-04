-- Prediction metrics dashboard
-- Predictions per day, class distribution, average confidence

-- 1. Predictions per day
SELECT
    DATE_TRUNC('day', timestamp) AS day,
    COUNT(*) AS total_predictions,
    COUNT(CASE WHEN predicted_class = 'mild' THEN 1 END) AS mild_count,
    COUNT(CASE WHEN predicted_class = 'moderate' THEN 1 END) AS moderate_count,
    COUNT(CASE WHEN predicted_class = 'severe' THEN 1 END) AS severe_count,
    ROUND(AVG(confidence)::NUMERIC, 4) AS avg_confidence,
    ROUND(AVG(latency_ms)::NUMERIC, 2) AS avg_latency_ms
FROM prediction_logs
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1 DESC;

-- 2. Overall class distribution (last 7 days)
SELECT
    predicted_class,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM prediction_logs
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY predicted_class
ORDER BY count DESC;

-- 3. Hourly request volume (last 24 hours)
SELECT
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) AS requests,
    ROUND(AVG(confidence)::NUMERIC, 4) AS avg_confidence,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::NUMERIC, 2) AS p95_latency_ms
FROM prediction_logs
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY 1
ORDER BY 1 DESC;

-- 4. Average confidence by class
SELECT
    predicted_class,
    ROUND(AVG(confidence)::NUMERIC, 4) AS avg_confidence,
    ROUND(MIN(confidence)::NUMERIC, 4) AS min_confidence,
    ROUND(MAX(confidence)::NUMERIC, 4) AS max_confidence,
    COUNT(*) AS num_predictions
FROM prediction_logs
GROUP BY predicted_class;

-- 5. Predictions by model version
SELECT
    model_version,
    COUNT(*) AS predictions,
    ROUND(AVG(confidence)::NUMERIC, 4) AS avg_confidence,
    MIN(timestamp) AS first_seen,
    MAX(timestamp) AS last_seen
FROM prediction_logs
GROUP BY model_version
ORDER BY last_seen DESC;
