-- Model performance dashboard
-- Model version history, accuracy, F1 score, evaluation date

-- 1. Latest evaluation per model version
SELECT
    model_name,
    model_version,
    split,
    ROUND(accuracy::NUMERIC, 4) AS accuracy,
    ROUND(f1_macro::NUMERIC, 4) AS f1_macro,
    ROUND(precision_macro::NUMERIC, 4) AS precision_macro,
    ROUND(recall_macro::NUMERIC, 4) AS recall_macro,
    ROUND(auroc::NUMERIC, 4) AS auroc,
    evaluated_at
FROM model_performance
ORDER BY evaluated_at DESC;

-- 2. Model performance over time (accuracy trend)
SELECT
    DATE_TRUNC('day', evaluated_at) AS eval_date,
    model_version,
    ROUND(AVG(accuracy)::NUMERIC, 4) AS avg_accuracy,
    ROUND(AVG(f1_macro)::NUMERIC, 4) AS avg_f1_macro
FROM model_performance
WHERE split = 'test'
GROUP BY 1, 2
ORDER BY 1 DESC;

-- 3. Best performing model
SELECT
    model_name,
    model_version,
    ROUND(accuracy::NUMERIC, 4) AS accuracy,
    ROUND(f1_macro::NUMERIC, 4) AS f1_macro,
    evaluated_at
FROM model_performance
WHERE split = 'test'
ORDER BY f1_macro DESC
LIMIT 5;

-- 4. Model performance comparison
SELECT
    model_version,
    ROUND(AVG(accuracy)::NUMERIC, 4) AS accuracy,
    ROUND(AVG(f1_macro)::NUMERIC, 4) AS f1_macro,
    ROUND(AVG(auroc)::NUMERIC, 4) AS auroc,
    COUNT(*) AS eval_count
FROM model_performance
GROUP BY model_version
ORDER BY AVG(f1_macro) DESC;
