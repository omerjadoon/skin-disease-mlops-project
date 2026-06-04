#!/usr/bin/env bash
# Initialize MLflow experiment and verify connectivity
set -euo pipefail

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-skin-severity-experiment}"

echo "================================================"
echo "  MLflow Initialization"
echo "  Tracking URI: ${MLFLOW_TRACKING_URI}"
echo "================================================"

# Wait for MLflow to be ready
echo "Waiting for MLflow to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${MLFLOW_TRACKING_URI}/health" > /dev/null 2>&1; then
        echo "✓ MLflow is ready"
        break
    fi
    echo "  Attempt ${i}/30..."
    sleep 3
done

# Create experiment via Python
python -c "
import mlflow
import os

tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '${MLFLOW_TRACKING_URI}')
experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', '${EXPERIMENT_NAME}')

mlflow.set_tracking_uri(tracking_uri)

# Create or get experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    exp_id = mlflow.create_experiment(
        experiment_name,
        artifact_location='s3://mlflow-artifacts',
        tags={'project': 'skin-severity-mlops', 'disclaimer': 'educational-research-only'}
    )
    print(f'✓ Created experiment: {experiment_name} (id={exp_id})')
else:
    print(f'✓ Experiment already exists: {experiment_name} (id={experiment.experiment_id})')
"

echo ""
echo "✅ MLflow initialization complete!"
echo "   UI: ${MLFLOW_TRACKING_URI}"
