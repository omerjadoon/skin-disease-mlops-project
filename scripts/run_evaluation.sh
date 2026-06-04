#!/usr/bin/env bash
set -euo pipefail
echo "📊 Running model evaluation..."
docker compose run --rm trainer python training/evaluate.py --config configs/train.yaml "$@"
echo "✅ Evaluation complete"
