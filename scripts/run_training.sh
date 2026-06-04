#!/usr/bin/env bash
set -euo pipefail
echo "🚀 Starting model training..."
docker compose run --rm trainer python training/train.py --config configs/train.yaml "$@"
echo "✅ Training complete"
