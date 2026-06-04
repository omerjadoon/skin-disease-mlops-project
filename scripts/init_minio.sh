#!/usr/bin/env bash
# Initialize MinIO buckets for the MLOps project
# Requires: mc (MinIO client) to be installed and configured

set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin123}"
ALIAS="local-minio"

BUCKETS=(
    "mlflow-artifacts"
    "dvc-storage"
    "model-artifacts"
    "raw-images"
    "processed-images"
)

echo "================================================"
echo "  MinIO Bucket Initialization"
echo "  Endpoint: ${MINIO_ENDPOINT}"
echo "================================================"

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${MINIO_ENDPOINT}/minio/health/live" > /dev/null 2>&1; then
        echo "✓ MinIO is ready"
        break
    fi
    echo "  Attempt ${i}/30..."
    sleep 2
done

# Configure mc alias
mc alias set "${ALIAS}" "${MINIO_ENDPOINT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" \
    --api S3v4 > /dev/null 2>&1

echo "✓ MinIO client configured"

# Create buckets
for bucket in "${BUCKETS[@]}"; do
    if mc ls "${ALIAS}/${bucket}" > /dev/null 2>&1; then
        echo "  Bucket already exists: ${bucket}"
    else
        mc mb "${ALIAS}/${bucket}"
        echo "  ✓ Created bucket: ${bucket}"
    fi

    # Set public read policy on mlflow-artifacts for convenience
    if [ "${bucket}" = "mlflow-artifacts" ]; then
        mc anonymous set download "${ALIAS}/${bucket}" > /dev/null 2>&1 || true
    fi
done

echo ""
echo "✅ MinIO initialization complete!"
echo "   Buckets: ${BUCKETS[*]}"
echo "   UI: ${MINIO_ENDPOINT/9000/9001}"
