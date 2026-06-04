FROM python:3.11-slim

LABEL maintainer="skin-severity-mlops"
LABEL description="FastAPI inference API for skin severity classification (educational use only)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install fastapi "uvicorn[standard]" pydantic pydantic-settings \
        mlflow==2.11.1 boto3 psycopg2-binary pyyaml pillow python-multipart \
        python-dotenv pytorch-lightning torchmetrics && \
    rm -rf /root/.cache/pip

# Copy source
COPY api/ ./api/
COPY training/ ./training/
COPY configs/ ./configs/

RUN mkdir -p /app/artifacts

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
