FROM python:3.11-slim

LABEL maintainer="skin-severity-mlops"
LABEL description="Training container for skin severity classifier (educational use only)"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install pytorch-lightning torchmetrics mlflow boto3 pyyaml omegaconf \
        pandas scikit-learn scipy pillow psycopg2-binary python-dotenv && \
    rm -rf /root/.cache/pip

# Copy source code
COPY training/ ./training/
COPY configs/ ./configs/
COPY data/ ./data/

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "training/train.py", "--config", "configs/train.yaml"]
