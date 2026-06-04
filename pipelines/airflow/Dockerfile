FROM apache/airflow:2.8.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pytorch-lightning torchmetrics mlflow==2.11.1 boto3 pyyaml omegaconf \
        pandas scikit-learn scipy pillow psycopg2-binary python-dotenv
