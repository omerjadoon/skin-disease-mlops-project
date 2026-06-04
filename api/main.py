"""
FastAPI inference service for skin disease severity classification.

DISCLAIMER: This API provides AI-assisted severity estimates only.
It is NOT a medical diagnosis tool. For educational/research purposes only.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.inference import build_prediction_response, preprocess_image, run_inference
from api.model_loader import get_model_loader
from api.schemas import (
    DISCLAIMER,
    HealthResponse,
    ModelInfoResponse,
    PredictionLog,
    PredictionResponse,
)

# ─── Config ───────────────────────────────────────────────────


def load_config() -> dict:
    config_path = Path("configs/api.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


APP_CONFIG = load_config()

# ─── Prediction Logger ────────────────────────────────────────

_db_conn = None


def get_db_connection():
    """Get or create PostgreSQL connection for prediction logging."""
    global _db_conn
    if _db_conn is not None:
        try:
            _db_conn.cursor().execute("SELECT 1")
            return _db_conn
        except Exception:
            _db_conn = None

    try:
        import psycopg2

        db_cfg = APP_CONFIG.get("database", {})
        _db_conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", db_cfg.get("host", "postgres")),
            port=int(os.environ.get("POSTGRES_PORT", db_cfg.get("port", 5432))),
            dbname=os.environ.get("POSTGRES_DB", db_cfg.get("name", "mlops_db")),
            user=os.environ.get("POSTGRES_USER", db_cfg.get("user", "mlops")),
            password=os.environ.get("POSTGRES_PASSWORD", db_cfg.get("password", "")),
        )
        _db_conn.autocommit = True
        _ensure_table(_db_conn)
        return _db_conn
    except Exception as e:
        print(f"⚠ DB connection failed: {e}")
        return None


def _ensure_table(conn) -> None:
    """Create prediction_logs table if it doesn't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id              SERIAL PRIMARY KEY,
        prediction_id   TEXT,
        timestamp       TIMESTAMPTZ DEFAULT NOW(),
        image_hash      TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence      FLOAT NOT NULL,
        prob_mild       FLOAT,
        prob_moderate   FLOAT,
        prob_severe     FLOAT,
        model_name      TEXT,
        model_version   TEXT,
        latency_ms      FLOAT,
        source_ip       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
        ON prediction_logs (timestamp);
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version
        ON prediction_logs (model_version);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


def log_prediction_to_db(log: PredictionLog, source_ip: str | None = None) -> None:
    """Insert prediction record into PostgreSQL."""
    conn = get_db_connection()
    if conn is None:
        return
    try:
        probs = log.class_probabilities
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_logs
                    (prediction_id, timestamp, image_hash, predicted_class,
                     confidence, prob_mild, prob_moderate, prob_severe,
                     model_name, model_version, latency_ms, source_ip)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    log.prediction_id,
                    log.timestamp,
                    log.image_hash,
                    log.predicted_class,
                    log.confidence,
                    probs.get("mild"),
                    probs.get("moderate"),
                    probs.get("severe"),
                    log.model_name,
                    log.model_version,
                    log.latency_ms,
                    source_ip,
                ),
            )
    except Exception as e:
        print(f"⚠ Failed to log prediction: {e}")


# ─── Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup."""
    print("=" * 55)
    print("  Skin Severity MLOps — Inference API")
    print("  DISCLAIMER: Not a medical diagnosis tool.")
    print("  Educational/research use only.")
    print("=" * 55)

    loader = get_model_loader()
    loader.load(APP_CONFIG)

    if loader.is_loaded:
        print(f"✓ Model ready: {loader.model_name} v{loader.model_version}")
    else:
        print("⚠ No model loaded — /predict will return 503 until model is available")

    yield

    print("Shutting down API...")


# ─── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Skin Severity MLOps API",
    description=(
        "AI-assisted skin disease severity classification. "
        "**DISCLAIMER: NOT a medical diagnosis tool. "
        "For educational and research purposes only.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_CONFIG.get("cors", {}).get("allow_origins", ["*"]),
    allow_methods=APP_CONFIG.get("cors", {}).get("allow_methods", ["GET", "POST"]),
    allow_headers=APP_CONFIG.get("cors", {}).get("allow_headers", ["*"]),
)

# ─── Routes ───────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API and model health."""
    loader = get_model_loader()
    return HealthResponse(
        status="ok",
        model_loaded=loader.is_loaded,
        model_name=loader.model_name,
        model_version=loader.model_version,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info() -> ModelInfoResponse:
    """Return model metadata and configuration."""
    loader = get_model_loader()
    if not loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = loader.model_info
    return ModelInfoResponse(
        model_name=info.get("model_name", loader.model_name),
        model_version=info.get("model_version", loader.model_version),
        backbone=info.get("backbone", "resnet18"),
        num_classes=info.get("num_classes", 3),
        class_names=info.get("class_names", ["mild", "moderate", "severe"]),
        input_size=info.get("input_size", [224, 224]),
        mlflow_tracking_uri=info.get("mlflow_tracking_uri"),
        disclaimer=DISCLAIMER,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """
    Classify skin disease severity from an uploaded image.

    Returns: severity (mild/moderate/severe), confidence, probabilities, and model info.

    **IMPORTANT:** This is an AI-assisted estimate only. NOT a medical diagnosis.
    """
    loader = get_model_loader()
    if not loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure a trained model is loaded.",
        )

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}",
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    # Time the inference
    start_time = time.perf_counter()

    try:
        # Preprocess
        image_size = APP_CONFIG.get("image_size", 224)
        norm = APP_CONFIG.get("normalize", {})
        tensor, image_hash = preprocess_image(
            image_bytes,
            image_size=image_size,
            normalize_mean=norm.get("mean"),
            normalize_std=norm.get("std"),
        )

        # Run inference
        model = loader.get_model()
        class_names = loader.model_info.get(
            "class_names", APP_CONFIG.get("class_names", ["mild", "moderate", "severe"])
        )
        result = run_inference(
            model=model,
            image_tensor=tensor,
            class_names=class_names,
            model_name=loader.model_name,
            model_version=loader.model_version,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}") from e

    latency_ms = (time.perf_counter() - start_time) * 1000
    prediction_id = str(uuid.uuid4())

    response = build_prediction_response(result, prediction_id, round(latency_ms, 2))

    # Log prediction to PostgreSQL (non-blocking, ignore errors)
    if APP_CONFIG.get("log_predictions", True):
        try:
            log = PredictionLog(
                prediction_id=prediction_id,
                image_hash=image_hash,
                predicted_class=result["severity"],
                confidence=result["confidence"],
                class_probabilities=result["class_probabilities"],
                model_name=loader.model_name,
                model_version=loader.model_version,
                latency_ms=round(latency_ms, 2),
            )
            source_ip = request.client.host if request.client else None
            log_prediction_to_db(log, source_ip)
        except Exception as e:
            print(f"⚠ Prediction logging failed: {e}")

    return response


@app.get("/", tags=["System"])
async def root() -> dict:
    """API root — redirect to docs."""
    return {
        "message": "Skin Severity MLOps API",
        "disclaimer": DISCLAIMER,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc!s}"},
    )


# ─── Entrypoint ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=APP_CONFIG.get("host", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", APP_CONFIG.get("port", 8000))),
        reload=APP_CONFIG.get("reload", False),
    )
