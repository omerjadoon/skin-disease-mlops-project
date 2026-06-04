.DEFAULT_GOAL := help
COMPOSE = docker compose
PROJECT = skin-severity-mlops

.PHONY: help build up down logs seed-data train evaluate api-test test lint format dvc-repro clean init-dvc

help: ## Show this help message
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║        Skin Severity MLOps Makefile          ║"
	@echo "║  Educational/Research use only               ║"
	@echo "╚══════════════════════════════════════════════╝"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Docker ───────────────────────────────────────────────────

build: ## Build all Docker images
	@echo "🔨 Building all Docker images..."
	cp .env.example .env 2>/dev/null || true
	$(COMPOSE) build

up: ## Start all services
	@echo "🚀 Starting all services..."
	cp .env.example .env 2>/dev/null || true
	$(COMPOSE) up -d
	@echo ""
	@echo "✅ Services started:"
	@echo "   FastAPI:   http://localhost:8000"
	@echo "   MLflow:    http://localhost:5000"
	@echo "   Airflow:   http://localhost:8080  (admin/admin)"
	@echo "   MinIO:     http://localhost:9001  (minioadmin/minioadmin123)"
	@echo "   Metabase:  http://localhost:3000"
	@echo "   FiftyOne:  http://localhost:5151"

down: ## Stop all services
	@echo "🛑 Stopping all services..."
	$(COMPOSE) down

down-volumes: ## Stop all services and remove volumes
	@echo "🛑 Stopping all services and removing volumes..."
	$(COMPOSE) down -v

logs: ## Tail logs from all services
	$(COMPOSE) logs -f

logs-api: ## Tail API logs
	$(COMPOSE) logs -f api

logs-trainer: ## Tail trainer logs
	$(COMPOSE) logs -f trainer

logs-mlflow: ## Tail MLflow logs
	$(COMPOSE) logs -f mlflow

# ─── Data ─────────────────────────────────────────────────────

seed-data: ## Generate synthetic sample dataset
	@echo "🌱 Seeding sample data..."
	python scripts/seed_sample_data.py
	@echo "✅ Sample data created in data/raw/ and data/processed/"

# ─── Training & Evaluation ────────────────────────────────────

train: ## Run model training (via Docker)
	@echo "🧠 Starting training..."
	$(COMPOSE) run --rm trainer python training/train.py --config configs/train.yaml

train-local: ## Run model training locally (no Docker)
	@echo "🧠 Starting local training..."
	python training/train.py --config configs/train.yaml

evaluate: ## Run model evaluation
	@echo "📊 Running evaluation..."
	$(COMPOSE) run --rm trainer python training/evaluate.py --config configs/train.yaml

# ─── API ──────────────────────────────────────────────────────

api-test: ## Test the FastAPI prediction endpoint with a sample image
	@echo "🔬 Testing API endpoint..."
	@if [ -f data/sample/test_image.jpg ]; then \
		curl -X POST http://localhost:8000/predict \
			-F "file=@data/sample/test_image.jpg" \
			-H "accept: application/json" | python -m json.tool; \
	else \
		echo "⚠️  No test image found. Run 'make seed-data' first."; \
	fi

api-health: ## Check API health
	curl -s http://localhost:8000/health | python -m json.tool

api-model-info: ## Get model info from API
	curl -s http://localhost:8000/model-info | python -m json.tool

# ─── Quality ──────────────────────────────────────────────────

test: ## Run pytest test suite
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --tb=short --cov=training --cov=api --cov=monitoring --cov-report=term-missing

test-docker: ## Run tests inside Docker
	$(COMPOSE) run --rm test-runner

lint: ## Run ruff linter
	@echo "🔍 Running ruff..."
	python -m ruff check training/ api/ monitoring/ scripts/ tests/

format: ## Auto-format code with ruff
	@echo "✨ Formatting code..."
	python -m ruff format training/ api/ monitoring/ scripts/ tests/
	python -m ruff check --fix training/ api/ monitoring/ scripts/ tests/

typecheck: ## Run mypy type checker
	@echo "🔎 Running mypy..."
	python -m mypy training/ api/ monitoring/ --ignore-missing-imports

# ─── DVC ──────────────────────────────────────────────────────

init-dvc: ## Initialize DVC and configure remote
	dvc init --no-scm 2>/dev/null || true
	dvc remote add -d minio s3://dvc-storage --force
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id minioadmin
	dvc remote modify minio secret_access_key minioadmin123

dvc-repro: ## Reproduce DVC pipeline
	@echo "♻️  Running DVC pipeline..."
	dvc repro

dvc-push: ## Push data/artifacts to MinIO via DVC
	dvc push

dvc-pull: ## Pull data/artifacts from MinIO via DVC
	dvc pull

# ─── Monitoring ───────────────────────────────────────────────

monitor: ## Run performance monitoring
	python monitoring/performance_monitor.py

drift-check: ## Check for prediction drift
	python monitoring/drift.py

# ─── Cleanup ──────────────────────────────────────────────────

clean: ## Remove generated artifacts
	@echo "🧹 Cleaning up..."
	rm -rf artifacts/model.ckpt artifacts/metrics.json artifacts/eval_metrics.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
