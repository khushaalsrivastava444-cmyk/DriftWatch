# DriftWatch — Production MLOps Makefile

.PHONY: help install lint typecheck test benchmark demo ui api docker-build docker-up docker-down clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install      Install project in editable mode with development dependencies"
	@echo "  make lint         Run ruff linter checks across all source code"
	@echo "  make typecheck    Run mypy static type checking on driftwatch core"
	@echo "  make test         Execute complete pytest suite"
	@echo "  make benchmark    Run statistical detector benchmark (FPR, power, latency, Evidently)"
	@echo "  make demo         Execute end-to-end autonomous demonstration (< 2 min CPU-only)"
	@echo "  make ui           Launch 7-page dark charcoal Streamlit dashboard"
	@echo "  make api          Launch FastAPI serving and monitoring daemon"
	@echo "  make docker-build Build lean multi-stage production Docker container"
	@echo "  make docker-up    Spin up full Docker Compose stack (API, Postgres, MLflow, Dashboard)"
	@echo "  make docker-down  Tear down Docker Compose containers and networks"

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ scripts/

typecheck:
	mypy src/driftwatch

test:
	pytest tests/ -v

benchmark:
	$(PYTHON) monitoring/benchmark/run_benchmark.py

demo:
	$(PYTHON) scripts/run_demo.py

ui:
	streamlit run src/driftwatch/ui/dashboard_app.py

api:
	uvicorn driftwatch.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache mlruns reports/*.html
