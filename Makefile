# DriftWatch Makefile

.PHONY: help install lint typecheck test benchmark demo ui api docker-build docker-up docker-down clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install      Install project in editable mode with development dependencies"
	@echo "  make lint         Run ruff checks across src, tests, and scripts"
	@echo "  make typecheck    Run mypy type checking on src/driftwatch"
	@echo "  make test         Run pytest suite"
	@echo "  make benchmark    Run detector benchmark (FPR, power, latency, Evidently)"
	@echo "  make demo         Run end-to-end demo (< 1 min CPU-only)"
	@echo "  make ui           Launch Streamlit dashboard"
	@echo "  make api          Launch FastAPI server"
	@echo "  make docker-build Build Docker container"
	@echo "  make docker-up    Start Docker Compose stack (API, Postgres, MLflow, Dashboard)"
	@echo "  make docker-down  Tear down Docker Compose containers"

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
