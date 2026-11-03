# Multi-stage production Dockerfile for DriftWatch Platform

FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# Final lean runtime image
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure unprivileged user
RUN useradd -m -u 1000 appuser

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source and scripts
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/

# Create runtime directories
RUN mkdir -p artifacts/models reports mlruns data \
    && chown -R appuser:appuser /app

USER appuser

# Install editable package pointer in runtime container
RUN pip install --no-deps -e .

EXPOSE 8000 8501 5000

ENV DRIFTWATCH_DB_URL="sqlite:///driftwatch.db" \
    MLFLOW_TRACKING_URI="sqlite:///artifacts/mlflow.db" \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "driftwatch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
