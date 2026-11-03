"""FastAPI Application for DriftWatch Inference, Monitoring, and Model Management."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from driftwatch.api.auth import verify_api_key
from driftwatch.api.schemas import (
    ActiveModelResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    GroundTruthIngestRequest,
    GroundTruthIngestResponse,
    HealthResponse,
    ModelDetailResponse,
    ModelVersionSummary,
    PredictRequest,
    PredictResponse,
    StructuredErrorResponse,
)
from driftwatch.api.service import serving_service
from driftwatch.db.base import get_db
from driftwatch.db.models import (
    AuditLogModel,
    Model,
    ModelVersion,
)
from driftwatch.db.retention import init_db
from driftwatch.models.xgboost_forecaster import XGBoostDemandForecaster

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database schemas on server startup."""
    init_db()
    yield


app = FastAPI(
    title="DriftWatch ML Serving & Monitoring API",
    version="0.1.0",
    description="Real-time production model serving, drift surveillance, and delayed-label performance evaluation.",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_and_timer_middleware(request: Request, call_next: Any) -> JSONResponse:
    """Inject X-Request-ID and X-Response-Time-Ms into every response."""
    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    request.state.request_id = request_id

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def structured_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Structured JSON error response for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown_req")
    logger.exception("Unhandled error processing request %s: %s", request_id, exc)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=StructuredErrorResponse(
                error_code="HTTP_ERROR",
                message=str(exc.detail),
                request_id=request_id,
            ).model_dump(),
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=StructuredErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message=str(exc),
            request_id=request_id,
        ).model_dump(),
    )


# ============================================================================
# Core Inference Endpoints
# ============================================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="Generate single demand forecast and log prediction to database",
)
def predict_single(
    request: PredictRequest,
    req_http: Request,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Serve a single point forecast using the active champion model."""
    request_id = req_http.state.request_id
    return serving_service.predict_single(request, db, request_id)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    summary="Generate batch demand forecasts (max 5,000 items)",
)
def predict_batch(
    request: BatchPredictRequest,
    req_http: Request,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """Serve batch point forecasts."""
    request_id = req_http.state.request_id
    return serving_service.predict_batch(request, db, request_id)


# ============================================================================
# Ground-Truth Label Association
# ============================================================================

@app.post(
    "/api/v1/labels",
    response_model=GroundTruthIngestResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Ingest delayed ground-truth labels and associate with predictions",
)
def ingest_ground_truth(
    request: GroundTruthIngestRequest,
    db: Session = Depends(get_db),
) -> GroundTruthIngestResponse:
    """Ingest delayed ground-truth labels (auth protected)."""
    ingested, matched = serving_service.ingest_labels(request.labels, db)
    return GroundTruthIngestResponse(
        ingested_count=ingested,
        matched_predictions_count=matched,
        status="success",
    )


# ============================================================================
# Model Registry & Health Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, summary="System health check")
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Verify API and database operational status."""
    active_version = serving_service.ensure_default_model(db)
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database="connected",
        active_model_id=active_version,
    )


@app.get("/models", response_model=list[ModelDetailResponse], summary="List all registered models")
def list_models(db: Session = Depends(get_db)) -> list[ModelDetailResponse]:
    """Retrieve all models and their version history from the registry."""
    models = db.execute(select(Model)).scalars().all()
    results = []
    for m in models:
        versions = [
            ModelVersionSummary(
                id=v.id,
                version=v.version,
                is_active_champion=v.is_active_champion,
                created_at=v.created_at.isoformat(),
                artifact_uri=v.artifact_uri,
            )
            for v in m.versions
        ]
        results.append(
            ModelDetailResponse(
                id=m.id,
                name=m.name,
                task_type=m.task_type,
                versions=versions,
            )
        )
    return results


@app.get("/models/active", response_model=ActiveModelResponse, summary="Retrieve active champion model")
def get_active_model(db: Session = Depends(get_db)) -> ActiveModelResponse:
    """Retrieve details of currently active champion model."""
    serving_service.ensure_default_model(db)
    stmt = (
        select(ModelVersion)
        .join(Model)
        .where(ModelVersion.is_active_champion == True)
        .order_by(desc(ModelVersion.created_at))
    )
    active_version = db.execute(stmt).scalars().first()
    if not active_version:
        raise HTTPException(status_code=404, detail="No active champion model configured.")

    return ActiveModelResponse(
        model_id=active_version.model_id,
        model_name=active_version.model.name,
        version_id=active_version.id,
        version=active_version.version,
        is_active_champion=True,
    )


@app.get("/models/{model_id}", response_model=ModelDetailResponse, summary="Get model by ID")
def get_model(model_id: str, db: Session = Depends(get_db)) -> ModelDetailResponse:
    """Retrieve details and version history of a specific model."""
    m = db.get(Model, model_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    versions = [
        ModelVersionSummary(
            id=v.id,
            version=v.version,
            is_active_champion=v.is_active_champion,
            created_at=v.created_at.isoformat(),
            artifact_uri=v.artifact_uri,
        )
        for v in m.versions
    ]
    return ModelDetailResponse(
        id=m.id,
        name=m.name,
        task_type=m.task_type,
        versions=versions,
    )


# ============================================================================
# Admin Promotion & Rollback Endpoints
# ============================================================================

@app.post(
    "/api/v1/models/{version_id}/promote",
    dependencies=[Depends(verify_api_key)],
    summary="Promote a candidate model version to Active Champion",
)
def promote_candidate(
    version_id: str,
    actor: str = "admin",
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Promote a candidate model to Active Champion and demote previous champion (audited)."""
    target = db.get(ModelVersion, version_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Model version '{version_id}' not found.")

    # Find incumbent champion
    incumbents = (
        db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == target.model_id,
                ModelVersion.is_active_champion == True,
            )
        )
        .scalars()
        .all()
    )
    for inc in incumbents:
        inc.is_active_champion = False

    target.is_active_champion = True

    # Audit log entry
    audit = AuditLogModel(
        id=f"audit_{uuid.uuid4().hex[:12]}",
        action="PROMOTE",
        actor=actor,
        entity_type="ModelVersion",
        entity_id=version_id,
        details_json=f'{{"previous_champion": "{incumbents[0].id if incumbents else "none"}"}}',
    )
    db.add(audit)
    db.commit()

    serving_service.active_version_id = target.id
    serving_service.active_model_id = target.model_id
    art_path = Path(target.artifact_uri)
    if not art_path.exists():
        fallback = Path("artifacts/models") / art_path.name
        if fallback.exists():
            art_path = fallback
    if art_path.exists():
        forecaster = XGBoostDemandForecaster()
        forecaster.load_model(str(art_path))
        serving_service.active_forecaster = forecaster

    return {
        "status": "promoted",
        "new_champion_id": target.id,
        "version": target.version,
        "model_id": target.model_id,
    }


@app.post(
    "/api/v1/models/{model_id}/rollback",
    dependencies=[Depends(verify_api_key)],
    summary="Rollback active champion to previous champion version",
)
def rollback_champion(
    model_id: str,
    actor: str = "admin",
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Rollback to the previous stable champion version (audited)."""
    # Find current champion
    current = (
        db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.is_active_champion == True,
            )
        )
        .scalars()
        .first()
    )

    # Find previous version
    candidates = (
        db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id, ModelVersion.is_active_champion == False)
            .order_by(desc(ModelVersion.version))
        )
        .scalars()
        .all()
    )

    if not candidates:
        raise HTTPException(status_code=400, detail="No prior model version available for rollback.")

    prev = candidates[0]
    if current:
        current.is_active_champion = False
    prev.is_active_champion = True

    audit = AuditLogModel(
        id=f"audit_{uuid.uuid4().hex[:12]}",
        action="ROLLBACK",
        actor=actor,
        entity_type="ModelVersion",
        entity_id=prev.id,
        details_json=f'{{"rolled_back_from": "{current.id if current else "none"}"}}',
    )
    db.add(audit)
    db.commit()

    serving_service.active_version_id = prev.id
    art_path = Path(prev.artifact_uri)
    if not art_path.exists():
        fallback = Path("artifacts/models") / art_path.name
        if fallback.exists():
            art_path = fallback
    if art_path.exists():
        forecaster = XGBoostDemandForecaster()
        forecaster.load_model(str(art_path))
        serving_service.active_forecaster = forecaster

    return {
        "status": "rolled_back",
        "restored_champion_id": prev.id,
        "version": prev.version,
    }
