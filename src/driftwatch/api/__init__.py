"""FastAPI serving application and schemas."""

from driftwatch.api.app import app
from driftwatch.api.schemas import (
    ActiveModelResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    GroundTruthIngestRequest,
    GroundTruthIngestResponse,
    HealthResponse,
    ModelDetailResponse,
    PredictRequest,
    PredictResponse,
)

__all__ = [
    "ActiveModelResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "GroundTruthIngestRequest",
    "GroundTruthIngestResponse",
    "HealthResponse",
    "ModelDetailResponse",
    "PredictRequest",
    "PredictResponse",
    "app",
]
