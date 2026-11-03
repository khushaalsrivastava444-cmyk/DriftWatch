"""Pydantic v2 Schemas for DriftWatch REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeaturePayload(BaseModel):
    """Features provided for an individual retail demand forecast."""
    lag_1: float
    lag_2: float = 0.0
    lag_7: float = 0.0
    lag_14: float = 0.0
    lag_28: float = 0.0
    rolling_mean_7: float = 0.0
    rolling_std_7: float = 0.0
    rolling_mean_14: float = 0.0
    rolling_mean_28: float = 0.0
    day_of_week: int = Field(ge=0, le=6)
    day_of_month: int = Field(ge=1, le=31)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    sell_price: float = Field(gt=0.0)
    price_discount_ratio: float = Field(default=1.0, ge=0.0)
    on_promotion: int = Field(default=0, ge=0, le=1)
    store_code: int = 0
    category_code: int = 0


class PredictRequest(BaseModel):
    """Single prediction request payload."""
    prediction_id: str | None = None
    features: FeaturePayload
    store_id: str | None = "STORE_01"
    item_id: str | None = "ITEM_001"
    date: str | None = None


class PredictResponse(BaseModel):
    """Single prediction response."""
    prediction_id: str
    request_id: str
    model_version_id: str
    prediction: float
    latency_ms: float
    timestamp: str


class BatchPredictRequest(BaseModel):
    """Batch prediction request with strict input size limit."""
    predictions: list[PredictRequest] = Field(max_length=5000)


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictResponse]
    total_count: int
    execution_time_ms: float
    request_id: str


class LabelItem(BaseModel):
    prediction_id: str
    true_label: float = Field(ge=0.0)
    observed_at: str | None = None


class GroundTruthIngestRequest(BaseModel):
    labels: list[LabelItem] = Field(max_length=10000)


class GroundTruthIngestResponse(BaseModel):
    ingested_count: int
    matched_predictions_count: int
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    active_model_id: str | None


class ModelVersionSummary(BaseModel):
    id: str
    version: int
    is_active_champion: bool
    created_at: str
    artifact_uri: str


class ModelDetailResponse(BaseModel):
    id: str
    name: str
    task_type: str
    versions: list[ModelVersionSummary]


class ActiveModelResponse(BaseModel):
    model_id: str
    model_name: str
    version_id: str
    version: int
    is_active_champion: bool


class StructuredErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)
