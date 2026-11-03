"""SQLAlchemy ORM Models for DriftWatch Platform.

Entities:
- Model & ModelVersion (champion/challenger registry metadata)
- Dataset & ReferenceDataset (baseline feature moments and quantiles)
- PredictionLog (operational inferences with request tracking)
- GroundTruthLabel (delayed true outcomes)
- DriftReport & FeatureDriftResultModel (statistical drift history)
- PerformanceReportModel (bootstrap CI performance evaluation)
- AlertModel (tiered alerts with evidence classification)
- RetrainingJobModel & ModelEvaluationModel (champion-challenger audits)
- AuditLogModel (immutable audit trail)
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from driftwatch.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class Model(Base):
    __tablename__ = "models"

    id = Column(String(64), primary_key=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    task_type = Column(String(64), default="demand_forecasting", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(64), primary_key=True)
    model_id = Column(String(64), ForeignKey("models.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    artifact_uri = Column(String(512), nullable=False)  # Path or MLflow URI (no binary in DB)
    params_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    is_active_champion = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    model = relationship("Model", back_populates="versions")
    predictions = relationship("PredictionLog", back_populates="model_version")

    __table_args__ = (
        Index("ix_model_versions_active", "model_id", "is_active_champion"),
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False, index=True)
    num_rows = Column(Integer, nullable=False)
    dataset_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class ReferenceDataset(Base):
    __tablename__ = "reference_datasets"

    id = Column(String(64), primary_key=True)
    model_version_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False, index=True)
    dataset_id = Column(String(64), ForeignKey("datasets.id"), nullable=False)
    quantiles_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    prediction_id = Column(String(64), primary_key=True, index=True)
    model_version_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False, index=True)
    request_id = Column(String(64), nullable=False, index=True)
    features_json = Column(Text, nullable=False)
    prediction = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)

    model_version = relationship("ModelVersion", back_populates="predictions")
    ground_truth = relationship("GroundTruthLabel", back_populates="prediction_log", uselist=False)

    __table_args__ = (
        Index("ix_pred_logs_model_time", "model_version_id", "timestamp"),
    )


class GroundTruthLabel(Base):
    __tablename__ = "ground_truth_labels"

    prediction_id = Column(String(64), ForeignKey("prediction_logs.prediction_id"), primary_key=True, index=True)
    true_label = Column(Float, nullable=False)
    observed_at = Column(DateTime, default=utc_now, nullable=False, index=True)
    delay_seconds = Column(Float, nullable=False)

    prediction_log = relationship("PredictionLog", back_populates="ground_truth")


class DriftReportModel(Base):
    __tablename__ = "drift_reports"

    id = Column(String(64), primary_key=True)
    dataset_id = Column(String(64), nullable=False, index=True)
    reference_id = Column(String(64), nullable=False)
    window_id = Column(String(128), nullable=False, index=True)
    config_hash = Column(String(32), nullable=False)
    is_drift_detected = Column(Boolean, nullable=False, index=True)
    drifted_features_count = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)

    feature_results = relationship("FeatureDriftResultModel", back_populates="report", cascade="all, delete-orphan")


class FeatureDriftResultModel(Base):
    __tablename__ = "feature_drift_results"

    id = Column(String(64), primary_key=True)
    report_id = Column(String(64), ForeignKey("drift_reports.id"), nullable=False, index=True)
    feature = Column(String(128), nullable=False, index=True)
    test = Column(String(64), nullable=False)
    statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    adjusted_p_value = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    decision = Column(Boolean, nullable=False, index=True)
    sample_size_ref = Column(Integer, nullable=False)
    sample_size_curr = Column(Integer, nullable=False)
    evidence_class = Column(String(64), nullable=False)
    details_json = Column(Text, nullable=True)

    report = relationship("DriftReportModel", back_populates="feature_results")


class PerformanceReportModel(Base):
    __tablename__ = "performance_reports"

    id = Column(String(64), primary_key=True)
    model_version_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False, index=True)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    sample_size = Column(Integer, nullable=False)
    labeled_count = Column(Integer, nullable=False)
    label_coverage_ratio = Column(Float, nullable=False)
    status = Column(String(32), nullable=False)  # VERIFIED or UNVERIFIED
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    wape = Column(Float, nullable=True)
    mase = Column(Float, nullable=True)
    bias = Column(Float, nullable=True)
    ci_lower = Column(Float, nullable=True)
    ci_upper = Column(Float, nullable=True)
    degradation_breached = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

    __table_args__ = (
        Index("ix_perf_reports_window", "model_version_id", "window_start", "window_end"),
    )


class AlertModel(Base):
    __tablename__ = "alerts"

    id = Column(String(64), primary_key=True)
    alert_type = Column(String(64), nullable=False, index=True)  # DRIFT or PERFORMANCE_DEGRADATION
    evidence_class = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False, index=True)    # INFO, WARNING, CRITICAL
    status = Column(String(32), default="ACTIVE", nullable=False, index=True)  # ACTIVE, ACKNOWLEDGED, RESOLVED
    affected_features_json = Column(Text, nullable=True)
    reason = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)


class RetrainingJobModel(Base):
    __tablename__ = "retraining_jobs"

    id = Column(String(64), primary_key=True)
    trigger_policy = Column(String(64), nullable=False)  # SUSTAINED_DEGRADATION, MANUAL, SCHEDULED
    status = Column(String(32), default="PENDING", nullable=False, index=True)
    champion_version_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False)
    candidate_version_id = Column(String(64), nullable=True)
    metrics_comparison_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)


class ModelEvaluationModel(Base):
    __tablename__ = "model_evaluations"

    id = Column(String(64), primary_key=True)
    champion_version_id = Column(String(64), nullable=False)
    candidate_version_id = Column(String(64), nullable=False)
    metric_name = Column(String(32), nullable=False)
    champion_metric = Column(Float, nullable=False)
    candidate_metric = Column(Float, nullable=False)
    p_value_paired = Column(Float, nullable=False)
    is_stat_significant = Column(Boolean, nullable=False)
    decision = Column(String(32), nullable=False)  # APPROVED or REJECTED
    created_at = Column(DateTime, default=utc_now, nullable=False)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True)
    action = Column(String(64), nullable=False, index=True)  # PROMOTE, ROLLBACK, RETRAIN_TRIGGER, ALERT_ACK
    actor = Column(String(64), nullable=False)
    entity_type = Column(String(64), nullable=False)
    entity_id = Column(String(64), nullable=False)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
