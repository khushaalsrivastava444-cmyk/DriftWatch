"""Streamlit UI Database Helper and Shared Components.

Provides safe database querying, explicit empty states, and styled evidence-class badges.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from driftwatch.db.base import SessionLocal
from driftwatch.db.models import (
    AlertModel,
    DriftReportModel,
    FeatureDriftResultModel,
    GroundTruthLabel,
    Model,
    ModelEvaluationModel,
    ModelVersion,
    PerformanceReportModel,
    PredictionLog,
    RetrainingJobModel,
)
from driftwatch.db.retention import init_db


def get_db_session() -> Session:
    """Ensure tables exist and return active session."""
    init_db()
    return SessionLocal()


def render_evidence_badge(evidence_class: str) -> str:
    """Render an HTML badge for evidence classes with distinct colors."""
    colors = {
        "STATISTICAL_SHIFT": "#1e88e5",  # Blue
        "PREDICTION_SHIFT": "#8e24aa",   # Purple
        "ESTIMATED_PERFORMANCE": "#f57c00",  # Orange
        "VERIFIED_PERFORMANCE": "#00d26a",   # Vibrant Green
    }
    color = colors.get(evidence_class, "#78909c")
    return (
        f'<span style="background-color: {color}; color: #ffffff; padding: 3px 8px; '
        f'border-radius: 4px; font-weight: 600; font-size: 0.82em; letter-spacing: 0.5px;">'
        f'{evidence_class}</span>'
    )


def render_severity_badge(severity: str) -> str:
    """Render an HTML badge for alert severities."""
    colors = {
        "INFO": "#0288d1",      # Light Blue
        "WARNING": "#ffa000",   # Amber
        "CRITICAL": "#e53935",  # Red
    }
    color = colors.get(severity, "#78909c")
    return (
        f'<span style="background-color: {color}; color: #ffffff; padding: 3px 8px; '
        f'border-radius: 4px; font-weight: 700; font-size: 0.82em;">'
        f'{severity}</span>'
    )


def render_status_badge(status_str: str) -> str:
    """Render an HTML badge for status flags (ACTIVE, VERIFIED, etc.)."""
    colors = {
        "ACTIVE": "#00d26a",
        "ACKNOWLEDGED": "#ffa000",
        "RESOLVED": "#546e7a",
        "VERIFIED": "#00d26a",
        "UNVERIFIED": "#e53935",
        "APPROVED": "#00d26a",
        "REJECTED": "#e53935",
        "COMPLETED": "#00d26a",
        "PENDING": "#ffa000",
    }
    color = colors.get(status_str, "#78909c")
    return (
        f'<span style="background-color: {color}22; color: {color}; border: 1px solid {color}; '
        f'padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 0.82em;">'
        f'{status_str}</span>'
    )


def query_system_overview(db: Session) -> dict[str, Any]:
    """Retrieve top-level summary metrics for system overview dashboard."""
    # Active Champion (most recently activated champion)
    active_version = db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_active_champion.is_(True))
        .order_by(desc(ModelVersion.created_at))
        .limit(1)
    ).scalar_one_or_none()

    total_predictions = db.query(func.count(PredictionLog.prediction_id)).scalar() or 0
    total_labels = db.query(func.count(GroundTruthLabel.prediction_id)).scalar() or 0
    label_coverage = (total_labels / total_predictions) if total_predictions > 0 else 0.0

    active_alerts_count = db.query(func.count(AlertModel.id)).filter(
        AlertModel.status == "ACTIVE"
    ).scalar() or 0

    critical_alerts_count = db.query(func.count(AlertModel.id)).filter(
        AlertModel.status == "ACTIVE", AlertModel.severity == "CRITICAL"
    ).scalar() or 0

    recent_perf = db.execute(
        select(PerformanceReportModel).order_by(desc(PerformanceReportModel.created_at)).limit(1)
    ).scalar_one_or_none()

    latest_drift_report = db.execute(
        select(DriftReportModel).order_by(desc(DriftReportModel.timestamp)).limit(1)
    ).scalar_one_or_none()

    return {
        "champion_version": active_version.version if active_version else None,
        "champion_model_id": active_version.model_id if active_version else None,
        "total_predictions": total_predictions,
        "total_labels": total_labels,
        "label_coverage": label_coverage,
        "active_alerts": active_alerts_count,
        "critical_alerts": critical_alerts_count,
        "recent_mae": recent_perf.mae if (recent_perf and recent_perf.mae is not None) else None,
        "recent_status": recent_perf.status if recent_perf else "NO_EVALUATION",
        "latest_drift_detected": latest_drift_report.is_drift_detected if latest_drift_report else False,
        "latest_drift_features_count": latest_drift_report.drifted_features_count if latest_drift_report else 0,
    }


def query_active_alerts(db: Session) -> list[AlertModel]:
    """Query currently ACTIVE alerts ordered by severity and recency."""
    return db.execute(
        select(AlertModel)
        .where(AlertModel.status == "ACTIVE")
        .order_by(desc(AlertModel.severity), desc(AlertModel.created_at))
    ).scalars().all()


def query_all_alerts(
    db: Session,
    severity: str | None = None,
    status: str | None = None,
) -> list[AlertModel]:
    """Query all alerts with optional filtering."""
    stmt = select(AlertModel)
    if severity and severity != "ALL":
        stmt = stmt.where(AlertModel.severity == severity)
    if status and status != "ALL":
        stmt = stmt.where(AlertModel.status == status)
    stmt = stmt.order_by(desc(AlertModel.created_at))
    return db.execute(stmt).scalars().all()


def query_drift_reports(db: Session, limit: int = 50) -> list[DriftReportModel]:
    """Query recent statistical drift reports."""
    return db.execute(
        select(DriftReportModel).order_by(desc(DriftReportModel.timestamp)).limit(limit)
    ).scalars().all()


def query_feature_drift_results(db: Session, report_id: str | None = None) -> list[FeatureDriftResultModel]:
    """Query feature-level drift results."""
    stmt = select(FeatureDriftResultModel)
    if report_id:
        stmt = stmt.where(FeatureDriftResultModel.report_id == report_id)
    stmt = stmt.order_by(desc(FeatureDriftResultModel.decision), desc(FeatureDriftResultModel.statistic))
    return db.execute(stmt).scalars().all()


def query_performance_reports(db: Session, limit: int = 50) -> list[PerformanceReportModel]:
    """Query chronological performance reports."""
    return db.execute(
        select(PerformanceReportModel).order_by(desc(PerformanceReportModel.window_start)).limit(limit)
    ).scalars().all()


def query_model_registry(db: Session) -> list[dict[str, Any]]:
    """Query registered models and their version history."""
    models = db.execute(select(Model)).scalars().all()
    results = []
    for m in models:
        versions = db.execute(
            select(ModelVersion).where(ModelVersion.model_id == m.id).order_by(desc(ModelVersion.version))
        ).scalars().all()
        results.append({
            "model": m,
            "versions": versions,
        })
    return results


def query_retraining_jobs(db: Session, limit: int = 50) -> list[RetrainingJobModel]:
    """Query retraining jobs."""
    return db.execute(
        select(RetrainingJobModel).order_by(desc(RetrainingJobModel.created_at)).limit(limit)
    ).scalars().all()


def query_model_evaluations(db: Session, limit: int = 50) -> list[ModelEvaluationModel]:
    """Query statistical champion-challenger evaluations."""
    return db.execute(
        select(ModelEvaluationModel).order_by(desc(ModelEvaluationModel.created_at)).limit(limit)
    ).scalars().all()
