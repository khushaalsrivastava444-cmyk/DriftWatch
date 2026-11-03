"""Tests for Alert and Decision Engine.

Enforces:
1. Drift alone NEVER raises a CRITICAL alert (always WARNING with STATISTICAL_SHIFT).
2. CRITICAL severity strictly requires VERIFIED_PERFORMANCE degradation.
3. UNVERIFIED performance data cannot raise CRITICAL alerts.
4. State transitions: ACTIVE -> ACKNOWLEDGED -> RESOLVED.
5. Pluggable notifiers correctly receive structured alert payloads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from driftwatch.alerts.engine import AlertEngine, AlertSeverity, AlertStatus
from driftwatch.alerts.notifiers import BaseNotifier
from driftwatch.db.models import AlertModel, Base
from driftwatch.drift.engine import DriftReport
from driftwatch.drift.results import EvidenceClass, FeatureDriftResult, TestType
from driftwatch.monitoring.metrics import ForecastingMetrics
from driftwatch.monitoring.performance import (
    BootstrapConfidenceInterval,
    EvaluatedPerformanceWindow,
)


class MockRecordingNotifier(BaseNotifier):
    """Notifier that records dispatched payloads in memory for testing."""

    def __init__(self) -> None:
        self.dispatched: list[dict[str, Any]] = []

    def send_alert(self, payload: dict[str, Any]) -> None:
        self.dispatched.append(payload)


@pytest.fixture
def db_session():
    """In-memory SQLite database session fixture."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_drift_alert_severity_and_evidence_class(db_session):
    """Verifies feature drift creates WARNING alert with STATISTICAL_SHIFT, never CRITICAL."""
    notifier = MockRecordingNotifier()
    engine = AlertEngine(notifiers=[notifier])

    drift_res = FeatureDriftResult(
        dataset_id="test_data",
        reference_id="ref_data",
        window="w_2026_01",
        feature="lag_7",
        test=TestType.PSI,
        statistic=0.25,
        p_value=0.001,
        adjusted_p_value=0.002,
        effect_size=0.25,
        threshold=0.10,
        decision=True,  # Drift detected!
        sample_size_ref=1000,
        sample_size_curr=200,
        timestamp="2026-01-01T00:00:00Z",
        config_hash="abc12345",
        evidence_class=EvidenceClass.STATISTICAL_SHIFT,
    )
    report = DriftReport(
        dataset_id="test_data",
        reference_id="ref_data",
        window_id="w_2026_01",
        timestamp="2026-01-01T00:00:00Z",
        config_hash="abc12345",
        results=[drift_res],
    )

    alerts = engine.process_drift_report(report, db_session)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == AlertSeverity.WARNING.value
    assert alert.evidence_class == EvidenceClass.STATISTICAL_SHIFT.value
    assert alert.status == AlertStatus.ACTIVE.value
    assert "lag_7" in json.loads(alert.affected_features_json)
    assert len(notifier.dispatched) == 1
    assert notifier.dispatched[0]["severity"] == "WARNING"


def test_verified_performance_degradation_raises_critical(db_session):
    """Verifies statistically verified degradation breach raises CRITICAL alert."""
    notifier = MockRecordingNotifier()
    engine = AlertEngine(notifiers=[notifier])

    metrics = ForecastingMetrics(
        mae=12.5,
        rmse=16.8,
        wape=0.38,
        mase=1.2,
        bias=1.5,
        wrmsse=1.45,
    )
    ci = BootstrapConfidenceInterval(
        metric_name="MAE",
        point_estimate=12.5,
        ci_lower=11.2,
        ci_upper=13.8,
    )
    window = EvaluatedPerformanceWindow(
        window_id="window_001",
        window_start="2026-01-01T00:00:00Z",
        window_end="2026-01-07T00:00:00Z",
        total_predictions=100,
        labeled_count=95,
        label_coverage_ratio=0.95,
        mean_delay_seconds=3600.0,
        median_delay_seconds=3200.0,
        status="VERIFIED",
        metrics=metrics,
        mae_ci=ci,
        degradation_breached=True,
        details={},
    )

    alert = engine.process_performance_evaluation(window, db_session)

    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL.value
    assert alert.evidence_class == EvidenceClass.VERIFIED_PERFORMANCE.value
    assert alert.status == AlertStatus.ACTIVE.value
    assert "target_sales_error" in json.loads(alert.affected_features_json)
    assert "12.50" in alert.reason
    assert len(notifier.dispatched) == 1
    assert notifier.dispatched[0]["severity"] == "CRITICAL"


def test_unverified_performance_cannot_raise_critical(db_session):
    """Verifies that unverified performance (insufficient labels) never triggers CRITICAL."""
    notifier = MockRecordingNotifier()
    engine = AlertEngine(notifiers=[notifier])

    window = EvaluatedPerformanceWindow(
        window_id="window_unverified",
        window_start="2026-01-01T00:00:00Z",
        window_end="2026-01-07T00:00:00Z",
        total_predictions=50,
        labeled_count=10,
        label_coverage_ratio=0.20,  # Below 80% threshold
        mean_delay_seconds=7200.0,
        median_delay_seconds=7200.0,
        status="UNVERIFIED",
        metrics=None,
        mae_ci=None,
        degradation_breached=False,
        details={},
    )

    alert = engine.process_performance_evaluation(window, db_session)

    assert alert is not None
    # Crucial rule: UNVERIFIED is INFO, never CRITICAL
    assert alert.severity == AlertSeverity.INFO.value
    assert alert.evidence_class == EvidenceClass.ESTIMATED_PERFORMANCE.value
    assert alert.severity != AlertSeverity.CRITICAL.value


def test_alert_state_transitions_acknowledge_and_resolve(db_session):
    """Tests alert lifecycle: ACTIVE -> ACKNOWLEDGED -> RESOLVED."""
    engine = AlertEngine(notifiers=[])

    alert = AlertModel(
        id="alert_test_lifecycle",
        alert_type="FEATURE_DRIFT",
        evidence_class=EvidenceClass.STATISTICAL_SHIFT.value,
        severity=AlertSeverity.WARNING.value,
        status=AlertStatus.ACTIVE.value,
        affected_features_json=json.dumps(["price"]),
        reason="Drift detected",
        recommended_action="Inspect source",
        created_at=datetime.now(UTC),
    )
    db_session.add(alert)
    db_session.commit()

    # Step 1: Acknowledge
    ack_res = engine.acknowledge_alert("alert_test_lifecycle", db_session)
    assert ack_res is True
    persisted = db_session.get(AlertModel, "alert_test_lifecycle")
    assert persisted.status == AlertStatus.ACKNOWLEDGED.value

    # Cannot acknowledge again while in ACKNOWLEDGED status
    assert engine.acknowledge_alert("alert_test_lifecycle", db_session) is False

    # Step 2: Resolve
    resolve_res = engine.resolve_alert("alert_test_lifecycle", db_session)
    assert resolve_res is True
    persisted = db_session.get(AlertModel, "alert_test_lifecycle")
    assert persisted.status == AlertStatus.RESOLVED.value
    assert persisted.resolved_at is not None
