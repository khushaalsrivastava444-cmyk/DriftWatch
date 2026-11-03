"""Alert and Decision Engine with Evidence Classification."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from enum import Enum

from sqlalchemy.orm import Session

from driftwatch.alerts.notifiers import BaseNotifier, LogNotifier
from driftwatch.db.models import AlertModel
from driftwatch.drift.engine import DriftReport
from driftwatch.drift.results import EvidenceClass
from driftwatch.drift.windowing import HysteresisStateTracker
from driftwatch.monitoring.performance import EvaluatedPerformanceWindow

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class AlertEngine:
    """Evaluates drift and performance reports, enforcing strict evidence classifications and hysteresis."""

    def __init__(
        self,
        notifiers: list[BaseNotifier] | None = None,
        k_hysteresis: int = 1,
        n_hysteresis: int = 1,
        hysteresis_margin: float = 0.03,
    ) -> None:
        self.notifiers = notifiers if notifiers is not None else [LogNotifier()]
        self.k_hysteresis = k_hysteresis
        self.n_hysteresis = n_hysteresis
        self.hysteresis_margin = hysteresis_margin
        self.trackers: dict[str, HysteresisStateTracker] = {}

    def process_drift_report(
        self,
        report: DriftReport,
        db: Session,
    ) -> list[AlertModel]:
        """Process statistical drift results and generate appropriate WARNING alerts.
        
        Hard rules:
        - Drift alone NEVER raises a CRITICAL alert (strictly WARNING).
        - Stateful hysteresis filters transient spikes (requires k-of-n window escalation).
        """
        created_alerts: list[AlertModel] = []

        escalated_feats: list[str] = []
        for r in report.results:
            tracker = self.trackers.setdefault(
                r.feature,
                HysteresisStateTracker(
                    k=self.k_hysteresis,
                    n=self.n_hysteresis,
                    threshold=r.threshold,
                    hysteresis_margin=self.hysteresis_margin,
                ),
            )
            val = float(r.effect_size) if r.decision else 0.0
            is_alerting, _ = tracker.update(key=r.feature, value=val)
            if is_alerting or (self.k_hysteresis == 1 and self.n_hysteresis == 1 and r.decision):
                escalated_feats.append(r.feature)

        if not escalated_feats:
            return created_alerts

        affected_feats = sorted(list(set(escalated_feats)))
        now_dt = datetime.now(UTC)
        alert_id = f"alert_drift_{uuid.uuid4().hex[:10]}"

        # Feature drift represents a statistical shift
        alert = AlertModel(
            id=alert_id,
            alert_type="FEATURE_DRIFT",
            evidence_class=EvidenceClass.STATISTICAL_SHIFT.value,
            severity=AlertSeverity.WARNING.value,  # Drift alone is WARNING, never CRITICAL
            status=AlertStatus.ACTIVE.value,
            affected_features_json=json.dumps(affected_feats),
            reason=f"Significant statistical drift confirmed via hysteresis in {len(affected_feats)} feature(s): {', '.join(affected_feats[:5])}",
            recommended_action="Inspect data ingestion pipelines for schema changes or upstream sensor shifts; evaluate delayed labels.",
            created_at=now_dt,
        )
        db.add(alert)
        db.commit()

        self._dispatch_alert(alert)
        created_alerts.append(alert)
        return created_alerts

    def process_performance_evaluation(
        self,
        window: EvaluatedPerformanceWindow,
        db: Session,
    ) -> AlertModel | None:
        """Process performance evaluation window and generate alerts.
        
        Hard rule: CRITICAL severity strictly requires VERIFIED_PERFORMANCE degradation.
        Unverified data or low-coverage drops cannot raise CRITICAL alerts.
        """
        now_dt = datetime.now(UTC)

        # 1. Check if degradation was verified
        if window.status == "VERIFIED" and window.degradation_breached:
            mae_val = window.mae_ci.point_estimate if window.mae_ci else 0.0
            ci_low = window.mae_ci.ci_lower if window.mae_ci else 0.0
            ci_high = window.mae_ci.ci_upper if window.mae_ci else 0.0

            alert_id = f"alert_perf_{uuid.uuid4().hex[:10]}"
            alert = AlertModel(
                id=alert_id,
                alert_type="PERFORMANCE_DEGRADATION",
                evidence_class=EvidenceClass.VERIFIED_PERFORMANCE.value,
                severity=AlertSeverity.CRITICAL.value,  # Strictly verified degradation triggers CRITICAL
                status=AlertStatus.ACTIVE.value,
                affected_features_json=json.dumps(["target_sales_error"]),
                reason=f"Verified performance degradation breach: MAE {mae_val:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}]) exceeds threshold.",
                recommended_action="Trigger champion-challenger model retraining pipeline with verified ground-truth labels.",
                created_at=now_dt,
            )
            db.add(alert)
            db.commit()
            self._dispatch_alert(alert)
            return alert

        # 2. Check for low label maturity (Unverified state warning)
        elif window.status == "UNVERIFIED" and window.total_predictions >= 20:
            alert_id = f"alert_unverif_{uuid.uuid4().hex[:10]}"
            alert = AlertModel(
                id=alert_id,
                alert_type="LABEL_MATURITY_WARNING",
                evidence_class=EvidenceClass.ESTIMATED_PERFORMANCE.value,
                severity=AlertSeverity.INFO.value,
                status=AlertStatus.ACTIVE.value,
                affected_features_json=json.dumps(["ground_truth_labels"]),
                reason=f"Window has inadequate label maturity: coverage is {window.label_coverage_ratio:.1%}, status remains UNVERIFIED.",
                recommended_action="Wait for asynchronous retail receipts and delivery confirmations before drawing performance conclusions.",
                created_at=now_dt,
            )
            db.add(alert)
            db.commit()
            self._dispatch_alert(alert)
            return alert

        return None

    def acknowledge_alert(self, alert_id: str, db: Session) -> bool:
        """Transition alert state to ACKNOWLEDGED."""
        alert = db.get(AlertModel, alert_id)
        if not alert or alert.status != AlertStatus.ACTIVE.value:
            return False
        alert.status = AlertStatus.ACKNOWLEDGED.value
        db.commit()
        logger.info("Alert %s transitioned to ACKNOWLEDGED", alert_id)
        return True

    def resolve_alert(self, alert_id: str, db: Session) -> bool:
        """Transition alert state to RESOLVED."""
        alert = db.get(AlertModel, alert_id)
        if not alert:
            return False
        alert.status = AlertStatus.RESOLVED.value
        alert.resolved_at = datetime.now(UTC)
        db.commit()
        logger.info("Alert %s transitioned to RESOLVED", alert_id)
        return True

    def _dispatch_alert(self, alert: AlertModel) -> None:
        payload = {
            "id": alert.id,
            "type": alert.alert_type,
            "severity": alert.severity,
            "evidence_class": alert.evidence_class,
            "reason": alert.reason,
            "recommended_action": alert.recommended_action,
            "created_at": alert.created_at.isoformat(),
        }
        for notifier in self.notifiers:
            try:
                notifier.send_alert(payload)
            except Exception as e:
                logger.error("Notifier failed to dispatch alert: %s", e)
