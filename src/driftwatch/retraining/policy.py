"""Retraining Trigger Policies and Governance Rules.

Enforces:
1. Drift alone NEVER auto-retrains (policy explicitly rejects statistical drift without verified degradation).
2. SUSTAINED_DEGRADATION requires consecutive verified performance breaches (k-of-n).
3. Explicit support for MANUAL and SCHEDULED triggers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from driftwatch.db.models import PerformanceReportModel

logger = logging.getLogger(__name__)


class RetrainingTriggerPolicy(str, Enum):
    SUSTAINED_DEGRADATION = "SUSTAINED_DEGRADATION"
    MANUAL = "MANUAL"
    SCHEDULED = "SCHEDULED"
    DRIFT_ONLY = "DRIFT_ONLY"


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """Outcome of evaluating a retraining trigger policy."""
    should_retrain: bool
    policy: RetrainingTriggerPolicy
    reason: str
    details: dict[str, Any]


def evaluate_retraining_trigger(
    policy: RetrainingTriggerPolicy,
    recent_performance_reports: list[PerformanceReportModel] | None = None,
    k_required: int = 2,
    n_recent: int = 3,
) -> PolicyEvaluationResult:
    """Evaluate whether a retraining pipeline should be launched.
    
    Hard rule: Drift alone NEVER auto-retrains.
    """
    if policy == RetrainingTriggerPolicy.DRIFT_ONLY:
        return PolicyEvaluationResult(
            should_retrain=False,
            policy=policy,
            reason="REJECTED: Statistical drift alone never auto-retrains without verified degradation.",
            details={"rule": "drift_is_not_degradation"},
        )

    if policy == RetrainingTriggerPolicy.MANUAL:
        return PolicyEvaluationResult(
            should_retrain=True,
            policy=policy,
            reason="Manual retraining triggered by authorized operator.",
            details={"trigger_source": "manual"},
        )

    if policy == RetrainingTriggerPolicy.SCHEDULED:
        return PolicyEvaluationResult(
            should_retrain=True,
            policy=policy,
            reason="Scheduled periodic retraining trigger activated.",
            details={"trigger_source": "scheduler"},
        )

    if policy == RetrainingTriggerPolicy.SUSTAINED_DEGRADATION:
        if not recent_performance_reports:
            return PolicyEvaluationResult(
                should_retrain=False,
                policy=policy,
                reason="No performance reports available to verify degradation.",
                details={"verified_reports_count": 0},
            )

        # Consider only VERIFIED reports
        verified_reports = [
            r for r in recent_performance_reports
            if r.status == "VERIFIED"
        ][:n_recent]

        if len(verified_reports) < k_required:
            return PolicyEvaluationResult(
                should_retrain=False,
                policy=policy,
                reason=f"Insufficient verified performance windows ({len(verified_reports)}/{k_required} required).",
                details={"verified_count": len(verified_reports)},
            )

        breached_count = sum(1 for r in verified_reports if r.degradation_breached)
        if breached_count >= k_required:
            return PolicyEvaluationResult(
                should_retrain=True,
                policy=policy,
                reason=f"Sustained verified degradation breached: {breached_count} of last {len(verified_reports)} windows breached CI band.",
                details={"breached_count": breached_count, "windows_inspected": len(verified_reports)},
            )
        else:
            return PolicyEvaluationResult(
                should_retrain=False,
                policy=policy,
                reason=f"Degradation is transient ({breached_count}/{k_required} breached). Hysteresis suppresses retraining.",
                details={"breached_count": breached_count, "windows_inspected": len(verified_reports)},
            )

    return PolicyEvaluationResult(
        should_retrain=False,
        policy=policy,
        reason=f"Unknown policy: {policy}",
        details={},
    )
