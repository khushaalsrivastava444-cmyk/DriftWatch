"""Drift-Impact Attribution and Feature Importance Gating.

Suppresses alert fatigue by evaluating whether drifted features actually impact
the champion model's forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from driftwatch.alerts.engine import AlertSeverity
from driftwatch.drift.results import EvidenceClass, FeatureDriftResult
from driftwatch.models.xgboost_forecaster import XGBoostDemandForecaster

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttributedDriftResult:
    """Statistical drift result enriched with model importance attribution."""
    feature: str
    is_drifted: bool
    importance_score: float
    importance_percent: float
    is_impactful: bool
    recommended_severity: AlertSeverity
    evidence_class: EvidenceClass
    attribution_reason: str


class DriftAttributionEngine:
    """Evaluates the operational impact of statistical feature drift using model feature importance."""

    def __init__(
        self,
        forecaster: XGBoostDemandForecaster,
        min_importance_threshold: float = 0.02,  # 2% of total predictive importance
    ) -> None:
        self.forecaster = forecaster
        self.min_importance_threshold = min_importance_threshold
        self.feature_importances_ = self._extract_feature_importances()

    def _extract_feature_importances(self) -> dict[str, float]:
        """Extract normalized feature importances from fitted XGBoost model."""
        if not self.forecaster.is_fitted_ or self.forecaster.model_ is None:
            return {}

        raw_imp = self.forecaster.model_.feature_importances_
        features = self.forecaster.feature_names_
        total = float(np.sum(raw_imp))

        if total <= 0:
            return {f: 1.0 / len(features) for f in features}

        return {f: float(imp / total) for f, imp in zip(features, raw_imp)}

    def attribute_drift_result(self, result: FeatureDriftResult) -> AttributedDriftResult:
        """Attribute predictive significance to an individual feature drift result."""
        importance = self.feature_importances_.get(result.feature, 0.0)
        importance_pct = round(importance * 100.0, 2)
        is_impactful = importance >= self.min_importance_threshold

        if not result.decision:
            return AttributedDriftResult(
                feature=result.feature,
                is_drifted=False,
                importance_score=importance,
                importance_percent=importance_pct,
                is_impactful=is_impactful,
                recommended_severity=AlertSeverity.INFO,
                evidence_class=EvidenceClass.STATISTICAL_SHIFT,
                attribution_reason="Feature is statistically stable; no alert required.",
            )

        if is_impactful:
            # Significant drift in a high-impact feature -> WARNING
            return AttributedDriftResult(
                feature=result.feature,
                is_drifted=True,
                importance_score=importance,
                importance_percent=importance_pct,
                is_impactful=True,
                recommended_severity=AlertSeverity.WARNING,
                evidence_class=EvidenceClass.STATISTICAL_SHIFT,
                attribution_reason=(
                    f"Drift detected in high-importance feature '{result.feature}' "
                    f"({importance_pct}% of predictive weight). Model predictions likely affected."
                ),
            )
        else:
            # Significant drift in an unimportant feature -> Demote to INFO
            return AttributedDriftResult(
                feature=result.feature,
                is_drifted=True,
                importance_score=importance,
                importance_percent=importance_pct,
                is_impactful=False,
                recommended_severity=AlertSeverity.INFO,
                evidence_class=EvidenceClass.STATISTICAL_SHIFT,
                attribution_reason=(
                    f"Drift detected in low-importance feature '{result.feature}' "
                    f"({importance_pct}% < {self.min_importance_threshold:.0%}). "
                    f"WARNING suppressed to INFO to prevent alert fatigue."
                ),
            )
