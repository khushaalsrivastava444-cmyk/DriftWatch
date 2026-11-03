"""Label-Free Performance Estimation via Density-Ratio Importance Weighting.

Estimates potential degradation on unlabeled streaming data before delayed labels arrive.
Strictly stamped as ESTIMATED_PERFORMANCE with documented assumptions and failure modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from driftwatch.drift.results import EvidenceClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelFreeEstimate:
    """Estimated model error on unlabeled production data prior to ground-truth maturity."""
    estimated_mae: float
    baseline_val_mae: float
    estimated_degradation_pct: float
    effective_sample_size: float
    weight_variance: float
    evidence_class: EvidenceClass
    stated_assumptions: list[str]
    failure_modes: list[str]
    details: dict[str, Any]


class LabelFreePerformanceEstimator:
    """Estimates performance decay under the Covariate Shift Assumption."""

    def __init__(self, clip_weight_max: float = 10.0) -> None:
        self.clip_weight_max = clip_weight_max

    def estimate_performance(
        self,
        val_features: np.ndarray,
        val_errors: np.ndarray,
        curr_features: np.ndarray,
        seed: int = 42,
    ) -> LabelFreeEstimate:
        """Estimate production MAE by re-weighting validation errors according to density ratios.
        
        Density ratio w(x) = P_curr(x) / P_val(x) is derived via a probabilistic domain classifier:
            w(x) = p(curr|x) / (1 - p(curr|x)) * (N_val / N_curr)
        """
        n_val = len(val_features)
        n_curr = len(curr_features)

        if n_val < 20 or n_curr < 20:
            raise ValueError(f"Insufficient samples for density ratio estimation: val={n_val}, curr={n_curr}")

        baseline_mae = float(np.mean(val_errors))

        # Train domain discriminator: 0 = Validation/Reference, 1 = Current Production
        X = np.vstack([val_features, curr_features])
        y = np.hstack([np.zeros(n_val, dtype=int), np.ones(n_curr, dtype=int)])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=200, random_state=seed, solver="lbfgs")
        clf.fit(X_scaled, y)

        # Predict probability of belonging to current distribution for validation instances
        X_val_scaled = X_scaled[:n_val]
        p_curr = np.clip(clf.predict_proba(X_val_scaled)[:, 1], 1e-4, 1.0 - 1e-4)

        # Odds ratio / density weights
        raw_weights = (p_curr / (1.0 - p_curr)) * (float(n_val) / float(n_curr))
        weights = np.clip(raw_weights, 0.0, self.clip_weight_max)
        weights_normalized = weights / np.sum(weights)

        # Weighted MAE estimate
        estimated_mae = float(np.sum(weights_normalized * val_errors))
        deg_pct = ((estimated_mae - baseline_mae) / baseline_mae * 100.0) if baseline_mae > 0 else 0.0

        # Effective Sample Size (Kish's formula)
        ess = float(1.0 / np.sum(weights_normalized ** 2))

        stated_assumptions = [
            "Covariate Shift Assumption: P(Y|X) is strictly invariant across reference and production.",
            "Common Support Assumption: Reference feature distribution covers the support of production data.",
            "Unbiased Probability Calibration: Domain classifier produces well-calibrated odds ratios.",
        ]

        failure_modes = [
            "Concept Drift: If consumer behavior or retail elasticities change, P(Y|X) shifts and the estimate is invalid.",
            "Extreme Non-Overlap: If production data enters unobserved feature space, weights explode or clip, biasing estimates.",
            "Unobserved Confounders: Macroeconomic shifts or competitor stockouts that are not present in input features.",
        ]

        return LabelFreeEstimate(
            estimated_mae=round(estimated_mae, 4),
            baseline_val_mae=round(baseline_mae, 4),
            estimated_degradation_pct=round(deg_pct, 2),
            effective_sample_size=round(ess, 1),
            weight_variance=round(float(np.var(weights)), 4),
            evidence_class=EvidenceClass.ESTIMATED_PERFORMANCE,
            stated_assumptions=stated_assumptions,
            failure_modes=failure_modes,
            details={
                "max_weight": round(float(np.max(weights)), 4),
                "min_weight": round(float(np.min(weights)), 4),
                "n_val": n_val,
                "n_curr": n_curr,
            },
        )
