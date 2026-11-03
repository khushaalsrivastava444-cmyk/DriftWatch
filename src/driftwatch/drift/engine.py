"""Central Statistical Drift Detection Engine.

Orchestrates multi-feature drift profiling, multiple-testing control,
dual-gating decision rules, and evidence classification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from driftwatch.drift.multiple_testing import (
    MultipleTestingMethod,
    adjust_p_values,
    evaluate_dual_gating_decision,
)
from driftwatch.drift.psi import calculate_psi
from driftwatch.drift.results import EvidenceClass, FeatureDriftResult, TestType
from driftwatch.drift.statistical_tests import (
    calculate_chi_square,
    calculate_jensen_shannon,
    calculate_ks_test,
    calculate_mean_shift,
    calculate_missing_rate_shift,
    calculate_variance_shift,
    calculate_wasserstein,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftEngineConfig:
    """Configurable thresholds and options for drift detection."""
    alpha: float = 0.05
    psi_threshold: float = 0.10
    ks_threshold: float = 0.10
    js_threshold: float = 0.10
    wasserstein_threshold: float = 0.25
    chi_square_threshold: float = 0.15
    mean_shift_threshold: float = 0.30
    variance_shift_threshold: float = 0.50
    missing_rate_threshold: float = 0.05
    multiple_testing_method: MultipleTestingMethod = MultipleTestingMethod.BENJAMINI_HOCHBERG
    active_tests: list[TestType] = field(default_factory=lambda: [
        TestType.PSI,
        TestType.KOLMOGOROV_SMIRNOV,
        TestType.JENSEN_SHANNON,
        TestType.WASSERSTEIN,
    ])

    def hash(self) -> str:
        """Deterministic hash of current configuration."""
        d = {
            "alpha": self.alpha,
            "psi": self.psi_threshold,
            "ks": self.ks_threshold,
            "js": self.js_threshold,
            "wass": self.wasserstein_threshold,
            "mt": self.multiple_testing_method.value,
            "tests": [t.value for t in self.active_tests],
        }
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class DriftReport:
    """Aggregate report of multi-feature drift testing across a window."""
    dataset_id: str
    reference_id: str
    window_id: str
    timestamp: str
    config_hash: str
    results: list[FeatureDriftResult] = field(default_factory=list)

    @property
    def drifted_features(self) -> set[str]:
        return {r.feature for r in self.results if r.decision}

    @property
    def is_drift_detected(self) -> bool:
        return len(self.drifted_features) > 0

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "window_id": self.window_id,
            "total_tests": len(self.results),
            "drifted_features_count": len(self.drifted_features),
            "drifted_features": sorted(list(self.drifted_features)),
            "is_drift_detected": self.is_drift_detected,
        }


class DriftEngine:
    """Multi-test, multiple-testing controlled statistical drift engine."""

    def __init__(self, config: DriftEngineConfig | None = None) -> None:
        self.config = config or DriftEngineConfig()

    def run_drift_analysis(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: list[str],
        dataset_id: str = "production_stream",
        reference_id: str = "training_baseline",
        window_id: str = "window_default",
        prediction_col: str | None = None,
    ) -> DriftReport:
        """Execute all configured drift tests and apply multiple-testing FDR adjustment."""
        config_hash = self.config.hash()
        raw_results: list[dict[str, Any]] = []

        # 1. Feature-level tests
        for feat in feature_columns:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue

            ref_series = reference_df[feat]
            curr_series = current_df[feat]
            is_numeric = pd.api.types.is_numeric_dtype(ref_series)

            if is_numeric:
                # Continuous numerical tests
                if TestType.PSI in self.config.active_tests:
                    stat, details = calculate_psi(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.PSI,
                        "statistic": stat,
                        "p_value": None,
                        "effect_size": stat,
                        "threshold": self.config.psi_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

                if TestType.KOLMOGOROV_SMIRNOV in self.config.active_tests:
                    stat, p_val, effect, details = calculate_ks_test(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.KOLMOGOROV_SMIRNOV,
                        "statistic": stat,
                        "p_value": p_val,
                        "effect_size": effect,
                        "threshold": self.config.ks_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

                if TestType.JENSEN_SHANNON in self.config.active_tests:
                    stat, _, effect, details = calculate_jensen_shannon(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.JENSEN_SHANNON,
                        "statistic": stat,
                        "p_value": None,
                        "effect_size": effect,
                        "threshold": self.config.js_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

                if TestType.WASSERSTEIN in self.config.active_tests:
                    stat, _, effect, details = calculate_wasserstein(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.WASSERSTEIN,
                        "statistic": stat,
                        "p_value": None,
                        "effect_size": effect,
                        "threshold": self.config.wasserstein_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

                if TestType.MEAN_SHIFT in self.config.active_tests:
                    stat, p_val, effect, details = calculate_mean_shift(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.MEAN_SHIFT,
                        "statistic": stat,
                        "p_value": p_val,
                        "effect_size": effect,
                        "threshold": self.config.mean_shift_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

                if TestType.VARIANCE_SHIFT in self.config.active_tests:
                    stat, p_val, effect, details = calculate_variance_shift(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.VARIANCE_SHIFT,
                        "statistic": stat,
                        "p_value": p_val,
                        "effect_size": effect,
                        "threshold": self.config.variance_shift_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })
            else:
                # Categorical tests
                if TestType.PSI in self.config.active_tests:
                    stat, details = calculate_psi(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.PSI,
                        "statistic": stat,
                        "p_value": None,
                        "effect_size": stat,
                        "threshold": self.config.psi_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

                if TestType.CHI_SQUARE in self.config.active_tests:
                    stat, p_val, effect, details = calculate_chi_square(ref_series, curr_series)
                    raw_results.append({
                        "feature": feat,
                        "test": TestType.CHI_SQUARE,
                        "statistic": stat,
                        "p_value": p_val,
                        "effect_size": effect,
                        "threshold": self.config.chi_square_threshold,
                        "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                        "details": details,
                    })

            # Check missing rate shifts
            if TestType.MISSING_RATE in self.config.active_tests:
                stat, p_val, effect, details = calculate_missing_rate_shift(ref_series, curr_series)
                raw_results.append({
                    "feature": feat,
                    "test": TestType.MISSING_RATE,
                    "statistic": stat,
                    "p_value": p_val,
                    "effect_size": effect,
                    "threshold": self.config.missing_rate_threshold,
                    "evidence_class": EvidenceClass.STATISTICAL_SHIFT,
                    "details": details,
                })

        # 2. Prediction distribution drift (if prediction column provided)
        if prediction_col and prediction_col in reference_df.columns and prediction_col in current_df.columns:
            ref_preds = reference_df[prediction_col]
            curr_preds = current_df[prediction_col]
            stat, p_val, effect, details = calculate_ks_test(ref_preds, curr_preds)
            raw_results.append({
                "feature": prediction_col,
                "test": TestType.PREDICTION_DRIFT,
                "statistic": stat,
                "p_value": p_val,
                "effect_size": effect,
                "threshold": self.config.ks_threshold,
                "evidence_class": EvidenceClass.PREDICTION_SHIFT,
                "details": details,
            })

        # 3. Apply multiple testing adjustment on all p-values simultaneously
        raw_p_values = [r["p_value"] for r in raw_results]
        adjusted_p_values = adjust_p_values(raw_p_values, method=self.config.multiple_testing_method)

        # 4. Apply dual-gating decisions
        final_results: list[FeatureDriftResult] = []
        now_ts = datetime.now(UTC).isoformat()

        for raw_r, adj_p in zip(raw_results, adjusted_p_values):
            feat = raw_r["feature"]
            p_for_decision = adj_p if adj_p is not None else raw_r["p_value"]
            decision, reason = evaluate_dual_gating_decision(
                p_value=p_for_decision,
                effect_size=raw_r["effect_size"],
                alpha=self.config.alpha,
                effect_size_threshold=raw_r["threshold"],
            )

            res = FeatureDriftResult(
                dataset_id=dataset_id,
                reference_id=reference_id,
                window=window_id,
                feature=feat,
                test=raw_r["test"],
                statistic=raw_r["statistic"],
                p_value=raw_r["p_value"],
                adjusted_p_value=adj_p,
                effect_size=raw_r["effect_size"],
                threshold=raw_r["threshold"],
                decision=decision,
                sample_size_ref=len(reference_df),
                sample_size_curr=len(current_df),
                evidence_class=raw_r["evidence_class"],
                timestamp=now_ts,
                config_hash=config_hash,
                details={**raw_r["details"], "decision_reason": reason},
            )
            final_results.append(res)

        return DriftReport(
            dataset_id=dataset_id,
            reference_id=reference_id,
            window_id=window_id,
            timestamp=now_ts,
            config_hash=config_hash,
            results=final_results,
        )
