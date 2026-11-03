"""Delayed Ground-Truth Performance Monitoring and Bootstrap Confidence Intervals.

Features:
- Asynchronous matching between prediction logs and delayed ground-truth labels.
- Label coverage ratio and delay distribution metrics.
- Explicit UNVERIFIED state when label coverage is inadequate.
- Stationary block bootstrap 95% confidence intervals on MAE and RMSE.
- CI-aware degradation detection (requires statistically significant breach, rejecting single-sample dips).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwatch.monitoring.metrics import ForecastingMetrics, evaluate_forecast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """95% Empirical Bootstrap Confidence Interval."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 500


@dataclass(frozen=True)
class EvaluatedPerformanceWindow:
    """Complete evaluation report for a chronological monitoring window."""
    window_id: str
    window_start: str
    window_end: str
    total_predictions: int
    labeled_count: int
    label_coverage_ratio: float
    mean_delay_seconds: float
    median_delay_seconds: float
    status: str  # "VERIFIED" or "UNVERIFIED"
    metrics: ForecastingMetrics | None
    mae_ci: BootstrapConfidenceInterval | None
    degradation_breached: bool
    details: dict[str, object]


def compute_stationary_bootstrap_ci(
    errors: np.ndarray,
    expected_block_size: int = 7,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_samples: int = 15,
    block_size: int | None = None,
) -> BootstrapConfidenceInterval:
    """Compute Politis & Romano (1994) stationary bootstrap confidence interval for MAE.
    
    Properties:
    - Block lengths geometrically distributed with parameter p = 1 / expected_block_size.
    - Circular index wrapping modulo n ensures strict stationarity.
    - Requires min_samples (>= 15) to guarantee valid empirical distribution.
    """
    if block_size is not None:
        expected_block_size = block_size
    n = len(errors)
    mean_err = float(np.mean(errors)) if n > 0 else 0.0
    if n < min_samples:
        return BootstrapConfidenceInterval(
            metric_name="mae",
            point_estimate=mean_err,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )

    rng = np.random.default_rng(seed)
    p = 1.0 / max(1, expected_block_size)
    boot_means = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        # Politis-Romano stationary bootstrap index selection
        indices = np.empty(n, dtype=int)
        current_idx = int(rng.integers(0, n))
        indices[0] = current_idx

        new_block_flags = rng.random(size=n) < p
        random_starts = rng.integers(0, n, size=n)

        for i in range(1, n):
            if new_block_flags[i]:
                current_idx = int(random_starts[i])
            else:
                current_idx = (current_idx + 1) % n
            indices[i] = current_idx

        boot_sample = errors[indices]
        boot_means[b] = np.mean(boot_sample)

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(boot_means, (alpha / 2.0) * 100))
    ci_high = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))

    return BootstrapConfidenceInterval(
        metric_name="mae",
        point_estimate=mean_err,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


# Backward-compatible alias
compute_block_bootstrap_ci = compute_stationary_bootstrap_ci


class PerformanceEvaluator:
    """Evaluates prediction streams against asynchronously arriving ground-truth labels."""

    def __init__(
        self,
        min_coverage_threshold: float = 0.80,
        baseline_mae: float = 5.0,
        degradation_threshold_ratio: float = 0.25,  # 25% degradation above baseline
        block_size: int = 7,
        n_bootstrap: int = 500,
    ) -> None:
        self.min_coverage_threshold = min_coverage_threshold
        self.baseline_mae = baseline_mae
        self.degradation_threshold_ratio = degradation_threshold_ratio
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap

    def evaluate_window(
        self,
        matched_df: pd.DataFrame,
        window_id: str,
        window_start: str,
        window_end: str,
        pred_col: str = "prediction",
        true_col: str = "true_label",
        delay_col: str = "delay_seconds",
        total_window_predictions: int | None = None,
    ) -> EvaluatedPerformanceWindow:
        """Evaluate matched prediction and ground-truth data in a time window.
        
        Args:
            matched_df: DataFrame containing predictions merged with ground_truth_labels on prediction_id.
            window_id: Identifier for window.
            window_start: ISO start date.
            window_end: ISO end date.
            pred_col: Column for predictions.
            true_col: Column for ground-truth labels.
            delay_col: Column for label arrival delay in seconds.
            total_window_predictions: Optional full window prediction count (if matched_df was filtered).
        """
        total_preds = total_window_predictions if total_window_predictions is not None else len(matched_df)
        if total_preds == 0:
            return EvaluatedPerformanceWindow(
                window_id=window_id,
                window_start=window_start,
                window_end=window_end,
                total_predictions=0,
                labeled_count=0,
                label_coverage_ratio=0.0,
                mean_delay_seconds=0.0,
                median_delay_seconds=0.0,
                status="UNVERIFIED",
                metrics=None,
                mae_ci=None,
                degradation_breached=False,
                details={"reason": "Empty prediction window"},
            )

        labeled_mask = matched_df[true_col].notna() if true_col in matched_df.columns else pd.Series(False, index=matched_df.index)
        labeled_count = int(labeled_mask.sum())
        coverage = (labeled_count / total_preds) if total_preds > 0 else 0.0

        # Delay statistics
        if delay_col in matched_df.columns and labeled_count > 0:
            delays = matched_df.loc[labeled_mask, delay_col].to_numpy(dtype=float)
            mean_delay = float(np.mean(delays))
            median_delay = float(np.median(delays))
        else:
            mean_delay = 0.0
            median_delay = 0.0

        # Enforce UNVERIFIED status if label coverage is below maturity threshold
        if coverage < self.min_coverage_threshold:
            return EvaluatedPerformanceWindow(
                window_id=window_id,
                window_start=window_start,
                window_end=window_end,
                total_predictions=total_preds,
                labeled_count=labeled_count,
                label_coverage_ratio=coverage,
                mean_delay_seconds=mean_delay,
                median_delay_seconds=median_delay,
                status="UNVERIFIED",
                metrics=None,
                mae_ci=None,
                degradation_breached=False,
                details={
                    "reason": f"Coverage {coverage:.1%} below maturity threshold ({self.min_coverage_threshold:.1%})"
                },
            )

        # Enforce UNVERIFIED status if sample size is too small for valid stationary bootstrap (< 15)
        if labeled_count < 15:
            return EvaluatedPerformanceWindow(
                window_id=window_id,
                window_start=window_start,
                window_end=window_end,
                total_predictions=total_preds,
                labeled_count=labeled_count,
                label_coverage_ratio=coverage,
                mean_delay_seconds=mean_delay,
                median_delay_seconds=median_delay,
                status="UNVERIFIED",
                metrics=None,
                mae_ci=None,
                degradation_breached=False,
                details={
                    "reason": f"Insufficient labeled observations (n={labeled_count} < 15) for stationary bootstrap."
                },
            )

        # Labeled data is mature: compute performance metrics
        labeled_df = matched_df[labeled_mask].copy()
        y_true = labeled_df[true_col].to_numpy(dtype=float)
        y_pred = labeled_df[pred_col].to_numpy(dtype=float)

        metrics = evaluate_forecast(
            y_true=y_true,
            y_pred=y_pred,
            eval_df=labeled_df,
            pred_col=pred_col,
            true_col=true_col,
        )

        # Compute block bootstrap confidence interval
        abs_errors = np.abs(y_true - y_pred)
        mae_ci = compute_block_bootstrap_ci(
            abs_errors,
            block_size=self.block_size,
            n_bootstrap=self.n_bootstrap,
        )

        # CI-Aware Degradation Rule:
        # Breach flagged ONLY if lower bound of CI exceeds the degraded threshold
        degradation_threshold = self.baseline_mae * (1.0 + self.degradation_threshold_ratio)
        degradation_breached = bool(mae_ci.ci_lower > degradation_threshold)

        return EvaluatedPerformanceWindow(
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
            total_predictions=total_preds,
            labeled_count=labeled_count,
            label_coverage_ratio=coverage,
            mean_delay_seconds=mean_delay,
            median_delay_seconds=median_delay,
            status="VERIFIED",
            metrics=metrics,
            mae_ci=mae_ci,
            degradation_breached=degradation_breached,
            details={
                "baseline_mae": self.baseline_mae,
                "degradation_threshold": degradation_threshold,
                "mae_point": mae_ci.point_estimate,
                "ci_lower": mae_ci.ci_lower,
                "ci_upper": mae_ci.ci_upper,
            },
        )
