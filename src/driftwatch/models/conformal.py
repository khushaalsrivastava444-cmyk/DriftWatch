"""Split Conformal Prediction and Empirical Coverage Monitoring.

Provides distribution-free, finite-sample valid prediction intervals for demand forecasts
and continuous calibration coverage tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwatch.models.xgboost_forecaster import XGBoostDemandForecaster

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformalIntervalResult:
    """Prediction point estimates alongside distribution-free conformal bounds."""
    point_predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    half_width: float
    confidence_level: float


@dataclass(frozen=True)
class ConformalCoverageReport:
    """Audit report of empirical conformal coverage over delayed true outcomes."""
    nominal_confidence: float
    empirical_coverage: float
    coverage_breached: bool
    average_interval_width: float
    total_samples: int
    uncovered_samples: int
    details: dict[str, object]


class ConformalDemandForecaster:
    """Wraps demand forecaster with split inductive conformal prediction intervals."""

    def __init__(
        self,
        base_forecaster: XGBoostDemandForecaster,
        default_confidence_level: float = 0.90,
    ) -> None:
        self.base_forecaster = base_forecaster
        self.default_confidence_level = default_confidence_level
        self.calibrated_quantiles_: dict[float, float] = {}
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series | np.ndarray,
        confidence_levels: tuple[float, ...] | None = (0.80, 0.90, 0.95),
    ) -> ConformalDemandForecaster:
        """Calibrate nonconformity scores on an independent chronological holdout set.
        
        Guarantees marginal coverage P(Y in C(X)) >= 1 - alpha for exchangeable sequences.
        """
        if not self.base_forecaster.is_fitted_:
            raise ValueError("Base forecaster must be fitted before conformal calibration.")

        y_true = np.asarray(y_cal, dtype=float)
        y_pred = self.base_forecaster.predict(X_cal)

        # Residual absolute errors as nonconformity scores
        scores = np.abs(y_true - y_pred)
        n = len(scores)
        if n < 20:
            raise ValueError(f"Calibration set too small for valid conformal quantiles: {n} < 20")

        self.calibrated_quantiles_ = {}
        for conf in (confidence_levels or (self.default_confidence_level,)):
            alpha = 1.0 - conf
            # Standard finite-sample conformal correction quantile formula
            q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
            q_val = float(np.quantile(scores, q_level, method="higher"))
            self.calibrated_quantiles_[conf] = q_val
            logger.info("Calibrated conformal radius for %.0f%% confidence: +/- %.3f units", conf * 100, q_val)

        self.is_calibrated_ = True
        return self

    def predict_intervals(
        self,
        X: pd.DataFrame,
        confidence_level: float | None = None,
    ) -> ConformalIntervalResult:
        """Generate point forecasts bounded by calibrated prediction intervals."""
        if not self.is_calibrated_:
            raise ValueError("Conformal forecaster must be calibrated before generating intervals.")

        conf = confidence_level or self.default_confidence_level
        if conf not in self.calibrated_quantiles_:
            # Fallback interpolation if custom confidence requested
            q_val = self.calibrated_quantiles_.get(self.default_confidence_level, 5.0)
        else:
            q_val = self.calibrated_quantiles_[conf]

        preds = self.base_forecaster.predict(X)
        lower = np.maximum(0.0, preds - q_val)  # Retail demand is strictly non-negative
        upper = preds + q_val

        return ConformalIntervalResult(
            point_predictions=preds,
            lower_bounds=lower,
            upper_bounds=upper,
            half_width=q_val,
            confidence_level=conf,
        )

    @staticmethod
    def evaluate_coverage(
        y_true: pd.Series | np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        nominal_confidence: float = 0.90,
        breach_tolerance: float = 0.10,
    ) -> ConformalCoverageReport:
        """Audit empirical coverage on delayed true outcomes.
        
        Breach condition: empirical coverage < nominal - breach_tolerance (e.g. < 80% on 90% nominal).
        """
        y_vals = np.asarray(y_true, dtype=float)
        n = len(y_vals)
        if n == 0:
            raise ValueError("Zero samples provided for coverage evaluation.")

        covered = (y_vals >= lower_bounds) & (y_vals <= upper_bounds)
        emp_coverage = float(np.mean(covered))
        widths = upper_bounds - lower_bounds
        avg_width = float(np.mean(widths))

        # Degradation condition on uncertainty intervals:
        # If real coverage falls significantly below nominal confidence, intervals are undercalibrated!
        is_breached = emp_coverage < (nominal_confidence - breach_tolerance)

        return ConformalCoverageReport(
            nominal_confidence=nominal_confidence,
            empirical_coverage=round(emp_coverage, 4),
            coverage_breached=is_breached,
            average_interval_width=round(avg_width, 4),
            total_samples=n,
            uncovered_samples=int(np.sum(~covered)),
            details={
                "breach_tolerance": breach_tolerance,
                "coverage_gap": round(nominal_confidence - emp_coverage, 4),
            },
        )
