"""XGBoost Production Demand Forecaster.

Provides a robust gradient boosted tree regressor tailored for tabular retail demand,
equipped with feature importance introspection, early stopping, and CPU optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from driftwatch.features.pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class XGBoostForecasterConfig:
    """Hyperparameter and runtime configuration for XGBoost Forecaster."""
    n_estimators: int = 150
    max_depth: int = 5
    learning_rate: float = 0.08
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: float = 3.0
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"


class XGBoostDemandForecaster:
    """Gradient boosted decision tree forecaster for retail item demand."""

    def __init__(self, config: XGBoostForecasterConfig | None = None) -> None:
        self.config = config or XGBoostForecasterConfig()
        self.feature_names_: list[str] = list(FEATURE_COLUMNS)
        self.model_: xgb.XGBRegressor | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        eval_set: list[tuple[pd.DataFrame, pd.Series | np.ndarray]] | None = None,
        early_stopping_rounds: int | None = 15,
    ) -> XGBoostDemandForecaster:
        """Fit XGBoost model on tabular feature matrix.
        
        Args:
            X: Feature matrix containing columns listed in self.feature_names_.
            y: Target demand vector (sales).
            eval_set: Validation split for early stopping.
            early_stopping_rounds: Number of rounds without validation improvement before stopping.
        """
        # Ensure only expected features are used
        X_mat = X[self.feature_names_].copy()
        y_vec = np.asarray(y, dtype=float)

        self.model_ = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            tree_method=self.config.tree_method,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
        )

        xgb_eval_set = None
        if eval_set is not None:
            xgb_eval_set = [
                (val_X[self.feature_names_], np.asarray(val_y, dtype=float))
                for val_X, val_y in eval_set
            ]

        self.model_.fit(
            X_mat,
            y_vec,
            eval_set=xgb_eval_set,
            verbose=False,
        )
        self.is_fitted_ = True
        logger.info("XGBoost forecaster fitted successfully on %d rows", len(X_mat))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate non-negative demand predictions."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("XGBoostDemandForecaster must be fitted before predict().")

        X_mat = X[self.feature_names_].copy()
        raw_preds = self.model_.predict(X_mat)
        # Sales are strictly non-negative
        return np.maximum(0.0, raw_preds)

    def get_feature_importances(self) -> dict[str, float]:
        """Return normalized gain feature importances."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Model is not fitted.")

        importances = self.model_.feature_importances_
        return {
            feat: float(imp)
            for feat, imp in zip(self.feature_names_, importances)
        }

    def save_model(self, path: str) -> None:
        """Save model artifact to disk in native XGBoost format."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        self.model_.save_model(path)

    def load_model(self, path: str) -> XGBoostDemandForecaster:
        """Load model artifact from disk."""
        self.model_ = xgb.XGBRegressor()
        self.model_.load_model(path)
        self.is_fitted_ = True
        return self
