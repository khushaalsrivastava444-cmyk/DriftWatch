"""Leak-Free Feature Engineering Pipeline for Retail Demand Forecasting.

Features generated:
- Backward-looking lag sales: t-1, t-2, t-7, t-14, t-28.
- Shifted rolling statistics: 7-day, 14-day, 28-day mean, std, min, max (shift=1 strictly enforced).
- Calendar attributes: day_of_week, day_of_month, month, is_weekend.
- Price dynamics: sell_price, discount_ratio (relative to training-only base price).
- Categorical mappings: fitted exclusively on the training partition.
- Reference statistics: quantiles and moments fit strictly on the training partition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "sell_price",
    "price_discount_ratio",
    "on_promotion",
    "store_code",
    "category_code",
]


@dataclass
class DemandFeaturePipeline:
    """Production feature engineering pipeline guaranteed free of lookahead leakage.
    
    Attributes:
        store_map_: Mapping from store_id to integer code.
        category_map_: Mapping from category name to integer code.
        base_prices_: Training-only maximum/base price per (store_id, item_id).
        impute_medians_: Training-only feature medians for cold-start imputation.
        reference_stats_: Summary moments and quantiles fit strictly on training set.
        is_fitted_: Boolean indicating whether fit() has completed.
    """
    store_map_: dict[str, int] = field(default_factory=dict)
    category_map_: dict[str, int] = field(default_factory=dict)
    base_prices_: dict[tuple[str, str], float] = field(default_factory=dict)
    impute_medians_: dict[str, float] = field(default_factory=dict)
    reference_stats_: dict[str, dict[str, float]] = field(default_factory=dict)
    is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> DemandFeaturePipeline:
        """Fit pipeline parameters and reference distribution baselines on training data.
        
        Args:
            train_df: Chronologically isolated training partition.
            
        Returns:
            self
        """
        logger.info("Fitting DemandFeaturePipeline on %d training rows", len(train_df))
        
        # 1. Deterministic categorical code mappings
        unique_stores = sorted(train_df["store_id"].unique())
        self.store_map_ = {s: i for i, s in enumerate(unique_stores)}

        unique_cats = sorted(train_df["category"].unique())
        self.category_map_ = {c: i for i, c in enumerate(unique_cats)}

        # 2. Historical base price: maximum price observed during training per item/store
        price_group = train_df.groupby(["store_id", "item_id"])["sell_price"].max()
        self.base_prices_ = {(s, i): float(p) for (s, i), p in price_group.items()}
        self.overall_max_price_ = float(train_df["sell_price"].max())

        # 3. Compute training features to learn medians and reference quantiles
        transformed_train = self._compute_raw_features(train_df, is_train=True)
        
        # Learn imputation medians from training features
        for col in FEATURE_COLUMNS:
            if col in transformed_train.columns:
                med = float(transformed_train[col].median(skipna=True))
                self.impute_medians_[col] = med if not np.isnan(med) else 0.0

        # Impute training features and store reference statistics
        imputed_train = self._impute(transformed_train)
        self.reference_stats_ = {}
        for col in FEATURE_COLUMNS:
            vals = imputed_train[col].values
            self.reference_stats_[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if float(np.std(vals)) > 0 else 1.0,
                "q25": float(np.quantile(vals, 0.25)),
                "q50": float(np.quantile(vals, 0.50)),
                "q75": float(np.quantile(vals, 0.75)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        self.is_fitted_ = True
        return self

    def get_feature_config(self) -> dict[str, Any]:
        """Return deterministic dictionary of feature names and pipeline metadata."""
        return {
            "feature_columns": list(FEATURE_COLUMNS),
            "num_features": len(FEATURE_COLUMNS),
            "store_classes": len(self.store_map_),
            "category_classes": len(self.category_map_),
            "is_fitted": self.is_fitted_,
        }

    def transform(self, df: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
        """Transform panel data into feature matrix.
        
        Args:
            df: Input DataFrame (must be chronologically ordered within series).
            drop_na: Whether to drop rows where initial lags are undefined (e.g. first 28 days).
            
        Returns:
            pd.DataFrame containing metadata + engineered features.
        """
        if not self.is_fitted_:
            raise RuntimeError("DemandFeaturePipeline must be fitted before calling transform().")

        transformed = self._compute_raw_features(df, is_train=False)
        imputed = self._impute(transformed)

        if drop_na:
            imputed = imputed.dropna().reset_index(drop=True)

        return imputed

    def _compute_raw_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Calculate features using strictly backward-looking lag and rolling operations."""
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        # Ensure ordering
        out = out.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        # 1. Calendar features
        out["day_of_week"] = out["date"].dt.dayofweek
        out["day_of_month"] = out["date"].dt.day
        out["month"] = out["date"].dt.month
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

        # 2. Categorical encodings
        out["store_code"] = out["store_id"].map(self.store_map_).fillna(-1).astype(int)
        out["category_code"] = out["category"].map(self.category_map_).fillna(-1).astype(int)

        # 3. Price discount relative to training base price
        def get_discount_ratio(row: pd.Series) -> float:
            key = (row["store_id"], row["item_id"])
            base_p = self.base_prices_.get(key, getattr(self, "overall_max_price_", row["sell_price"]))
            if base_p <= 0.0:
                return 1.0
            return float(row["sell_price"] / base_p)

        out["price_discount_ratio"] = out.apply(get_discount_ratio, axis=1)

        # 4. Lag and rolling features per (store_id, item_id) series
        # CRITICAL: shift(1) is applied before any rolling computation to guarantee zero lookahead leakage
        grouped_sales = out.groupby(["store_id", "item_id"])["sales"]

        out["lag_1"] = grouped_sales.shift(1)
        out["lag_2"] = grouped_sales.shift(2)
        out["lag_7"] = grouped_sales.shift(7)
        out["lag_14"] = grouped_sales.shift(14)
        out["lag_28"] = grouped_sales.shift(28)

        # Shifted rolling aggregations: day t only uses t-1 back
        # Note: transform with rolling requires grouping by the series keys
        out["rolling_mean_7"] = (
            out.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )
        out["rolling_std_7"] = (
            out.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
        )
        out["rolling_mean_14"] = (
            out.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        )
        out["rolling_mean_28"] = (
            out.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
        )

        # Resort by date to preserve temporal evaluation sequence
        out = out.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
        return out

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using training medians strictly."""
        out = df.copy()
        for col, med in self.impute_medians_.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        return out

    def save(self, path: str | Path) -> None:
        """Serialize fitted feature pipeline to disk."""
        import joblib
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(target))

    @classmethod
    def load(cls, path: str | Path) -> DemandFeaturePipeline:
        """Load serialized feature pipeline from disk."""
        import joblib
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Feature pipeline artifact not found at {target}")
        pipeline = joblib.load(str(target))
        if not isinstance(pipeline, DemandFeaturePipeline):
            raise TypeError(f"Loaded artifact is not a DemandFeaturePipeline: {type(pipeline)}")
        return pipeline
