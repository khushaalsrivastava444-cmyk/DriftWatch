"""Seasonal-Naive Baseline Demand Forecaster.

In retail time-series, the standard gold-standard operational benchmark is
Seasonal Naive: y_hat(t) = y(t - m), where m = 7 days for daily data.
Falls back to lag 1 or series mean when lag 7 is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaiveForecaster:
    """Seasonal-naive forecaster adhering to scikit-learn estimator interface."""

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length
        self.series_history_: dict[tuple[str, str], list[float]] = {}
        self.global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame, target_col: str = "sales") -> SeasonalNaiveForecaster:
        """Store historical series from the training partition.
        
        Args:
            df: Training DataFrame containing [store_id, item_id, date, sales].
            target_col: Column name of target demand.
        """
        self.global_mean_ = float(df[target_col].mean()) if not df.empty else 0.0
        self.series_history_ = {}
        
        # Sort chronologically
        sorted_df = df.sort_values(["store_id", "item_id", "date"])
        for (store_id, item_id), grp in sorted_df.groupby(["store_id", "item_id"]):
            self.series_history_[(str(store_id), str(item_id))] = grp[target_col].tolist()
            
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate seasonal-naive point predictions for the given DataFrame.
        
        Uses feature columns lag_7 (or lag_1) if present; otherwise references stored history.
        """
        # If feature matrix already contains lag_7, directly utilize it
        if "lag_7" in df.columns:
            preds = df["lag_7"].copy()
            if "lag_1" in df.columns:
                preds = preds.fillna(df["lag_1"])
            preds = preds.fillna(self.global_mean_)
            return np.maximum(0.0, preds.to_numpy(dtype=float))

        # Otherwise evaluate from stored history
        preds_list: list[float] = []
        for _, row in df.iterrows():
            key = (str(row["store_id"]), str(row["item_id"]))
            hist = self.series_history_.get(key, [])
            if len(hist) >= self.season_length:
                val = hist[-self.season_length]
            elif len(hist) >= 1:
                val = hist[-1]
            else:
                val = self.global_mean_
            preds_list.append(max(0.0, float(val)))
            
        return np.array(preds_list, dtype=float)
