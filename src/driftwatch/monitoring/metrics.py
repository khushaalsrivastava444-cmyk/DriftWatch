"""Forecasting Performance Evaluation Metrics.

Computes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(y))
- MASE (Mean Absolute Scaled Error: scaled by seasonal-naive in-sample benchmark)
- Bias (Mean Error: mean(y_hat - y))
- WRMSSE (Weighted Root Mean Squared Scaled Error, M5 competition specification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastingMetrics:
    """Forecast evaluation metrics."""
    mae: float
    rmse: float
    wape: float
    mase: float
    bias: float
    wrmsse: float | None = None
    sample_size: int = 0

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = {
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "wape": round(self.wape, 4),
            "mase": round(self.mase, 4),
            "bias": round(self.bias, 4),
            "sample_size": float(self.sample_size),
        }
        if self.wrmsse is not None:
            d["wrmsse"] = round(self.wrmsse, 4)
        return d


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Weighted Absolute Percentage Error: sum(|y - y_hat|) / max(sum(y), eps)."""
    denom = max(float(np.sum(y_true)), epsilon)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 7,
    epsilon: float = 1e-6,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).
    
    If y_train is provided, scale is in-sample seasonal-naive MAE.
    Otherwise, scale is calculated over y_true.
    """
    scale_series = y_train if (y_train is not None and len(y_train) > seasonality) else y_true
    if len(scale_series) <= seasonality:
        scale = max(float(np.mean(np.abs(np.diff(scale_series)))), epsilon)
    else:
        diffs = np.abs(scale_series[seasonality:] - scale_series[:-seasonality])
        scale = max(float(np.mean(diffs)), epsilon)

    mae = calculate_mae(y_true, y_pred)
    return float(mae / scale)


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean directional error: mean(y_pred - y_true)."""
    return float(np.mean(y_pred - y_true))


def calculate_wrmsse_approximation(
    eval_df: pd.DataFrame,
    pred_col: str = "pred",
    true_col: str = "sales",
    price_col: str = "sell_price",
    series_keys: tuple[str, ...] = ("store_id", "item_id"),
    train_df: pd.DataFrame | None = None,
    epsilon: float = 1e-6,
) -> float:
    """Weighted Root Mean Squared Scaled Error (WRMSSE) approximation across series.
    
    Weights each series by dollar volume (sales * price) and normalizes MSE by
    historical lag-1 diff scale.
    """
    rmsses: list[float] = []
    dollar_volumes: list[float] = []

    for _, grp in eval_df.groupby(list(series_keys)):
        y_eval = grp[true_col].to_numpy(dtype=float)
        y_hat = grp[pred_col].to_numpy(dtype=float)
        prices = grp[price_col].to_numpy(dtype=float) if price_col in grp.columns else np.ones_like(y_eval)

        # Scale calculation: in-sample squared differences
        if len(y_eval) < 2:
            scale = 1.0
        else:
            scale = max(float(np.mean((y_eval[1:] - y_eval[:-1]) ** 2)), epsilon)

        # RMSSE for this series
        mse = float(np.mean((y_eval - y_hat) ** 2))
        rmsse = float(np.sqrt(mse / scale))

        volume = float(np.sum(y_eval * prices))
        
        rmsses.append(rmsse)
        dollar_volumes.append(volume)

    total_vol = sum(dollar_volumes)
    if total_vol <= 0.0:
        return float(np.mean(rmsses))

    weights = [v / total_vol for v in dollar_volumes]
    return float(np.sum(np.array(weights) * np.array(rmsses)))


def evaluate_forecast(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series | None = None,
    eval_df: pd.DataFrame | None = None,
    pred_col: str = "pred",
    true_col: str = "sales",
    price_col: str = "sell_price",
) -> ForecastingMetrics:
    """Calculate all standard forecasting performance metrics in a single call."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    y_tr = np.asarray(y_train, dtype=float) if y_train is not None else None

    mae = calculate_mae(y_t, y_p)
    rmse = calculate_rmse(y_t, y_p)
    wape = calculate_wape(y_t, y_p)
    mase = calculate_mase(y_t, y_p, y_train=y_tr)
    bias = calculate_bias(y_t, y_p)

    wrmsse = None
    if eval_df is not None and "store_id" in eval_df.columns and "item_id" in eval_df.columns:
        df_copy = eval_df.copy()
        df_copy[pred_col] = y_p
        df_copy[true_col] = y_t
        try:
            wrmsse = calculate_wrmsse_approximation(
                df_copy,
                pred_col=pred_col,
                true_col=true_col,
                price_col=price_col,
            )
        except Exception as e:
            logger.warning("WRMSSE calculation encountered exception: %s", e)

    return ForecastingMetrics(
        mae=mae,
        rmse=rmse,
        wape=wape,
        mase=mase,
        bias=bias,
        wrmsse=wrmsse,
        sample_size=len(y_t),
    )
