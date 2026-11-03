"""Forecasting Benchmark Tests: Seasonal-Naive vs XGBoost Forecaster.

Verifies end-to-end forecasting pipeline and produces real comparative metrics.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from driftwatch.data.split import temporal_train_val_test_split
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.features.pipeline import DemandFeaturePipeline
from driftwatch.models.baseline import SeasonalNaiveForecaster
from driftwatch.models.xgboost_forecaster import (
    XGBoostDemandForecaster,
    XGBoostForecasterConfig,
)
from driftwatch.monitoring.metrics import evaluate_forecast

logger = logging.getLogger(__name__)


def test_baseline_vs_xgboost_forecasting_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    """Benchmark Seasonal-Naive baseline vs XGBoost model and assert superior performance."""
    # 1. Generate full year of synthetic demand across 3 stores and 8 items
    cfg = SyntheticDemandConfig(
        start_date="2024-01-01",
        num_days=300,
        num_stores=3,
        num_items=8,
        seed=42,
    )
    raw_df = generate_synthetic_demand(cfg)
    assert len(raw_df) == 300 * 3 * 8

    # 2. Strict chronological partition
    partitions = temporal_train_val_test_split(raw_df, train_ratio=0.70, val_ratio=0.15)
    train_raw = partitions.train
    val_raw = partitions.val
    test_raw = partitions.test

    assert len(train_raw) > 0 and len(val_raw) > 0 and len(test_raw) > 0

    # 3. Fit pipeline on train only, transform train, val, test
    pipeline = DemandFeaturePipeline()
    pipeline.fit(train_raw)

    train_feats = pipeline.transform(train_raw)
    val_feats = pipeline.transform(val_raw)
    test_feats = pipeline.transform(test_raw)

    # 4. Seasonal Naive baseline evaluation
    baseline = SeasonalNaiveForecaster(season_length=7)
    baseline.fit(train_raw)
    baseline_preds = baseline.predict(test_feats)

    baseline_metrics = evaluate_forecast(
        y_true=test_feats["sales"],
        y_pred=baseline_preds,
        y_train=train_feats["sales"],
        eval_df=test_feats,
    )

    # 5. XGBoost model training and evaluation
    xgb_config = XGBoostForecasterConfig(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        tree_method="hist",
    )
    xgb_model = XGBoostDemandForecaster(config=xgb_config)
    xgb_model.fit(
        X=train_feats,
        y=train_feats["sales"],
        eval_set=[(val_feats, val_feats["sales"])],
        early_stopping_rounds=15,
    )

    xgb_preds = xgb_model.predict(test_feats)
    xgb_metrics = evaluate_forecast(
        y_true=test_feats["sales"],
        y_pred=xgb_preds,
        y_train=train_feats["sales"],
        eval_df=test_feats,
    )

    # 6. Format metrics comparison table
    table_data = [
        {
            "Model": "Seasonal-Naive (Benchmark)",
            "MAE": baseline_metrics.mae,
            "RMSE": baseline_metrics.rmse,
            "WAPE": baseline_metrics.wape,
            "MASE": baseline_metrics.mase,
            "Bias": baseline_metrics.bias,
            "WRMSSE": baseline_metrics.wrmsse,
        },
        {
            "Model": "XGBoost Regressor",
            "MAE": xgb_metrics.mae,
            "RMSE": xgb_metrics.rmse,
            "WAPE": xgb_metrics.wape,
            "MASE": xgb_metrics.mase,
            "Bias": xgb_metrics.bias,
            "WRMSSE": xgb_metrics.wrmsse,
        },
    ]
    metrics_table = pd.DataFrame(table_data)
    
    with capsys.disabled():
        print("\n" + "=" * 70)
        print("FORECASTING BENCHMARK RESULTS (TEST PARTITION)")
        print("=" * 70)
        print(metrics_table.to_string(index=False))
        print("=" * 70)

    # Assertions:
    # (a) Both models output valid non-negative numbers
    assert (baseline_preds >= 0.0).all()
    assert (xgb_preds >= 0.0).all()

    # (b) XGBoost beats seasonal-naive on MAE, RMSE, and WAPE
    assert xgb_metrics.mae < baseline_metrics.mae, f"XGBoost MAE ({xgb_metrics.mae}) not lower than baseline ({baseline_metrics.mae})"
    assert xgb_metrics.rmse < baseline_metrics.rmse, f"XGBoost RMSE ({xgb_metrics.rmse}) not lower than baseline ({baseline_metrics.rmse})"
    assert xgb_metrics.wape < baseline_metrics.wape, f"XGBoost WAPE ({xgb_metrics.wape}) not lower than baseline ({baseline_metrics.wape})"

    # (c) MASE is < 1.0 (indicating better forecast than seasonal persistence)
    assert xgb_metrics.mase < 1.0, f"XGBoost MASE ({xgb_metrics.mase}) should be < 1.0"
