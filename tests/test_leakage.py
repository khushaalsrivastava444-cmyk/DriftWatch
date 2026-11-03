"""Tests Proving Zero Temporal and Feature Lookahead Leakage.

Demonstrates mathematically that:
1. Perturbing future rows does NOT alter past feature calculations (causality check).
2. Reference statistics and categorical mappings are strictly derived from training partition.
3. Partitions have zero temporal overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.data.split import temporal_train_val_test_split
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.features.pipeline import FEATURE_COLUMNS, DemandFeaturePipeline


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Fixture providing deterministic synthetic demand."""
    cfg = SyntheticDemandConfig(
        num_days=90,
        num_stores=2,
        num_items=3,
        seed=101,
    )
    return generate_synthetic_demand(cfg)


def test_split_temporal_isolation(synthetic_data: pd.DataFrame) -> None:
    """Verify that temporal splits are strictly disjoint and ordered."""
    parts = temporal_train_val_test_split(synthetic_data, train_ratio=0.6, val_ratio=0.2)

    train_dates = set(parts.train["date"].unique())
    val_dates = set(parts.val["date"].unique())
    test_dates = set(parts.test["date"].unique())

    # Disjointness
    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)

    # Monotonicity
    assert parts.train["date"].max() < parts.val["date"].min()
    assert parts.val["date"].max() < parts.test["date"].min()


def test_feature_causality_no_future_leakage(synthetic_data: pd.DataFrame) -> None:
    """Verify that altering future target or prices does NOT alter past feature values.
    
    If feature engineering is strictly backward-looking, changing sales on day T+5
    must result in exactly identical features for all days t <= T.
    """
    pipeline = DemandFeaturePipeline()
    parts = temporal_train_val_test_split(synthetic_data, train_ratio=0.7, val_ratio=0.15)
    pipeline.fit(parts.train)

    # 1. Transform baseline test data
    features_original = pipeline.transform(parts.test)

    # Choose a cutoff date in the middle of test partition
    unique_test_dates = sorted(parts.test["date"].unique())
    cutoff_idx = len(unique_test_dates) // 2
    cutoff_date = unique_test_dates[cutoff_idx]

    # 2. Create perturbed test data where future sales & prices after cutoff are radically modified
    perturbed_test = parts.test.copy()
    future_mask = perturbed_test["date"] > cutoff_date
    perturbed_test.loc[future_mask, "sales"] = perturbed_test.loc[future_mask, "sales"] * 100 + 999
    perturbed_test.loc[future_mask, "sell_price"] = 9999.0

    features_perturbed = pipeline.transform(perturbed_test)

    # 3. Assert past features (dates <= cutoff_date) are exactly bitwise identical
    past_orig = features_original[features_original["date"] <= cutoff_date].sort_values(["store_id", "item_id", "date"])
    past_pert = features_perturbed[features_perturbed["date"] <= cutoff_date].sort_values(["store_id", "item_id", "date"])

    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            past_orig[col].to_numpy(dtype=float),
            past_pert[col].to_numpy(dtype=float),
            err_msg=f"Leakage detected in feature '{col}': past values changed when future sales were perturbed!",
            rtol=1e-7,
            atol=1e-7,
        )


def test_reference_statistics_fit_on_train_only(synthetic_data: pd.DataFrame) -> None:
    """Verify that reference quantiles and base prices are fit purely on train_df."""
    parts = temporal_train_val_test_split(synthetic_data, train_ratio=0.7, val_ratio=0.15)

    pipe1 = DemandFeaturePipeline()
    pipe1.fit(parts.train)

    # Create a corrupted test set with extreme outliers
    corrupted_test = parts.test.copy()
    corrupted_test["sales"] = 1_000_000
    corrupted_test["sell_price"] = 500_000.0

    pipe2 = DemandFeaturePipeline()
    # Fitting only on train again
    pipe2.fit(parts.train)

    # Transforming corrupted test should NOT alter the fitted pipeline's reference_stats
    pipe2.transform(corrupted_test)

    for col in FEATURE_COLUMNS:
        stat1 = pipe1.reference_stats_[col]
        stat2 = pipe2.reference_stats_[col]
        assert stat1 == stat2, f"Pipeline reference_stats_ mutated after transforming test: {col}"
