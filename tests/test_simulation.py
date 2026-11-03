"""Tests for Replay Simulator, Injectors, and Benchmark Suite."""

from __future__ import annotations

import pandas as pd
import pytest

from driftwatch.data.split import temporal_train_val_test_split
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.db.base import Base, SessionLocal, engine
from driftwatch.features.pipeline import DemandFeaturePipeline
from driftwatch.simulation.injectors import (
    DriftInjectionType,
    InjectionConfig,
    apply_drift_injection,
)
from driftwatch.simulation.simulator import ReplaySimulator, SimulationConfig


@pytest.fixture(scope="module", autouse=True)
def setup_db() -> None:
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def base_demand() -> pd.DataFrame:
    cfg = SyntheticDemandConfig(num_days=45, num_stores=2, num_items=3, seed=42)
    return generate_synthetic_demand(cfg)


def test_stable_injector_preserves_hash(base_demand: pd.DataFrame) -> None:
    """Stable injection must return bitwise identical data and matching hash."""
    cfg = InjectionConfig(injection_type=DriftInjectionType.STABLE)
    out_df, meta = apply_drift_injection(base_demand, cfg)

    assert meta["before_hash"] == meta["after_hash"]
    pd.testing.assert_frame_equal(base_demand, out_df)


def test_mean_shift_injector(base_demand: pd.DataFrame) -> None:
    """Mean shift must modify target column mean and produce distinct hash."""
    cfg = InjectionConfig(
        injection_type=DriftInjectionType.MEAN_SHIFT,
        target_columns=["sales"],
        shift_magnitude=2.0,
    )
    out_df, meta = apply_drift_injection(base_demand, cfg)

    assert meta["before_hash"] != meta["after_hash"]
    assert out_df["sales"].mean() > base_demand["sales"].mean()


def test_missing_rate_injector(base_demand: pd.DataFrame) -> None:
    """Missing rate injector must introduce nulls in target columns."""
    cfg = InjectionConfig(
        injection_type=DriftInjectionType.MISSING_RATE_INCREASE,
        target_columns=["sales", "sell_price"],
        dropout_ratio=0.30,
    )
    out_df, meta = apply_drift_injection(base_demand, cfg)

    assert out_df["sales"].isna().sum() > 0
    assert out_df["sell_price"].isna().sum() > 0


def test_concept_drift_injector(base_demand: pd.DataFrame) -> None:
    """Concept drift modifies sales relationship while preserving features."""
    cfg = InjectionConfig(
        injection_type=DriftInjectionType.CONCEPT_DRIFT,
    )
    out_df, meta = apply_drift_injection(base_demand, cfg)

    assert meta["before_hash"] != meta["after_hash"]
    assert out_df["sales"].sum() < base_demand["sales"].sum()


def test_replay_simulator_end_to_end(base_demand: pd.DataFrame) -> None:
    """Verify ReplaySimulator streams batches, queues delayed labels, and runs drift reports."""
    parts = temporal_train_val_test_split(base_demand, train_ratio=0.6, val_ratio=0.2)

    pipeline = DemandFeaturePipeline()
    pipeline.fit(parts.train)
    ref_feats = pipeline.transform(parts.train)

    sim_config = SimulationConfig(
        label_lag_days=3,
        injection_config=InjectionConfig(
            injection_type=DriftInjectionType.MEAN_SHIFT,
            target_columns=["sales"],
            shift_magnitude=2.0,
        ),
    )
    simulator = ReplaySimulator(sim_config)

    db = SessionLocal()
    try:
        result = simulator.run_simulation(
            production_df=parts.test,
            pipeline=pipeline,
            db=db,
            reference_features_df=ref_feats,
        )
        assert result.total_days > 0
        assert result.total_predictions > 0
        assert result.total_labels_released > 0
        assert len(result.drift_reports) >= 1
        assert len(result.performance_reports) >= 1
    finally:
        db.close()
