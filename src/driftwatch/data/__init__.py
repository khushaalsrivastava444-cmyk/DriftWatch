"""Data generation, loading, and partitioning modules."""

from driftwatch.data.m5_loader import load_m5_data
from driftwatch.data.split import (
    TemporalPartitions,
    rolling_origin_splits,
    temporal_train_val_test_split,
)
from driftwatch.data.synthetic import (
    SyntheticDemandConfig,
    generate_synthetic_demand,
)

__all__ = [
    "SyntheticDemandConfig",
    "TemporalPartitions",
    "generate_synthetic_demand",
    "load_m5_data",
    "rolling_origin_splits",
    "temporal_train_val_test_split",
]
