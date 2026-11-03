"""Statistical drift engine, property-tested tests, and windowing."""

from driftwatch.drift.engine import DriftEngine, DriftEngineConfig, DriftReport
from driftwatch.drift.multiple_testing import (
    MultipleTestingMethod,
    adjust_p_values,
    evaluate_dual_gating_decision,
)
from driftwatch.drift.psi import calculate_psi, calculate_psi_categorical, calculate_psi_continuous
from driftwatch.drift.results import EvidenceClass, FeatureDriftResult, TestType
from driftwatch.drift.statistical_tests import (
    calculate_chi_square,
    calculate_jensen_shannon,
    calculate_ks_test,
    calculate_mean_shift,
    calculate_missing_rate_shift,
    calculate_variance_shift,
    calculate_wasserstein,
)
from driftwatch.drift.windowing import (
    HysteresisStateTracker,
    WindowSlice,
    slice_sliding_windows,
    slice_tumbling_windows,
)

__all__ = [
    "DriftEngine",
    "DriftEngineConfig",
    "DriftReport",
    "EvidenceClass",
    "FeatureDriftResult",
    "HysteresisStateTracker",
    "MultipleTestingMethod",
    "TestType",
    "WindowSlice",
    "adjust_p_values",
    "calculate_chi_square",
    "calculate_jensen_shannon",
    "calculate_ks_test",
    "calculate_mean_shift",
    "calculate_missing_rate_shift",
    "calculate_psi",
    "calculate_psi_categorical",
    "calculate_psi_continuous",
    "calculate_variance_shift",
    "calculate_wasserstein",
    "evaluate_dual_gating_decision",
    "slice_sliding_windows",
    "slice_tumbling_windows",
]
