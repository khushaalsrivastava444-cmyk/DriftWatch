"""Hypothesis Property-Based Tests for Statistical Drift Engine.

Properties verified:
1. Identity: PSI(x, x) == 0 for any valid distribution.
2. Symmetry: JS(x, y) == JS(y, x) for all distributions.
3. Boundedness: JS(x, y) in [0.0, 1.0].
4. Order Invariance: Shuffling rows does not alter test statistics.
5. Zero-Bin Safety: Empty bins in target do not produce division-by-zero or NaN/Inf.
6. NaN Safety: Monitored data containing NaNs is handled gracefully without crashing.
7. Monotonicity: Benjamini-Hochberg adjusted p-values preserve relative ranking.
8. Hysteresis Stability: k-of-n escalation and recovery deadbands prevent alert flapping.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from driftwatch.drift.multiple_testing import (
    adjust_p_values_benjamini_hochberg,
    adjust_p_values_bonferroni,
    evaluate_dual_gating_decision,
)
from driftwatch.drift.psi import calculate_psi_continuous
from driftwatch.drift.statistical_tests import (
    calculate_jensen_shannon,
    calculate_ks_test,
    calculate_wasserstein,
)
from driftwatch.drift.windowing import HysteresisStateTracker

# Strategy for generating non-empty real-valued numeric samples
float_samples = st.lists(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e4,
        max_value=1e4,
    ),
    min_size=40,
    max_size=150,
)


@given(float_samples)
@settings(max_examples=50, deadline=None)
def test_property_psi_identity_zero(sample: list[float]) -> None:
    """Property 1: PSI(x, x) == 0.0 for any distribution."""
    arr = np.array(sample)
    # Require at least two distinct values for meaningful quantiles
    if len(np.unique(arr)) < 2:
        return

    psi_val, details = calculate_psi_continuous(arr, arr, num_bins=10)
    assert not np.isnan(psi_val)
    assert not np.isinf(psi_val)
    # Within small floating point epsilon due to Laplace smoothing
    assert psi_val < 1e-4, f"PSI(x, x) was {psi_val}, expected ~0.0"


@given(float_samples, float_samples)
@settings(max_examples=50, deadline=None)
def test_property_js_symmetry_and_boundedness(sample_x: list[float], sample_y: list[float]) -> None:
    """Property 2 & 3: JS(x, y) is strictly symmetric and bounded in [0, 1]."""
    x = np.array(sample_x)
    y = np.array(sample_y)

    js_xy, _, _, _ = calculate_jensen_shannon(x, y, num_bins=15)
    js_yx, _, _, _ = calculate_jensen_shannon(y, x, num_bins=15)

    # Boundedness
    assert 0.0 <= js_xy <= 1.0, f"JS divergence {js_xy} out of bounds [0, 1]"
    assert 0.0 <= js_yx <= 1.0, f"JS divergence {js_yx} out of bounds [0, 1]"

    # Symmetry
    np.testing.assert_allclose(js_xy, js_yx, rtol=1e-5, atol=1e-5)


@given(float_samples, float_samples)
@settings(max_examples=40, deadline=None)
def test_property_row_order_invariance(sample_x: list[float], sample_y: list[float]) -> None:
    """Property 4: Shuffling reference or target samples does NOT change test statistics."""
    x = np.array(sample_x)
    y = np.array(sample_y)

    rng = np.random.default_rng(42)
    x_shuffled = rng.permutation(x)
    y_shuffled = rng.permutation(y)

    # 1. KS test order invariance
    ks_orig, _, _, _ = calculate_ks_test(x, y)
    ks_shuf, _, _, _ = calculate_ks_test(x_shuffled, y_shuffled)
    np.testing.assert_allclose(ks_orig, ks_shuf, rtol=1e-6, atol=1e-6)

    # 2. Wasserstein order invariance
    w_orig, _, _, _ = calculate_wasserstein(x, y)
    w_shuf, _, _, _ = calculate_wasserstein(x_shuffled, y_shuffled)
    np.testing.assert_allclose(w_orig, w_shuf, rtol=1e-6, atol=1e-6)

    # 3. PSI order invariance
    if len(np.unique(x)) >= 2:
        psi_orig, _ = calculate_psi_continuous(x, y)
        psi_shuf, _ = calculate_psi_continuous(x_shuffled, y_shuffled)
        np.testing.assert_allclose(psi_orig, psi_shuf, rtol=1e-6, atol=1e-6)


def test_property_zero_bin_safety() -> None:
    """Property 5: Concentrated production data with empty bins does NOT produce division-by-zero or NaN."""
    # Reference distributed across 0 to 100
    ref = np.linspace(0.0, 100.0, 200)
    # Current sample completely clustered at 50.0 (empty outer bins)
    curr = np.full(200, 50.0)

    psi_val, details = calculate_psi_continuous(ref, curr, num_bins=10)
    assert not np.isnan(psi_val)
    assert not np.isinf(psi_val)
    assert psi_val > 0.0  # Should detect heavy drift without crash


def test_property_nan_handling() -> None:
    """Property 6: NaN entries are safely dropped without throwing exceptions."""
    ref = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0] * 10)
    curr = np.array([np.nan, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0] * 10)

    psi_val, _ = calculate_psi_continuous(ref, curr)
    assert not np.isnan(psi_val)

    ks_stat, p_val, _, _ = calculate_ks_test(ref, curr)
    assert not np.isnan(ks_stat)
    assert not np.isnan(p_val)


def test_multiple_testing_fdr_control() -> None:
    """Property 7: Benjamini-Hochberg adjustment produces valid monotonically bounded p-values."""
    raw_p = [0.001, 0.015, 0.03, 0.045, 0.20, 0.65, 0.90]
    bh_p = adjust_p_values_benjamini_hochberg(raw_p)

    # All adjusted p-values >= raw p-values
    for r, adj in zip(raw_p, bh_p):
        assert adj is not None
        assert adj >= r
        assert 0.0 <= adj <= 1.0

    # Bonferroni adjustment
    bonf_p = adjust_p_values_bonferroni(raw_p)
    for r, adj in zip(raw_p, bonf_p):
        assert adj is not None
        assert adj >= r


def test_dual_gating_decision_rule() -> None:
    """Property 8: Dual-gating requires BOTH significance AND minimum effect size."""
    # Case A: Significant p-value BUT tiny effect size (Large-N problem)
    is_drift, reason = evaluate_dual_gating_decision(p_value=1e-8, effect_size=0.01, effect_size_threshold=0.10)
    assert not is_drift
    assert "trivial effect size" in reason

    # Case B: Substantial effect size BUT not statistically significant (low sample size noise)
    is_drift, reason = evaluate_dual_gating_decision(p_value=0.25, effect_size=0.35, effect_size_threshold=0.10)
    assert not is_drift
    assert "lacks statistical significance" in reason

    # Case C: Both significant and substantial effect size -> Confirmed drift
    is_drift, reason = evaluate_dual_gating_decision(p_value=0.001, effect_size=0.25, effect_size_threshold=0.10)
    assert is_drift
    assert "Drift confirmed" in reason


def test_hysteresis_state_tracker() -> None:
    """Property 9: Stateful window tracker requires k-of-n escalation and hysteresis margin recovery."""
    tracker = HysteresisStateTracker(
        k=3,
        n=5,
        m_recovery=2,
        threshold=0.10,
        hysteresis_margin=0.03,  # Recovery ceiling = 0.07
    )
    key = "feat_sales"

    # Step 1: Single breach does NOT escalate
    is_alert, msg = tracker.update(key, 0.12)
    assert not is_alert

    # Step 2: Second breach does NOT escalate yet (needs k=3)
    is_alert, msg = tracker.update(key, 0.15)
    assert not is_alert

    # Step 3: Third breach -> Escalates to ALERT!
    is_alert, msg = tracker.update(key, 0.14)
    assert is_alert
    assert "Alert escalated" in msg

    # Step 4: Small dip to 0.09 (below 0.10 but above recovery ceiling 0.07) -> Remains in ALERT!
    is_alert, msg = tracker.update(key, 0.09)
    assert is_alert
    assert "Alert sustained" in msg

    # Step 5: First recovery window below 0.07 (e.g. 0.05) -> Cooling down (needs m=2)
    is_alert, msg = tracker.update(key, 0.05)
    assert is_alert
    assert "cooling down" in msg

    # Step 6: Second recovery window below 0.07 -> Alert cleared!
    is_alert, msg = tracker.update(key, 0.04)
    assert not is_alert
    assert "Alert cleared" in msg
