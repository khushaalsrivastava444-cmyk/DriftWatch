"""Statistical Drift Tests and Effect Size Estimators.

Implements standard continuous and discrete two-sample hypothesis tests:
- Kolmogorov-Smirnov (KS)
- Jensen-Shannon (JS) Divergence
- Wasserstein (Earth Mover's Distance)
- Chi-Square Test of Independence
- Mean Shift (Welch's t-test + Cohen's d)
- Variance Shift (Levene test + Log Variance Ratio)
- Missingness Shift (Two-proportion z-test)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import spatial, stats

logger = logging.getLogger(__name__)


def calculate_ks_test(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float, float, dict[str, Any]]:
    """Two-sample Kolmogorov-Smirnov test.
    
    Returns:
        (ks_statistic, p_value, effect_size, details)
        Effect size is directly the KS statistic D in [0, 1].
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, 1.0, 0.0, {"error": "Empty sample"}

    res = stats.ks_2samp(ref, curr)
    stat = float(res.statistic)
    p_val = float(res.pvalue)
    effect_size = stat  # D is bounded in [0, 1]

    return stat, p_val, effect_size, {"n_ref": len(ref), "n_curr": len(curr)}


def calculate_jensen_shannon(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_bins: int = 20,
    epsilon: float = 1e-5,
) -> tuple[float, float | None, float, dict[str, Any]]:
    """Jensen-Shannon Divergence using joint support binning with base-2 logarithm.
    
    Returns:
        (js_divergence, None, effect_size, details)
        Guaranteed properties:
        - Symmetric: JS(x, y) == JS(y, x)
        - Bounded: JS in [0, 1]
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, None, 0.0, {"error": "Empty sample"}

    # Joint support
    min_val = min(float(np.min(ref)), float(np.min(curr)))
    max_val = max(float(np.max(ref)), float(np.max(curr)))

    if min_val == max_val:
        return 0.0, None, 0.0, {"note": "Degenerate single-value distribution"}

    bins = np.linspace(min_val, max_val, num_bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bins)
    curr_counts, _ = np.histogram(curr, bins=bins)

    # Laplace smoothing
    p = (ref_counts + epsilon) / (len(ref) + (num_bins * epsilon))
    q = (curr_counts + epsilon) / (len(curr) + (num_bins * epsilon))

    # scipy jensenshannon returns the distance (sqrt of divergence)
    js_dist = float(spatial.distance.jensenshannon(p, q, base=2.0))
    js_divergence = float(js_dist ** 2)
    js_divergence = max(0.0, min(1.0, js_divergence))

    return js_divergence, None, js_divergence, {"js_distance": js_dist, "n_bins": num_bins}


def calculate_wasserstein(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float | None, float, dict[str, Any]]:
    """Wasserstein Distance (Earth Mover's Distance).
    
    Returns:
        (wasserstein_dist, None, normalized_effect_size, details)
        Effect size is standardized by reference standard deviation.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, None, 0.0, {"error": "Empty sample"}

    dist = float(stats.wasserstein_distance(ref, curr))
    ref_std = float(np.std(ref))
    norm_effect = (dist / ref_std) if ref_std > 1e-6 else dist

    return dist, None, norm_effect, {"ref_std": ref_std}


def calculate_chi_square(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    epsilon: float = 1.0,
) -> tuple[float, float, float, dict[str, Any]]:
    """Chi-Square Test of Independence for categorical distributions.
    
    Returns:
        (chi2_statistic, p_value, cramers_v, details)
    """
    ref_s = pd.Series(reference).dropna().astype(str)
    curr_s = pd.Series(current).dropna().astype(str)

    if len(ref_s) == 0 or len(curr_s) == 0:
        return 0.0, 1.0, 0.0, {"error": "Empty sample"}

    all_cats = sorted(set(ref_s.unique()) | set(curr_s.unique()))
    ref_counts = ref_s.value_counts()
    curr_counts = curr_s.value_counts()

    # Form contingency table with Laplace smoothing
    obs = np.array([
        [ref_counts.get(c, 0) + epsilon for c in all_cats],
        [curr_counts.get(c, 0) + epsilon for c in all_cats],
    ])

    chi2, p_val, dof, _ = stats.chi2_contingency(obs)
    n_total = float(np.sum(obs))
    # Cramér's V = sqrt(chi2 / (n * min(r-1, c-1)))
    min_dim = min(obs.shape[0] - 1, obs.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n_total * min_dim))) if (n_total > 0 and min_dim > 0) else 0.0

    return float(chi2), float(p_val), float(cramers_v), {"dof": dof, "categories": all_cats}


def calculate_mean_shift(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float, float, dict[str, Any]]:
    """Welch's Two-Sample t-test for difference in means and Cohen's d effect size.
    
    Returns:
        (t_statistic, p_value, cohens_d, details)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) < 2 or len(curr) < 2:
        return 0.0, 1.0, 0.0, {"error": "Insufficient sample"}

    t_stat, p_val = stats.ttest_ind(curr, ref, equal_var=False)

    # Cohen's d: (mean_curr - mean_ref) / pooled_std
    mean_ref, mean_curr = float(np.mean(ref)), float(np.mean(curr))
    var_ref, var_curr = float(np.var(ref, ddof=1)), float(np.var(curr, ddof=1))
    pooled_sd = np.sqrt(((len(ref) - 1) * var_ref + (len(curr) - 1) * var_curr) / (len(ref) + len(curr) - 2))
    cohens_d = float(abs(mean_curr - mean_ref) / pooled_sd) if pooled_sd > 1e-6 else 0.0

    return float(t_stat), float(p_val), cohens_d, {"mean_ref": mean_ref, "mean_curr": mean_curr}


def calculate_variance_shift(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float, float, dict[str, Any]]:
    """Levene test for equality of variance and log variance ratio effect size."""
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) < 3 or len(curr) < 3:
        return 0.0, 1.0, 0.0, {"error": "Insufficient sample"}

    stat, p_val = stats.levene(ref, curr, center="median")
    var_ref = float(np.var(ref, ddof=1))
    var_curr = float(np.var(curr, ddof=1))

    log_ratio = float(abs(np.log(var_curr / var_ref))) if (var_ref > 1e-8 and var_curr > 1e-8) else 0.0
    return float(stat), float(p_val), log_ratio, {"var_ref": var_ref, "var_curr": var_curr}


def calculate_missing_rate_shift(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
) -> tuple[float, float, float, dict[str, Any]]:
    """Two-proportion test for changes in null/missing rate."""
    ref_s = pd.Series(reference)
    curr_s = pd.Series(current)

    n_ref = len(ref_s)
    n_curr = len(curr_s)
    if n_ref == 0 or n_curr == 0:
        return 0.0, 1.0, 0.0, {"error": "Empty sample"}

    null_ref = int(ref_s.isna().sum())
    null_curr = int(curr_s.isna().sum())

    rate_ref = null_ref / n_ref
    rate_curr = null_curr / n_curr
    effect_size = abs(rate_curr - rate_ref)

    # Pooled proportion test
    p_pool = (null_ref + null_curr) / (n_ref + n_curr)
    if p_pool == 0.0 or p_pool == 1.0:
        return 0.0, 1.0, effect_size, {"rate_ref": rate_ref, "rate_curr": rate_curr}

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_ref + 1 / n_curr))
    z_stat = (rate_curr - rate_ref) / se if se > 1e-8 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return float(z_stat), float(p_val), effect_size, {"rate_ref": rate_ref, "rate_curr": rate_curr}
