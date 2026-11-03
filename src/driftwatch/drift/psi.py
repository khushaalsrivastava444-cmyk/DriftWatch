"""Population Stability Index (PSI) Implementation.

Features:
- Bins derived strictly from reference quantiles (never target data).
- Laplace smoothing (epsilon = 1e-4) to avoid zero-bin division.
- Native categorical distribution support.
- Fully vectorized with zero memory leaks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_psi_continuous(
    reference: np.ndarray | pd.Series | list[float],
    current: np.ndarray | pd.Series | list[float],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    """Compute Population Stability Index for continuous numerical distributions.
    
    Bin edges are computed strictly from reference partition quantiles:
        PSI = sum_{b=1}^B (Actual_b - Expected_b) * ln(Actual_b / Expected_b)
    
    Args:
        reference: Historical reference baseline sample.
        current: Current monitored production sample.
        num_bins: Number of quantile bins to form on reference data.
        epsilon: Additive Laplace smoothing parameter for empty bins.
        
    Returns:
        Tuple of (psi_statistic, details_dict)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    # Filter NaNs
    ref_clean = ref[~np.isnan(ref)]
    curr_clean = curr[~np.isnan(curr)]

    n_ref = len(ref_clean)
    n_curr = len(curr_clean)

    if n_ref == 0 or n_curr == 0:
        return 0.0, {"error": "Empty sample after NaN removal", "n_ref": n_ref, "n_curr": n_curr}

    # 1. Fit quantile bins strictly on reference
    percentiles = np.linspace(0.0, 100.0, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    # Remove duplicate edges for degenerate distributions with repeated values
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        # All reference values are identical
        unique_edges = np.array([unique_edges[0] - 1.0, unique_edges[0] + 1.0])

    # Extend lower and upper boundaries to cover all real values [-inf, +inf]
    unique_edges[0] = -np.inf
    unique_edges[-1] = np.inf

    n_bins_actual = len(unique_edges) - 1

    # 2. Histogram counts
    ref_counts, _ = np.histogram(ref_clean, bins=unique_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=unique_edges)

    # 3. Laplace smoothed probabilities
    p_ref = (ref_counts + epsilon) / (n_ref + (n_bins_actual * epsilon))
    q_curr = (curr_counts + epsilon) / (n_curr + (n_bins_actual * epsilon))

    # 4. PSI calculation: sum((Q - P) * ln(Q / P))
    psi_components = (q_curr - p_ref) * np.log(q_curr / p_ref)
    psi_value = float(np.sum(psi_components))
    psi_value = max(0.0, psi_value)  # Numerical non-negativity guard

    details = {
        "n_ref": n_ref,
        "n_curr": n_curr,
        "n_bins": n_bins_actual,
        "ref_proportions": p_ref.tolist(),
        "curr_proportions": q_curr.tolist(),
        "bin_psi_contributions": psi_components.tolist(),
    }
    return psi_value, details


def calculate_psi_categorical(
    reference: pd.Series | list[str] | np.ndarray,
    current: pd.Series | list[str] | np.ndarray,
    epsilon: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    """Compute Population Stability Index for categorical distributions.
    
    Category categories are determined strictly by reference partition,
    with an explicit fallback bin for unseen production categories.
    """
    ref_series = pd.Series(reference).dropna().astype(str)
    curr_series = pd.Series(current).dropna().astype(str)

    n_ref = len(ref_series)
    n_curr = len(curr_series)

    if n_ref == 0 or n_curr == 0:
        return 0.0, {"error": "Empty sample after NaN removal", "n_ref": n_ref, "n_curr": n_curr}

    # Reference category vocabulary + unseen fallback category
    ref_cats = sorted(ref_series.unique().tolist())
    all_bins = ref_cats + ["__UNSEEN__"]
    n_bins = len(all_bins)

    ref_counts = ref_series.value_counts()
    curr_counts = curr_series.value_counts()

    ref_vec = np.zeros(n_bins, dtype=float)
    curr_vec = np.zeros(n_bins, dtype=float)

    for i, cat in enumerate(ref_cats):
        ref_vec[i] = ref_counts.get(cat, 0)
        curr_vec[i] = curr_counts.get(cat, 0)

    # Count unseen categories in current sample
    unseen_count = sum(cnt for cat, cnt in curr_counts.items() if cat not in ref_cats)
    curr_vec[-1] = unseen_count
    ref_vec[-1] = 0.0  # By definition, reference has 0 unseen categories

    # Laplace smoothing
    p_ref = (ref_vec + epsilon) / (n_ref + (n_bins * epsilon))
    q_curr = (curr_vec + epsilon) / (n_curr + (n_bins * epsilon))

    psi_components = (q_curr - p_ref) * np.log(q_curr / p_ref)
    psi_value = float(np.sum(psi_components))
    psi_value = max(0.0, psi_value)

    details = {
        "n_ref": n_ref,
        "n_curr": n_curr,
        "categories": all_bins,
        "unseen_count": unseen_count,
        "ref_proportions": p_ref.tolist(),
        "curr_proportions": q_curr.tolist(),
    }
    return psi_value, details


def calculate_psi(
    reference: np.ndarray | pd.Series | list[Any],
    current: np.ndarray | pd.Series | list[Any],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    """Universal PSI dispatcher detecting numeric vs categorical data automatically."""
    ref_s = pd.Series(reference)
    if pd.api.types.is_numeric_dtype(ref_s):
        return calculate_psi_continuous(reference, current, num_bins=num_bins, epsilon=epsilon)
    return calculate_psi_categorical(reference, current, epsilon=epsilon)
