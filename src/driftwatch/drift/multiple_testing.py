"""Multiple Testing Corrections and Dual-Gating Decision Logic.

Implements:
- Benjamini-Hochberg (BH) False Discovery Rate (FDR) control
- Bonferroni Family-Wise Error Rate (FWER) control
- Dual-Gating decision rules combining significance and effect size to solve the large-N problem
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


class MultipleTestingMethod(str, Enum):
    BENJAMINI_HOCHBERG = "BENJAMINI_HOCHBERG"
    BONFERRONI = "BONFERRONI"
    NONE = "NONE"


def adjust_p_values_benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Adjust raw p-values using the Benjamini-Hochberg (1995) FDR control procedure.
    
    Preserves exact input order and handles None values safely.
    
    Formula:
        p_adj(i) = min(1.0, min_{j >= i} (m / j) * p_(j))
    """
    valid_indices = [i for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    m = len(valid_indices)
    if m == 0:
        return list(p_values)

    raw_vals = [float(p_values[i]) for i in valid_indices]  # type: ignore[arg-type]
    
    # Sort ascending
    sort_order = np.argsort(raw_vals)
    sorted_p = np.array(raw_vals)[sort_order]
    
    # Compute BH adjustment factor: (m / rank)
    ranks = np.arange(1, m + 1)
    adjusted_factors = (m / ranks) * sorted_p
    
    # Monotonize backwards from the highest p-value
    monotonized = np.minimum.accumulate(adjusted_factors[::-1])[::-1]
    monotonized = np.clip(monotonized, 0.0, 1.0)
    
    # Invert sorting to match original order
    inverse_order = np.empty(m, dtype=int)
    inverse_order[sort_order] = np.arange(m)
    final_adj = monotonized[inverse_order]

    result: list[float | None] = list(p_values)
    for orig_idx, adj_val in zip(valid_indices, final_adj):
        result[orig_idx] = float(adj_val)

    return result


def adjust_p_values_bonferroni(p_values: Sequence[float | None]) -> list[float | None]:
    """Adjust raw p-values using Bonferroni single-step FWER correction."""
    valid_indices = [i for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    m = len(valid_indices)
    if m == 0:
        return list(p_values)

    result: list[float | None] = list(p_values)
    for idx in valid_indices:
        raw_p = float(p_values[idx])  # type: ignore[arg-type]
        result[idx] = min(1.0, float(raw_p * m))

    return result


def adjust_p_values(
    p_values: Sequence[float | None],
    method: MultipleTestingMethod = MultipleTestingMethod.BENJAMINI_HOCHBERG,
) -> list[float | None]:
    """Dispatcher applying chosen multiple testing adjustment."""
    if method == MultipleTestingMethod.BENJAMINI_HOCHBERG:
        return adjust_p_values_benjamini_hochberg(p_values)
    elif method == MultipleTestingMethod.BONFERRONI:
        return adjust_p_values_bonferroni(p_values)
    return list(p_values)


def evaluate_dual_gating_decision(
    p_value: float | None,
    effect_size: float,
    alpha: float = 0.05,
    effect_size_threshold: float = 0.10,
) -> tuple[bool, str]:
    """Dual-gating drift decision rule to mitigate the large-N sample size collapse.
    
    Requirement:
        A drift detection alert triggers IF AND ONLY IF:
        1. The difference is statistically significant (adjusted p-value < alpha, if p-value exists).
        2. AND the magnitude is substantively meaningful (effect_size >= effect_size_threshold).
        
    Note: Thresholds are labeled domain heuristics.
    """
    if p_value is None:
        is_effect_substantial = effect_size >= effect_size_threshold
        if is_effect_substantial:
            reason = (
                f"Drift confirmed (effect-size detector): Effect size {effect_size:.4f} >= "
                f"{effect_size_threshold:.4f} (Heuristic threshold detector without hypothesis test)"
            )
        else:
            reason = (
                f"Drift rejected (effect-size detector): Effect size {effect_size:.4f} < "
                f"{effect_size_threshold:.4f}"
            )
        return is_effect_substantial, reason

    is_significant = p_value < alpha
    is_effect_substantial = effect_size >= effect_size_threshold
    is_drift = bool(is_significant and is_effect_substantial)

    if is_drift:
        reason = (
            f"Drift confirmed (dual-gated): Statistically significant (p={p_value:.6f} < {alpha}) "
            f"AND substantive effect size ({effect_size:.4f} >= {effect_size_threshold:.4f})"
        )
    elif not is_significant and is_effect_substantial:
        reason = (
            f"Drift rejected: Effect size ({effect_size:.4f}) is substantial, but lacks "
            f"statistical significance (p={p_value:.6f} >= {alpha})"
        )
    elif is_significant and not is_effect_substantial:
        reason = (
            f"Drift rejected (large-N guard): Statistically significant (p={p_value:.6f} < {alpha}) "
            f"but trivial effect size ({effect_size:.4f} < {effect_size_threshold:.4f})"
        )
    else:
        reason = (
            f"Drift rejected: Neither statistically significant (p={p_value:.6f}) "
            f"nor substantial effect ({effect_size:.4f})"
        )

    return is_drift, reason
