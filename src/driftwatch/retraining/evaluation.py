"""Champion vs Challenger Paired Statistical Evaluation.

Performs paired-error hypothesis testing to verify whether a retrained candidate
is genuinely superior to the active champion before approval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelComparisonResult:
    """Rigorous statistical comparison result between Champion and Candidate."""
    champion_metric: float
    candidate_metric: float
    metric_name: str
    absolute_improvement: float
    relative_improvement_pct: float
    test_statistic: float
    p_value: float
    is_stat_significant: bool
    decision: str  # "APPROVED" or "REJECTED"
    sample_size: int
    test_method: str
    details: dict[str, object]


def diebold_mariano_hln_test(
    champ_errors: np.ndarray,
    cand_errors: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN 1997) autocorrelation adjustment.
    
    H0: E[d_t] >= 0 (Candidate error is not smaller than Champion)
    H1: E[d_t] < 0 (Candidate error is strictly smaller than Champion)
    where d_t = cand_error_t - champ_error_t.
    """
    d = np.asarray(cand_errors, dtype=float) - np.asarray(champ_errors, dtype=float)
    t_len = len(d)
    if t_len < 5:
        raise ValueError(f"Insufficient samples for DM test: {t_len} < 5")

    mean_d = float(np.mean(d))
    d_centered = d - mean_d

    gamma_0 = float(np.mean(d_centered ** 2))
    autocov_sum = 0.0
    max_lag = max(1, h - 1)
    for k in range(1, max_lag):
        gamma_k = float(np.sum(d_centered[k:] * d_centered[:-k]) / t_len)
        w = 1.0 - (k / (max_lag + 1))
        autocov_sum += 2.0 * w * gamma_k

    long_run_var = max(1e-12, (gamma_0 + autocov_sum) / t_len)
    dm_stat = mean_d / np.sqrt(long_run_var)

    # Harvey, Leybourne, and Newbold (1997) small-sample & autocorrelation correction
    hln_factor = np.sqrt(max(0.01, (t_len + 1 - 2 * h + (h * (h - 1) / t_len)) / t_len))
    stat_hln = float(dm_stat * hln_factor)

    df = max(1, t_len - 1)
    p_val = float(stats.t.cdf(stat_hln, df=df))
    return stat_hln, p_val


def compare_champion_challenger(
    y_true: np.ndarray,
    champion_preds: np.ndarray,
    candidate_preds: np.ndarray,
    metric_name: str = "MAE",
    alpha: float = 0.05,
    min_improvement_pct: float = 0.0,
    forecast_horizon: int = 1,
) -> ModelComparisonResult:
    """Paired statistical hypothesis test comparing candidate against champion.
    
    H0: Candidate error >= Champion error (Candidate is NOT better)
    H1: Candidate error < Champion error (Candidate has significantly smaller error)
    """
    y_true = np.asarray(y_true, dtype=float)
    champ_preds = np.asarray(champion_preds, dtype=float)
    cand_preds = np.asarray(candidate_preds, dtype=float)

    n_samples = len(y_true)
    if n_samples < 5:
        raise ValueError(f"Insufficient samples for statistical comparison: {n_samples} < 5")

    # Compute absolute paired errors
    champ_errors = np.abs(y_true - champ_preds)
    cand_errors = np.abs(y_true - cand_preds)

    champ_mean_mae = float(np.mean(champ_errors))
    cand_mean_mae = float(np.mean(cand_errors))
    abs_improvement = champ_mean_mae - cand_mean_mae
    rel_improvement_pct = (abs_improvement / champ_mean_mae * 100.0) if champ_mean_mae > 0 else 0.0

    # Paired differences: candidate - champion (negative values indicate candidate was better)
    diffs = cand_errors - champ_errors

    # If differences are all zero (e.g. identical models), reject
    if np.allclose(diffs, 0.0):
        return ModelComparisonResult(
            champion_metric=champ_mean_mae,
            candidate_metric=cand_mean_mae,
            metric_name=metric_name,
            absolute_improvement=0.0,
            relative_improvement_pct=0.0,
            test_statistic=0.0,
            p_value=1.0,
            is_stat_significant=False,
            decision="REJECTED",
            sample_size=n_samples,
            test_method="None (Identical errors)",
            details={"reason": "Candidate produces identical errors to champion"},
        )

    # Diebold-Mariano test with Harvey-Leybourne-Newbold autocorrelation correction
    test_method = "Diebold-Mariano Test (HLN 1997 Corrected)"
    try:
        stat, p_val = diebold_mariano_hln_test(champ_errors, cand_errors, h=forecast_horizon)
    except Exception as e:
        logger.info("DM-HLN test fallback to Wilcoxon (%s); running non-parametric paired test", e)
        test_method = "Wilcoxon Signed-Rank Test"
        try:
            # Wilcoxon requires non-zero differences
            res = stats.wilcoxon(cand_errors, champ_errors, alternative="less", zero_method="pratt")
            stat = float(res.statistic)
            p_val = float(res.pvalue)
        except Exception as e_w:
            logger.warning("Wilcoxon test encountered fallback condition (%s); using paired t-test", e_w)
            test_method = "Paired Student's t-test"
            res_t = stats.ttest_rel(cand_errors, champ_errors, alternative="less")
            stat = float(res_t.statistic)
            p_val = float(res_t.pvalue)

    # Decision criteria:
    # 1. p-value < alpha (statistically significant superiority)
    # 2. Candidate MAE must be strictly less than Champion MAE
    # 3. Relative improvement >= min_improvement_pct
    is_significant = (p_val < alpha) and (cand_mean_mae < champ_mean_mae)
    meets_threshold = rel_improvement_pct >= min_improvement_pct

    decision = "APPROVED" if (is_significant and meets_threshold) else "REJECTED"

    return ModelComparisonResult(
        champion_metric=round(champ_mean_mae, 4),
        candidate_metric=round(cand_mean_mae, 4),
        metric_name=metric_name,
        absolute_improvement=round(abs_improvement, 4),
        relative_improvement_pct=round(rel_improvement_pct, 2),
        test_statistic=round(stat, 4),
        p_value=round(p_val, 6),
        is_stat_significant=is_significant,
        decision=decision,
        sample_size=n_samples,
        test_method=test_method,
        details={
            "alpha": alpha,
            "min_improvement_pct": min_improvement_pct,
            "mean_paired_difference": float(np.mean(diffs)),
            "std_paired_difference": float(np.std(diffs, ddof=1)),
        },
    )
