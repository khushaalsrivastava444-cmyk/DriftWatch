"""Multivariate Drift Detection using Domain Classifier AUC and MMD.

Identifies complex joint distribution shifts P(X_1, ..., X_p) that evade univariate tests,
calibrated with rigorous permutation testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from driftwatch.drift.results import EvidenceClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultivariateDriftResult:
    """Multivariate drift evaluation results."""
    method: str
    statistic: float  # AUC or MMD^2
    p_value: float
    threshold: float
    decision: bool  # True if multivariate drift detected
    n_reference: int
    n_current: int
    evidence_class: EvidenceClass
    details: dict[str, Any]


def compute_domain_classifier_auc(
    ref_features: np.ndarray,
    curr_features: np.ndarray,
    n_permutations: int = 50,
    auc_threshold: float = 0.65,
    alpha: float = 0.05,
    seed: int = 42,
) -> MultivariateDriftResult:
    """Detect joint feature distribution shift using a Domain Classifier with permutation calibration.
    
    Labels: 0 = Reference data, 1 = Current data.
    If distributions are identical, AUC ~ 0.50.
    Drift flagged if AUC > auc_threshold AND permutation p-value < alpha.
    """
    rng = np.random.RandomState(seed)

    n_ref = len(ref_features)
    n_curr = len(curr_features)

    # Subsample if large to maintain sub-second runtime
    max_samples = 1000
    if n_ref > max_samples:
        idx_ref = rng.choice(n_ref, size=max_samples, replace=False)
        ref_features = ref_features[idx_ref]
        n_ref = len(ref_features)
    if n_curr > max_samples:
        idx_curr = rng.choice(n_curr, size=max_samples, replace=False)
        curr_features = curr_features[idx_curr]
        n_curr = len(curr_features)

    X = np.vstack([ref_features, curr_features])
    y = np.hstack([np.zeros(n_ref, dtype=int), np.ones(n_curr, dtype=int)])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3-Fold Cross-Validated Out-Of-Fold AUC
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in cv.split(X_scaled, y):
        clf = LogisticRegression(max_iter=200, random_state=seed, solver="lbfgs")
        clf.fit(X_scaled[train_idx], y[train_idx])
        oof_preds[val_idx] = clf.predict_proba(X_scaled[val_idx])[:, 1]

    observed_auc = float(roc_auc_score(y, oof_preds))

    # Permutation Calibration: Null distribution under H0 (no domain separation)
    perm_aucs = []
    clf_perm = LogisticRegression(max_iter=100, random_state=seed, solver="lbfgs")
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        clf_perm.fit(X_scaled, y_perm)
        p_scores = clf_perm.predict_proba(X_scaled)[:, 1]
        perm_aucs.append(float(roc_auc_score(y_perm, p_scores)))

    # Empirical permutation p-value
    perm_aucs = np.array(perm_aucs)
    p_val = float((1 + np.sum(perm_aucs >= observed_auc)) / (n_permutations + 1))

    # Dual criteria: statistical significance AND substantive separation
    decision = (observed_auc >= auc_threshold) and (p_val < alpha)

    return MultivariateDriftResult(
        method="Domain Classifier (Cross-Validated AUC)",
        statistic=round(observed_auc, 4),
        p_value=round(p_val, 4),
        threshold=auc_threshold,
        decision=decision,
        n_reference=n_ref,
        n_current=n_curr,
        evidence_class=EvidenceClass.STATISTICAL_SHIFT,
        details={
            "null_auc_mean": round(float(np.mean(perm_aucs)), 4),
            "null_auc_std": round(float(np.std(perm_aucs)), 4),
            "permutations": n_permutations,
        },
    )


def compute_mmd(
    ref_features: np.ndarray,
    curr_features: np.ndarray,
    n_permutations: int = 50,
    alpha: float = 0.05,
    seed: int = 42,
) -> MultivariateDriftResult:
    """Maximum Mean Discrepancy (MMD) with RBF kernel and permutation test.
    
    Tests H0: P = Q using the median distance heuristic for kernel bandwidth.
    """
    rng = np.random.RandomState(seed)

    # Subsample for computational efficiency (O(N^2) kernel matrix)
    n_max = 300
    X = ref_features[:n_max]
    Y = curr_features[:n_max]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.transform(Y)

    # Median heuristic for gamma
    dists = pdist(X)
    med_dist = np.median(dists) if len(dists) > 0 and np.median(dists) > 0 else 1.0
    gamma = 1.0 / (2 * (med_dist ** 2))

    def rbf_kernel(A, B):
        # A: (n, d), B: (m, d)
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        return np.exp(-gamma * sq_dist)

    def calc_mmd2(A, B):
        n_a, n_b = len(A), len(B)
        K_aa = rbf_kernel(A, A)
        K_bb = rbf_kernel(B, B)
        K_ab = rbf_kernel(A, B)

        # Unbiased estimator (exclude diagonals)
        np.fill_diagonal(K_aa, 0)
        np.fill_diagonal(K_bb, 0)
        term_aa = np.sum(K_aa) / (n_a * (n_a - 1)) if n_a > 1 else 0
        term_bb = np.sum(K_bb) / (n_b * (n_b - 1)) if n_b > 1 else 0
        term_ab = 2.0 * np.sum(K_ab) / (n_a * n_b)
        return float(term_aa + term_bb - term_ab)

    observed_mmd2 = calc_mmd2(X, Y)

    # Permutation test
    combined = np.vstack([X, Y])
    n_x = len(X)
    null_mmd2s = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        null_mmd2s.append(calc_mmd2(shuffled[:n_x], shuffled[n_x:]))

    null_mmd2s = np.array(null_mmd2s)
    p_val = float((1 + np.sum(null_mmd2s >= observed_mmd2)) / (n_permutations + 1))
    threshold = float(np.quantile(null_mmd2s, 1.0 - alpha))
    decision = (observed_mmd2 > threshold) and (p_val < alpha)

    return MultivariateDriftResult(
        method="Maximum Mean Discrepancy (MMD RBF)",
        statistic=round(max(0.0, observed_mmd2), 6),
        p_value=round(p_val, 4),
        threshold=round(max(0.0, threshold), 6),
        decision=decision,
        n_reference=len(X),
        n_current=len(Y),
        evidence_class=EvidenceClass.STATISTICAL_SHIFT,
        details={
            "gamma": round(gamma, 5),
            "median_pairwise_distance": round(med_dist, 4),
            "permutations": n_permutations,
        },
    )
