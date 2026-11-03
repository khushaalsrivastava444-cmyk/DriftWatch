"""Comprehensive Verification Suite for Tier 2 Advanced Capabilities.

Covers (in exact brief priority order):
1. Drift-impact attribution (permutation/feature importance gates alert severity).
2. Conformal prediction intervals and empirical coverage breach detection.
3. Multivariate drift (Domain Classifier cross-validated AUC + MMD with permutation testing).
4. Shadow deployment of challenger models.
5. Label-free performance estimation with stated assumptions and concept drift failure modes.
6. Evidently 0.7+ HTML report export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftwatch.alerts.engine import AlertSeverity
from driftwatch.api.schemas import PredictResponse
from driftwatch.api.shadow import ShadowDeploymentManager
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.drift.attribution import DriftAttributionEngine
from driftwatch.drift.multivariate import compute_domain_classifier_auc, compute_mmd
from driftwatch.drift.results import EvidenceClass, FeatureDriftResult, TestType
from driftwatch.features.pipeline import DemandFeaturePipeline
from driftwatch.models.conformal import ConformalDemandForecaster
from driftwatch.models.xgboost_forecaster import (
    XGBoostDemandForecaster,
    XGBoostForecasterConfig,
)
from driftwatch.monitoring.evidently_export import generate_evidently_html_report
from driftwatch.monitoring.label_free import LabelFreePerformanceEstimator


@pytest.fixture(scope="module")
def trained_forecaster_and_data():
    """Module fixture providing trained forecaster and leak-free feature partitions."""
    cfg = SyntheticDemandConfig(num_days=120, num_stores=2, num_items=5, seed=42)
    raw = generate_synthetic_demand(cfg)
    split_idx = int(len(raw) * 0.7)
    train_df = raw.iloc[:split_idx].copy()
    test_df = raw.iloc[split_idx:].copy()

    pipeline = DemandFeaturePipeline().fit(train_df)
    train_feats = pipeline.transform(train_df)
    test_feats = pipeline.transform(test_df)

    forecaster = XGBoostDemandForecaster(XGBoostForecasterConfig(n_estimators=50, max_depth=3))
    forecaster.fit(train_feats, train_feats["sales"])

    return forecaster, pipeline, train_feats, test_feats


def test_drift_attribution_suppresses_unimportant_feature_warning(trained_forecaster_and_data):
    """Verifies drift in low-importance features is demoted to INFO, preventing alert fatigue."""
    forecaster, _, _, _ = trained_forecaster_and_data
    attribution_engine = DriftAttributionEngine(forecaster, min_importance_threshold=0.03)

    # Pick the most important and least important features
    sorted_features = sorted(
        attribution_engine.feature_importances_.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    high_imp_feat = sorted_features[0][0]
    low_imp_feat = sorted_features[-1][0]

    # Drift in high importance feature -> WARNING
    drift_high = FeatureDriftResult(
        dataset_id="d1", reference_id="r1", window="w1",
        feature=high_imp_feat, test=TestType.PSI, statistic=0.30,
        p_value=0.001, adjusted_p_value=0.002, effect_size=0.30,
        threshold=0.10, decision=True, sample_size_ref=500, sample_size_curr=100,
        timestamp="2026-01-01T00:00:00Z", config_hash="h1",
        evidence_class=EvidenceClass.STATISTICAL_SHIFT,
    )
    res_high = attribution_engine.attribute_drift_result(drift_high)
    assert res_high.is_impactful is True
    assert res_high.recommended_severity == AlertSeverity.WARNING

    # Drift in low importance feature -> Demoted to INFO
    drift_low = FeatureDriftResult(
        dataset_id="d1", reference_id="r1", window="w1",
        feature=low_imp_feat, test=TestType.PSI, statistic=0.30,
        p_value=0.001, adjusted_p_value=0.002, effect_size=0.30,
        threshold=0.10, decision=True, sample_size_ref=500, sample_size_curr=100,
        timestamp="2026-01-01T00:00:00Z", config_hash="h1",
        evidence_class=EvidenceClass.STATISTICAL_SHIFT,
    )
    res_low = attribution_engine.attribute_drift_result(drift_low)
    assert res_low.is_impactful is False
    assert res_low.recommended_severity == AlertSeverity.INFO
    assert "suppressed to INFO" in res_low.attribution_reason


def test_split_conformal_prediction_and_coverage_breach(trained_forecaster_and_data):
    """Verifies conformal interval coverage and detects coverage degradation breach."""
    forecaster, _, train_feats, test_feats = trained_forecaster_and_data

    # Use holdout partition of train_feats as calibration set (inductive conformal)
    n_train = len(train_feats)
    cal_split_idx = int(n_train * 0.6)
    cal_feats = train_feats.iloc[cal_split_idx:].copy()

    conformal = ConformalDemandForecaster(forecaster, default_confidence_level=0.90)
    conformal.calibrate(cal_feats, cal_feats["sales"].values)

    # 1. Evaluate nominal coverage on in-distribution test set
    intervals = conformal.predict_intervals(test_feats, confidence_level=0.90)
    assert len(intervals.point_predictions) == len(test_feats)
    assert np.all(intervals.lower_bounds >= 0.0)  # Demand non-negative
    assert np.all(intervals.upper_bounds >= intervals.lower_bounds)

    report_in_dist = ConformalDemandForecaster.evaluate_coverage(
        y_true=test_feats["sales"].values,
        lower_bounds=intervals.lower_bounds,
        upper_bounds=intervals.upper_bounds,
        nominal_confidence=0.90,
    )
    assert report_in_dist.empirical_coverage >= 0.80  # Well-calibrated
    assert report_in_dist.coverage_breached is False

    # 2. Simulate severe unexpected demand shock (coverage breach)
    shocked_y = test_feats["sales"].values + 40.0
    report_breached = ConformalDemandForecaster.evaluate_coverage(
        y_true=shocked_y,
        lower_bounds=intervals.lower_bounds,
        upper_bounds=intervals.upper_bounds,
        nominal_confidence=0.90,
    )
    assert report_breached.coverage_breached is True
    assert report_breached.empirical_coverage < 0.50


def test_multivariate_domain_classifier_auc_and_mmd():
    """Verifies multivariate drift detection via Domain Classifier AUC and MMD."""
    np.random.seed(42)
    n = 200

    # Clean / stationary data: X_ref and X_curr from identical multivariate Gaussian
    mean = [0, 0, 0]
    cov = [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]
    X_ref = np.random.multivariate_normal(mean, cov, size=n)
    X_curr_stable = np.random.multivariate_normal(mean, cov, size=n)

    res_stable_auc = compute_domain_classifier_auc(X_ref, X_curr_stable, n_permutations=30, seed=42)
    assert res_stable_auc.decision is False
    assert res_stable_auc.statistic < 0.65  # Near 0.50

    res_stable_mmd = compute_mmd(X_ref, X_curr_stable, n_permutations=30, seed=42)
    assert res_stable_mmd.decision is False

    # Shifted joint distribution: Covariance & mean shift
    X_curr_shifted = np.random.multivariate_normal([1.2, -1.0, 0.8], cov, size=n)
    res_shifted_auc = compute_domain_classifier_auc(X_ref, X_curr_shifted, n_permutations=30, seed=42)
    assert res_shifted_auc.decision is True
    assert res_shifted_auc.statistic > 0.70
    assert res_shifted_auc.p_value < 0.05

    res_shifted_mmd = compute_mmd(X_ref, X_curr_shifted, n_permutations=30, seed=42)
    assert res_shifted_mmd.decision is True
    assert res_shifted_mmd.p_value < 0.05


def test_shadow_challenger_deployment(trained_forecaster_and_data):
    """Verifies shadow candidate inference executes without altering champion response."""
    champ, _, _, test_feats = trained_forecaster_and_data

    # Create distinct challenger
    challenger = XGBoostDemandForecaster(XGBoostForecasterConfig(n_estimators=20, max_depth=2))
    challenger.fit(test_feats, test_feats["sales"])

    mgr = ShadowDeploymentManager()
    mgr.set_shadow_challenger(challenger, challenger_version_id="mv_cand_shadow_99")

    # Simulate live champion response
    sample_feat = test_feats.iloc[[0]]
    champ_pred = float(champ.predict(sample_feat)[0])
    champ_resp = PredictResponse(
        prediction_id="pred_shadow_001",
        request_id="req_shadow_001",
        model_version_id="mv_champ_v1",
        prediction=champ_pred,
        latency_ms=1.2,
        timestamp="2026-01-01T00:00:00Z",
    )

    shadow_log = mgr.execute_shadow_inference(sample_feat, champ_resp)

    assert shadow_log is not None
    assert shadow_log.champion_version_id == "mv_champ_v1"
    assert shadow_log.challenger_version_id == "mv_cand_shadow_99"
    assert shadow_log.challenger_prediction > 0.0
    assert shadow_log.challenger_latency_ms >= 0.0
    assert len(mgr.shadow_logs) == 1


def test_label_free_performance_estimation_and_disclaimer():
    """Verifies label-free density-ratio estimation and explicit concept drift disclaimer."""
    np.random.seed(42)
    n = 150

    val_features = np.random.normal(0, 1, size=(n, 3))
    # Simulated validation errors
    val_errors = np.abs(np.random.normal(3.0, 1.0, size=n))

    # Shifted production features (covariate shift)
    curr_features = np.random.normal(0.8, 1.2, size=(n, 3))

    estimator = LabelFreePerformanceEstimator()
    est = estimator.estimate_performance(val_features, val_errors, curr_features, seed=42)

    assert est.estimated_mae > 0.0
    assert est.effective_sample_size > 0.0
    assert est.evidence_class == EvidenceClass.ESTIMATED_PERFORMANCE

    # Check stated assumptions and failure modes
    assumptions_str = " ".join(est.stated_assumptions)
    assert "Covariate Shift Assumption" in assumptions_str

    failure_modes_str = " ".join(est.failure_modes)
    assert "Concept Drift" in failure_modes_str


def test_evidently_html_report_export(tmp_path: Path):
    """Verifies Evidently 0.7+ standalone HTML report generation and file persistence."""
    np.random.seed(42)
    df_ref = pd.DataFrame({
        "sales": np.random.normal(20, 5, 100),
        "sell_price": np.random.uniform(2, 10, 100),
    })
    df_curr = pd.DataFrame({
        "sales": np.random.normal(28, 6, 100),  # Drifted
        "sell_price": np.random.uniform(2, 10, 100),
    })

    out_file = tmp_path / "evidently_test_report.html"
    summary = generate_evidently_html_report(
        reference_df=df_ref,
        current_df=df_curr,
        features=["sales", "sell_price"],
        output_html_path=str(out_file),
    )

    assert summary["status"] == "success"
    assert out_file.exists()
    assert out_file.stat().st_size > 500  # Non-trivial HTML file
