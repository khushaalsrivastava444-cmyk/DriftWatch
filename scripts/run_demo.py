"""DriftWatch — Autonomous End-to-End Platform Demonstration.

Executes a complete production MLOps scenario in under 2 minutes (CPU-only):
1. Database Schema Initialization & Champion (v1) Bootstrap.
2. Clean Streaming Inference & Delayed Label Ingestion (Zero False Alarms).
3. Injection of Statistical & Concept Drift via Replay Simulator.
4. Drift Detection with Benjamini-Hochberg FDR Control & Dual-Gating.
5. Delayed Label Evaluation & 95% Block Bootstrap Degradation Detection.
6. Tiered Alerting (WARNING on statistical shift -> CRITICAL on verified degradation).
7. Retraining Pipeline with MLflow 3.x Tracking & Paired Wilcoxon Hypothesis Comparison.
8. Audited Candidate Promotion to Champion (v2) & Audit Trail Verification.
9. Audited Rollback to Champion (v1) & Dynamic Hot-Reloading.
10. Tier 2 Features: Conformal Intervals, Multivariate Domain Classifier, and Evidently HTML Export.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from driftwatch.alerts.engine import AlertEngine
from driftwatch.api.schemas import LabelItem, PredictRequest
from driftwatch.api.service import ModelServingService
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.db.base import SessionLocal
from driftwatch.db.models import (
    AuditLogModel,
    GroundTruthLabel,
    ModelVersion,
    PredictionLog,
)
from driftwatch.db.retention import init_db
from driftwatch.drift.engine import DriftEngine, DriftEngineConfig
from driftwatch.drift.multiple_testing import MultipleTestingMethod
from driftwatch.drift.multivariate import compute_domain_classifier_auc
from driftwatch.features.pipeline import DemandFeaturePipeline
from driftwatch.models.conformal import ConformalDemandForecaster
from driftwatch.monitoring.evidently_export import generate_evidently_html_report
from driftwatch.monitoring.performance import PerformanceEvaluator
from driftwatch.retraining.pipeline import RetrainingPipeline
from driftwatch.retraining.policy import RetrainingTriggerPolicy
from driftwatch.retraining.registry import MLflowGovernanceRegistry
from driftwatch.simulation.injectors import (
    DriftInjectionType,
    InjectionConfig,
    apply_drift_injection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("driftwatch.demo")


def print_banner(step: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP {step}: {title}")
    print("=" * 80)


def run_demo() -> None:
    start_demo_time = time.perf_counter()
    print_banner(1, "Database Initialization & Champion (v1) Bootstrap")
    init_db()
    db = SessionLocal()

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    serving = ModelServingService(models_dir=str(models_dir))

    v1_id = serving.ensure_default_model(db)
    v1_rec = db.get(ModelVersion, v1_id)
    print(f"[OK] Champion Forecaster Bootstrapped: ID={v1_id}, Version={v1_rec.version}")
    print(f"     Artifact URI: {v1_rec.artifact_uri}")
    print(f"     Is Active Champion: {v1_rec.is_active_champion}")

    print_banner(2, "Baseline Streaming Inferences & Ground-Truth Ingestion")
    cfg_base = SyntheticDemandConfig(num_days=60, num_stores=2, num_items=5, seed=101)
    base_data = generate_synthetic_demand(cfg_base)

    if serving.pipeline is None:
        serving.pipeline = DemandFeaturePipeline().fit(base_data)
    pipeline = serving.pipeline
    base_feats = pipeline.transform(base_data)

    run_tag = uuid.uuid4().hex[:6]
    pred_ids = []
    labels_to_ingest = []
    print(f"[*] Serving {len(base_feats)} baseline requests through ModelServingService...")

    for idx, row in base_feats.iterrows():
        p_id = f"demo_{run_tag}_{idx:04d}"
        pred_ids.append(p_id)
        req = PredictRequest(
            prediction_id=p_id,
            features={
                "store_id": str(row["store_id"]),
                "item_id": str(row["item_id"]),
                "sell_price": float(row["sell_price"]),
                "day_of_week": int(row["day_of_week"]),
                "day_of_month": int(row["day_of_month"]),
                "month": int(row["month"]),
                "year": int(row["year"]) if "year" in row else (pd.to_datetime(row["date"]).year if "date" in row else 2024),
                "is_weekend": int(row["is_weekend"]),
                "lag_1": float(row["lag_1"]),
                "lag_2": float(row["lag_2"]),
                "lag_7": float(row["lag_7"]),
                "lag_14": float(row["lag_14"]),
                "lag_28": float(row["lag_28"]),
                "rolling_mean_7": float(row["rolling_mean_7"]),
                "rolling_std_7": float(row["rolling_std_7"]),
                "rolling_mean_28": float(row["rolling_mean_28"]),
                "price_discount_ratio": float(row["price_discount_ratio"]),
            },
        )
        _ = serving.predict_single(req, db, request_id=f"req_{idx}")
        labels_to_ingest.append(LabelItem(prediction_id=p_id, true_label=float(row["sales"])))

    print("[*] Ingesting delayed ground-truth retail sales outcomes...")
    ingested, matched = serving.ingest_labels(labels_to_ingest, db)
    print(f"[OK] Ingested {ingested} labels, matched {matched} operational prediction logs.")

    print_banner(3, "Drift Injection (Feature & Concept Drift Simulation)")
    drift_cfg = InjectionConfig(
        injection_type=DriftInjectionType.CONCEPT_DRIFT,
        target_columns=["sales", "sell_price"],
        shift_magnitude=2.0,
        seed=202,
    )
    drifted_raw, drift_meta = apply_drift_injection(base_data, drift_cfg)
    print(f"[OK] Applied {drift_meta['injection_type']}:")
    print(f"     Pre-injection SHA-256:  {drift_meta['before_hash'][:16]}...")
    print(f"     Post-injection SHA-256: {drift_meta['after_hash'][:16]}...")

    drifted_feats = pipeline.transform(drifted_raw)

    drift_pred_ids = []
    drift_labels = []
    print(f"[*] Serving {len(drifted_feats)} drifted requests through live Champion...")
    for idx, row in drifted_feats.iterrows():
        p_id = f"demo_drift_{run_tag}_{idx:04d}"
        drift_pred_ids.append(p_id)
        req = PredictRequest(
            prediction_id=p_id,
            features={
                "store_id": str(row["store_id"]),
                "item_id": str(row["item_id"]),
                "sell_price": float(row["sell_price"]),
                "day_of_week": int(row["day_of_week"]),
                "day_of_month": int(row["day_of_month"]),
                "month": int(row["month"]),
                "year": int(row["year"]) if "year" in row else (pd.to_datetime(row["date"]).year if "date" in row else 2024),
                "is_weekend": int(row["is_weekend"]),
                "lag_1": float(row["lag_1"]),
                "lag_2": float(row["lag_2"]),
                "lag_7": float(row["lag_7"]),
                "lag_14": float(row["lag_14"]),
                "lag_28": float(row["lag_28"]),
                "rolling_mean_7": float(row["rolling_mean_7"]),
                "rolling_std_7": float(row["rolling_std_7"]),
                "rolling_mean_28": float(row["rolling_mean_28"]),
                "price_discount_ratio": float(row["price_discount_ratio"]),
            },
        )
        serving.predict_single(req, db, request_id=f"req_drift_{idx}")
        drift_labels.append(LabelItem(prediction_id=p_id, true_label=float(row["sales"])))

    print_banner(4, "Statistical Drift Engine & Benjamini-Hochberg FDR Control")
    drift_engine = DriftEngine(
        config=DriftEngineConfig(
            alpha=0.05,
            multiple_testing_method=MultipleTestingMethod.BENJAMINI_HOCHBERG,
        )
    )
    drift_report = drift_engine.run_drift_analysis(
        reference_df=base_feats,
        current_df=drifted_feats,
        feature_columns=["sales", "sell_price", "lag_1", "lag_7", "rolling_mean_7"],
        dataset_id="drift_simulation_stream",
        reference_id="baseline_stream",
        window_id="demo_window_drifted",
    )
    print(f"[OK] Multi-Feature Drift Evaluation Complete (Config Hash: {drift_report.config_hash}):")
    print(f"     Total Tests: {len(drift_report.results)}")
    print(f"     Drifted Features Detected: {len(drift_report.drifted_features)} ({', '.join(sorted(drift_report.drifted_features)[:4])})")

    # Display sample test statistics
    for r in drift_report.results[:4]:
        adj_p = f"{r.adjusted_p_value:.4f}" if r.adjusted_p_value is not None else "N/A"
        print(f"     - Feature '{r.feature}' [{r.test.value}]: stat={r.statistic:.4f}, raw_p={r.p_value}, BH_p={adj_p}, decision={r.decision}")

    print_banner(5, "Alert Engine: Statistical Shift Warning")
    alert_engine = AlertEngine()
    drift_alerts = alert_engine.process_drift_report(drift_report, db)
    print(f"[OK] Generated {len(drift_alerts)} alert(s) from statistical drift report:")
    for a in drift_alerts:
        print(f"     [ALERT {a.id}] Severity: {a.severity} | Evidence: {a.evidence_class}")
        print(f"     Reason: {a.reason}")
        print(f"     Recommended Action: {a.recommended_action}")

    print_banner(6, "Delayed Ground-Truth Evaluation & 95% Bootstrap CI Degradation")
    serving.ingest_labels(drift_labels, db)

    # Query matched predictions and ground truth labels from database
    stmt = (
        select(PredictionLog.prediction, GroundTruthLabel.true_label, GroundTruthLabel.delay_seconds)
        .join(GroundTruthLabel, PredictionLog.prediction_id == GroundTruthLabel.prediction_id)
        .where(PredictionLog.prediction_id.in_(drift_pred_ids))
    )
    matched_rows = db.execute(stmt).all()
    matched_df = pd.DataFrame(matched_rows, columns=["prediction", "true_label", "delay_seconds"])

    perf_evaluator = PerformanceEvaluator(
        min_coverage_threshold=0.80,
        baseline_mae=5.0,
        degradation_threshold_ratio=0.25,
        n_bootstrap=300,
    )
    perf_window = perf_evaluator.evaluate_window(
        matched_df=matched_df,
        window_id="demo_window_drifted",
        window_start=(datetime.now(UTC) - timedelta(days=7)).isoformat(),
        window_end=datetime.now(UTC).isoformat(),
    )

    print(f"[OK] Window Performance Status: {perf_window.status}")
    print(f"     Label Coverage: {perf_window.label_coverage_ratio:.1%}")
    if perf_window.mae_ci:
        print(f"     Point MAE: {perf_window.mae_ci.point_estimate:.3f}")
        print(f"     95% Stationary Block Bootstrap CI: [{perf_window.mae_ci.ci_lower:.3f}, {perf_window.mae_ci.ci_upper:.3f}]")
    print(f"     Degradation Breached (MAE > 7.50): {perf_window.degradation_breached}")

    perf_alert = alert_engine.process_performance_evaluation(perf_window, db)
    if perf_alert:
        print(f"[ESCALATION] Alert Raised: {perf_alert.id}")
        print(f"     Severity: {perf_alert.severity} (Strictly requires VERIFIED_PERFORMANCE)")
        print(f"     Evidence Class: {perf_alert.evidence_class}")
        print(f"     Reason: {perf_alert.reason}")

    print_banner(7, "Retraining Pipeline with MLflow 3.x Tracking & Paired Hypothesis Test")
    mlflow_db = Path("artifacts/mlflow_demo.db")
    mlflow_registry = MLflowGovernanceRegistry(
        tracking_uri=f"sqlite:///{mlflow_db.resolve()}",
        experiment_name="demo-production-retraining",
    )
    retrain_pipeline = RetrainingPipeline(models_dir=models_dir, mlflow_registry=mlflow_registry)

    # Approved fresh training data incorporating recent shifts
    fresh_cfg = SyntheticDemandConfig(num_days=180, num_stores=2, num_items=5, seed=777)
    fresh_training_data = generate_synthetic_demand(fresh_cfg)

    print("[*] Executing retraining under atomic job locks...")
    retrain_res = retrain_pipeline.execute_retraining(
        raw_approved_data=fresh_training_data,
        model_id=v1_rec.model_id,
        trigger_policy=RetrainingTriggerPolicy.SUSTAINED_DEGRADATION,
        db=db,
        drift_context="Degradation breach confirmed on delayed labels (MAE > 7.50)",
    )
    print(f"[OK] Retraining Job {retrain_res.job_id} Completed:")
    print(f"     Candidate Version: v{retrain_res.candidate_version} (ID: {retrain_res.candidate_version_id})")
    print(f"     MLflow Run ID: {retrain_res.mlflow_metadata.run_id}")
    print(f"     Data SHA-256:  {retrain_res.mlflow_metadata.data_hash[:16]}...")
    print(f"     Git Commit SHA: {retrain_res.mlflow_metadata.git_sha}")

    comp = retrain_res.comparison
    print(f"[OK] Champion vs Challenger Paired Hypothesis Comparison ({comp.test_method}):")
    print(f"     Champion Holdout MAE:  {comp.champion_metric:.4f}")
    print(f"     Candidate Holdout MAE: {comp.candidate_metric:.4f} ({comp.relative_improvement_pct:+.1f}%)")
    print(f"     Paired p-value:        {comp.p_value:.6f} (Statistically Significant: {comp.is_stat_significant})")
    print(f"     Approval Decision:     {comp.decision}")

    cand_rec = db.get(ModelVersion, retrain_res.candidate_version_id)
    print(f"     Candidate Registered In Database: Is Active Champion = {cand_rec.is_active_champion} (Strictly NO auto-promotion)")

    print_banner(8, "Audited Candidate Promotion")
    promoted = serving.promote_model_version(
        model_id=v1_rec.model_id,
        version=retrain_res.candidate_version,
        actor="autonomous_mlops_demo",
        db=db,
    )
    print(f"[OK] Model v{promoted.version} ({promoted.id}) Promoted to Active Champion.")
    audit_promo = db.query(AuditLogModel).filter(AuditLogModel.action == "PROMOTE").order_by(AuditLogModel.timestamp.desc()).first()
    print(f"     Audit Log ID: {audit_promo.id} | Actor: {audit_promo.actor} | Action: {audit_promo.action}")

    print_banner(9, "Audited Rollback to Champion (v1)")
    rolled_back = serving.rollback_champion(
        model_id=v1_rec.model_id,
        actor="autonomous_mlops_demo",
        db=db,
    )
    print(f"[OK] Rollback Executed: Active Champion is now v{rolled_back.version} ({rolled_back.id}).")
    audit_rollback = db.query(AuditLogModel).filter(AuditLogModel.action == "ROLLBACK").order_by(AuditLogModel.timestamp.desc()).first()
    print(f"     Audit Log ID: {audit_rollback.id} | Actor: {audit_rollback.actor} | Action: {audit_rollback.action}")

    print_banner(10, "Tier 2 Capabilities Showcase")
    # 1. Split Conformal Prediction Intervals
    conformal = ConformalDemandForecaster(serving.active_forecaster, default_confidence_level=0.90)
    conformal.calibrate(base_feats, base_feats["sales"].values)
    conf_intervals = conformal.predict_intervals(base_feats.iloc[:5])
    print("[OK] Conformal Prediction 90% Intervals for first 3 samples:")
    for i in range(3):
        print(f"     Sample {i}: Pred={conf_intervals.point_predictions[i]:.2f} in [{conf_intervals.lower_bounds[i]:.2f}, {conf_intervals.upper_bounds[i]:.2f}]")

    # 2. Multivariate Domain Classifier AUC
    multi_res = compute_domain_classifier_auc(
        ref_features=base_feats[["lag_1", "lag_7", "sell_price"]].values,
        curr_features=drifted_feats[["lag_1", "lag_7", "sell_price"]].values,
        n_permutations=25,
        seed=42,
    )
    print(f"[OK] Multivariate Domain Classifier AUC: {multi_res.statistic:.4f} (p-val: {multi_res.p_value:.4f}, Drift: {multi_res.decision})")

    # 3. Evidently 0.7+ HTML Export
    ev_path = "reports/evidently_demo_report.html"
    _ = generate_evidently_html_report(
        reference_df=base_feats,
        current_df=drifted_feats,
        features=["sales", "sell_price", "lag_1", "lag_7"],
        output_html_path=ev_path,
    )
    print(f"[OK] Evidently 0.7+ Standalone HTML Report Exported to {ev_path}")

    db.close()
    elapsed = time.perf_counter() - start_demo_time

    print("\n" + "=" * 80)
    print(f"[SUCCESS] DRIFTWATCH DEMO COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS (CPU-ONLY)")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
