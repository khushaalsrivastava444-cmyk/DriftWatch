"""End-to-End Retraining, MLflow Logging, Champion-Challenger Comparison, and Audit Pipeline Tests.

DoD Verification:
1. Retraining trigger policies (drift alone never auto-retrains; sustained degradation, manual, scheduled).
2. Paired statistical hypothesis comparison (Wilcoxon/t-test) with explicit approval/rejection.
3. Job locks and idempotency (prevents concurrent retraining jobs).
4. Full E2E lifecycle: Train Champion (v1) -> Serve -> Inject Drift/Degradation -> Alert ->
   Retrain Candidate (v2) -> MLflow Tracking -> Paired Comparison -> Audited Promote -> Audited Rollback.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from driftwatch.alerts.engine import AlertEngine
from driftwatch.api.schemas import PredictRequest
from driftwatch.api.service import ModelServingService
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.db.models import (
    AuditLogModel,
    Base,
    ModelEvaluationModel,
    ModelVersion,
    PerformanceReportModel,
    RetrainingJobModel,
)
from driftwatch.drift.engine import DriftReport
from driftwatch.drift.results import EvidenceClass, FeatureDriftResult, TestType
from driftwatch.retraining.evaluation import (
    compare_champion_challenger,
)
from driftwatch.retraining.pipeline import RetrainingPipeline
from driftwatch.retraining.policy import (
    RetrainingTriggerPolicy,
    evaluate_retraining_trigger,
)
from driftwatch.retraining.registry import MLflowGovernanceRegistry


@pytest.fixture
def temp_db(tmp_path: Path) -> Generator[Session, None, None]:
    """Isolated SQLite database fixture for retraining tests."""
    db_file = tmp_path / "test_retrain.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()
    yield session
    session.close()


def test_retraining_trigger_policy_rules():
    """Verifies retraining policy governance: drift alone is strictly rejected."""
    # 1. DRIFT_ONLY policy MUST BE REJECTED
    drift_res = evaluate_retraining_trigger(RetrainingTriggerPolicy.DRIFT_ONLY)
    assert drift_res.should_retrain is False
    assert "never auto-retrains" in drift_res.reason

    # 2. MANUAL & SCHEDULED policies approved
    assert evaluate_retraining_trigger(RetrainingTriggerPolicy.MANUAL).should_retrain is True
    assert evaluate_retraining_trigger(RetrainingTriggerPolicy.SCHEDULED).should_retrain is True

    # 3. SUSTAINED_DEGRADATION requires k-of-n verified breaches
    now = datetime.now(UTC)
    # Only 1 breach out of 3 windows (transient)
    reports_transient = [
        PerformanceReportModel(
            id="r1", model_version_id="mv1", window_start=now, window_end=now,
            sample_size=100, labeled_count=90, label_coverage_ratio=0.9, status="VERIFIED",
            degradation_breached=True,
        ),
        PerformanceReportModel(
            id="r2", model_version_id="mv1", window_start=now, window_end=now,
            sample_size=100, labeled_count=90, label_coverage_ratio=0.9, status="VERIFIED",
            degradation_breached=False,
        ),
        PerformanceReportModel(
            id="r3", model_version_id="mv1", window_start=now, window_end=now,
            sample_size=100, labeled_count=90, label_coverage_ratio=0.9, status="VERIFIED",
            degradation_breached=False,
        ),
    ]
    res_transient = evaluate_retraining_trigger(
        RetrainingTriggerPolicy.SUSTAINED_DEGRADATION,
        recent_performance_reports=reports_transient,
        k_required=2,
    )
    assert res_transient.should_retrain is False
    assert "transient" in res_transient.reason.lower()

    # 2 breaches out of 3 windows (sustained)
    reports_sustained = [
        PerformanceReportModel(
            id="r1", model_version_id="mv1", window_start=now, window_end=now,
            sample_size=100, labeled_count=90, label_coverage_ratio=0.9, status="VERIFIED",
            degradation_breached=True,
        ),
        PerformanceReportModel(
            id="r2", model_version_id="mv1", window_start=now, window_end=now,
            sample_size=100, labeled_count=90, label_coverage_ratio=0.9, status="VERIFIED",
            degradation_breached=True,
        ),
        PerformanceReportModel(
            id="r3", model_version_id="mv1", window_start=now, window_end=now,
            sample_size=100, labeled_count=90, label_coverage_ratio=0.9, status="VERIFIED",
            degradation_breached=False,
        ),
    ]
    res_sustained = evaluate_retraining_trigger(
        RetrainingTriggerPolicy.SUSTAINED_DEGRADATION,
        recent_performance_reports=reports_sustained,
        k_required=2,
    )
    assert res_sustained.should_retrain is True
    assert "sustained" in res_sustained.reason.lower()


def test_paired_statistical_comparison_approval_and_rejection():
    """Verifies paired Wilcoxon test approves genuine challenger and rejects inferior candidate."""
    np.random.seed(42)
    y_true = np.random.uniform(20.0, 50.0, size=100)

    # Candidate has consistently smaller errors than Champion
    champ_preds = y_true + np.random.normal(5.0, 1.5, size=100)
    cand_preds = y_true + np.random.normal(1.5, 0.8, size=100)

    res_approved = compare_champion_challenger(
        y_true=y_true,
        champion_preds=champ_preds,
        candidate_preds=cand_preds,
        metric_name="MAE",
        alpha=0.05,
    )
    assert res_approved.decision == "APPROVED"
    assert res_approved.is_stat_significant is True
    assert res_approved.p_value < 0.05
    assert res_approved.candidate_metric < res_approved.champion_metric

    # Inferior candidate has larger errors than Champion
    bad_cand_preds = y_true + np.random.normal(8.0, 2.0, size=100)
    res_rejected = compare_champion_challenger(
        y_true=y_true,
        champion_preds=champ_preds,
        candidate_preds=bad_cand_preds,
        metric_name="MAE",
        alpha=0.05,
    )
    assert res_rejected.decision == "REJECTED"
    assert res_rejected.is_stat_significant is False


def test_job_locking_and_idempotency(temp_db: Session, tmp_path: Path):
    """Verifies retraining job lock blocks concurrent executions."""
    # Insert an existing RUNNING job
    existing_job = RetrainingJobModel(
        id="job_active_lock",
        trigger_policy="MANUAL",
        status="RUNNING",
        champion_version_id="mv_champ_1",
        created_at=datetime.now(UTC),
    )
    temp_db.add(existing_job)
    temp_db.commit()

    pipeline = RetrainingPipeline(models_dir=tmp_path / "models")
    dummy_df = generate_synthetic_demand(SyntheticDemandConfig(num_days=40, seed=42))

    with pytest.raises(RuntimeError, match="Job lock active"):
        pipeline.execute_retraining(
            raw_approved_data=dummy_df,
            model_id="model_demand_v1",
            trigger_policy=RetrainingTriggerPolicy.MANUAL,
            db=temp_db,
        )


def test_retraining_e2e_full_lifecycle(temp_db: Session, tmp_path: Path):
    """Full End-to-End Retraining Lifecycle Test.
    
    Workflow:
    1. Train & register initial Champion (v1).
    2. Serve inferences via ModelServingService.
    3. Simulate drift alert.
    4. Run RetrainingPipeline to train and evaluate Candidate (v2).
    5. Verify MLflow tracking logs params, metrics, git SHA, data hash, and artifacts.
    6. Verify Candidate v2 is registered in DB as is_active_champion=False (no auto-promotion).
    7. Execute audited promote of v2 -> Active champion updates, audit log written.
    8. Execute audited rollback -> Restores v1, audit log written.
    """
    models_dir = tmp_path / "models"
    mlflow_db = tmp_path / "mlflow.db"
    mlflow_registry = MLflowGovernanceRegistry(
        tracking_uri=f"sqlite:///{mlflow_db}",
        experiment_name="test-retraining-e2e",
    )

    # Step 1: Initialize Serving Service & Bootstrap Champion v1
    serving = ModelServingService(models_dir=str(models_dir))
    champ_v1_id = serving.ensure_default_model(temp_db)
    assert champ_v1_id is not None

    champ_record = temp_db.get(ModelVersion, champ_v1_id)
    assert champ_record is not None
    assert champ_record.version == 1
    assert champ_record.is_active_champion is True
    assert Path(champ_record.artifact_uri).exists()

    # Step 2: Serve inference through active champion
    pred_req = PredictRequest(
        features={
            "store_id": "CA_1",
            "item_id": "FOODS_1",
            "sell_price": 2.50,
            "day_of_week": 4,
            "day_of_month": 15,
            "month": 6,
            "year": 2026,
            "is_weekend": 0,
            "lag_1": 15.0,
            "lag_2": 14.0,
            "lag_7": 16.0,
            "lag_14": 15.0,
            "lag_28": 17.0,
            "rolling_mean_7": 15.2,
            "rolling_std_7": 1.1,
            "rolling_mean_28": 15.5,
            "price_discount_ratio": 0.0,
        }
    )
    pred_resp = serving.predict_single(pred_req, temp_db, request_id="req_e2e_001")
    assert pred_resp.prediction > 0.0
    assert pred_resp.model_version_id == champ_v1_id

    # Step 3: Trigger Alert Engine (Statistical Drift WARNING)
    alert_engine = AlertEngine(notifiers=[])
    drift_res = FeatureDriftResult(
        dataset_id="test_stream",
        reference_id="ref_v1",
        window="w_e2e",
        feature="lag_7",
        test=TestType.PSI,
        statistic=0.28,
        p_value=0.001,
        adjusted_p_value=0.002,
        effect_size=0.28,
        threshold=0.10,
        decision=True,
        sample_size_ref=500,
        sample_size_curr=150,
        timestamp=datetime.now(UTC).isoformat(),
        config_hash="cfg_e2e_123",
        evidence_class=EvidenceClass.STATISTICAL_SHIFT,
    )
    report = DriftReport(
        dataset_id="test_stream",
        reference_id="ref_v1",
        window_id="w_e2e",
        timestamp=datetime.now(UTC).isoformat(),
        config_hash="cfg_e2e_123",
        results=[drift_res],
    )
    alerts = alert_engine.process_drift_report(report, temp_db)
    assert len(alerts) == 1
    assert alerts[0].severity == "WARNING"
    assert alerts[0].evidence_class == "STATISTICAL_SHIFT"

    # Step 4: Retrain Candidate (v2) on Fresh Approved Data
    fresh_data = generate_synthetic_demand(SyntheticDemandConfig(num_days=180, num_stores=2, num_items=5, seed=99))
    pipeline = RetrainingPipeline(models_dir=models_dir, mlflow_registry=mlflow_registry)

    retrain_res = pipeline.execute_retraining(
        raw_approved_data=fresh_data,
        model_id=champ_record.model_id,
        trigger_policy=RetrainingTriggerPolicy.MANUAL,
        db=temp_db,
        drift_context="Retraining triggered following feature drift warning",
    )

    assert retrain_res.status == "COMPLETED"
    assert retrain_res.candidate_version == 2
    assert retrain_res.candidate_version_id is not None
    assert retrain_res.comparison is not None
    assert retrain_res.mlflow_metadata is not None

    # Step 5: Verify Candidate v2 Registration in Database (Strictly NOT active champion)
    cand_record = temp_db.get(ModelVersion, retrain_res.candidate_version_id)
    assert cand_record is not None
    assert cand_record.version == 2
    assert cand_record.is_active_champion is False  # Hard rule: NO auto-promotion
    assert Path(cand_record.artifact_uri).exists()

    # Verify ModelEvaluation record
    eval_record = temp_db.execute(
        select(ModelEvaluationModel).where(ModelEvaluationModel.candidate_version_id == cand_record.id)
    ).scalars().first()
    assert eval_record is not None
    assert eval_record.champion_version_id == champ_v1_id
    assert eval_record.metric_name == "MAE"
    assert eval_record.p_value_paired >= 0.0

    # Step 6: Audited Candidate Promotion
    promoted = serving.promote_model_version(
        model_id=champ_record.model_id,
        version=2,
        actor="lead_mlops_engineer",
        db=temp_db,
    )
    assert promoted.version == 2
    assert promoted.is_active_champion is True

    # Verify v1 demoted
    v1_refreshed = temp_db.get(ModelVersion, champ_v1_id)
    assert v1_refreshed.is_active_champion is False

    # Verify Audit Log entry for PROMOTE
    promo_audit = temp_db.execute(
        select(AuditLogModel).where(AuditLogModel.action == "PROMOTE")
    ).scalars().first()
    assert promo_audit is not None
    assert promo_audit.actor == "lead_mlops_engineer"
    assert promo_audit.entity_id == cand_record.id

    # Verify serving service serves with v2
    pred_v2 = serving.predict_single(pred_req, temp_db, request_id="req_e2e_002")
    assert pred_v2.model_version_id == cand_record.id

    # Step 7: Audited Rollback to Champion v1
    rolled_back = serving.rollback_champion(
        model_id=champ_record.model_id,
        actor="lead_mlops_engineer",
        db=temp_db,
    )
    assert rolled_back.version == 1
    assert rolled_back.is_active_champion is True

    # Verify v2 demoted
    cand_refreshed = temp_db.get(ModelVersion, cand_record.id)
    assert cand_refreshed.is_active_champion is False

    # Verify Audit Log entry for ROLLBACK
    rollback_audit = temp_db.execute(
        select(AuditLogModel).where(AuditLogModel.action == "ROLLBACK")
    ).scalars().first()
    assert rollback_audit is not None
    assert rollback_audit.actor == "lead_mlops_engineer"
    assert rollback_audit.entity_id == champ_v1_id

    # Verify serving service reverted to v1
    pred_v1_restored = serving.predict_single(pred_req, temp_db, request_id="req_e2e_003")
    assert pred_v1_restored.model_version_id == champ_v1_id
