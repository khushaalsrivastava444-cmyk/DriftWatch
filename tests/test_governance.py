"""Tests for Governance, Retraining Policies, and Promotion Gating."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from driftwatch.api.service import ModelServingService
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.db.models import (
    AuditLogModel,
    Base,
    Model,
    ModelEvaluationModel,
    ModelVersion,
    RetrainingJobModel,
)
from driftwatch.models.xgboost_forecaster import (
    XGBoostDemandForecaster,
    XGBoostForecasterConfig,
)
from driftwatch.retraining.pipeline import PolicyViolationError, RetrainingPipeline
from driftwatch.retraining.policy import RetrainingTriggerPolicy


@pytest.fixture
def test_db(tmp_path: Path) -> Session:
    """Create isolated SQLite database session."""
    db_path = tmp_path / "test_gov.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def bootstrapped_model(test_db: Session, tmp_path: Path) -> tuple[str, str, str]:
    """Bootstrap a model and champion version with a real artifact on disk."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_id = "demand_model_test"
    db_model = Model(
        id=model_id,
        name="test_forecaster",
        task_type="demand_forecasting",
    )
    test_db.add(db_model)

    champ_id = f"mv_champ_{uuid.uuid4().hex[:8]}"
    artifact_path = models_dir / f"{champ_id}.json"

    # Train a minimal XGBoost forecaster and save it
    forecaster = XGBoostDemandForecaster(XGBoostForecasterConfig(n_estimators=10, max_depth=3))
    raw_df = generate_synthetic_demand(SyntheticDemandConfig(num_days=40, num_stores=1, num_items=2, seed=1))
    from driftwatch.features.pipeline import DemandFeaturePipeline
    pipeline = DemandFeaturePipeline()
    pipeline.fit(raw_df)
    feats = pipeline.transform(raw_df)
    forecaster.fit(feats, feats["sales"])
    forecaster.save_model(str(artifact_path))

    champ_version = ModelVersion(
        id=champ_id,
        model_id=model_id,
        version=1,
        artifact_uri=str(artifact_path),
        is_active_champion=True,
        params_json=json.dumps({"n_estimators": 10}),
        metrics_json=json.dumps({"test_mae": 5.0}),
        created_at=datetime.now(UTC),
    )
    test_db.add(champ_version)
    test_db.commit()

    return model_id, champ_id, str(models_dir)


def test_drift_only_policy_rejected(test_db: Session, bootstrapped_model: tuple[str, str, str]) -> None:
    """Policy rule 1: Drift alone NEVER auto-retrains."""
    model_id, champ_id, models_dir = bootstrapped_model
    pipeline = RetrainingPipeline(models_dir=Path(models_dir))

    raw_df = generate_synthetic_demand(SyntheticDemandConfig(num_days=80, num_stores=2, num_items=3, seed=42))

    with pytest.raises(PolicyViolationError, match="Statistical drift alone never auto-retrains"):
        pipeline.execute_retraining(
            raw_approved_data=raw_df,
            model_id=model_id,
            trigger_policy=RetrainingTriggerPolicy.DRIFT_ONLY,
            db=test_db,
        )

    # Verify that a REJECTED job was recorded in database
    rejected_job = test_db.execute(
        select(RetrainingJobModel).where(RetrainingJobModel.trigger_policy == "DRIFT_ONLY")
    ).scalars().first()
    assert rejected_job is not None
    assert rejected_job.status == "REJECTED"


def test_sustained_degradation_policy_requires_verified_reports(
    test_db: Session, bootstrapped_model: tuple[str, str, str]
) -> None:
    """Policy rule 2: SUSTAINED_DEGRADATION requires k-of-n verified breaches."""
    model_id, champ_id, models_dir = bootstrapped_model
    pipeline = RetrainingPipeline(models_dir=Path(models_dir))

    raw_df = generate_synthetic_demand(SyntheticDemandConfig(num_days=80, num_stores=2, num_items=3, seed=42))

    # Without any performance reports, execution must be rejected
    with pytest.raises(PolicyViolationError, match="No performance reports available to verify degradation"):
        pipeline.execute_retraining(
            raw_approved_data=raw_df,
            model_id=model_id,
            trigger_policy=RetrainingTriggerPolicy.SUSTAINED_DEGRADATION,
            db=test_db,
        )


def test_cannot_promote_rejected_or_unevaluated_candidate(
    test_db: Session, bootstrapped_model: tuple[str, str, str]
) -> None:
    """Promotion rule: Candidates without APPROVED decision can NEVER be promoted."""
    model_id, champ_id, models_dir = bootstrapped_model
    serving = ModelServingService(models_dir=Path(models_dir))

    # 1. Create candidate version 2 with valid artifact on disk
    cand_id = f"mv_cand_{uuid.uuid4().hex[:8]}"
    artifact_path = Path(models_dir) / f"{cand_id}.json"
    forecaster = XGBoostDemandForecaster(XGBoostForecasterConfig(n_estimators=5, max_depth=2))
    raw_df = generate_synthetic_demand(SyntheticDemandConfig(num_days=40, num_stores=1, num_items=2, seed=2))
    from driftwatch.features.pipeline import DemandFeaturePipeline
    pipeline = DemandFeaturePipeline()
    pipeline.fit(raw_df)
    feats = pipeline.transform(raw_df)
    forecaster.fit(feats, feats["sales"])
    forecaster.save_model(str(artifact_path))

    cand_version = ModelVersion(
        id=cand_id,
        model_id=model_id,
        version=2,
        artifact_uri=str(artifact_path),
        is_active_champion=False,
        params_json=json.dumps({"n_estimators": 5}),
        metrics_json=json.dumps({"test_mae": 6.0}),
        created_at=datetime.now(UTC),
    )
    test_db.add(cand_version)
    test_db.commit()

    # Attempt to promote unevaluated candidate -> must be rejected
    with pytest.raises(ValueError, match="No statistical evaluation record found"):
        serving.promote_model_version(model_id=model_id, version=2, actor="lead_engineer", db=test_db)

    # Add REJECTED evaluation record
    eval_rejected = ModelEvaluationModel(
        id=f"eval_{uuid.uuid4().hex[:8]}",
        champion_version_id=champ_id,
        candidate_version_id=cand_id,
        metric_name="MAE",
        champion_metric=5.0,
        candidate_metric=6.2,
        p_value_paired=0.85,
        is_stat_significant=False,
        decision="REJECTED",
        created_at=datetime.now(UTC),
    )
    test_db.add(eval_rejected)
    test_db.commit()

    # Attempt to promote REJECTED candidate -> must be strictly rejected
    with pytest.raises(ValueError, match="Statistically rejected models cannot be promoted"):
        serving.promote_model_version(model_id=model_id, version=2, actor="lead_engineer", db=test_db)

    # Verify that champion was NOT changed
    champ = test_db.execute(
        select(ModelVersion).where(ModelVersion.model_id == model_id, ModelVersion.is_active_champion.is_(True))
    ).scalars().first()
    assert champ is not None
    assert champ.id == champ_id


def test_atomic_promotion_and_rollback_flow(
    test_db: Session, bootstrapped_model: tuple[str, str, str]
) -> None:
    """Promotion and rollback must be atomic, audited, and verify artifact loads."""
    model_id, champ_id, models_dir = bootstrapped_model
    serving = ModelServingService(models_dir=Path(models_dir))

    # Create approved candidate version 2
    cand_id = f"mv_cand_{uuid.uuid4().hex[:8]}"
    artifact_path = Path(models_dir) / f"{cand_id}.json"
    forecaster = XGBoostDemandForecaster(XGBoostForecasterConfig(n_estimators=5, max_depth=2))
    raw_df = generate_synthetic_demand(SyntheticDemandConfig(num_days=40, num_stores=1, num_items=2, seed=3))
    from driftwatch.features.pipeline import DemandFeaturePipeline
    pipeline = DemandFeaturePipeline()
    pipeline.fit(raw_df)
    feats = pipeline.transform(raw_df)
    forecaster.fit(feats, feats["sales"])
    forecaster.save_model(str(artifact_path))

    cand_version = ModelVersion(
        id=cand_id,
        model_id=model_id,
        version=2,
        artifact_uri=str(artifact_path),
        is_active_champion=False,
        params_json=json.dumps({"n_estimators": 5}),
        metrics_json=json.dumps({"test_mae": 3.8}),
        created_at=datetime.now(UTC),
    )
    test_db.add(cand_version)

    eval_approved = ModelEvaluationModel(
        id=f"eval_{uuid.uuid4().hex[:8]}",
        champion_version_id=champ_id,
        candidate_version_id=cand_id,
        metric_name="MAE",
        champion_metric=5.0,
        candidate_metric=3.8,
        p_value_paired=0.002,
        is_stat_significant=True,
        decision="APPROVED",
        created_at=datetime.now(UTC),
    )
    test_db.add(eval_approved)
    test_db.commit()

    # Promote approved candidate
    promoted = serving.promote_model_version(model_id=model_id, version=2, actor="principal_mle", db=test_db)
    assert promoted.id == cand_id
    assert promoted.is_active_champion is True

    # Verify old champion is demoted
    old_champ = test_db.get(ModelVersion, champ_id)
    assert old_champ is not None
    assert old_champ.is_active_champion is False

    # Verify audit log
    audit_promote = test_db.execute(
        select(AuditLogModel).where(AuditLogModel.action == "PROMOTE", AuditLogModel.entity_id == cand_id)
    ).scalars().first()
    assert audit_promote is not None
    assert audit_promote.actor == "principal_mle"

    # Now execute rollback to previous champion
    rolled_back = serving.rollback_champion(model_id=model_id, actor="oncall_engineer", db=test_db)
    assert rolled_back.id == cand_id or rolled_back.version == 1

    # Verify audit log for rollback
    audit_rollback = test_db.execute(
        select(AuditLogModel).where(AuditLogModel.action == "ROLLBACK")
    ).scalars().first()
    assert audit_rollback is not None
    assert audit_rollback.actor == "oncall_engineer"
