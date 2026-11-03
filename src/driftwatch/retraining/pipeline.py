"""Retraining Pipeline with MLflow Tracking, Temporal Evaluation, and Job Locks."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from driftwatch.data.split import rolling_origin_splits, temporal_train_val_test_split
from driftwatch.db.models import (
    ModelEvaluationModel,
    ModelVersion,
    PerformanceReportModel,
    RetrainingJobModel,
)
from driftwatch.features.pipeline import DemandFeaturePipeline
from driftwatch.models.xgboost_forecaster import (
    XGBoostDemandForecaster,
    XGBoostForecasterConfig,
)
from driftwatch.monitoring.metrics import evaluate_forecast
from driftwatch.retraining.evaluation import (
    ModelComparisonResult,
    compare_champion_challenger,
)
from driftwatch.retraining.policy import (
    RetrainingTriggerPolicy,
    evaluate_retraining_trigger,
)
from driftwatch.retraining.registry import (
    MLflowGovernanceRegistry,
    MLflowRunMetadata,
)
from driftwatch.validation.engine import DataValidationEngine, ValidationConfig

logger = logging.getLogger(__name__)


class PolicyViolationError(ValueError):
    """Raised when a retraining trigger policy rejects pipeline execution."""
    pass


@dataclass(frozen=True)
class RetrainingExecutionResult:
    """Complete summary of retraining pipeline execution."""
    job_id: str
    status: str
    champion_version_id: str
    candidate_version_id: str | None
    candidate_version: int | None
    comparison: ModelComparisonResult | None
    mlflow_metadata: MLflowRunMetadata | None


class RetrainingPipeline:
    """Autonomous retraining pipeline enforcing validation, temporal holdout, and statistical comparison."""

    def __init__(
        self,
        models_dir: Path | None = None,
        mlflow_registry: MLflowGovernanceRegistry | None = None,
    ) -> None:
        self.models_dir = models_dir or Path("./artifacts/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_registry = mlflow_registry or MLflowGovernanceRegistry()
        self.validation_engine = DataValidationEngine(ValidationConfig())

    def execute_retraining(
        self,
        raw_approved_data: pd.DataFrame,
        model_id: str,
        trigger_policy: RetrainingTriggerPolicy,
        db: Session,
        drift_context: str = "Automated retraining trigger",
        candidate_config: XGBoostForecasterConfig | None = None,
        recent_performance_reports: list[PerformanceReportModel] | None = None,
        k_required: int = 2,
        n_recent: int = 3,
    ) -> RetrainingExecutionResult:
        """Run full end-to-end retraining pipeline under atomic job locks."""
        # 1. Job Lock check (Idempotency) - reject concurrent executions immediately
        active_job = db.execute(
            select(RetrainingJobModel).where(RetrainingJobModel.status == "RUNNING")
        ).scalars().first()
        if active_job:
            raise RuntimeError(
                f"Job lock active: Retraining job {active_job.id} is already running."
            )

        # 2. Identify active Champion
        champion_version = db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.is_active_champion.is_(True),
            ).order_by(desc(ModelVersion.created_at))
        ).scalars().first()

        if not champion_version:
            raise ValueError(f"No active champion found for model {model_id} to evaluate against.")

        # 3. Retraining Policy Enforcement (Hard rule: Drift alone NEVER auto-retrains)
        if trigger_policy == RetrainingTriggerPolicy.SUSTAINED_DEGRADATION and recent_performance_reports is None:
            recent_performance_reports = db.execute(
                select(PerformanceReportModel)
                .where(PerformanceReportModel.model_version_id == champion_version.id)
                .order_by(desc(PerformanceReportModel.created_at))
                .limit(10)
            ).scalars().all()

        policy_result = evaluate_retraining_trigger(
            policy=trigger_policy,
            recent_performance_reports=recent_performance_reports,
            k_required=k_required,
            n_recent=n_recent,
        )

        if not policy_result.should_retrain:
            job_id = f"retrain_{uuid.uuid4().hex[:10]}"
            rejected_job = RetrainingJobModel(
                id=job_id,
                trigger_policy=trigger_policy.value,
                status="REJECTED",
                champion_version_id=champion_version.id,
                metrics_comparison_json=json.dumps({
                    "decision": "REJECTED",
                    "reason": policy_result.reason,
                    "details": policy_result.details,
                }),
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
            db.add(rejected_job)
            db.commit()
            logger.warning(
                "Retraining execution rejected by policy %s: %s",
                trigger_policy.value,
                policy_result.reason,
            )
            raise PolicyViolationError(
                f"Retraining execution rejected by policy '{trigger_policy.value}': {policy_result.reason}"
            )

        # 4. Load Champion forecaster from disk
        champion_forecaster = XGBoostDemandForecaster()
        artifact_path = Path(champion_version.artifact_uri)
        if not artifact_path.exists():
            fallback = Path("artifacts/models") / artifact_path.name
            if fallback.exists():
                artifact_path = fallback
            else:
                raise FileNotFoundError(f"Champion artifact missing at {champion_version.artifact_uri}")
        champion_forecaster.load_model(str(artifact_path))

        # 4. Register RUNNING job in DB
        job_id = f"retrain_{uuid.uuid4().hex[:10]}"
        job = RetrainingJobModel(
            id=job_id,
            trigger_policy=trigger_policy.value,
            status="RUNNING",
            champion_version_id=champion_version.id,
            created_at=datetime.now(UTC),
        )
        db.add(job)
        db.commit()

        try:
            # 5. Ingestion Validation
            report, clean_df, quarantined_df = self.validation_engine.validate(raw_approved_data)
            if len(clean_df) < 50:
                raise ValueError(f"Insufficient validated rows for retraining: {len(clean_df)} < 50")

            data_hash = hashlib.sha256(clean_df.to_json(orient="records", date_format="iso").encode()).hexdigest()

            # 6. Temporal Train/Val/Test Split + Rolling-Origin Walk-Forward CV
            parts = temporal_train_val_test_split(clean_df, train_ratio=0.70, val_ratio=0.15)

            pipeline = DemandFeaturePipeline()
            pipeline.fit(parts.train)
            train_feats = pipeline.transform(parts.train)
            val_feats = pipeline.transform(parts.val)
            test_feats = pipeline.transform(parts.test)

            # 7. Train Candidate Model
            cfg = candidate_config or XGBoostForecasterConfig(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
            )
            candidate_forecaster = XGBoostDemandForecaster(config=cfg)
            candidate_forecaster.fit(
                train_feats,
                train_feats["sales"],
                eval_set=[(val_feats, val_feats["sales"])],
            )

            # Rolling-origin walk-forward CV check
            cv_champ_errors: list[float] = []
            cv_cand_errors: list[float] = []
            try:
                for fold_train, fold_test in rolling_origin_splits(
                    clean_df, n_splits=3, horizon_days=14, min_train_days=45
                ):
                    if len(fold_train) >= 30 and len(fold_test) >= 5:
                        f_pipe = DemandFeaturePipeline()
                        f_pipe.fit(fold_train)
                        f_test_feats = f_pipe.transform(fold_test)
                        if "sales" in f_test_feats.columns:
                            y_f = f_test_feats["sales"].to_numpy(dtype=float)
                            c_preds = champion_forecaster.predict(f_test_feats)
                            k_preds = candidate_forecaster.predict(f_test_feats)
                            cv_champ_errors.extend(list(np.abs(y_f - c_preds)))
                            cv_cand_errors.extend(list(np.abs(y_f - k_preds)))
            except Exception as cv_err:
                logger.info("Rolling origin CV fold notice: %s", cv_err)

            # 8. Evaluate Champion and Candidate on Unseen Test Partition
            y_test = test_feats["sales"].values
            champ_preds = champion_forecaster.predict(test_feats)
            cand_preds = candidate_forecaster.predict(test_feats)

            cand_metrics = evaluate_forecast(y_test, cand_preds)
            comparison = compare_champion_challenger(
                y_true=y_test,
                champion_preds=champ_preds,
                candidate_preds=cand_preds,
                metric_name="MAE",
                alpha=0.05,
            )

            # 8. Determine Next Version Number
            max_v = db.execute(
                select(ModelVersion.version)
                .where(ModelVersion.model_id == model_id)
                .order_by(desc(ModelVersion.version))
            ).scalars().first()
            next_version = (max_v or 1) + 1

            # 9. MLflow Tracking & Artifact Logging
            run_metadata = self.mlflow_registry.log_retraining_run(
                forecaster=candidate_forecaster,
                model_version=next_version,
                params={
                    "n_estimators": cfg.n_estimators,
                    "max_depth": cfg.max_depth,
                    "learning_rate": cfg.learning_rate,
                },
                metrics={
                    "test_mae": cand_metrics.mae,
                    "test_rmse": cand_metrics.rmse,
                    "test_wape": cand_metrics.wape,
                    "test_mase": cand_metrics.mase,
                    "champion_test_mae": comparison.champion_metric,
                    "paired_p_value": comparison.p_value,
                },
                data_hash=data_hash,
                feature_config=pipeline.get_feature_config(),
                drift_context=drift_context,
                models_dir=self.models_dir,
                pipeline=pipeline,
            )

            # 10. Register Candidate in DB (is_active_champion=False strictly!)
            candidate_id = f"mv_{uuid.uuid4().hex[:12]}"
            cand_version_record = ModelVersion(
                id=candidate_id,
                model_id=model_id,
                version=next_version,
                artifact_uri=run_metadata.artifact_uri,
                is_active_champion=False,  # Hard rule: NEVER auto-promoted without explicit promotion
                params_json=json.dumps({"n_estimators": cfg.n_estimators, "max_depth": cfg.max_depth}),
                metrics_json=json.dumps(cand_metrics.to_dict()),
                created_at=datetime.now(UTC),
            )
            db.add(cand_version_record)

            # 11. Record Model Evaluation
            eval_record = ModelEvaluationModel(
                id=f"eval_{uuid.uuid4().hex[:10]}",
                champion_version_id=champion_version.id,
                candidate_version_id=candidate_id,
                metric_name="MAE",
                champion_metric=comparison.champion_metric,
                candidate_metric=comparison.candidate_metric,
                p_value_paired=comparison.p_value,
                is_stat_significant=comparison.is_stat_significant,
                decision=comparison.decision,
                created_at=datetime.now(UTC),
            )
            db.add(eval_record)

            # 12. Finalize Retraining Job
            job.status = "COMPLETED"
            job.candidate_version_id = candidate_id
            job.metrics_comparison_json = json.dumps({
                "decision": comparison.decision,
                "champion_mae": comparison.champion_metric,
                "candidate_mae": comparison.candidate_metric,
                "p_value": comparison.p_value,
                "significant": comparison.is_stat_significant,
            })
            job.completed_at = datetime.now(UTC)
            db.commit()

            return RetrainingExecutionResult(
                job_id=job_id,
                status="COMPLETED",
                champion_version_id=champion_version.id,
                candidate_version_id=candidate_id,
                candidate_version=next_version,
                comparison=comparison,
                mlflow_metadata=run_metadata,
            )

        except Exception as e:
            db.rollback()
            job_fail = db.get(RetrainingJobModel, job_id)
            if job_fail:
                job_fail.status = "FAILED"
                job_fail.completed_at = datetime.now(UTC)
                db.commit()
            logger.error("Retraining job %s failed: %s", job_id, e, exc_info=True)
            raise
