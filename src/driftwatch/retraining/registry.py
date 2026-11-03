"""MLflow Tracking and Artifact Governance Integration."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from driftwatch.models.xgboost_forecaster import XGBoostDemandForecaster

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLflowRunMetadata:
    """Audit metadata recorded during an MLflow retraining run."""
    run_id: str
    experiment_id: str
    artifact_uri: str
    git_sha: str
    data_hash: str
    pipeline_uri: str | None = None


def get_git_commit_sha() -> str:
    """Safely obtain current Git commit SHA or fallback to unknown."""
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return res.decode().strip()[:10]
    except Exception:
        return "uncommitted_local"


class MLflowGovernanceRegistry:
    """Manages MLflow runs, metric/param logging, and model artifact tracking."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "driftwatch-forecasting") -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_name = experiment_name
        self._ensure_experiment()

    def _ensure_experiment(self) -> None:
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def log_retraining_run(
        self,
        forecaster: XGBoostDemandForecaster,
        model_version: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        data_hash: str,
        feature_config: dict[str, Any],
        drift_context: str,
        models_dir: Path,
        pipeline: Any | None = None,
    ) -> MLflowRunMetadata:
        """Execute a fully auditable MLflow logging transaction."""
        models_dir.mkdir(parents=True, exist_ok=True)
        git_sha = get_git_commit_sha()

        with mlflow.start_run(run_name=f"candidate_v{model_version}") as run:
            run_id = run.info.run_id
            exp_id = run.info.experiment_id

            # 1. Log Hyperparameters
            for k, v in params.items():
                mlflow.log_param(k, v)

            # 2. Log Evaluation Metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # 3. Log Provenance Tags
            mlflow.set_tag("model_version", str(model_version))
            mlflow.set_tag("data_hash", data_hash)
            mlflow.set_tag("git_sha", git_sha)
            mlflow.set_tag("drift_context", drift_context)
            mlflow.set_tag("feature_config", json.dumps(feature_config))

            # 4. Save and Log Model Artifact
            artifact_file = models_dir / f"model_v{model_version}_{run_id[:8]}.json"
            forecaster.save_model(str(artifact_file))
            mlflow.log_artifact(str(artifact_file), artifact_path="model")
            local_artifact_uri = str(artifact_file.resolve())

            # 5. Save and Log Feature Pipeline Artifact
            pipeline_uri = None
            if pipeline is not None and hasattr(pipeline, "save"):
                pipeline_file = models_dir / f"pipeline_v{model_version}_{run_id[:8]}.joblib"
                pipeline.save(pipeline_file)
                mlflow.log_artifact(str(pipeline_file), artifact_path="feature_pipeline")
                pipeline_uri = str(pipeline_file.resolve())

            logger.info(
                "Logged MLflow run %s for model v%d (Model: %s, Pipeline: %s)",
                run_id, model_version, local_artifact_uri, pipeline_uri
            )

            return MLflowRunMetadata(
                run_id=run_id,
                experiment_id=exp_id,
                artifact_uri=local_artifact_uri,
                git_sha=git_sha,
                data_hash=data_hash,
                pipeline_uri=pipeline_uri,
            )
