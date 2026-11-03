"""Model Management and Serving Logic for FastAPI Service."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from driftwatch.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    LabelItem,
    PredictRequest,
    PredictResponse,
)
from driftwatch.data.split import temporal_train_val_test_split
from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.db.models import (
    AuditLogModel,
    GroundTruthLabel,
    Model,
    ModelVersion,
    PredictionLog,
)
from driftwatch.features.pipeline import FEATURE_COLUMNS, DemandFeaturePipeline
from driftwatch.models.xgboost_forecaster import (
    XGBoostDemandForecaster,
    XGBoostForecasterConfig,
)

logger = logging.getLogger(__name__)


class ModelServingService:
    """Manages active models, genuine inference, and database persistence."""

    def __init__(self, models_dir: str = "artifacts/models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: DemandFeaturePipeline | None = None
        self.active_forecaster: XGBoostDemandForecaster | None = None
        self.active_version_id: str | None = None
        self.active_model_id: str | None = None

    def ensure_default_model(self, db: Session) -> str:
        """Ensure at least one active champion model is registered and loaded in memory."""
        # 1. Query for active champion in DB
        stmt = (
            select(ModelVersion)
            .join(Model)
            .where(ModelVersion.is_active_champion == True)
            .order_by(desc(ModelVersion.created_at))
        )
        active_version = db.execute(stmt).scalars().first()

        if active_version:
            self.active_version_id = active_version.id
            self.active_model_id = active_version.model_id
            # Load from artifact URI if not already in memory
            art_path = Path(active_version.artifact_uri)
            if not art_path.exists():
                fallback = self.models_dir / art_path.name
                if fallback.exists():
                    art_path = fallback
            if self.active_forecaster is None and art_path.exists():
                forecaster = XGBoostDemandForecaster()
                forecaster.load_model(str(art_path))
                self.active_forecaster = forecaster
                logger.info("Loaded active champion model %s from %s", active_version.id, str(art_path))
            return active_version.id

        # 2. If no active champion exists, train and register initial Champion (v1)
        logger.info("No active champion found in registry. Bootstrapping initial Champion model...")
        cfg = SyntheticDemandConfig(num_days=180, num_stores=2, num_items=5, seed=42)
        raw_df = generate_synthetic_demand(cfg)
        parts = temporal_train_val_test_split(raw_df, train_ratio=0.75, val_ratio=0.15)

        pipeline = DemandFeaturePipeline()
        pipeline.fit(parts.train)
        self.pipeline = pipeline

        train_feats = pipeline.transform(parts.train)
        val_feats = pipeline.transform(parts.val)

        forecaster = XGBoostDemandForecaster(config=XGBoostForecasterConfig(n_estimators=80, max_depth=4))
        forecaster.fit(train_feats, train_feats["sales"], eval_set=[(val_feats, val_feats["sales"])])

        model_id = "model_demand_v1"
        version_id = f"mv_{uuid.uuid4().hex[:12]}"
        artifact_path = str(self.models_dir / f"{version_id}.json")
        forecaster.save_model(artifact_path)

        # Register in database
        db_model = db.get(Model, model_id)
        if not db_model:
            db_model = Model(
                id=model_id,
                name="retail_demand_forecaster",
                description="Production XGBoost Retail Demand Forecaster",
                task_type="demand_forecasting",
            )
            db.add(db_model)
            db.flush()

        db_version = ModelVersion(
            id=version_id,
            model_id=model_id,
            version=1,
            artifact_uri=artifact_path,
            is_active_champion=True,
            params_json=json.dumps({"n_estimators": 80, "max_depth": 4}),
            metrics_json=json.dumps({"status": "bootstrapped"}),
        )
        db.add(db_version)
        db.commit()

        self.active_forecaster = forecaster
        self.active_version_id = version_id
        self.active_model_id = model_id
        logger.info("Registered and loaded initial champion version %s", version_id)
        return version_id

    def predict_single(
        self,
        req: PredictRequest,
        db: Session,
        request_id: str,
    ) -> PredictResponse:
        """Serve single demand prediction with duplicate prediction_id handling."""
        start_time = time.perf_counter()
        self.ensure_default_model(db)

        pred_id = req.prediction_id or f"pred_{uuid.uuid4().hex}"

        # 1. Check for duplicate prediction_id (Idempotency)
        existing_log = db.get(PredictionLog, pred_id)
        if existing_log is not None:
            logger.info("Duplicate prediction ID '%s' received. Returning stored prediction.", pred_id)
            return PredictResponse(
                prediction_id=existing_log.prediction_id,
                request_id=request_id,
                model_version_id=existing_log.model_version_id,
                prediction=existing_log.prediction,
                latency_ms=existing_log.latency_ms,
                timestamp=existing_log.timestamp.isoformat(),
            )

        # 2. Genuine inference execution using active model
        feat_dict = req.features.model_dump()
        feat_df = pd.DataFrame([feat_dict])
        
        # Ensure all expected columns are present
        for col in FEATURE_COLUMNS:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        if self.active_forecaster is not None:
            raw_pred = self.active_forecaster.predict(feat_df)[0]
        else:
            raw_pred = float(feat_dict.get("lag_1", 10.0))

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        now_dt = datetime.now(UTC)

        # 3. Log prediction genuinely to Postgres/SQLite
        log_entry = PredictionLog(
            prediction_id=pred_id,
            model_version_id=self.active_version_id or "mv_unknown",
            request_id=request_id,
            features_json=json.dumps(feat_dict),
            prediction=float(raw_pred),
            latency_ms=round(latency_ms, 2),
            timestamp=now_dt,
        )
        db.add(log_entry)
        db.commit()

        return PredictResponse(
            prediction_id=pred_id,
            request_id=request_id,
            model_version_id=self.active_version_id or "mv_unknown",
            prediction=float(raw_pred),
            latency_ms=round(latency_ms, 2),
            timestamp=now_dt.isoformat(),
        )

    def predict_batch(
        self,
        batch_req: BatchPredictRequest,
        db: Session,
        request_id: str,
    ) -> BatchPredictResponse:
        """Serve batch prediction requests."""
        start_time = time.perf_counter()
        responses: list[PredictResponse] = []

        for req in batch_req.predictions:
            resp = self.predict_single(req, db, request_id)
            responses.append(resp)

        total_ms = (time.perf_counter() - start_time) * 1000.0
        return BatchPredictResponse(
            predictions=responses,
            total_count=len(responses),
            execution_time_ms=round(total_ms, 2),
            request_id=request_id,
        )

    def ingest_labels(self, labels: list[LabelItem], db: Session) -> tuple[int, int]:
        """Ingest delayed ground truth labels and attach them to prediction logs."""
        ingested = 0
        matched = 0

        for item in labels:
            existing = db.get(GroundTruthLabel, item.prediction_id)
            if existing is not None:
                # Update existing
                existing.true_label = item.true_label
                ingested += 1
                continue

            pred_log = db.get(PredictionLog, item.prediction_id)
            delay_sec = 0.0
            now_dt = datetime.now(UTC)
            if pred_log is not None:
                delay_sec = max(0.0, (now_dt - pred_log.timestamp.replace(tzinfo=UTC)).total_seconds())
                matched += 1

            label_record = GroundTruthLabel(
                prediction_id=item.prediction_id,
                true_label=item.true_label,
                observed_at=now_dt,
                delay_seconds=delay_sec,
            )
            db.add(label_record)
            ingested += 1

        db.commit()
        return ingested, matched

    def promote_model_version(
        self,
        model_id: str,
        version: int | None = None,
        version_id: str | None = None,
        actor: str = "admin",
        db: Session = None,
    ) -> ModelVersion:
        """Promote a specific candidate model version to active champion (audited)."""
        if version_id:
            target = db.get(ModelVersion, version_id)
        elif version is not None:
            target = db.execute(
                select(ModelVersion).where(
                    ModelVersion.model_id == model_id,
                    ModelVersion.version == version,
                )
            ).scalars().first()
        else:
            raise ValueError("Either version or version_id must be provided.")

        if not target:
            raise ValueError(f"Model version not found (id={version_id}, v={version}).")

        # Unset existing champions
        incumbents = db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.is_active_champion.is_(True),
            )
        ).scalars().all()
        for inc in incumbents:
            inc.is_active_champion = False

        target.is_active_champion = True

        audit = AuditLogModel(
            id=f"audit_{uuid.uuid4().hex[:12]}",
            action="PROMOTE",
            actor=actor,
            entity_type="ModelVersion",
            entity_id=target.id,
            details_json=f'{{"version": {target.version}, "previous_champion": "{incumbents[0].id if incumbents else "none"}"}}',
        )
        db.add(audit)
        db.commit()

        self.active_version_id = target.id
        self.active_model_id = target.model_id
        art_path = Path(target.artifact_uri)
        if not art_path.exists():
            fallback = self.models_dir / art_path.name
            if fallback.exists():
                art_path = fallback
        if art_path.exists():
            forecaster = XGBoostDemandForecaster()
            forecaster.load_model(str(art_path))
            self.active_forecaster = forecaster

        logger.info("Promoted model %s v%d (ID: %s) to active champion", model_id, version, target.id)
        return target

    def rollback_champion(
        self,
        model_id: str,
        actor: str,
        db: Session,
    ) -> ModelVersion:
        """Rollback active champion to previous champion version (audited)."""
        current = db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.is_active_champion.is_(True),
            )
        ).scalars().first()

        candidates = db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id, ModelVersion.is_active_champion.is_(False))
            .order_by(desc(ModelVersion.version))
        ).scalars().all()

        if not candidates:
            raise ValueError("No prior model version available for rollback.")

        prev = candidates[0]
        if current:
            current.is_active_champion = False
        prev.is_active_champion = True

        audit = AuditLogModel(
            id=f"audit_{uuid.uuid4().hex[:12]}",
            action="ROLLBACK",
            actor=actor,
            entity_type="ModelVersion",
            entity_id=prev.id,
            details_json=f'{{"restored_version": {prev.version}, "rolled_back_from": "{current.id if current else "none"}"}}',
        )
        db.add(audit)
        db.commit()

        self.active_version_id = prev.id
        self.active_model_id = prev.model_id
        art_path = Path(prev.artifact_uri)
        if not art_path.exists():
            fallback = self.models_dir / art_path.name
            if fallback.exists():
                art_path = fallback
        if art_path.exists():
            forecaster = XGBoostDemandForecaster()
            forecaster.load_model(str(art_path))
            self.active_forecaster = forecaster

        logger.info("Rolled back model %s to champion v%d (ID: %s)", model_id, prev.version, prev.id)
        return prev


# Singleton instance
serving_service = ModelServingService()
