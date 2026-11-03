"""Time-Ordered Production Replay Simulator with Delayed Label Queues."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from driftwatch.api.schemas import BatchPredictRequest, LabelItem, PredictRequest
from driftwatch.api.service import serving_service
from driftwatch.drift.engine import DriftEngine, DriftEngineConfig, DriftReport
from driftwatch.features.pipeline import FEATURE_COLUMNS, DemandFeaturePipeline
from driftwatch.monitoring.performance import EvaluatedPerformanceWindow, PerformanceEvaluator
from driftwatch.simulation.injectors import InjectionConfig, apply_drift_injection

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for production replay simulation."""
    label_lag_days: int = 7
    window_days: int = 14
    injection_config: InjectionConfig | None = None
    drift_engine_config: DriftEngineConfig | None = None


@dataclass
class SimulationResult:
    """Output summary from a production replay simulation run."""
    total_days: int
    total_predictions: int
    total_labels_released: int
    drift_reports: list[DriftReport] = field(default_factory=list)
    performance_reports: list[EvaluatedPerformanceWindow] = field(default_factory=list)
    injection_metadata: dict[str, Any] = field(default_factory=dict)


class ReplaySimulator:
    """Simulates real-world production data streams through genuine APIs with delayed label queues."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.drift_engine = DriftEngine(self.config.drift_engine_config)
        self.perf_evaluator = PerformanceEvaluator()

    def run_simulation(
        self,
        production_df: pd.DataFrame,
        pipeline: DemandFeaturePipeline,
        db: Session,
        reference_features_df: pd.DataFrame,
    ) -> SimulationResult:
        """Run production streaming simulation chronologically day by day."""
        serving_service.ensure_default_model(db)
        
        # 1. Apply drift injection if configured
        injection_meta: dict[str, Any] = {}
        if self.config.injection_config is not None:
            sim_df, injection_meta = apply_drift_injection(production_df, self.config.injection_config)
        else:
            sim_df = production_df.copy()

        # Transform to feature matrix
        transformed_df = pipeline.transform(sim_df)
        transformed_df["date"] = pd.to_datetime(transformed_df["date"])

        sorted_dates = sorted(transformed_df["date"].unique())
        total_days = len(sorted_dates)
        logger.info("Starting replay simulation across %d days with label lag = %d days", total_days, self.config.label_lag_days)

        # Label queue: dict of release_date -> list of LabelItem
        label_queue: dict[pd.Timestamp, list[LabelItem]] = {}
        prediction_history: list[dict[str, Any]] = []

        drift_reports: list[DriftReport] = []
        performance_reports: list[EvaluatedPerformanceWindow] = []

        total_preds = 0
        total_labels_released = 0

        # Run day-by-day stream
        for current_date in sorted_dates:
            day_mask = transformed_df["date"] == current_date
            day_records = transformed_df[day_mask]

            # 1. Release mature labels whose delay has elapsed
            if current_date in label_queue:
                matured_labels = label_queue.pop(current_date)
                ingested, _ = serving_service.ingest_labels(matured_labels, db)
                total_labels_released += ingested

            # 2. Serve predictions through the real serving service
            predict_requests = []
            for _, row in day_records.iterrows():
                pred_id = f"sim_{row['store_id']}_{row['item_id']}_{current_date.strftime('%Y%m%d')}"
                feat_dict = {col: float(row[col]) for col in FEATURE_COLUMNS if col in row}
                
                req = PredictRequest(
                    prediction_id=pred_id,
                    features=feat_dict,  # type: ignore[arg-type]
                    store_id=str(row["store_id"]),
                    item_id=str(row["item_id"]),
                    date=current_date.strftime("%Y-%m-%d"),
                )
                predict_requests.append(req)

            batch_req = BatchPredictRequest(predictions=predict_requests)
            batch_resp = serving_service.predict_batch(batch_req, db, request_id=f"sim_{current_date.strftime('%Y%m%d')}")
            total_preds += len(batch_resp.predictions)

            # 3. Schedule ground-truth labels into the delayed arrival queue
            release_date = current_date + pd.Timedelta(days=self.config.label_lag_days)
            if release_date not in label_queue:
                label_queue[release_date] = []

            for p_resp, (_, row) in zip(batch_resp.predictions, day_records.iterrows()):
                label_queue[release_date].append(
                    LabelItem(
                        prediction_id=p_resp.prediction_id,
                        true_label=float(row["sales"]),
                    )
                )
                prediction_history.append({
                    "prediction_id": p_resp.prediction_id,
                    "date": current_date,
                    "prediction": p_resp.prediction,
                    "true_label": float(row["sales"]),
                    "features": {col: float(row[col]) for col in FEATURE_COLUMNS if col in row},
                })

        # Process any remaining labels at end of simulation
        for remaining_labels in label_queue.values():
            ingested, _ = serving_service.ingest_labels(remaining_labels, db)
            total_labels_released += ingested

        # 4. Execute post-simulation windowed drift surveillance and performance evaluation
        hist_df = pd.DataFrame(prediction_history)
        if not hist_df.empty:
            # Evaluate drift across full production stream vs reference
            curr_feat_df = pd.DataFrame([p["features"] for p in prediction_history])
            report = self.drift_engine.run_drift_analysis(
                reference_df=reference_features_df,
                current_df=curr_feat_df,
                feature_columns=FEATURE_COLUMNS,
                dataset_id="production_replay",
                window_id="FULL_PRODUCTION_RUN",
            )
            drift_reports.append(report)

            # Evaluate performance
            perf_window = self.perf_evaluator.evaluate_window(
                matched_df=hist_df,
                window_id="PERF_PRODUCTION_RUN",
                window_start=sorted_dates[0].strftime("%Y-%m-%d"),
                window_end=sorted_dates[-1].strftime("%Y-%m-%d"),
                pred_col="prediction",
                true_col="true_label",
            )
            performance_reports.append(perf_window)

        logger.info(
            "Replay completed: %d predictions served, %d labels attached, %d drift reports generated",
            total_preds,
            total_labels_released,
            len(drift_reports),
        )

        return SimulationResult(
            total_days=total_days,
            total_predictions=total_preds,
            total_labels_released=total_labels_released,
            drift_reports=drift_reports,
            performance_reports=performance_reports,
            injection_metadata=injection_meta,
        )
