"""Shadow Deployment Engine for Risk-Free Challenger Evaluation.

Executes asynchronous candidate inference in parallel with the production Champion
without affecting live request latency or customer-facing responses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from driftwatch.api.schemas import PredictResponse
from driftwatch.models.xgboost_forecaster import XGBoostDemandForecaster

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowPredictionLog:
    """Telemetry captured from candidate shadow execution."""
    prediction_id: str
    request_id: str
    champion_version_id: str
    champion_prediction: float
    challenger_version_id: str
    challenger_prediction: float
    challenger_latency_ms: float
    prediction_delta: float


class ShadowDeploymentManager:
    """Manages shadow candidate forecasters and records comparative inference logs."""

    def __init__(self) -> None:
        self.challenger_forecaster: XGBoostDemandForecaster | None = None
        self.challenger_version_id: str | None = None
        self.shadow_logs: list[ShadowPredictionLog] = []

    def set_shadow_challenger(
        self,
        forecaster: XGBoostDemandForecaster,
        challenger_version_id: str,
    ) -> None:
        """Register or update active shadow challenger."""
        self.challenger_forecaster = forecaster
        self.challenger_version_id = challenger_version_id
        logger.info("Shadow challenger active: version %s", challenger_version_id)

    def remove_shadow_challenger(self) -> None:
        """Remove active shadow challenger."""
        self.challenger_forecaster = None
        self.challenger_version_id = None

    def execute_shadow_inference(
        self,
        features_df: Any,
        champion_resp: PredictResponse,
    ) -> ShadowPredictionLog | None:
        """Execute shadow prediction if a challenger is registered."""
        if self.challenger_forecaster is None or self.challenger_version_id is None:
            return None

        start = time.perf_counter()
        try:
            raw_pred = self.challenger_forecaster.predict(features_df)[0]
            cand_pred = float(raw_pred)
            latency_ms = (time.perf_counter() - start) * 1000.0

            log_entry = ShadowPredictionLog(
                prediction_id=champion_resp.prediction_id,
                request_id=champion_resp.request_id,
                champion_version_id=champion_resp.model_version_id,
                champion_prediction=champion_resp.prediction,
                challenger_version_id=self.challenger_version_id,
                challenger_prediction=cand_pred,
                challenger_latency_ms=round(latency_ms, 2),
                prediction_delta=round(cand_pred - champion_resp.prediction, 4),
            )
            self.shadow_logs.append(log_entry)
            return log_entry
        except Exception as e:
            logger.error("Shadow inference failed: %s", e)
            return None
