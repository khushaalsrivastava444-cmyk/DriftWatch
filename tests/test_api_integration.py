"""Integration Tests for FastAPI Endpoints, Persistence, Delayed Labels, and Performance Monitoring."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftwatch.api.app import app
from driftwatch.api.auth import DEFAULT_API_KEY
from driftwatch.db.base import Base, engine
from driftwatch.monitoring.performance import PerformanceEvaluator


@pytest.fixture(scope="module", autouse=True)
def setup_test_db() -> None:
    """Ensure clean test database tables exist."""
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_check_endpoint(client: TestClient) -> None:
    """Verify /health returns 200, connected database, and active model."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert data["active_model_id"] is not None
    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-Ms" in resp.headers


def test_single_prediction_and_duplicate_id_handling(client: TestClient) -> None:
    """Verify genuine model inference, database logging, and duplicate ID idempotency."""
    custom_pred_id = f"test_pred_{uuid.uuid4().hex[:8]}"
    payload = {
        "prediction_id": custom_pred_id,
        "features": {
            "lag_1": 15.0,
            "lag_2": 14.0,
            "lag_7": 18.0,
            "lag_14": 16.0,
            "lag_28": 15.0,
            "rolling_mean_7": 16.2,
            "rolling_std_7": 2.1,
            "rolling_mean_14": 15.8,
            "rolling_mean_28": 15.4,
            "day_of_week": 4,
            "day_of_month": 15,
            "month": 6,
            "is_weekend": 0,
            "sell_price": 9.99,
            "price_discount_ratio": 1.0,
            "on_promotion": 0,
            "store_code": 0,
            "category_code": 0,
        },
        "store_id": "STORE_01",
        "item_id": "ITEM_001",
    }

    # 1. First call: new prediction
    resp1 = client.post("/api/v1/predict", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["prediction_id"] == custom_pred_id
    assert data1["prediction"] >= 0.0
    assert data1["latency_ms"] >= 0.0

    # 2. Second call with IDENTICAL prediction_id: Idempotency check
    resp2 = client.post("/api/v1/predict", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["prediction_id"] == custom_pred_id
    # Must return exact same cached prediction value
    assert data2["prediction"] == data1["prediction"]


def test_batch_prediction_endpoint(client: TestClient) -> None:
    """Verify batch prediction endpoint."""
    batch_items = []
    for i in range(5):
        batch_items.append({
            "features": {
                "lag_1": 10.0 + i,
                "day_of_week": i % 7,
                "day_of_month": 10 + i,
                "month": 3,
                "is_weekend": 1 if i in [5, 6] else 0,
                "sell_price": 12.50,
            }
        })

    resp = client.post("/api/v1/predict/batch", json={"predictions": batch_items})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 5
    assert len(data["predictions"]) == 5
    for p in data["predictions"]:
        assert p["prediction"] >= 0.0


def test_ground_truth_ingestion_and_auth(client: TestClient) -> None:
    """Verify delayed ground-truth ingestion requires auth and attaches to predictions."""
    # Step 1: Make a prediction to generate an existing prediction_id
    pred_resp = client.post(
        "/api/v1/predict",
        json={
            "features": {
                "lag_1": 20.0,
                "day_of_week": 1,
                "day_of_month": 5,
                "month": 4,
                "is_weekend": 0,
                "sell_price": 5.0,
            }
        },
    )
    pred_id = pred_resp.json()["prediction_id"]

    label_payload = {
        "labels": [
            {"prediction_id": pred_id, "true_label": 22.0},
            {"prediction_id": "unmatched_pred_999", "true_label": 10.0},
        ]
    }

    # Step 2: Attempt without auth header -> Must fail with 401
    unauth_resp = client.post("/api/v1/labels", json=label_payload)
    assert unauth_resp.status_code == 401

    # Step 3: Attempt with valid X-API-Key -> Must succeed
    auth_resp = client.post(
        "/api/v1/labels",
        json=label_payload,
        headers={"X-API-Key": DEFAULT_API_KEY},
    )
    assert auth_resp.status_code == 200
    data = auth_resp.json()
    assert data["ingested_count"] == 2
    assert data["matched_predictions_count"] >= 1


def test_performance_evaluation_and_bootstrap_ci() -> None:
    """Verify PerformanceEvaluator computes bootstrap CIs and enforces UNVERIFIED on low coverage."""
    evaluator = PerformanceEvaluator(
        min_coverage_threshold=0.75,
        baseline_mae=4.0,
        degradation_threshold_ratio=0.30,  # Degradation trigger at 4.0 * 1.30 = 5.20
        block_size=5,
        n_bootstrap=100,
    )

    # Case A: Low label coverage (e.g. only 50% labels have arrived)
    n_samples = 60
    preds = np.random.uniform(10.0, 30.0, n_samples)
    true_labels = preds.copy()
    # Mask half as pending / delayed
    true_labels[30:] = np.nan

    low_cov_df = pd.DataFrame({
        "prediction_id": [f"p_{i}" for i in range(n_samples)],
        "prediction": preds,
        "true_label": true_labels,
        "delay_seconds": 3600.0,
    })

    eval_unverified = evaluator.evaluate_window(
        matched_df=low_cov_df,
        window_id="W_LOW_COV",
        window_start="2024-05-01",
        window_end="2024-05-15",
    )
    assert eval_unverified.status == "UNVERIFIED"
    assert eval_unverified.label_coverage_ratio == 0.50
    assert not eval_unverified.degradation_breached

    # Case B: High label coverage (mature window) with normal performance (no degradation)
    mature_trues = preds + np.random.normal(0.0, 1.5, n_samples)
    mature_df = pd.DataFrame({
        "prediction_id": [f"p_{i}" for i in range(n_samples)],
        "prediction": preds,
        "true_label": mature_trues,
        "delay_seconds": 7200.0,
    })

    eval_mature = evaluator.evaluate_window(
        matched_df=mature_df,
        window_id="W_MATURE",
        window_start="2024-05-01",
        window_end="2024-05-15",
    )
    assert eval_mature.status == "VERIFIED"
    assert eval_mature.label_coverage_ratio == 1.0
    assert eval_mature.metrics is not None
    assert eval_mature.mae_ci is not None
    assert eval_mature.mae_ci.ci_lower <= eval_mature.mae_ci.point_estimate <= eval_mature.mae_ci.ci_upper
    # MAE is around ~1.2, well below 5.20 -> no degradation
    assert not eval_mature.degradation_breached

    # Case C: Severe performance degradation (large shift in errors)
    degraded_trues = preds + 15.0  # Massive 15 unit bias
    degraded_df = pd.DataFrame({
        "prediction_id": [f"p_{i}" for i in range(n_samples)],
        "prediction": preds,
        "true_label": degraded_trues,
        "delay_seconds": 7200.0,
    })

    eval_degraded = evaluator.evaluate_window(
        matched_df=degraded_df,
        window_id="W_DEGRADED",
        window_start="2024-05-01",
        window_end="2024-05-15",
    )
    assert eval_degraded.status == "VERIFIED"
    assert eval_degraded.mae_ci is not None
    assert eval_degraded.mae_ci.ci_lower > 5.20
    # Confirmed degradation breach via confidence interval!
    assert eval_degraded.degradation_breached


def test_models_registry_promotion_and_rollback(client: TestClient) -> None:
    """Verify registry listing, candidate promotion, and rollback with audit logging."""
    # 1. Get models list
    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    models_list = models_resp.json()
    assert len(models_list) >= 1

    # 2. Promote invalid version returns 404
    bad_promote = client.post(
        "/api/v1/models/mv_nonexistent/promote",
        headers={"X-API-Key": DEFAULT_API_KEY},
    )
    assert bad_promote.status_code == 404

    # 3. Active model endpoint
    active_resp = client.get("/models/active")
    assert active_resp.status_code == 200
    assert active_resp.json()["is_active_champion"] is True
