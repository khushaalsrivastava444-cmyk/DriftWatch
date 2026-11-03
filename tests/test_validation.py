"""Tests for Data Quality Validation Engine and Quarantine Routing."""

from __future__ import annotations

import pandas as pd
import pytest

from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand
from driftwatch.validation.engine import DataValidationEngine, ValidationConfig


@pytest.fixture
def clean_dataset() -> pd.DataFrame:
    cfg = SyntheticDemandConfig(num_days=30, num_stores=2, num_items=3, seed=123)
    return generate_synthetic_demand(cfg)


def test_clean_dataset_passes_validation(clean_dataset: pd.DataFrame) -> None:
    engine = DataValidationEngine()
    report, clean_df, quar_df = engine.validate(clean_dataset)

    assert report.is_valid
    assert not report.has_errors
    assert len(quar_df) == 0
    assert len(clean_df) == len(clean_dataset)


def test_missing_required_columns_fails(clean_dataset: pd.DataFrame) -> None:
    corrupted = clean_dataset.drop(columns=["sales", "sell_price"])
    engine = DataValidationEngine()
    report, clean_df, quar_df = engine.validate(corrupted)

    assert not report.is_valid
    assert report.has_errors
    assert any(v.rule_name == "required_columns" for v in report.violations)
    assert len(clean_df) == 0
    assert len(quar_df) == len(clean_dataset)


def test_negative_sales_and_price_quarantine(clean_dataset: pd.DataFrame) -> None:
    corrupted = clean_dataset.copy()
    # Inject negative sales into first 5 rows
    corrupted.loc[:4, "sales"] = -10
    # Inject 0 price into rows 10-14
    corrupted.loc[10:14, "sell_price"] = 0.0

    engine = DataValidationEngine()
    report, clean_df, quar_df = engine.validate(corrupted)

    assert not report.is_valid
    assert report.has_errors
    assert len(quar_df) == 10
    assert len(clean_df) == len(clean_dataset) - 10

    # Ensure clean df contains no negative values
    assert (clean_df["sales"] >= 0).all()
    assert (clean_df["sell_price"] > 0).all()


def test_duplicate_primary_keys_quarantine(clean_dataset: pd.DataFrame) -> None:
    # Duplicate first 3 rows
    dups = clean_dataset.iloc[:3].copy()
    corrupted = pd.concat([clean_dataset, dups], ignore_index=True)

    engine = DataValidationEngine()
    report, clean_df, quar_df = engine.validate(corrupted)

    assert not report.is_valid
    assert any(v.rule_name == "duplicate_primary_key" for v in report.violations)
    # Both original and duplicate instances should be flagged
    assert len(quar_df) >= 3


def test_unexpected_categories_warning(clean_dataset: pd.DataFrame) -> None:
    corrupted = clean_dataset.copy()
    corrupted.loc[0, "category"] = "UNSEEN_SPACE_GEAR"

    config = ValidationConfig(
        allowed_categories={"category": {"FOODS", "HOUSEHOLD", "BEVERAGES"}}
    )
    engine = DataValidationEngine(config)
    report, clean_df, quar_df = engine.validate(corrupted)

    # Unexpected category is a WARNING, not an ERROR, so dataset remains valid and not quarantined
    assert report.is_valid
    assert report.has_warnings
    assert not report.has_errors
    assert any(v.rule_name == "unexpected_categories" for v in report.violations)
    assert len(clean_df) == len(clean_dataset)
