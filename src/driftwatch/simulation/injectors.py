"""Controlled Drift and Corruption Injectors.

Provides mathematical operators to perturb chronological production streams:
- Stable (Identity baseline)
- Mean Shift (delta * sigma)
- Variance Shift (variance scaling factor)
- Categorical Shift (category frequency redistribution)
- Missing Rate Increase (random dropout)
- Prediction Drift (targeted shift on high-importance input)
- Concept Drift (alteration of P(Y|X) conditional distribution)

All injections compute before/after SHA-256 dataset hashes and document seeds.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftInjectionType(str, Enum):
    STABLE = "STABLE"
    MEAN_SHIFT = "MEAN_SHIFT"
    VARIANCE_SHIFT = "VARIANCE_SHIFT"
    CATEGORICAL_SHIFT = "CATEGORICAL_SHIFT"
    MISSING_RATE_INCREASE = "MISSING_RATE_INCREASE"
    PREDICTION_DRIFT = "PREDICTION_DRIFT"
    CONCEPT_DRIFT = "CONCEPT_DRIFT"


def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash of DataFrame contents."""
    encoded = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class InjectionConfig:
    """Configuration for drift injection."""
    injection_type: DriftInjectionType = DriftInjectionType.STABLE
    target_columns: list[str] = field(default_factory=lambda: ["sales", "sell_price"])
    shift_magnitude: float = 1.5      # standard deviations for mean shift, or variance multiplier
    dropout_ratio: float = 0.20       # for missing rate increase
    seed: int = 42
    start_date: str | None = None  # If set, only perturb records on/after this date


def apply_drift_injection(
    df: pd.DataFrame,
    config: InjectionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply chosen drift transformation to DataFrame and return before/after provenance."""
    out = df.copy()
    before_hash = compute_df_hash(out)
    rng = np.random.default_rng(config.seed)

    # Determine subset mask (e.g. only perturb production period)
    if config.start_date and "date" in out.columns:
        mask = pd.to_datetime(out["date"]) >= pd.to_datetime(config.start_date)
    else:
        mask = pd.Series(True, index=out.index)

    rows_affected = int(mask.sum())
    meta: dict[str, Any] = {
        "injection_type": config.injection_type.value,
        "seed": config.seed,
        "rows_affected": rows_affected,
        "target_columns": config.target_columns,
        "shift_magnitude": config.shift_magnitude,
    }

    if config.injection_type == DriftInjectionType.STABLE or rows_affected == 0:
        after_hash = compute_df_hash(out)
        meta["before_hash"] = before_hash
        meta["after_hash"] = after_hash
        return out, meta

    elif config.injection_type == DriftInjectionType.MEAN_SHIFT:
        for col in config.target_columns:
            if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
                is_int = pd.api.types.is_integer_dtype(out[col])
                std_val = float(out.loc[~mask, col].std()) if not out[~mask].empty else float(out[col].std())
                std_val = std_val if std_val > 0 else 1.0
                shift = config.shift_magnitude * std_val
                if is_int:
                    out[col] = out[col].astype(float)
                out.loc[mask, col] = np.maximum(0.0, out.loc[mask, col] + shift)
                if is_int:
                    out[col] = out[col].round().astype(int)

    elif config.injection_type == DriftInjectionType.VARIANCE_SHIFT:
        for col in config.target_columns:
            if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
                is_int = pd.api.types.is_integer_dtype(out[col])
                if is_int:
                    out[col] = out[col].astype(float)
                mean_val = float(out.loc[mask, col].mean())
                diff = out.loc[mask, col] - mean_val
                out.loc[mask, col] = np.maximum(0.0, mean_val + (diff * config.shift_magnitude))
                if is_int:
                    out[col] = out[col].round().astype(int)

    elif config.injection_type == DriftInjectionType.CATEGORICAL_SHIFT:
        # Heavily skew categories or introduce unseen category
        for col in config.target_columns:
            if col in out.columns:
                unique_cats = out[col].unique().tolist()
                if unique_cats:
                    # Skew 80% of affected records towards the last category
                    skew_cat = unique_cats[-1]
                    replace_mask = mask & (rng.random(len(out)) < 0.75)
                    out.loc[replace_mask, col] = skew_cat

    elif config.injection_type == DriftInjectionType.MISSING_RATE_INCREASE:
        for col in config.target_columns:
            if col in out.columns:
                drop_mask = mask & (rng.random(len(out)) < config.dropout_ratio)
                out.loc[drop_mask, col] = np.nan

    elif config.injection_type == DriftInjectionType.PREDICTION_DRIFT:
        # Perturb primary lag feature which drives prediction shifts
        for col in ["lag_1", "sell_price"]:
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] * 2.5 + 10.0

    elif config.injection_type == DriftInjectionType.CONCEPT_DRIFT:
        # Concept drift alters the relationship P(Y|X):
        # E.g. Consumers suddenly stop responding to promotions, or price sensitivity flips
        if "sales" in out.columns:
            # Sales drop by 60% despite features remaining identical
            out.loc[mask, "sales"] = np.maximum(0, (out.loc[mask, "sales"] * 0.40).astype(int))

    after_hash = compute_df_hash(out)
    meta["before_hash"] = before_hash
    meta["after_hash"] = after_hash

    logger.info(
        "Applied %s injection on %d rows. Hashes: %s -> %s",
        config.injection_type.value,
        rows_affected,
        before_hash,
        after_hash,
    )
    return out, meta
