"""Temporal Data Partitioning Utilities.

Enforces strictly chronological, non-overlapping train/validation/test partitions.
Shuffled or random splits are strictly disallowed to prevent lookahead data leakage.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalPartitions:
    """Container for chronologically isolated partitions."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_column: str = "date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> TemporalPartitions:
    """Split a DataFrame strictly by chronological order.
    
    Args:
        df: Input panel DataFrame.
        date_column: Column name containing datetime or date objects.
        train_ratio: Proportion of unique dates allocated to training partition.
        val_ratio: Proportion of unique dates allocated to validation partition.
        
    Returns:
        TemporalPartitions(train, val, test)
        
    Raises:
        ValueError: If partitions overlap in time or ratios are invalid.
    """
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}. Must sum to < 1.0.")

    sorted_dates = pd.Series(df[date_column].unique()).sort_values().reset_index(drop=True)
    n_dates = len(sorted_dates)
    if n_dates < 10:
        raise ValueError(f"Insufficient unique dates for temporal split: {n_dates} dates.")

    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_dates = set(sorted_dates.iloc[:train_end_idx])
    val_dates = set(sorted_dates.iloc[train_end_idx:val_end_idx])
    test_dates = set(sorted_dates.iloc[val_end_idx:])

    train_df = df[df[date_column].isin(train_dates)].copy().sort_values(date_column).reset_index(drop=True)
    val_df = df[df[date_column].isin(val_dates)].copy().sort_values(date_column).reset_index(drop=True)
    test_df = df[df[date_column].isin(test_dates)].copy().sort_values(date_column).reset_index(drop=True)

    # Verification: assert zero temporal overlap
    max_train = train_df[date_column].max()
    min_val = val_df[date_column].min()
    max_val = val_df[date_column].max()
    min_test = test_df[date_column].min()

    if max_train >= min_val:
        raise ValueError(f"Leakage detected: max train date ({max_train}) >= min val date ({min_val})")
    if max_val >= min_test:
        raise ValueError(f"Leakage detected: max val date ({max_val}) >= min test date ({min_test})")

    return TemporalPartitions(train=train_df, val=val_df, test=test_df)


def rolling_origin_splits(
    df: pd.DataFrame,
    date_column: str = "date",
    n_splits: int = 4,
    horizon_days: int = 14,
    min_train_days: int = 60,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Generate walk-forward expanding window cross-validation splits.
    
    Yields:
        (train_fold, test_fold) where test_fold immediately follows train_fold in time.
    """
    sorted_dates = pd.Series(df[date_column].unique()).sort_values().reset_index(drop=True)
    n_dates = len(sorted_dates)

    required_dates = min_train_days + (n_splits * horizon_days)
    if n_dates < required_dates:
        # Scale back parameters if dataset is shorter
        horizon_days = max(3, (n_dates - min_train_days) // (n_splits + 1))

    for split_idx in range(n_splits):
        cutoff_idx = min_train_days + (split_idx * horizon_days)
        test_end_idx = min(cutoff_idx + horizon_days, n_dates)
        
        train_dates = set(sorted_dates.iloc[:cutoff_idx])
        test_dates = set(sorted_dates.iloc[cutoff_idx:test_end_idx])

        train_fold = df[df[date_column].isin(train_dates)].copy().sort_values(date_column).reset_index(drop=True)
        test_fold = df[df[date_column].isin(test_dates)].copy().sort_values(date_column).reset_index(drop=True)
        
        yield train_fold, test_fold
