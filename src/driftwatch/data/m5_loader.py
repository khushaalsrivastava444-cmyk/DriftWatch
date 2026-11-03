"""M5 Forecasting Dataset Loader with Synthetic Fallback.

Checks for local M5 Kaggle competition files in `data/m5/`:
    - sales_train_evaluation.csv (or sales_train_validation.csv)
    - calendar.csv
    - sell_prices.csv

If present, reshapes the wide M5 format into long-format daily panel data.
If absent, transparently falls back to `generate_synthetic_demand()` with
clear diagnostic logging and provenance markers (`is_synthetic=True`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from driftwatch.data.synthetic import SyntheticDemandConfig, generate_synthetic_demand

logger = logging.getLogger(__name__)

DEFAULT_M5_DIR = Path("data/m5")


def load_m5_data(
    m5_dir: Path | str = DEFAULT_M5_DIR,
    max_series: int = 50,
    max_days: int = 365,
    fallback_config: SyntheticDemandConfig | None = None,
) -> pd.DataFrame:
    """Load local M5 retail data or gracefully fallback to synthetic generation.
    
    Args:
        m5_dir: Directory containing local M5 CSV files.
        max_series: Subsample limit for store/item combinations to keep CPU runtimes fast.
        max_days: Maximum recent history days to extract.
        fallback_config: Configuration used if fallback to synthetic data is triggered.
        
    Returns:
        pd.DataFrame standardized to [date, store_id, item_id, category, sell_price, on_promotion, sales, is_synthetic]
    """
    m5_path = Path(m5_dir)
    sales_file = None
    for candidate in ["sales_train_evaluation.csv", "sales_train_validation.csv"]:
        if (m5_path / candidate).is_file():
            sales_file = m5_path / candidate
            break

    calendar_file = m5_path / "calendar.csv"
    prices_file = m5_path / "sell_prices.csv"

    if sales_file is None or not calendar_file.is_file():
        logger.info(
            "Local M5 dataset files not found in '%s'. Falling back to reproducible "
            "synthetic retail-demand generator (provenance: seed=%d, is_synthetic=True).",
            m5_path,
            (fallback_config.seed if fallback_config else 42),
        )
        return generate_synthetic_demand(fallback_config)

    logger.info("Loading local M5 data from %s", sales_file)
    sales_df = pd.read_csv(sales_file, nrows=max_series)
    cal_df = pd.read_csv(calendar_file)

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    if len(day_cols) > max_days:
        day_cols = day_cols[-max_days:]

    # Melt wide into long panel format
    melted = pd.melt(
        sales_df,
        id_vars=[c for c in id_vars if c in sales_df.columns],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # Merge calendar to obtain exact Gregorian dates
    cal_cols = ["d", "date", "event_name_1"]
    cal_subset = cal_df[[c for c in cal_cols if c in cal_df.columns]].copy()
    merged = pd.merge(melted, cal_subset, on="d", how="inner")

    # Map features to standardized DriftWatch schema
    merged["date"] = pd.to_datetime(merged["date"])
    merged["category"] = merged["cat_id"] if "cat_id" in merged.columns else "RETAIL"
    merged["on_promotion"] = merged["event_name_1"].notna().astype(int) if "event_name_1" in merged.columns else 0

    # Sell prices if available, otherwise synthetic proxy price
    if prices_file.is_file():
        prices_df = pd.read_csv(prices_file)
        cal_wm_yr = cal_df[["d", "wm_yr_wk"]].drop_duplicates()
        merged = pd.merge(merged, cal_wm_yr, on="d", how="left")
        merged = pd.merge(merged, prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        merged["sell_price"] = merged["sell_price"].fillna(9.99)
    else:
        merged["sell_price"] = 9.99

    merged["is_synthetic"] = False
    
    std_cols = ["date", "store_id", "item_id", "category", "sell_price", "on_promotion", "sales", "is_synthetic"]
    result = merged[std_cols].sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    logger.info("Successfully loaded and standardized %d M5 records", len(result))
    return result
