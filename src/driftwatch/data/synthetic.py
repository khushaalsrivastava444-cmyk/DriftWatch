"""Synthetic Retail-Demand Dataset Generator.

Generates reproducible, realistic daily retail demand time series with
explicit seasonal patterns, price elasticity, trend, and promotional effects.
All rows are explicitly marked with `is_synthetic = True`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticDemandConfig:
    """Configuration for synthetic retail-demand generation.
    
    Attributes:
        start_date: First date in the dataset.
        num_days: Total number of consecutive days to generate.
        num_stores: Number of distinct retail store locations.
        num_items: Number of distinct product SKUs.
        seed: Random seed for full reproducibility.
        base_demand_range: (min, max) baseline daily item demand.
        price_range: (min, max) base price per unit.
        promo_probability: Daily probability of an item being on promotion.
    """
    start_date: str = "2024-01-01"
    num_days: int = 365
    num_stores: int = 3
    num_items: int = 10
    seed: int = 42
    base_demand_range: tuple[float, float] = (5.0, 35.0)
    price_range: tuple[float, float] = (2.50, 45.0)
    promo_probability: float = 0.12


def generate_synthetic_demand(config: SyntheticDemandConfig | None = None) -> pd.DataFrame:
    """Generate a reproducible synthetic retail demand dataset.
    
    Generative Formulation:
        Demand(store, item, t) ~ Poisson( lambda(store, item, t) )
        where:
            lambda = BaseDemand(item)
                   * StoreMultiplier(store)
                   * DayOfWeekMultiplier(t)
                   * MonthMultiplier(t)
                   * (1 + Trend * t)
                   * PriceElasticityFactor(price_t / base_price)
                   * PromoBoost(promo_t)
                   + epsilon
                   
    Returns:
        pd.DataFrame containing columns:
            [date, store_id, item_id, category, sell_price, on_promotion, sales, is_synthetic]
    """
    cfg = config or SyntheticDemandConfig()
    rng = np.random.default_rng(cfg.seed)

    start = date.fromisoformat(cfg.start_date)
    dates = [start + timedelta(days=i) for i in range(cfg.num_days)]

    store_ids = [f"STORE_{i+1:02d}" for i in range(cfg.num_stores)]
    categories = ["FOODS", "HOUSEHOLD", "BEVERAGES"]
    
    # Store-specific scaling factors (e.g. store size / footfall)
    store_multipliers = {
        store_id: rng.uniform(0.75, 1.35) for store_id in store_ids
    }

    # Item metadata: item_id, category, base_demand, base_price, elasticity
    items_meta: list[dict[str, object]] = []
    for i in range(cfg.num_items):
        item_id = f"ITEM_{i+1:03d}"
        cat = categories[i % len(categories)]
        base_demand = rng.uniform(*cfg.base_demand_range)
        base_price = round(float(rng.uniform(*cfg.price_range)), 2)
        elasticity = rng.uniform(-1.5, -0.6)  # Standard negative price elasticity
        items_meta.append({
            "item_id": item_id,
            "category": cat,
            "base_demand": base_demand,
            "base_price": base_price,
            "elasticity": elasticity,
        })

    # Day-of-week multipliers: Mon=0 -> Sun=6 (retail spikes Friday-Sunday)
    dow_multipliers = np.array([0.85, 0.90, 0.92, 0.95, 1.15, 1.30, 1.20])

    # Monthly seasonal curve (annual cycle peaking in winter holiday season)
    month_multipliers = np.array([
        0.90, 0.92, 0.98, 1.00, 1.02, 1.05, 1.04, 1.03, 1.01, 1.06, 1.15, 1.25
    ])

    records: list[dict[str, object]] = []

    for d_idx, dt in enumerate(dates):
        dow = dt.weekday()
        dow_mult = dow_multipliers[dow]
        month_mult = month_multipliers[dt.month - 1]
        trend_mult = 1.0 + (0.05 * (d_idx / cfg.num_days))  # mild 5% annual growth

        for store_id in store_ids:
            store_mult = store_multipliers[store_id]

            for item in items_meta:
                item_id = str(item["item_id"])
                cat = str(item["category"])
                base_demand = float(item["base_demand"])
                base_price = float(item["base_price"])
                elasticity = float(item["elasticity"])

                # Promotion state (Bernoulli)
                on_promo = int(rng.random() < cfg.promo_probability)
                
                # Price discount when on promotion: 10% to 30% off
                if on_promo:
                    discount = rng.uniform(0.10, 0.30)
                    sell_price = round(base_price * (1.0 - discount), 2)
                    promo_boost = 1.35  # +35% demand lift
                else:
                    price_noise = rng.uniform(-0.02, 0.02)
                    sell_price = round(base_price * (1.0 + price_noise), 2)
                    promo_boost = 1.0

                # Price elasticity effect: (P / P_0) ^ elasticity
                price_ratio = max(0.5, sell_price / base_price)
                price_effect = price_ratio ** elasticity

                # Expected demand rate lambda
                expected_lambda = (
                    base_demand
                    * store_mult
                    * dow_mult
                    * month_mult
                    * trend_mult
                    * price_effect
                    * promo_boost
                )
                expected_lambda = max(0.1, expected_lambda)

                # Observed sales from Poisson with gamma heterogeneity (Negative Binomial)
                sales = int(rng.poisson(lam=expected_lambda))

                records.append({
                    "date": dt.isoformat(),
                    "store_id": store_id,
                    "item_id": item_id,
                    "category": cat,
                    "sell_price": float(sell_price),
                    "on_promotion": int(on_promo),
                    "sales": int(sales),
                    "is_synthetic": True,
                })

    df = pd.DataFrame(records)
    # Ensure strict datetime sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    
    logger.info(
        "Generated synthetic retail demand: %d rows (%s to %s) across %d stores, %d items",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
        cfg.num_stores,
        cfg.num_items,
    )
    return df
