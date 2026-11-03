"""Window Slicing and Stateful Hysteresis Tracking.

Implements:
- Tumbling and sliding window iterators over chronological panel data.
- k-of-n persistent window escalation.
- Hysteresis deadbands to prevent alert flapping.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class WindowSlice:
    """Chronological window partition."""
    window_id: str
    start_date: str
    end_date: str
    df: pd.DataFrame


def slice_tumbling_windows(
    df: pd.DataFrame,
    date_col: str = "date",
    window_size_days: int = 14,
) -> Generator[WindowSlice, None, None]:
    """Slice DataFrame into non-overlapping contiguous tumbling windows."""
    df_sorted = df.sort_values(date_col).copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    
    min_date = df_sorted[date_col].min().date()
    max_date = df_sorted[date_col].max().date()

    curr_start = min_date
    window_idx = 0
    while curr_start <= max_date:
        curr_end = curr_start + timedelta(days=window_size_days - 1)
        mask = (df_sorted[date_col].dt.date >= curr_start) & (df_sorted[date_col].dt.date <= curr_end)
        w_df = df_sorted[mask].copy().reset_index(drop=True)
        if not w_df.empty:
            yield WindowSlice(
                window_id=f"TUMBLING_{window_idx:03d}_{curr_start}_{curr_end}",
                start_date=curr_start.isoformat(),
                end_date=curr_end.isoformat(),
                df=w_df,
            )
            window_idx += 1
        curr_start = curr_end + timedelta(days=1)


def slice_sliding_windows(
    df: pd.DataFrame,
    date_col: str = "date",
    window_size_days: int = 14,
    step_days: int = 7,
) -> Generator[WindowSlice, None, None]:
    """Slice DataFrame into overlapping sliding windows."""
    df_sorted = df.sort_values(date_col).copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

    min_date = df_sorted[date_col].min().date()
    max_date = df_sorted[date_col].max().date()

    curr_start = min_date
    window_idx = 0
    while curr_start <= max_date:
        curr_end = curr_start + timedelta(days=window_size_days - 1)
        mask = (df_sorted[date_col].dt.date >= curr_start) & (df_sorted[date_col].dt.date <= curr_end)
        w_df = df_sorted[mask].copy().reset_index(drop=True)
        if not w_df.empty:
            yield WindowSlice(
                window_id=f"SLIDING_{window_idx:03d}_{curr_start}_{curr_end}",
                start_date=curr_start.isoformat(),
                end_date=curr_end.isoformat(),
                df=w_df,
            )
            window_idx += 1
        curr_start = curr_start + timedelta(days=step_days)


@dataclass
class HysteresisStateTracker:
    """Stateful k-of-n escalation and hysteresis deadband monitor.
    
    Rules:
    - Escalation to ALERT: At least k of the last n windows breach the alert threshold.
    - De-escalation to NORMAL: Value must fall below (threshold - hysteresis_margin)
      for m consecutive windows.
    """
    k: int = 3
    n: int = 5
    m_recovery: int = 2
    hysteresis_margin: float = 0.03
    threshold: float = 0.10

    # Internal state: key -> deque of recent decisions (bool)
    _window_history: dict[str, deque[bool]] = field(default_factory=dict)
    _recent_values: dict[str, deque[float]] = field(default_factory=dict)
    _alert_state: dict[str, bool] = field(default_factory=dict)
    _recovery_streak: dict[str, int] = field(default_factory=dict)

    def update(self, key: str, value: float) -> tuple[bool, str]:
        """Update tracker with the latest window metric and return current state.
        
        Args:
            key: Unique identifier (e.g. 'sales_lag_1' or 'model_mae').
            value: Observed metric in the current window (e.g. PSI or effect size).
            
        Returns:
            Tuple of (is_alerting, status_message)
        """
        if key not in self._window_history:
            self._window_history[key] = deque(maxlen=self.n)
            self._recent_values[key] = deque(maxlen=self.n)
            self._alert_state[key] = False
            self._recovery_streak[key] = 0

        hist = self._window_history[key]
        vals = self._recent_values[key]
        is_breaching = (value >= self.threshold)
        
        hist.append(is_breaching)
        vals.append(value)

        currently_alerting = self._alert_state[key]
        breach_count = sum(hist)

        if not currently_alerting:
            # Check for escalation: k of n breached
            if len(hist) >= self.k and breach_count >= self.k:
                self._alert_state[key] = True
                self._recovery_streak[key] = 0
                return True, f"Alert escalated: {breach_count}/{len(hist)} recent windows breached threshold ({self.threshold})"
            return False, f"Stable: {breach_count}/{len(hist)} windows breached (threshold={self.threshold})"

        # If currently alerting, check for hysteresis recovery
        # Must fall strictly below (threshold - hysteresis_margin)
        recovery_ceiling = self.threshold - self.hysteresis_margin
        if value < recovery_ceiling:
            self._recovery_streak[key] += 1
            if self._recovery_streak[key] >= self.m_recovery:
                self._alert_state[key] = False
                self._recovery_streak[key] = 0
                return False, f"Alert cleared: Value ({value:.4f}) stayed below recovery ceiling ({recovery_ceiling:.4f}) for {self.m_recovery} consecutive windows"
            return True, f"Alert cooling down: {self._recovery_streak[key]}/{self.m_recovery} recovery windows achieved"
        else:
            self._recovery_streak[key] = 0
            return True, f"Alert sustained: Current value ({value:.4f}) remains above recovery ceiling ({recovery_ceiling:.4f})"
