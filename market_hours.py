"""Helpers for not trusting a daily bar until the NSE session has finished.

yfinance returns the in-progress bar for "today" while the market is open. Stored
as-is, that partial close/volume would look like a finished day. These helpers
drop such bars so only completed sessions reach the database and the analytics.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
# NSE's regular session ends 15:30 IST; the buffer lets the feed publish the
# final close before today's bar is treated as complete.
SESSION_FINAL_AFTER = time(16, 0)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_bar_final(bar_date: date, now: datetime | None = None) -> bool:
    now = now.astimezone(IST) if now is not None else now_ist()
    today = now.date()
    if bar_date < today:
        return True
    if bar_date > today:
        return False
    return now.time() >= SESSION_FINAL_AFTER


def drop_incomplete_bars(df: pd.DataFrame, date_col: str = "date", now: datetime | None = None) -> pd.DataFrame:
    """Drop rows whose `date_col` (python dates) is a session not yet finished."""
    if df.empty:
        return df
    keep = [is_bar_final(d, now) for d in df[date_col]]
    return df[keep].copy()


def drop_incomplete_series(series: pd.Series, now: datetime | None = None) -> pd.Series:
    """Same, for a Series whose index holds timestamps (tz-aware or naive)."""
    if series.empty:
        return series
    idx: Any = pd.DatetimeIndex(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    keep = np.array([is_bar_final(d, now) for d in idx.date], dtype=bool)
    return series[keep]
