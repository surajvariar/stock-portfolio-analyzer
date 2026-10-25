from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd

# Policy choice (not a SEBI standard): a stock is flagged illiquid when its
# median daily traded value over the last LIQUIDITY_WINDOW sessions is below
# this many rupees. Default Rs 1 crore. Override with LIQUIDITY_MIN_TURNOVER_INR.
MIN_MEDIAN_DAILY_TURNOVER_INR = float(os.getenv("LIQUIDITY_MIN_TURNOVER_INR", "10000000"))
LIQUIDITY_WINDOW = 60


def median_daily_turnover(closes: Any, volumes: Any, window: int = LIQUIDITY_WINDOW) -> float | None:
    """Median of (close x volume) over the trailing `window` sessions, in rupees.

    Traded value, not share count: a Rs 25,000 stock trading 3,000 shares a day
    is far more liquid than a Rs 4 stock trading 150,000 shares. Median (not
    mean) so a single block-deal spike does not hide thin trading, and a recent
    window so a liquidity collapse is not averaged away by 3 years of history.
    """
    if closes is None or volumes is None:
        return None
    frame = pd.concat([pd.Series(closes), pd.Series(volumes)], axis=1, join="inner").dropna()
    if frame.empty:
        return None
    frame.columns = ["close", "volume"]
    frame = frame.tail(window)
    turnover = (frame["close"] * frame["volume"]).median()
    return float(turnover) if math.isfinite(turnover) else None


def liquidity_flag(median_daily_turnover_inr: Any, trade_to_trade_flag: Any = False) -> bool:
    """True means "treat as illiquid". Unknown turnover is treated as illiquid."""
    if bool(trade_to_trade_flag):
        return True

    try:
        turnover = float(median_daily_turnover_inr)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(turnover):
        return True

    return turnover < MIN_MEDIAN_DAILY_TURNOVER_INR


def liquidity_risk_score(flag: bool) -> int:
    return 80 if flag else 20
