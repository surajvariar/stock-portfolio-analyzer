from __future__ import annotations

import math
from typing import Any

import pandas as pd

TRADING_DAYS = 252
# The metric is stored as `volatility_1y`: use the trailing ~1 trading year.
VOLATILITY_WINDOW = 252
# Fewer daily returns than this is too little data to score honestly.
MIN_RETURN_OBSERVATIONS = 60


def annualized_volatility(
    prices: Any,
    window: int | None = VOLATILITY_WINDOW,
    min_observations: int = MIN_RETURN_OBSERVATIONS,
) -> float | None:
    """Annualized volatility (%) from a raw price series.

    Takes PRICES, not pre-differenced returns - it computes daily returns
    internally via pct_change(). Passing already-differenced returns in here
    will silently double-difference the data and produce a meaningless
    number, so callers must pass the price series itself.

    Uses the trailing `window` prices. Returns None (not 0.0, which would read
    as "Low risk") when there are fewer than `min_observations` daily returns
    or the result is not finite.
    """
    if prices is None:
        return None

    series = prices.dropna() if isinstance(prices, pd.Series) else pd.Series(prices).dropna()
    series = series[series > 0]
    if series.empty:
        return None

    if isinstance(series.index, pd.DatetimeIndex) and not series.index.is_monotonic_increasing:
        series = series.sort_index()
    if window:
        series = series.tail(window + 1)

    daily_returns = series.pct_change().dropna()
    if len(daily_returns) < min_observations:
        return None

    volatility = float(daily_returns.std(ddof=1) * (TRADING_DAYS**0.5) * 100)
    return volatility if math.isfinite(volatility) else None


def volatility_risk_bucket(volatility: float | None) -> str:
    if volatility is None or not math.isfinite(volatility):
        return "Unknown"
    if volatility < 20:
        return "Low"
    if volatility <= 35:
        return "Moderate"
    return "High"


def volatility_risk_score(volatility: float | None) -> int:
    bucket = volatility_risk_bucket(volatility)
    return {"Low": 20, "Moderate": 50, "High": 80}.get(bucket, 50)
