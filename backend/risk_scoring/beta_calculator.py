from __future__ import annotations

import math
from typing import Any, NamedTuple

import pandas as pd

# The metric is stored as `beta_1y`, so it is computed over ~1 trading year.
BETA_WINDOW = 252
# Below this many date-aligned daily returns the estimate is too noisy to score.
MIN_BETA_OBSERVATIONS = 60


class BetaEstimate(NamedTuple):
    beta: float | None
    observations: int


def to_date_indexed(series: Any) -> pd.Series:
    """Return a copy indexed by tz-naive, midnight-normalized calendar dates.

    The NIFTY series from yfinance carries a tz-aware (Asia/Kolkata) index while
    prices read back from Postgres are tz-naive. Joining the two directly matches
    zero rows, so both sides are reduced to plain dates before aligning.
    """
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    s = s.dropna()
    if s.empty:
        return s
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.Series(s.to_numpy(), index=idx.normalize())
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def estimate_beta(
    stock_returns: Any,
    market_returns: Any,
    *,
    window: int | None = BETA_WINDOW,
    min_observations: int = MIN_BETA_OBSERVATIONS,
) -> BetaEstimate:
    """Beta over the trailing `window` date-aligned daily returns.

    Returns beta=None (never a made-up 1.0) when there is not enough overlapping
    history, so callers can skip the stock and report why.
    """
    if stock_returns is None or market_returns is None:
        return BetaEstimate(None, 0)

    stock = to_date_indexed(stock_returns)
    market = to_date_indexed(market_returns)
    if stock.empty or market.empty:
        return BetaEstimate(None, 0)

    aligned = pd.concat([stock, market], axis=1, join="inner").dropna()
    aligned.columns = ["stock", "market"]
    if window:
        aligned = aligned.tail(window)

    n = int(aligned.shape[0])
    if n < min_observations:
        return BetaEstimate(None, n)

    variance = aligned["market"].var(ddof=1)
    if not math.isfinite(variance) or variance == 0:
        return BetaEstimate(None, n)

    beta = float(aligned["stock"].cov(aligned["market"]) / variance)
    if not math.isfinite(beta):
        return BetaEstimate(None, n)
    return BetaEstimate(beta, n)


def calculate_beta(stock_returns: Any, market_returns: Any) -> float | None:
    return estimate_beta(stock_returns, market_returns).beta


def beta_risk_bucket(beta: float | None) -> str:
    if beta is None or not math.isfinite(beta):
        return "Unknown"
    if beta < 0.8:
        return "Low"
    if beta <= 1.2:
        return "Moderate"
    return "High"


def beta_risk_score(beta: float | None) -> int:
    bucket = beta_risk_bucket(beta)
    return {"Low": 20, "Moderate": 50, "High": 80}.get(bucket, 50)
