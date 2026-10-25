from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import pandas as pd

from backend.risk_scoring.beta_calculator import to_date_indexed

# Reviews younger than this are noise, not evidence.
MIN_REVIEW_DAYS = 5
# A HOLD "worked" if the stock stayed within this many percentage points of Nifty.
HOLD_TOLERANCE_PCT = 5.0


def _benchmark_return(benchmark_prices: pd.Series | None, entry_date: Any, latest_date: Any) -> float | None:
    if benchmark_prices is None or entry_date is None or latest_date is None:
        return None
    series = to_date_indexed(benchmark_prices)
    if series.empty:
        return None
    entry = series.asof(pd.Timestamp(entry_date))
    latest = series.asof(pd.Timestamp(latest_date))
    if pd.isna(entry) or pd.isna(latest) or float(entry) <= 0:
        return None
    return round((float(latest) / float(entry) - 1) * 100, 2)


def _outcome(recommendation: str | None, reference_return: float, holding_days: int) -> str:
    """Judge the call against `reference_return` (excess vs Nifty when available)."""
    if holding_days < MIN_REVIEW_DAYS:
        return "too_early"
    if recommendation in {"BUY", "INCREASE"}:
        return "hit" if reference_return > 0 else "miss"
    if recommendation == "SELL":
        return "hit" if reference_return < 0 else "miss"
    if recommendation == "HOLD":
        return "hit" if abs(reference_return) <= HOLD_TOLERANCE_PCT else "miss"
    return "unknown"


def build_review_rows(
    recommendations: Iterable[dict[str, Any]],
    benchmark_prices: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Forward-return review of past recommendations.

    Price returns only (not dividend-adjusted). When `benchmark_prices` (Nifty 50
    closes) is supplied, each call is judged on EXCESS return over the benchmark
    across the same window, so a rising market does not make every BUY look good
    and every SELL look bad.
    """
    rows: list[dict[str, Any]] = []
    for recommendation in recommendations:
        entry_price = recommendation.get("entry_price")
        latest_price = recommendation.get("latest_price")
        entry_date = recommendation.get("entry_date")
        latest_date = recommendation.get("latest_date")

        forward_return = None
        if entry_price is not None and latest_price is not None and float(entry_price) > 0:
            forward_return = round((float(latest_price) / float(entry_price) - 1) * 100, 2)

        holding_days = None
        if isinstance(entry_date, date) and isinstance(latest_date, date):
            holding_days = (latest_date - entry_date).days

        benchmark_return = _benchmark_return(benchmark_prices, entry_date, latest_date)
        excess_return = None
        if forward_return is not None and benchmark_return is not None:
            excess_return = round(forward_return - benchmark_return, 2)

        outcome = None
        outcome_basis = None
        if forward_return is not None and holding_days is not None:
            reference = excess_return if excess_return is not None else forward_return
            outcome_basis = "excess_vs_nifty" if excess_return is not None else "absolute_return"
            outcome = _outcome(recommendation.get("recommendation"), reference, holding_days)

        rows.append(
            {
                "stock_symbol": recommendation.get("stock_symbol"),
                "recommendation": recommendation.get("recommendation"),
                "reasoning": recommendation.get("reasoning"),
                "inputs_snapshot": recommendation.get("inputs_snapshot", {}),
                "recommended_at": recommendation.get("recommended_at"),
                "entry_date": entry_date,
                "entry_price": float(entry_price) if entry_price is not None else None,
                "latest_date": latest_date,
                "latest_price": float(latest_price) if latest_price is not None else None,
                "holding_days": holding_days,
                "forward_price_return_pct": forward_return,
                "benchmark_return_pct": benchmark_return,
                "excess_return_pct": excess_return,
                "outcome": outcome,
                "outcome_basis": outcome_basis,
                "review_status": "ready" if forward_return is not None else "awaiting_price_data",
            }
        )
    return rows
