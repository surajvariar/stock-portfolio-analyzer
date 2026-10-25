from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from backend.market_hours import drop_incomplete_series
from backend.recommendation_agent import generate_ollama_recommendation
from backend.risk_scoring.beta_calculator import (
    beta_risk_score,
    estimate_beta,
    to_date_indexed,
)
from backend.risk_scoring.composite_risk_score import composite_risk_label
from backend.risk_scoring.liquidity_flagger import (
    liquidity_flag,
    liquidity_risk_score,
    median_daily_turnover,
)
from backend.risk_scoring.volatility_calculator import (
    annualized_volatility,
    volatility_risk_score,
)

logger = logging.getLogger(__name__)

NSE_ETF_URL = "https://www.nseindia.com/api/etf"
ETF_HISTORY_DAYS = 1095
ETF_ANALYSIS_WORKERS = max(1, int(os.getenv("ETF_ANALYSIS_WORKERS", "4")))
ETF_RISK_WEIGHTS = {
    "volatility": 1 / 6,
    "beta": 1 / 6,
    "liquidity": 1 / 6,
    "sortino": 1 / 6,
    "max_drawdown": 1 / 6,
    "tracking_error": 1 / 6,
}
# Scanning every ETF through the language model is slow; the plain-language
# templates are used unless this is switched on.
ETF_USE_LLM = os.getenv("ETF_USE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(record.get(key))
        if value is not None:
            return value
    return None


def is_small_cap_etf(record: dict[str, Any]) -> bool:
    underlying = str(record.get("underlying") or "").casefold()
    return "smallcap" in underlying or "small cap" in underlying


def infer_etf_asset_class(underlying: str) -> str:
    value = underlying.casefold()
    if any(word in value for word in ("gold", "silver", "commodity", "bullion")):
        return "Commodity"
    if any(word in value for word in ("bond", "g-sec", "gsec", "government securities", "liquid", "overnight", "rate index")):
        return "Debt"
    if any(word in value for word in ("nasdaq", "hang seng", "s&p 500", "nyse", "msci", "global")):
        return "International"
    if "hybrid" in value or "50:50" in value:
        return "Hybrid"
    return "Equity"


def normalize_nse_etf(record: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(record.get("symbol") or "").strip().upper()
    if not symbol:
        return None

    underlying = str(record.get("assets") or "").strip()
    return {
        "symbol": symbol,
        "underlying": underlying or None,
        "asset_class": infer_etf_asset_class(underlying),
        "last_price": _number(record.get("ltP")),
        "nav": _number(record.get("nav")),
        "traded_value": _number(record.get("trdVal")),
        "volume": _number(record.get("qty")),
        "change_30d_pct": _number(record.get("perChange30d")),
        "change_365d_pct": _number(record.get("perChange365d")),
        "inav": _first_number(record, "iNav", "iNAV", "inav"),
        "portfolio_turnover_ratio": _first_number(
            record, "portfolioTurnoverRatio", "portfolio_turnover_ratio", "turnoverRatio"
        ),
        "yield_pct": _first_number(record, "yield", "yieldPct", "yield_pct"),
        "repo_rate_pct": _first_number(record, "repoRate", "repo_rate", "repo_rate_pct"),
    }


def fetch_nse_etf_universe() -> tuple[list[dict[str, Any]], str | None]:
    response = requests.get(
        NSE_ETF_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nseindia.com/market-data/exchange-traded-funds-etf",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise TypeError("NSE ETF response did not contain a data list.")

    normalized = [normalize_nse_etf(record) for record in records if isinstance(record, dict)]
    return [record for record in normalized if record is not None], payload.get("timestamp")


def fetch_etf_market_history(symbol: str) -> pd.DataFrame:
    end = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    start = end - timedelta(days=ETF_HISTORY_DAYS)
    history = yf.Ticker(f"{symbol}.NS").history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
    )
    if history.empty or "Close" not in history or "Volume" not in history:
        return pd.DataFrame(columns=["Close", "Volume"])
    closes = history["Close"].dropna()
    if isinstance(closes.index, pd.DatetimeIndex) and closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
        history.index = history.index.tz_localize(None)
    complete_dates = drop_incomplete_series(closes).index
    return history.loc[complete_dates, ["Close", "Volume"]].dropna(subset=["Close"])


PLAIN_NOT_ENOUGH_HISTORY = "There is not enough price history to rate it fairly yet."
PLAIN_NO_PRICE_DATA = "We could not get its price data right now, so it was not rated."


def _sortino_ratio(prices: pd.Series, window: int = 252, min_observations: int = 60) -> float | None:
    returns = prices.dropna().pct_change().dropna().tail(window)
    if len(returns) < min_observations:
        return None
    downside_deviation = float(returns.clip(upper=0).pow(2).mean() ** 0.5)
    if not math.isfinite(downside_deviation) or downside_deviation == 0:
        return None
    ratio = float(returns.mean() / downside_deviation * (252**0.5))
    return ratio if math.isfinite(ratio) else None


def _max_drawdown_pct(prices: pd.Series, window: int = 252, min_observations: int = 60) -> float | None:
    series = prices.dropna().tail(window + 1)
    if len(series) - 1 < min_observations:
        return None
    drawdown = series / series.cummax() - 1
    maximum = float(-drawdown.min() * 100)
    return maximum if math.isfinite(maximum) else None


def _tracking_error_pct(
    prices: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    min_observations: int = 60,
) -> float | None:
    etf_returns = to_date_indexed(prices.pct_change().dropna())
    benchmark_returns = to_date_indexed(market_returns)
    if etf_returns.empty or benchmark_returns.empty:
        return None
    active_returns = pd.concat([etf_returns, benchmark_returns], axis=1, join="inner").dropna().tail(window)
    if len(active_returns) < min_observations:
        return None
    tracking_error = float(active_returns.iloc[:, 0].sub(active_returns.iloc[:, 1]).std(ddof=1) * (252**0.5) * 100)
    return tracking_error if math.isfinite(tracking_error) else None


def _sortino_risk_score(ratio: float | None) -> int | None:
    if ratio is None:
        return None
    if ratio >= 1.5:
        return 20
    if ratio >= 0.5:
        return 50
    if ratio > 0:
        return 80
    return 100


def _drawdown_risk_score(drawdown_pct: float | None) -> int | None:
    if drawdown_pct is None:
        return None
    if drawdown_pct < 15:
        return 20
    if drawdown_pct <= 30:
        return 50
    return 80


def _tracking_error_risk_score(tracking_error_pct: float | None) -> int | None:
    if tracking_error_pct is None:
        return None
    if tracking_error_pct < 5:
        return 20
    if tracking_error_pct <= 10:
        return 50
    return 80


def score_equity_etf(
    prices: pd.Series,
    market_returns: pd.Series,
    median_turnover: float | None,
    risk_tier: str,
    portfolio_summary: dict[str, Any],
    *,
    symbol: str = "This ETF",
    market_sentiment_label: str | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    volatility = annualized_volatility(prices)
    if volatility is None:
        return {
            "status": "skipped",
            "reason": "At least 60 daily returns are required to estimate ETF volatility.",
            "plain_reason": PLAIN_NOT_ENOUGH_HISTORY,
        }

    beta = estimate_beta(prices.pct_change().dropna(), market_returns)
    if beta.beta is None:
        return {
            "status": "skipped",
            "reason": f"Only {beta.observations} overlapping daily returns with the Nifty 50; at least 60 are required for beta.",
            "plain_reason": PLAIN_NOT_ENOUGH_HISTORY,
        }

    is_illiquid = liquidity_flag(median_turnover)
    sortino = _sortino_ratio(prices)
    max_drawdown = _max_drawdown_pct(prices)
    tracking_error = _tracking_error_pct(prices, market_returns)
    risk_components = {
        "volatility": volatility_risk_score(volatility),
        "beta": beta_risk_score(abs(beta.beta)),
        "liquidity": liquidity_risk_score(is_illiquid),
        "sortino": _sortino_risk_score(sortino),
        "max_drawdown": _drawdown_risk_score(max_drawdown),
        "tracking_error": _tracking_error_risk_score(tracking_error),
    }
    available_weight = sum(
        ETF_RISK_WEIGHTS[name] for name, component in risk_components.items() if component is not None
    )
    score = sum(
        ETF_RISK_WEIGHTS[name] * component
        for name, component in risk_components.items()
        if component is not None
    ) / available_weight
    composite = round(score, 2)
    label = composite_risk_label(score)
    metrics = {
        "risk_tier": risk_tier,
        "composite_score": composite,
        "risk_label": label,
        "volatility_1y": round(volatility, 2),
        "beta_1y": round(beta.beta, 3),
        "sortino_1y": round(sortino, 3) if sortino is not None else None,
        "max_drawdown_1y_pct": round(max_drawdown, 2) if max_drawdown is not None else None,
        "tracking_error_1y_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "liquidity_flag": is_illiquid,
        "portfolio_hhi": _number(portfolio_summary.get("portfolio_hhi")),
        "sector_hhi": _number(portfolio_summary.get("sector_hhi")),
        "market_cap_hhi": _number(portfolio_summary.get("market_cap_hhi")),
        "portfolio_risk_score": _number(portfolio_summary.get("weighted_average_risk_score")),
        "profile_alignment": portfolio_summary.get("profile_alignment"),
        "market_sentiment_label": market_sentiment_label,
        # Overall market mood is only meaningful for equity ETFs (all that is scored here).
        "use_market_mood": market_sentiment_label is not None,
    }
    advice = generate_ollama_recommendation(symbol, metrics, risk_tier, held=False, kind="ETF", use_llm=use_llm)
    return {
        "status": "scored",
        "volatility_1y": round(volatility, 2),
        "beta_1y": round(beta.beta, 3),
        "sortino_1y": round(sortino, 3) if sortino is not None else None,
        "max_drawdown_1y_pct": round(max_drawdown, 2) if max_drawdown is not None else None,
        "tracking_error_1y_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "beta_observations": beta.observations,
        "history_days": len(prices.dropna()),
        "liquidity_flag": is_illiquid,
        "composite_score": composite,
        "risk_label": label,
        "recommendation": advice["recommendation"],
        "reasoning": advice["reasoning"],
        "explanation": advice["explanation"],
        "portfolio_effect": advice.get("portfolio_effect"),
        "model_status": advice["model_status"],
        "market_sentiment_label": market_sentiment_label,
        "portfolio_hhi": _number(portfolio_summary.get("portfolio_hhi")),
        "sector_hhi": _number(portfolio_summary.get("sector_hhi")),
        "market_cap_hhi": _number(portfolio_summary.get("market_cap_hhi")),
    }


def analyze_equity_etfs(
    records: list[dict[str, Any]],
    market_prices: pd.Series,
    risk_tier: str,
    portfolio_summary: dict[str, Any],
    market_sentiment_label: str | None = None,
) -> list[dict[str, Any]]:
    market_returns = market_prices.pct_change().dropna()
    candidates = [
        record
        for record in records
        if record["asset_class"] == "Equity" and is_small_cap_etf(record)
    ]

    def analyze(record: dict[str, Any]) -> dict[str, Any]:
        try:
            history = fetch_etf_market_history(record["symbol"])
            prices = history["Close"]
            median_turnover = median_daily_turnover(prices, history["Volume"])
            result = score_equity_etf(
                prices,
                market_returns,
                median_turnover,
                risk_tier,
                portfolio_summary,
                symbol=record["symbol"],
                market_sentiment_label=market_sentiment_label,
                use_llm=ETF_USE_LLM,
            )
            result["median_daily_turnover"] = median_turnover
            yield_pct = _number(record.get("yield_pct"))
            repo_rate_pct = _number(record.get("repo_rate_pct"))
            last_price = _number(record.get("last_price"))
            inav = _number(record.get("inav"))
            result["yield_repo_gap_pct"] = (
                round(yield_pct - repo_rate_pct, 2)
                if yield_pct is not None and repo_rate_pct is not None
                else None
            )
            result["inav_premium_discount_pct"] = (
                round((last_price / inav - 1) * 100, 2)
                if last_price is not None and inav is not None and inav > 0
                else None
            )
        except Exception as exc:  # noqa: BLE001 - one provider failure should not stop the universe scan
            logger.warning("ETF analysis failed for %s (%s)", record["symbol"], type(exc).__name__)
            result = {
                "status": "skipped",
                "reason": "Price history could not be fetched from the market-data provider.",
                "plain_reason": PLAIN_NO_PRICE_DATA,
            }
        return {**record, **result}

    with ThreadPoolExecutor(max_workers=ETF_ANALYSIS_WORKERS) as pool:
        results = list(pool.map(analyze, candidates))
    return sorted(
        results,
        key=lambda item: (
            item.get("status") != "scored",
            {"BUY": 0, "WATCH": 1, "AVOID": 2}.get(item.get("recommendation"), 3),
            item.get("composite_score", 101),
            -(item.get("traded_value") or 0),
            item["symbol"],
        ),
    )