from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from backend.risk_questionnaire import TIER_RISK_LIMITS
from backend.risk_scoring.composite_risk_score import composite_risk_label


def _hhi_by_group(holdings: list[dict[str, Any]], group_key: str) -> float | None:
    """Herfindahl-Hirschman concentration index for one grouping (sector,
    market cap tier, ...).

    Computed over whatever subset of holdings has this metadata, not the
    whole portfolio: a single stock still missing `sector`/`market_cap_tier`
    (metadata import is a manual step, so partial coverage is the norm)
    would otherwise make the entire metric `None`. It is normalized against
    the value of the covered holdings instead.
    """
    grouped_values: dict[str, float] = {}
    covered_value = 0.0
    for holding in holdings:
        value = float(holding.get("position_value", 0) or 0)
        group = holding.get(group_key)
        if value <= 0 or not group:
            continue
        grouped_values[str(group)] = grouped_values.get(str(group), 0.0) + value
        covered_value += value
    if not grouped_values or covered_value <= 0:
        return None
    return round(sum((value / covered_value) ** 2 for value in grouped_values.values()), 4)


def group_weights_pct(holdings: Iterable[dict[str, Any]], group_key: str) -> dict[str, float]:
    """Share of TOTAL portfolio value (percent) held in each group. Holdings
    with no value for `group_key` are left out of the result but still count
    in the denominator."""
    items = [dict(item) for item in holdings]
    total = sum(float(item.get("position_value", 0) or 0) for item in items)
    if total <= 0:
        return {}
    weights: dict[str, float] = {}
    for item in items:
        group = item.get(group_key)
        value = float(item.get("position_value", 0) or 0)
        if not group or value <= 0:
            continue
        weights[str(group)] = weights.get(str(group), 0.0) + value / total * 100
    return {group: round(weight, 2) for group, weight in weights.items()}


def _empty_summary() -> dict[str, Any]:
    return {
        "weighted_average_risk_score": None,
        "portfolio_hhi": 0.0,
        "sector_hhi": None,
        "market_cap_hhi": None,
        "risk_label": "Low",
        "total_value": 0.0,
        "profile_alignment": "unavailable",
        "coverage_pct": 0.0,
        "scored_holdings": 0,
        "total_holdings": 0,
        "unscored_symbols": [],
    }


def compute_portfolio_summary(holdings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    holdings = [
        {**item, "position_value": item.get("position_value", item.get("quantity", 0))}
        for item in holdings
    ]
    if not holdings:
        return _empty_summary()

    total_value = sum(float(item.get("position_value", 0) or 0) for item in holdings)
    if total_value <= 0:
        summary = _empty_summary()
        summary["total_holdings"] = len(holdings)
        return summary

    scored_holdings = [item for item in holdings if item.get("risk_score") is not None]
    scored_value = sum(float(item.get("position_value", 0) or 0) for item in scored_holdings)
    weighted_average = None
    if scored_value > 0:
        weighted_average = sum(
            float(item.get("position_value", 0) or 0) * float(item["risk_score"])
            for item in scored_holdings
        ) / scored_value

    stock_hhi = round(
        sum((float(item.get("position_value", 0) or 0) / total_value) ** 2 for item in holdings),
        4,
    )
    sector_hhi = _hhi_by_group(holdings, "sector")
    market_cap_hhi = _hhi_by_group(holdings, "market_cap_tier")

    # Single source of truth for the Low/Moderate/High/Very High cut-offs.
    label = composite_risk_label(weighted_average) if weighted_average is not None else "Unavailable"

    risk_tier = next((item.get("risk_tier") for item in holdings if item.get("risk_tier")), None)
    profile_alignment = "unavailable"
    if weighted_average is not None and risk_tier in TIER_RISK_LIMITS:
        limit = TIER_RISK_LIMITS[risk_tier]
        profile_alignment = "above_profile" if weighted_average > limit else "within_profile"

    unscored_symbols = [
        str(item.get("symbol"))
        for item in holdings
        if item.get("risk_score") is None and item.get("symbol")
    ]

    return {
        "weighted_average_risk_score": round(weighted_average, 2) if weighted_average is not None else None,
        "portfolio_hhi": stock_hhi,
        "sector_hhi": sector_hhi,
        "market_cap_hhi": market_cap_hhi,
        "risk_label": label,
        "total_value": round(total_value, 2),
        "profile_alignment": profile_alignment,
        # Share of portfolio VALUE the risk average actually covers. Below 100
        # the weighted average describes only part of the portfolio.
        "coverage_pct": round(scored_value / total_value * 100, 2),
        "scored_holdings": len(scored_holdings),
        "total_holdings": len(holdings),
        "unscored_symbols": unscored_symbols,
    }
