from __future__ import annotations

from .beta_calculator import beta_risk_score
from .liquidity_flagger import liquidity_risk_score
from .market_cap_classifier import market_cap_risk_score
from .volatility_calculator import volatility_risk_score


def composite_risk_score(
    volatility: float,
    beta: float,
    market_cap_tier: str | None,
    liquidity_flag: bool,
) -> float:
    volatility_score = volatility_risk_score(volatility)
    beta_score = beta_risk_score(beta)
    market_cap_score = market_cap_risk_score(market_cap_tier)
    liquidity_score = liquidity_risk_score(liquidity_flag)

    weighted_total = (
        0.35 * volatility_score
        + 0.25 * beta_score
        + 0.25 * market_cap_score
        + 0.15 * liquidity_score
    )
    return float(max(0.0, min(100.0, weighted_total)))


def composite_risk_label(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Very High"
