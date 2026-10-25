from __future__ import annotations

from typing import Any


def classify_market_cap_rank(rank: Any) -> str:
    if rank is None:
        return "Unknown"

    try:
        value = int(rank)
    except (TypeError, ValueError):
        return "Unknown"

    if value < 1:
        return "Unknown"
    if value <= 100:
        return "Large Cap"
    if value <= 250:
        return "Mid Cap"
    return "Small Cap"


def market_cap_risk_score(tier: str | None) -> int:
    mapping = {
        "Large Cap": 20,
        "Mid Cap": 50,
        "Small Cap": 80,
        "Unknown": 50,
    }
    return mapping.get(tier or "Unknown", 50)
