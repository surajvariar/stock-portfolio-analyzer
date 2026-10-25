from __future__ import annotations

# Maximum swing (points) each signal can contribute around the neutral 50.
INDEX_WEIGHT = 25.0
VIX_WEIGHT = 20.0
FLOW_WEIGHT = 15.0
SMA50_WEIGHT = 5.0
SMA200_WEIGHT = 10.0


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _vix_contribution(vix_value: float) -> float:
    """Smooth replacement for the old 15 / 20 step function.

    +20 at VIX <= 13, falling linearly to -20 at VIX >= 25 (about +5 at 17.5, the
    middle of the old "calm" band), so 20.0 and 20.1 no longer differ by 25 points.
    """
    return _clamp(VIX_WEIGHT - (vix_value - 13.0) * (2 * VIX_WEIGHT / 12.0), -VIX_WEIGHT, VIX_WEIGHT)


def composite_sentiment_label(score: float) -> str:
    """Label for the blended (quant + news) sentiment score."""
    return "Positive" if score >= 65 else "Negative" if score <= 35 else "Neutral"


def compute_market_sentiment(
    index_return: float | None,
    vix_value: float | None,
    fii_net_flow: float | None = None,
    dii_net_flow: float | None = None,
    above_sma_50: bool | None = None,
    above_sma_200: bool | None = None,
) -> dict[str, float | str]:
    """Compute a 0-100 market sentiment score and a label.

    This is the deterministic branch of the sentiment layer. The LLM never
    computes these numbers; it only reasons over them.

    Each available signal contributes a signed amount up to its weight. The sum
    is divided by the total weight of the signals that were actually available
    and mapped onto 0-100 around 50, so (a) the score only reaches 0 or 100 when
    every available signal is at its extreme, instead of clamping once the raw
    sum overshoots, and (b) a missing input reweights the rest instead of being
    silently treated as neutral. `coverage` reports how much of the full signal
    set was present (1.0 = all five).
    """
    delta = 0.0
    available = 0.0

    if index_return is not None:
        available += INDEX_WEIGHT
        delta += INDEX_WEIGHT * _clamp(index_return * 10, -1.0, 1.0)

    if vix_value is not None:
        available += VIX_WEIGHT
        delta += _vix_contribution(vix_value)

    if fii_net_flow is not None and dii_net_flow is not None:
        available += FLOW_WEIGHT
        if fii_net_flow > 0 and dii_net_flow > 0:
            delta += FLOW_WEIGHT
        elif fii_net_flow < 0 and dii_net_flow < 0:
            delta -= FLOW_WEIGHT

    if above_sma_50 is not None:
        available += SMA50_WEIGHT
        delta += SMA50_WEIGHT if above_sma_50 else -SMA50_WEIGHT

    if above_sma_200 is not None:
        available += SMA200_WEIGHT
        delta += SMA200_WEIGHT if above_sma_200 else -SMA200_WEIGHT

    full_weight = INDEX_WEIGHT + VIX_WEIGHT + FLOW_WEIGHT + SMA50_WEIGHT + SMA200_WEIGHT
    score = 50.0 if available == 0 else _clamp(50.0 + 50.0 * (delta / available))

    # Cut-offs at 65 / 35 line up with composite_sentiment_label().
    if score >= 65:
        label = "Bullish"
    elif score >= 50:
        label = "Neutral"
    elif score >= 35:
        label = "Cautious"
    else:
        label = "Bearish"

    return {
        "score": round(score, 2),
        "label": label,
        "coverage": round(available / full_weight, 2),
    }
