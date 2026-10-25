"""Plain-language layer for recommendations.

Pure Python (no LLM, DB or network). It does three jobs:

  1. `plain_facts` turns the numeric metrics into short everyday-word facts.
     The LLM is shown ONLY these facts, never raw scores, so it has no
     jargon to echo back.
  2. `build_explanation` writes a template explanation per action. It is used
     when the model is unavailable, or when the model's text fails
     `validate_explanation` (jargon, decimals, or too long).
  3. `compose` joins headline / why / watch into the single `reasoning`
     string that is stored and shown.

The decision itself (BUY/SELL/HOLD...) never comes from here; it comes from
recommendation_policy. This module only words it.
"""

from __future__ import annotations

import re
from typing import Any

from backend.risk_questionnaire import TIER_RISK_LIMITS

DEFAULT_TIER = "Balanced"

# Words and phrases a non-expert would not understand. Matched case-insensitively
# on word boundaries, so "beta" does not trip on "better".
JARGON_TERMS = (
    "volatility", "volatile", "beta", "composite", "risk score", "score", "points",
    "tier", "deterministic", "policy", "benchmark", "nifty", "liquidity", "illiquid",
    "drawdown", "correlation", "sharpe", "alpha", "nav", "aum", "expense ratio",
    "tracking error", "hhi", "concentration", "overweight", "underweight",
    "sentiment", "quantitative", "annualized", "annualised", "standard deviation",
    "percentile", "rebalance", "rebalancing", "traded value", "turnover", "metric",
    "metrics", "threshold", "pro-forma", "pro forma", "weighted", "coverage",
    "unrealised", "unrealized", "pnl", "p&l", "llm", "inputs", "market cap",
    "large-cap", "mid-cap", "small-cap", "basis points", "bps",
)
_JARGON_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in sorted(JARGON_TERMS, key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
# 62.3, 0.85 and similar raw metric values are not meaningful to a layperson.
_DECIMAL_RE = re.compile(r"\d+\.\d+")

MAX_HEADLINE_CHARS = 140
MAX_WHY_CHARS = 380
MAX_WATCH_CHARS = 240


def find_jargon(text: str, *, allow_symbol: str | None = None) -> list[str]:
    cleaned = text or ""
    if allow_symbol:
        cleaned = re.sub(re.escape(allow_symbol), "", cleaned, flags=re.IGNORECASE)
    found = {match.group(1).lower() for match in _JARGON_RE.finditer(cleaned)}
    if _DECIMAL_RE.search(cleaned):
        found.add("decimal number")
    return sorted(found)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


# --- facts -------------------------------------------------------------------
def comfort_gap(metrics: dict[str, Any]) -> float | None:
    score = _num(metrics.get("composite_score"))
    if score is None:
        return None
    tier = metrics.get("risk_tier")
    limit = float(TIER_RISK_LIMITS.get(tier, TIER_RISK_LIMITS[DEFAULT_TIER]))
    return score - limit


def _comfort_phrase(gap: float | None) -> str | None:
    if gap is None:
        return None
    if gap > 15:
        return "much riskier than you said you are comfortable with"
    if gap > 0:
        return "a little riskier than you said you are comfortable with"
    if gap > -5:
        return "right around the risk you said you are comfortable with"
    return "well within the risk you said you are comfortable with"


def _swing_phrase(volatility: float | None) -> str | None:
    if volatility is None:
        return None
    if volatility < 20:
        return "its price usually moves up and down only gently"
    if volatility <= 35:
        return "its price has moderate ups and downs"
    return "its price swings up and down a lot"


def _market_link_phrase(beta: float | None) -> str | None:
    if beta is None:
        return None
    beta = abs(beta)
    if beta < 0.8:
        return "it tends to move less than the overall market"
    if beta <= 1.2:
        return "it tends to move about in step with the overall market"
    return "it tends to move more than the overall market"


def _selling_phrase(is_hard_to_sell: Any) -> str | None:
    if is_hard_to_sell is None:
        return None
    return (
        "it can be harder to sell quickly at a fair price"
        if is_hard_to_sell
        else "it is easy to buy and sell"
    )


def _share_phrase(position_weight_pct: float | None) -> str | None:
    if position_weight_pct is None:
        return None
    pct = max(1, round(position_weight_pct))
    if position_weight_pct >= 10:
        return f"it already makes up about {pct}% of your investments, reaching the 10% single-stock limit"
    if position_weight_pct >= 5:
        return f"it makes up about {pct}% of your investments, leaving moderate room below the 10% single-stock limit"
    return f"it only makes up about {pct}% of your investments, leaving plenty of room below the 10% single-stock limit"


def _industry_phrase(sector: str | None, sector_weight_pct: float | None) -> str | None:
    if sector_weight_pct is None:
        return None
    pct = max(1, round(sector_weight_pct))
    name = f" ({sector})" if sector else ""
    if sector_weight_pct >= 35:
        return f"about {pct}% of your money is already in the same industry{name}, reaching the 35% industry limit"
    return f"about {pct}% of your money is in this industry{name}, well within the 35% industry limit"


def _mood_phrase(label: str | None, subject: str) -> str | None:
    return {
        "Positive": f"{subject} looks upbeat right now",
        "Neutral": f"{subject} looks mixed right now",
        "Negative": f"{subject} looks gloomy right now",
    }.get(label or "")


def _gain_phrase(pnl_pct: float | None) -> str | None:
    if pnl_pct is None or abs(pnl_pct) < 1:
        return None
    return (
        f"you are currently up about {pnl_pct:.0f}% on what you paid"
        if pnl_pct > 0
        else f"you are currently down about {abs(pnl_pct):.0f}% on what you paid"
    )


def plain_facts(metrics: dict[str, Any]) -> dict[str, str]:
    """Everyday-word facts for the model. Missing data is simply omitted."""
    portfolio_gap = None
    portfolio_score = _num(metrics.get("portfolio_risk_score"))
    tier = metrics.get("risk_tier")
    if portfolio_score is not None:
        portfolio_gap = portfolio_score - float(TIER_RISK_LIMITS.get(tier, TIER_RISK_LIMITS[DEFAULT_TIER]))

    facts: dict[str, str | None] = {
        "main reason for this action": metrics.get("decision_driver"),
        "how risky it is for you": _comfort_phrase(comfort_gap(metrics)),
        "price ups and downs": _swing_phrase(_num(metrics.get("volatility_1y"))),
        "link to the overall market": _market_link_phrase(_num(metrics.get("beta_1y"))),
        "ease of selling": _selling_phrase(metrics.get("liquidity_flag")),
        "size in your investments": _share_phrase(_num(metrics.get("position_weight_pct"))),
        "same industry exposure": _industry_phrase(metrics.get("sector"), _num(metrics.get("sector_weight_pct"))),
        "overall market mood": _mood_phrase(metrics.get("market_sentiment_label"), "the overall market"),
        "news mood": _mood_phrase(metrics.get("stock_sentiment_label"), "recent news about it"),
        "your gain or loss so far (background only, not a reason to buy)": _gain_phrase(_num(metrics.get("unrealised_pnl_pct"))),
        "your whole portfolio": (
            None
            if portfolio_gap is None
            else (
                "your investments overall are riskier than you are comfortable with"
                if portfolio_gap > 0
                else "your investments overall are within the risk you are comfortable with"
            )
        ),
    }
    effect = metrics.get("portfolio_effect")
    if effect in {"lower", "same", "higher"}:
        facts["effect of adding it"] = {
            "lower": "adding a small amount would make your investments overall a little safer",
            "same": "adding a small amount would leave your overall risk about the same",
            "higher": "adding a small amount would make your investments overall a little riskier",
        }[effect]
    return {key: value for key, value in facts.items() if value}


# --- template explanations -----------------------------------------------------
def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _join(parts: list[str | None]) -> str:
    return " ".join(_sentence(part) for part in parts if part)


# Used when there are no facts to build a reason from, so "why" is never blank.
_DEFAULT_WHY = {
    "SELL": "Based on what you told us about the risk you are comfortable with, this looks like a poor fit right now",
    "INCREASE": "Based on what you told us about the risk you are comfortable with, there is room to add more",
    "HOLD": "Nothing here calls for a change right now",
    "BUY": "Based on what you told us about the risk you are comfortable with, this looks like a reasonable fit",
    "WATCH": "It is a close call, so it is better to wait",
    "AVOID": "Based on what you told us about the risk you are comfortable with, this looks like a poor fit right now",
}


def build_explanation(
    action: str,
    name: str,
    metrics: dict[str, Any],
    *,
    held: bool,
    block_reason: str | None = None,
) -> dict[str, str]:
    """Template headline / why / watch for one decision. `action` is the final
    action; `block_reason` is a plain-words reason an otherwise allowed
    increase/buy is being held back. Always returns three non-empty parts."""
    explanation = _build_explanation(action, name, metrics, held=held, block_reason=block_reason)
    if not explanation["why"].strip() and action in _DEFAULT_WHY:
        explanation["why"] = _sentence(_DEFAULT_WHY[action])
    return explanation


def _build_explanation(
    action: str,
    name: str,
    metrics: dict[str, Any],
    *,
    held: bool,
    block_reason: str | None = None,
) -> dict[str, str]:
    facts = plain_facts(metrics)
    comfort = facts.get("how risky it is for you")
    swings = facts.get("price ups and downs")
    market_link = facts.get("link to the overall market")
    selling = facts.get("ease of selling")
    share = facts.get("size in your investments")
    industry = facts.get("same industry exposure")
    market_mood = facts.get("overall market mood")
    effect = facts.get("effect of adding it")
    whole = facts.get("your whole portfolio")
    gain = facts.get("your gain or loss so far (background only, not a reason to buy)")
    gap = comfort_gap(metrics)

    if action == "SELL":
        return {
            "headline": f"Consider selling or trimming {name} because it is too risky for your comfort level.",
            "why": _join([
                f"{name} is {comfort}" if comfort else None,
                swings or market_link,
                share,
            ]),
            "watch": _join([
                gain,
                "You do not have to sell it all at once; trimming part of your position will immediately lower your overall portfolio risk",
            ]),
        }
    if action == "INCREASE":
        return {
            "headline": f"You can safely add a little more {name} because it is steady and you have room in your portfolio.",
            "why": _join([
                f"{name} is {comfort}" if comfort else None,
                swings or market_link,
                share or selling,
            ]),
            "watch": _join([
                industry,
                market_mood and f"Keep in mind that {market_mood}",
                "Add in small steps so no single stock grows above 10% of your investments",
            ]),
        }
    if action == "HOLD":
        if block_reason:
            headline = f"Keep holding {name} without adding more right now."
            why_parts = [
                f"{name} is {comfort}" if comfort else None,
                f"however, buying more is paused because {block_reason}",
            ]
        elif gap is not None and gap > 0:
            headline = f"Hold {name} for now, but watch it closely and do not add more."
            why_parts = [
                f"{name} is {comfort}",
                "it is close enough to your limit that you do not need to rush into selling, but any further rise in risk would tip it toward a sale",
                swings,
            ]
        else:
            headline = f"Keep holding your current {name} shares; your position is well balanced as is."
            why_parts = [
                f"{name} is {comfort}" if comfort else None,
                swings or market_link,
                share,
            ]
        return {
            "headline": headline,
            "why": _join(why_parts),
            "watch": _join([
                gain,
                None if block_reason else (market_mood and f"Also note that {market_mood}"),
                "Review again after the next risk update to make sure it stays within your comfort zone",
            ]),
        }
    if action == "BUY":
        return {
            "headline": f"{name} is a good fit to add because it matches your risk comfort and trades easily.",
            "why": _join([f"{name} is {comfort}" if comfort else None, effect, swings or selling]),
            "watch": _join([
                market_mood and f"Note that {market_mood}",
                "Consider buying in small steps rather than all at once",
            ]),
        }
    if action == "WATCH":
        reason = block_reason or (
            f"{name} is {comfort}" if comfort else "there is not enough clear evidence yet"
        )
        return {
            "headline": f"Wait on buying {name} for now.",
            "why": _join([f"It is on hold because {reason}" if block_reason else reason, whole if not held else None]),
            "watch": "Keep it on your watchlist and check again after the next update.",
        }
    if action == "AVOID":
        return {
            "headline": f"Skip {name} because its risk is far above what you are comfortable with.",
            "why": _join([f"{name} is {comfort}" if comfort else None, swings or market_link]),
            "watch": "Look for steadier options that stay within your risk comfort zone.",
        }
    return {
        "headline": f"We could not judge {name} yet.",
        "why": _sentence(block_reason or "There is not enough price history to rate it fairly"),
        "watch": "It will be rated automatically once enough history is available.",
    }


def compose(explanation: dict[str, str]) -> str:
    return " ".join(
        part.strip() for part in (explanation.get("headline"), explanation.get("why"), explanation.get("watch")) if part and part.strip()
    )


def validate_explanation(explanation: dict[str, str], *, symbol: str | None = None) -> list[str]:
    """Return the reasons a model-written explanation is unacceptable (empty = ok)."""
    problems: list[str] = []
    limits = {"headline": MAX_HEADLINE_CHARS, "why": MAX_WHY_CHARS, "watch": MAX_WATCH_CHARS}
    for key, limit in limits.items():
        text = (explanation.get(key) or "").strip()
        if not text:
            problems.append(f"{key} is empty")
            continue
        if len(text) > limit:
            problems.append(f"{key} is too long ({len(text)} > {limit})")
        jargon = find_jargon(text, allow_symbol=symbol)
        if jargon:
            problems.append(f"{key} uses technical words: {', '.join(jargon)}")
    return problems