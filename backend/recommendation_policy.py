"""Deterministic recommendation policy.

Pure Python (no LLM, no DB, no network). It is used in two ways:
  * as the fallback when the model is unavailable, and
  * as a hard bound on the model: the LLM may explain and may be MORE
    conservative than this policy, but it can never take a risk-increasing
    action the numbers do not support.

`composite_score` is a RISK score: higher = riskier. The user's tier sets the
highest score considered acceptable (TIER_RISK_LIMITS).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.risk_questionnaire import TIER_RISK_LIMITS

# SELL only when the stock is clearly above the investor's limit.
SELL_EXCESS_POINTS = 15.0
# INCREASE needs the stock to sit below the limit AND be low-to-moderate risk in absolute terms.
# 38.0 is the composite score of a liquid Large-Cap stock with moderate volatility and moderate beta.
INCREASE_MARGIN_POINTS = 5.0
INCREASE_MAX_SCORE = 40.0
# Concentration guards that block adding more.
MAX_SINGLE_POSITION_PCT = 10.0
MAX_SECTOR_PCT = 35.0

DEFAULT_TIER = "Balanced"


# ETF candidates the investor does NOT hold yet (BUY / WATCH / AVOID).
ETF_AVOID_EXCESS_POINTS = 15.0
ETF_BUY_MARGIN_POINTS = 5.0
# How much of the portfolio a new ETF is assumed to become when estimating its
# effect on overall risk (an investment amount can replace this later).
DEFAULT_NEW_POSITION_SHARE = 0.05
# Overall risk moving by less than this many points counts as "about the same".
PORTFOLIO_EFFECT_TOLERANCE = 0.5


@dataclass(frozen=True)
class PolicyDecision:
    recommendation: str  # SELL | HOLD | INCREASE | BUY | WATCH | AVOID
    rationale: str
    risk_limit: float
    excess: float
    # Plain-words reason a good-looking action was held back (None if not blocked).
    block_reason: str | None = None
    # lower | same | higher: effect of adding a small amount on overall portfolio risk.
    portfolio_effect: str | None = None
    # Plain-words causal summary justifying the chosen action.
    decision_driver: str | None = None


def evaluate_policy(
    score: float,
    risk_tier: str,
    *,
    position_weight_pct: float | None = None,
    sector_weight_pct: float | None = None,
    market_sentiment_label: str | None = None,
    stock_sentiment_label: str | None = None,
    is_hard_to_sell: bool = False,
) -> PolicyDecision:
    tier = risk_tier if risk_tier in TIER_RISK_LIMITS else DEFAULT_TIER
    limit = float(TIER_RISK_LIMITS[tier])
    excess = float(score) - limit

    if excess > SELL_EXCESS_POINTS:
        return PolicyDecision(
            "SELL",
            f"Risk score {score:.1f} is {excess:.1f} points above the {tier} limit of {limit:.0f}.",
            limit,
            excess,
            decision_driver=(
                "sell or trim this stock because its risk is far above what you are comfortable with, "
                "so keeping it exposes your investments to steeper drops than you signed up for"
            ),
        )
    if excess > 0:
        return PolicyDecision(
            "HOLD",
            f"Risk score {score:.1f} is slightly above the {tier} limit of {limit:.0f} "
            f"(by {excess:.1f}); within the tolerance band, so monitor rather than sell.",
            limit,
            excess,
            decision_driver=(
                "hold for now without adding more because its risk has crept slightly above your comfort zone, "
                "though not enough to force an immediate sale"
            ),
        )

    if excess <= -INCREASE_MARGIN_POINTS and score <= INCREASE_MAX_SCORE:
        blockers: list[str] = []
        plain_blockers: list[str] = []
        if is_hard_to_sell:
            blockers.append("stock is flagged as illiquid or trade-to-trade")
            plain_blockers.append("it can be hard to sell quickly at a fair price")
        if position_weight_pct is not None and position_weight_pct >= MAX_SINGLE_POSITION_PCT:
            blockers.append(f"position is already {position_weight_pct:.1f}% of the portfolio")
            plain_blockers.append(
                f"it already makes up about {max(1, round(position_weight_pct))}% of your investments "
                "(the safe cap for one stock is 10%)"
            )
        if sector_weight_pct is not None and sector_weight_pct >= MAX_SECTOR_PCT:
            blockers.append(f"sector is already {sector_weight_pct:.1f}% of the portfolio")
            plain_blockers.append(
                f"its industry already makes up about {max(1, round(sector_weight_pct))}% of your investments "
                "(the safe cap for one industry is 35%)"
            )
        if market_sentiment_label == "Negative":
            blockers.append("market sentiment is Negative")
            plain_blockers.append("the overall market looks gloomy right now")
        if stock_sentiment_label == "Negative":
            blockers.append("stock news sentiment is Negative")
            plain_blockers.append("recent news about it looks gloomy")
        if blockers:
            block_text = " and ".join(plain_blockers)
            return PolicyDecision(
                "HOLD",
                f"Risk score {score:.1f} leaves room under the {tier} limit of {limit:.0f}, "
                f"but adding is blocked: {'; '.join(blockers)}.",
                limit,
                excess,
                block_reason=block_text,
                decision_driver=(
                    "keep your current shares rather than buying more: even though the stock fits your risk comfort level, "
                    f"adding is paused because {block_text}"
                ),
            )
        pos_text = (
            f"it only makes up about {max(1, round(position_weight_pct))}% of your investments "
            "(well below the 10% single-stock cap)"
            if position_weight_pct is not None
            else "your current position size leaves room to add more"
        )
        return PolicyDecision(
            "INCREASE",
            f"Risk score {score:.1f} is low and comfortably under the {tier} limit of {limit:.0f}, "
            "with no concentration or sentiment blocker.",
            limit,
            excess,
            decision_driver=(
                "you can safely add more because its price behaviour is steady and well within your risk comfort zone, "
                f"{pos_text}, and neither industry exposure nor recent news stands in the way"
            ),
        )

    return PolicyDecision(
        "HOLD",
        f"Risk score {score:.1f} is within the {tier} limit of {limit:.0f}; no change warranted.",
        limit,
        excess,
        decision_driver=(
            "keep holding your current shares without buying or selling: its risk fits within your comfort zone, "
            "and your current position is already appropriately sized for its level of price swings"
        ),
    )


def pro_forma_score(portfolio_score: float, etf_score: float, share: float = DEFAULT_NEW_POSITION_SHARE) -> float:
    """Overall portfolio risk if `share` of the portfolio moved into the ETF."""
    return (1.0 - share) * float(portfolio_score) + share * float(etf_score)


def evaluate_etf_candidate(
    score: float,
    risk_tier: str,
    *,
    portfolio_score: float | None,
    is_hard_to_sell: bool = False,
    market_sentiment_label: str | None = None,
    use_market_mood: bool = False,
    share: float = DEFAULT_NEW_POSITION_SHARE,
) -> PolicyDecision:
    """BUY / WATCH / AVOID for an ETF the investor does not hold.

    Order of checks: the ETF's own risk against the investor's limit, whether
    it can be sold easily, then what adding it does to OVERALL portfolio risk
    (so a lower-risk ETF can still be a BUY when the portfolio is already above
    the investor's limit, and a comfortably safe ETF is not blocked when the
    portfolio is also well below the investor's limit). Market mood only applies
    when `use_market_mood` is set (equity ETFs) and can only hold a BUY back to WATCH.
    """
    tier = risk_tier if risk_tier in TIER_RISK_LIMITS else DEFAULT_TIER
    limit = float(TIER_RISK_LIMITS[tier])
    excess = float(score) - limit

    effect: str | None = None
    new_portfolio_score: float | None = None
    if portfolio_score is not None:
        new_portfolio_score = pro_forma_score(portfolio_score, score, share)
        change = new_portfolio_score - float(portfolio_score)
        effect = (
            "lower" if change < -PORTFOLIO_EFFECT_TOLERANCE
            else "higher" if change > PORTFOLIO_EFFECT_TOLERANCE
            else "same"
        )

    if excess > ETF_AVOID_EXCESS_POINTS:
        return PolicyDecision(
            "AVOID",
            f"Risk score {score:.1f} is {excess:.1f} points above the {tier} limit of {limit:.0f}.",
            limit,
            excess,
            portfolio_effect=effect,
            decision_driver="skip this ETF because its price swings and risk are far above what you are comfortable with",
        )
    if excess > 0:
        return PolicyDecision(
            "WATCH",
            f"Risk score {score:.1f} is above the {tier} limit of {limit:.0f}.",
            limit,
            excess,
            portfolio_effect=effect,
            decision_driver="wait on this ETF because its risk sits slightly above your comfort zone",
        )
    if is_hard_to_sell:
        return PolicyDecision(
            "WATCH",
            "Recent trading volume is below the liquidity threshold.",
            limit,
            excess,
            block_reason="it can be hard to sell quickly at a fair price",
            portfolio_effect=effect,
            decision_driver="wait on this ETF because low daily trading makes it harder to buy or sell quickly at a fair price",
        )

    fits_comfortably = excess <= -ETF_BUY_MARGIN_POINTS and (
        new_portfolio_score is None or new_portfolio_score <= limit
    )
    if effect is None and not fits_comfortably:
        return PolicyDecision(
            "WATCH",
            "Portfolio risk is unavailable and the ETF is not clearly below the limit.",
            limit,
            excess,
            block_reason="there is not enough information about your portfolio yet",
            decision_driver="wait until your existing investments are scored before adding this ETF",
        )
    if effect != "lower" and not fits_comfortably:
        return PolicyDecision(
            "WATCH",
            f"Risk score {score:.1f} is within the limit but adding it would not lower portfolio risk.",
            limit,
            excess,
            block_reason="adding it would not make your investments overall any safer",
            portfolio_effect=effect,
            decision_driver=(
                "wait on this ETF because your existing investments are already near or above your risk limit "
                "and adding this would not bring your overall risk down"
            ),
        )

    if use_market_mood and market_sentiment_label == "Negative":
        return PolicyDecision(
            "WATCH",
            "Risk fits, but market sentiment is Negative.",
            limit,
            excess,
            block_reason="the overall market looks gloomy right now",
            portfolio_effect=effect,
            decision_driver="wait before buying: the ETF fits your risk comfort zone, but the overall market looks gloomy right now",
        )
    return PolicyDecision(
        "BUY",
        f"Risk score {score:.1f} fits the {tier} limit of {limit:.0f} and adding it lowers or keeps portfolio risk in bounds.",
        limit,
        excess,
        portfolio_effect=effect,
        decision_driver=(
            "this ETF is a good candidate to buy because its risk fits comfortably within your comfort zone, "
            "it is easy to buy and sell, and your overall investments stay within your risk limit"
        ),
    )


def allowed_actions(policy_recommendation: str) -> set[str]:
    """What the model may answer given the policy result.
    The model may be more conservative than the policy, never riskier."""
    return {
        "SELL": {"SELL"},
        "INCREASE": {"INCREASE", "HOLD"},
        "BUY": {"BUY", "WATCH"},
        "WATCH": {"WATCH"},
        "AVOID": {"AVOID"},
    }.get(policy_recommendation, {"HOLD"})


def reconcile(llm_recommendation: str, decision: PolicyDecision, *, held: bool = True) -> tuple[str, bool]:
    """Return (final_recommendation, overridden). For a holding, BUY is read as
    INCREASE; for a candidate the investor does not hold, the actions are
    BUY / WATCH / AVOID and anything else is overridden."""
    suggested = "INCREASE" if (held and llm_recommendation == "BUY") else llm_recommendation
    if suggested in allowed_actions(decision.recommendation):
        return suggested, False
    return decision.recommendation, True