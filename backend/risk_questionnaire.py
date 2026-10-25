from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Scoring convention: every answer is 1-5 where HIGHER = MORE ABLE/WILLING TO
# TAKE RISK. The frontend must therefore reverse-code questions whose natural
# wording runs the other way - in particular `liquidity_needs`, where
# 5 = "I will not need this money for a long time" and 1 = "I may need it soon".
#
# Capacity = ability to bear losses; tolerance = willingness to bear them.
CAPACITY_QUESTIONS = (
    "income_stability",
    "time_horizon",
    "liquidity_needs",
)

TOLERANCE_QUESTIONS = (
    "investment_experience",
    "drawdown_reaction",
)

LEGACY_CAPACITY_QUESTIONS = ("income_stability", "emergency_fund", "time_horizon")
LEGACY_TOLERANCE_QUESTIONS = ("market_volatility_tolerance", "loss_aversion")

RISK_TIERS = [
    (9, "Conservative"),
    (13, "Moderate"),
    (17, "Balanced"),
    (21, "Growth"),
    (25, "Aggressive"),
]

# Highest portfolio/stock composite risk score (0-100, see risk_scoring) that is
# still considered consistent with each tier. Tunable policy, not a regulation.
TIER_RISK_LIMITS = {
    "Conservative": 25,
    "Moderate": 40,
    "Balanced": 55,
    "Growth": 70,
    "Aggressive": 100,
}


def validate_answer(value: Any, question_key: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Question '{question_key}' must be an integer from 1 to 5.") from exc

    if score < 1 or score > 5:
        raise ValueError(f"Question '{question_key}' must be between 1 and 5.")
    return score


def calculate_risk_profile(answers: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(answers, Mapping):
        raise TypeError("The answers payload must be a dictionary of question keys to scores.")

    if all(question in answers for question in (*CAPACITY_QUESTIONS, *TOLERANCE_QUESTIONS)):
        capacity_questions = CAPACITY_QUESTIONS
        tolerance_questions = TOLERANCE_QUESTIONS
    elif all(question in answers for question in (*LEGACY_CAPACITY_QUESTIONS, *LEGACY_TOLERANCE_QUESTIONS)):
        capacity_questions = LEGACY_CAPACITY_QUESTIONS
        tolerance_questions = LEGACY_TOLERANCE_QUESTIONS
    else:
        required = (*CAPACITY_QUESTIONS, *TOLERANCE_QUESTIONS)
        missing = [question for question in required if question not in answers]
        raise ValueError(f"Missing questionnaire answers: {', '.join(missing)}.")

    capacity_score = 0
    tolerance_score = 0

    for question in capacity_questions:
        capacity_score += validate_answer(answers[question], question)

    for question in tolerance_questions:
        tolerance_score += validate_answer(answers[question], question)

    total_score = capacity_score + tolerance_score

    risk_tier = "Aggressive"
    for threshold, tier in sorted(RISK_TIERS, key=lambda entry: entry[0]):
        if total_score <= threshold:
            risk_tier = tier
            break

    return {
        "capacity_score": capacity_score,
        "tolerance_score": tolerance_score,
        "total_score": total_score,
        "risk_tier": risk_tier,
    }
