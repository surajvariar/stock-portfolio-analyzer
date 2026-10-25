"""Plain-language recommendations: wording, policy bounds, and the ETF path."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import backend.recommendation_agent as agent
from backend.plain_language import (
    build_explanation,
    compose,
    find_jargon,
    plain_facts,
    validate_explanation,
)
from backend.recommendation_policy import (
    evaluate_etf_candidate,
    evaluate_policy,
    pro_forma_score,
    reconcile,
)

STOCK_METRICS = {
    "risk_tier": "Balanced",
    "composite_score": 30.0,
    "volatility_1y": 18.4,
    "beta_1y": 0.72,
    "liquidity_flag": False,
    "position_weight_pct": 4.0,
    "sector_weight_pct": 12.0,
    "unrealised_pnl_pct": 12.6,
    "market_sentiment_label": "Neutral",
    "stock_sentiment_label": "Positive",
    "portfolio_risk_score": 48.0,
}


def all_text(explanation: dict[str, str]) -> str:
    return " ".join(explanation.values())


# --- jargon guard -------------------------------------------------------------
def test_find_jargon_catches_technical_words_and_decimals():
    assert "volatility" in find_jargon("Its volatility is high")
    assert "beta" in find_jargon("Beta of 1 means in step")
    assert "decimal number" in find_jargon("It scored 62.3 today")


def test_find_jargon_ignores_words_that_merely_contain_a_term():
    assert find_jargon("This is a better choice and a good alphabet of options") == []
    assert find_jargon("It is easy to buy and sell.") == []


def test_validate_explanation_flags_jargon_empty_and_long():
    good = {"headline": "Keep holding it.", "why": "It is steady.", "watch": "Check back later."}
    assert validate_explanation(good) == []
    bad = {"headline": "", "why": "Volatility is 20.5 points.", "watch": "x" * 500}
    problems = " | ".join(validate_explanation(bad))
    assert "headline is empty" in problems
    assert "technical words" in problems
    assert "watch is too long" in problems


# --- facts shown to the model --------------------------------------------------
def test_plain_facts_contain_no_raw_numbers_except_whole_percentages():
    facts = plain_facts({**STOCK_METRICS, "portfolio_effect": "lower"})
    text = " ".join(facts.values())
    assert find_jargon(text) == []
    assert "18.4" not in text and "0.72" not in text and "30" not in text.replace("30%", "")


def test_plain_facts_omits_missing_data():
    assert plain_facts({"risk_tier": "Balanced"}) == {}


# --- templates: every action, held and not held, reads in everyday words --------
@pytest.mark.parametrize("action,held", [
    ("SELL", True), ("INCREASE", True), ("HOLD", True),
    ("BUY", False), ("WATCH", False), ("AVOID", False), ("UNRATED", False),
])
def test_every_template_is_jargon_free_and_within_length(action, held):
    metrics = {**STOCK_METRICS, "portfolio_effect": "lower", "composite_score": 30.0}
    explanation = build_explanation(action, "ABC", metrics, held=held)
    assert set(explanation) == {"headline", "why", "watch"}
    assert validate_explanation(explanation) == [], all_text(explanation)


def test_template_with_no_metrics_still_reads_ok():
    for action in ("SELL", "HOLD", "BUY", "WATCH", "AVOID", "UNRATED"):
        explanation = build_explanation(action, "ABC", {}, held=action in {"SELL", "HOLD"})
        assert validate_explanation(explanation) == [], (action, all_text(explanation))


def test_hold_back_reason_appears_in_plain_words():
    explanation = build_explanation(
        "HOLD", "ABC", STOCK_METRICS, held=True, block_reason="the overall market looks gloomy right now"
    )
    assert "gloomy" in explanation["why"]


def test_hold_does_not_repeat_the_reason_it_is_held_back():
    explanation = build_explanation(
        "HOLD", "ITC",
        {**STOCK_METRICS, "composite_score": 30.0, "market_sentiment_label": "Negative"},
        held=True, block_reason="the overall market looks gloomy right now",
    )
    assert all_text(explanation).lower().count("gloomy") == 1


def test_comfort_wording_does_not_repeat_itself():
    for score in (90.0, 60.0, 52.0, 30.0):
        why = build_explanation("HOLD", "ABC", {**STOCK_METRICS, "composite_score": score}, held=True)["why"]
        assert why.lower().count("comfortable") <= 1, why


def test_increase_watch_never_calls_a_small_holding_something_to_watch():
    explanation = build_explanation("INCREASE", "ABC", {**STOCK_METRICS, "composite_score": 30.0}, held=True)
    assert "small part" not in explanation["watch"]


def test_unrated_uses_the_supplied_plain_reason():
    explanation = build_explanation("UNRATED", "XYZ", {}, held=False, block_reason="Not enough price history yet.")
    assert "could not judge" in explanation["headline"].lower()
    assert "Not enough price history yet." in explanation["why"]


def test_compose_joins_parts():
    assert compose({"headline": "A.", "why": "B.", "watch": " C. "}) == "A. B. C."


# --- policy: stock path unchanged, ETF candidate path ---------------------------
def test_stock_policy_boundaries_unchanged():
    assert evaluate_policy(90, "Balanced").recommendation == "SELL"
    assert evaluate_policy(70, "Balanced").recommendation == "HOLD"
    assert evaluate_policy(30, "Balanced").recommendation == "INCREASE"


def test_stock_increase_blocked_gets_plain_reason():
    decision = evaluate_policy(30, "Balanced", market_sentiment_label="Negative")
    assert decision.recommendation == "HOLD"
    assert "gloomy" in (decision.block_reason or "")


def test_etf_buy_when_it_lowers_overall_risk_even_if_portfolio_is_above_limit():
    decision = evaluate_etf_candidate(30, "Balanced", portfolio_score=70)
    assert decision.recommendation == "BUY"
    assert decision.portfolio_effect == "lower"


def test_etf_watch_when_it_would_raise_overall_risk():
    decision = evaluate_etf_candidate(54, "Balanced", portfolio_score=40)
    assert decision.recommendation == "WATCH"
    assert decision.portfolio_effect == "higher"
    assert decision.block_reason


def test_etf_avoid_and_watch_by_distance_over_limit():
    assert evaluate_etf_candidate(75, "Balanced", portfolio_score=40).recommendation == "AVOID"
    assert evaluate_etf_candidate(60, "Balanced", portfolio_score=40).recommendation == "WATCH"


def test_etf_hard_to_sell_holds_back_a_buy():
    decision = evaluate_etf_candidate(20, "Balanced", portfolio_score=50, is_hard_to_sell=True)
    assert decision.recommendation == "WATCH"
    assert "hard" in (decision.block_reason or "")


def test_etf_market_mood_only_applies_when_enabled():
    quiet = evaluate_etf_candidate(20, "Balanced", portfolio_score=50, market_sentiment_label="Negative")
    assert quiet.recommendation == "BUY"
    gloomy = evaluate_etf_candidate(
        20, "Balanced", portfolio_score=50, market_sentiment_label="Negative", use_market_mood=True
    )
    assert gloomy.recommendation == "WATCH"


def test_etf_without_portfolio_score_needs_clear_margin():
    assert evaluate_etf_candidate(20, "Balanced", portfolio_score=None).recommendation == "BUY"
    assert evaluate_etf_candidate(53, "Balanced", portfolio_score=None).recommendation == "WATCH"


def test_pro_forma_score_is_a_weighted_mix():
    assert pro_forma_score(60, 20, share=0.25) == pytest.approx(50.0)


def test_reconcile_buy_reads_as_increase_only_for_holdings():
    decision = evaluate_policy(30, "Balanced")
    assert reconcile("BUY", decision, held=True) == ("INCREASE", False)
    candidate = evaluate_etf_candidate(30, "Balanced", portfolio_score=50)
    assert reconcile("SELL", candidate, held=False) == ("BUY", True)


# --- agent: the model can only word the decision, never change it ---------------
class FakeChat:
    model = "fake-model"

    def __init__(self, output):
        self.output = output

    def with_structured_output(self, schema, method=None):
        return SimpleNamespace(invoke=lambda messages, config=None: self.output)


def patch_model(monkeypatch, output):
    monkeypatch.setattr(agent, "create_chat_ollama", lambda temperature=0: FakeChat(output))
    monkeypatch.setattr(agent, "langfuse_callback_config", lambda *a, **k: None)


def test_prompt_never_contains_raw_scores_or_jargon():
    prompt = agent.build_ollama_prompt("ABC", STOCK_METRICS, "Balanced", held=True)
    facts_block = prompt.split("Facts (already in plain words):")[1].split("The investor's own")[0]
    assert "18.4" not in facts_block and "0.72" not in facts_block
    assert find_jargon(facts_block) == []
    assert "ABC" in prompt and "already owns" in prompt


def test_prompt_for_a_candidate_says_it_is_not_owned():
    prompt = agent.build_ollama_prompt("ETF1", {**STOCK_METRICS}, "Balanced", held=False, kind="ETF")
    assert "does NOT own this ETF" in prompt


def test_prompt_allows_self_explanatory_headlines_without_a_word_count_limit():
    prompt = agent.build_ollama_prompt("ABC", STOCK_METRICS, "Balanced")

    assert "there is no word-count limit" in prompt
    assert "saying what to do and the main reason" in prompt
    assert "at most 12 words" not in prompt


def test_good_model_text_is_used(monkeypatch):
    output = agent.RecommendationOutput(
        recommendation="HOLD",
        headline="Keep holding ABC.",
        why="It is steady and fits your comfort level.",
        watch="Check again next month.",
    )
    patch_model(monkeypatch, output)
    result = agent.generate_ollama_recommendation("ABC", {**STOCK_METRICS, "composite_score": 45.0}, "Balanced")
    assert result["model_status"] == "ollama"
    assert result["recommendation"] == "HOLD"
    assert result["reasoning"].startswith("Keep holding ABC.")


def test_longer_self_explanatory_model_headline_is_used(monkeypatch):
    output = agent.RecommendationOutput(
        recommendation="HOLD",
        headline="Keep holding ABC for now because it fits your comfort level and no urgent change is needed.",
        why="Its price usually moves up and down only gently.",
        watch="Check again after the next update.",
    )
    patch_model(monkeypatch, output)

    result = agent.generate_ollama_recommendation(
        "ABC", {**STOCK_METRICS, "composite_score": 45.0}, "Balanced"
    )

    assert result["model_status"] == "ollama"
    assert result["explanation"]["headline"] == output.headline


def test_model_text_with_jargon_is_replaced_by_plain_template(monkeypatch):
    output = agent.RecommendationOutput(
        recommendation="HOLD",
        headline="Hold ABC.",
        why="Volatility of 18.4 and a beta of 0.72 give a composite score of 45.0.",
        watch="Monitor.",
    )
    patch_model(monkeypatch, output)
    result = agent.generate_ollama_recommendation("ABC", {**STOCK_METRICS, "composite_score": 45.0}, "Balanced")
    assert result["model_status"] == "ollama_reworded"
    assert result["recommendation"] == "HOLD"
    assert find_jargon(result["reasoning"]) == []


def test_model_cannot_take_a_riskier_action_than_the_policy_allows(monkeypatch):
    output = agent.RecommendationOutput(
        recommendation="INCREASE", headline="Buy more.", why="It looks great.", watch="Nothing."
    )
    patch_model(monkeypatch, output)
    result = agent.generate_ollama_recommendation("ABC", {**STOCK_METRICS, "composite_score": 90.0}, "Balanced")
    assert result["model_status"] == "ollama_overridden"
    assert result["recommendation"] == "SELL"
    assert result["llm_recommendation"] == "INCREASE"
    assert "selling" in result["reasoning"].lower()


def test_model_failure_falls_back_to_plain_template(monkeypatch):
    def boom(temperature=0):
        raise RuntimeError("model down")

    monkeypatch.setattr(agent, "create_chat_ollama", boom)
    result = agent.generate_ollama_recommendation("ABC", {**STOCK_METRICS, "composite_score": 30.0}, "Balanced")
    assert result["model_status"] == "deterministic_fallback"
    assert result["recommendation"] == "INCREASE"
    assert validate_explanation(result["explanation"]) == []


def test_use_llm_false_never_calls_the_model(monkeypatch):
    def boom(temperature=0):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(agent, "create_chat_ollama", boom)
    result = agent.generate_ollama_recommendation(
        "ETF1", {**STOCK_METRICS, "composite_score": 30.0}, "Balanced", held=False, kind="ETF", use_llm=False
    )
    assert result["model_status"] == "deterministic_fallback"
    assert result["recommendation"] == "BUY"


# --- ETF scoring end to end ----------------------------------------------------
def synthetic_prices(days: int = 300, step: float = 0.001) -> pd.Series:
    index = pd.bdate_range("2025-01-01", periods=days)
    values = [100.0]
    for i in range(1, days):
        values.append(values[-1] * (1 + (step if i % 2 else -step / 2)))
    return pd.Series(values, index=index)


def test_score_equity_etf_returns_plain_wording(monkeypatch):
    import backend.etf_analyzer as etf_analyzer
    monkeypatch.setattr(etf_analyzer, "liquidity_flag", lambda turnover: False)
    prices = synthetic_prices()
    market_returns = (synthetic_prices(step=0.002)).pct_change().dropna()
    summary = {"weighted_average_risk_score": 55.0, "profile_alignment": "above_profile"}
    result = etf_analyzer.score_equity_etf(
        prices, market_returns, 1e8, "Balanced", summary, symbol="CALMBEES", market_sentiment_label="Neutral"
    )
    assert result["status"] == "scored"
    assert result["recommendation"] in {"BUY", "WATCH", "AVOID"}
    assert set(result["explanation"]) == {"headline", "why", "watch"}
    assert "CALMBEES" in result["explanation"]["headline"]
    assert validate_explanation(result["explanation"]) == []
    # The stored text is the plain text, not the old "Risk score 52.3 ..." wording.
    assert "Risk score" not in result["reasoning"]


def test_score_equity_etf_short_history_is_skipped_with_plain_reason():
    import backend.etf_analyzer as etf_analyzer
    prices = synthetic_prices(days=30)
    result = etf_analyzer.score_equity_etf(
        prices, prices.pct_change().dropna(), 1e8, "Balanced", {}, symbol="NEWETF"
    )
    assert result["status"] == "skipped"
    assert result["plain_reason"] == etf_analyzer.PLAIN_NOT_ENOUGH_HISTORY
    assert find_jargon(result["plain_reason"]) == []
