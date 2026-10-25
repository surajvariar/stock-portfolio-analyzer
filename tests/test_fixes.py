from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from market_hours import drop_incomplete_bars, drop_incomplete_series, is_bar_final
from portfolio_aggregation import compute_portfolio_summary, group_weights_pct
from price_history_sync import normalize_symbol, prices_rebased
from recommendation_policy import allowed_actions, evaluate_policy, reconcile
from review_harness import build_review_rows
from risk_questionnaire import calculate_risk_profile
from risk_scoring.beta_calculator import beta_risk_score, calculate_beta, estimate_beta, to_date_indexed
from risk_scoring.liquidity_flagger import liquidity_flag, median_daily_turnover
from risk_scoring.volatility_calculator import annualized_volatility, volatility_risk_score
from sentiment.news_sentiment import summarize_headlines
from sentiment.quant_sentiment import compute_market_sentiment

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------- beta / tz
def _series(n=300, beta=1.3, tz=None, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    market = rng.normal(0, 0.01, n)
    stock = beta * market + rng.normal(0, 0.003, n)
    m_idx = dates.tz_localize(tz) if tz else dates
    return pd.Series(stock, index=dates), pd.Series(market, index=m_idx)


def test_beta_aligns_tz_aware_market_with_naive_stock():
    stock, market_aware = _series(tz="Asia/Kolkata")
    est = estimate_beta(stock, market_aware)
    assert est.beta == pytest.approx(1.3, abs=0.1)
    assert est.observations == 252  # trailing one-year window


def test_beta_none_when_overlap_too_short():
    stock, market = _series(n=30)
    est = estimate_beta(stock, market)
    assert est.beta is None and est.observations == 30
    assert calculate_beta(stock, market) is None
    assert beta_risk_score(None) == 50  # neutral, never a fake 1.0


def test_to_date_indexed_strips_tz_and_time():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-05 00:00", "2026-01-06 00:00"]).tz_localize("Asia/Kolkata"))
    out = to_date_indexed(s)
    assert out.index.tz is None and list(out.index.date) == [date(2026, 1, 5), date(2026, 1, 6)]


# ------------------------------------------------------------- volatility
def test_volatility_none_for_short_history_not_nan_or_zero():
    assert annualized_volatility(pd.Series([100.0, 101.0])) is None
    assert annualized_volatility(pd.Series(dtype=float)) is None
    assert annualized_volatility(None) is None
    assert volatility_risk_score(None) == 50


def test_volatility_uses_trailing_year_only():
    rng = np.random.default_rng(1)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, 600))
    wild_tail = calm[-1] * np.cumprod(1 + rng.normal(0, 0.03, 252))
    prices = pd.Series(np.concatenate([calm, wild_tail]))
    recent = annualized_volatility(prices)
    everything = annualized_volatility(prices, window=None)
    assert recent > everything  # old calm data must not dilute the 1y figure


# --------------------------------------------------------------- liquidity
def test_liquidity_uses_traded_value_not_share_count():
    idx = pd.date_range("2026-01-01", periods=80, freq="B")
    pricey = median_daily_turnover(pd.Series(25000.0, index=idx), pd.Series(3000.0, index=idx))
    penny = median_daily_turnover(pd.Series(4.0, index=idx), pd.Series(150000.0, index=idx))
    assert liquidity_flag(pricey) is False  # Rs 7.5 cr/day
    assert liquidity_flag(penny) is True  # Rs 6 lakh/day


def test_liquidity_unknown_or_t2t_is_flagged():
    assert liquidity_flag(None) is True
    assert liquidity_flag(float("nan")) is True
    assert liquidity_flag(5e9, trade_to_trade_flag=True) is True


# ---------------------------------------------------------------- sentiment
def test_news_matches_whole_words_only():
    # "against" contains "gain", "group"/"update" contain "up": none are signals.
    neutral = summarize_headlines(["Regulator moves against promoter group", "Company to update investors on setup"])
    assert neutral["score"] == 50.0
    assert summarize_headlines(["Stock falls on weak results"])["score"] == 40.0
    assert summarize_headlines(["Shares rise as profit gains"])["score"] == 60.0


def test_quant_sentiment_not_saturated_and_reweights_missing_inputs():
    strong = compute_market_sentiment(0.02, 14.0, 500.0, 300.0, True, True)
    extreme = compute_market_sentiment(0.15, 11.0, 500.0, 300.0, True, True)
    assert strong["score"] < extreme["score"] == 100.0
    partial = compute_market_sentiment(0.15, 11.0)  # only index + vix available
    assert partial["coverage"] == pytest.approx(0.6, abs=0.01)
    assert partial["score"] == 100.0


def test_vix_has_no_cliff_at_20():
    a = compute_market_sentiment(0.0, 19.9)["score"]
    b = compute_market_sentiment(0.0, 20.1)["score"]
    assert abs(a - b) < 2


# ------------------------------------------------------------------ policy
@pytest.mark.parametrize(
    "score,tier,expected",
    [
        (85, "Conservative", "SELL"),
        (85, "Balanced", "SELL"),
        (90, "Growth", "SELL"),  # 20 above Growth's limit of 70
        (85, "Growth", "HOLD"),  # exactly 15 above: boundary is "more than 15"
        (85, "Aggressive", "HOLD"),  # limit 100: never "too risky"
        (20, "Conservative", "INCREASE"),
        (30, "Balanced", "INCREASE"),
        (50, "Balanced", "HOLD"),
        (60, "Balanced", "HOLD"),  # slightly above limit: monitor, don't sell
    ],
)
def test_policy_is_risk_aware(score, tier, expected):
    assert evaluate_policy(score, tier).recommendation == expected


def test_policy_blocks_increase_on_concentration_or_negative_sentiment():
    assert evaluate_policy(25, "Balanced", position_weight_pct=15).recommendation == "HOLD"
    assert evaluate_policy(25, "Balanced", sector_weight_pct=40).recommendation == "HOLD"
    assert evaluate_policy(25, "Balanced", market_sentiment_label="Negative").recommendation == "HOLD"
    assert evaluate_policy(25, "Balanced", stock_sentiment_label="Negative").recommendation == "HOLD"


def test_reconcile_only_lets_model_be_more_conservative():
    sell = evaluate_policy(90, "Balanced")
    hold = evaluate_policy(50, "Balanced")
    inc = evaluate_policy(25, "Balanced")
    assert reconcile("HOLD", sell) == ("HOLD", False)
    assert reconcile("INCREASE", sell) == ("SELL", True)
    assert reconcile("SELL", hold) == ("HOLD", True)
    assert reconcile("BUY", inc) == ("INCREASE", False)  # BUY read as INCREASE
    assert reconcile("SELL", inc) == ("INCREASE", True)
    assert allowed_actions("HOLD") == {"HOLD"}


# ---------------------------------------------------------------- portfolio
def test_portfolio_reports_coverage_and_unscored_symbols():
    summary = compute_portfolio_summary(
        [
            {"symbol": "A", "position_value": 300, "risk_score": 40, "risk_tier": "Balanced"},
            {"symbol": "B", "position_value": 100, "risk_score": None, "risk_tier": "Balanced"},
        ]
    )
    assert summary["coverage_pct"] == 75.0
    assert summary["unscored_symbols"] == ["B"]
    assert summary["profile_alignment"] == "within_profile"
    assert summary["risk_label"] == "Moderate"


def test_group_weights_use_total_portfolio_as_denominator():
    weights = group_weights_pct(
        [{"sector": "IT", "position_value": 50}, {"sector": None, "position_value": 50}], "sector"
    )
    assert weights == {"IT": 50.0}


# --------------------------------------------------------------- questionnaire
def test_questionnaire_tiers_unchanged_and_liquidity_counts_as_capacity():
    answers = dict(income_stability=3, investment_experience=3, time_horizon=3, drawdown_reaction=3, liquidity_needs=3)
    profile = calculate_risk_profile(answers)
    assert profile["total_score"] == 15 and profile["risk_tier"] == "Balanced"
    assert profile["capacity_score"] == 9 and profile["tolerance_score"] == 6


# ------------------------------------------------------------ price sync helpers
def test_symbol_normalization():
    assert normalize_symbol("reliance") == "RELIANCE.NS"
    assert normalize_symbol("500325", "BSE") == "500325.BO"
    assert normalize_symbol("XYZ-BE") == "XYZ.NS"
    assert normalize_symbol("M&M") == "M&M.NS"
    assert normalize_symbol("TCS.NS") == "TCS.NS"


def test_split_detection_triggers_on_basis_change_only():
    stored = {date(2026, 1, 5): 1000.0, date(2026, 1, 6): 1010.0}
    same = [{"date": date(2026, 1, 5), "close": 1000.4}, {"date": date(2026, 1, 7), "close": 1020.0}]
    split = [{"date": date(2026, 1, 5), "close": 500.0}]  # 1:2 split re-based history
    assert prices_rebased(stored, same) is False
    assert prices_rebased(stored, split) is True


def test_partial_intraday_bar_is_dropped():
    midday = datetime(2026, 10, 5, 11, 0, tzinfo=IST)  # Monday, market open
    after_close = datetime(2026, 10, 5, 16, 30, tzinfo=IST)
    assert is_bar_final(date(2026, 10, 2), midday) is True
    assert is_bar_final(date(2026, 10, 5), midday) is False
    assert is_bar_final(date(2026, 10, 5), after_close) is True

    df = pd.DataFrame({"date": [date(2026, 10, 2), date(2026, 10, 5)], "close": [1.0, 2.0]})
    assert list(drop_incomplete_bars(df, now=midday)["date"]) == [date(2026, 10, 2)]
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-02", "2026-10-05"]).tz_localize("Asia/Kolkata"))
    assert len(drop_incomplete_series(s, now=midday)) == 1


# ------------------------------------------------------------------- review
def test_review_judges_against_benchmark_and_waits_for_enough_days():
    bench = pd.Series(
        [100.0, 110.0], index=pd.to_datetime(["2026-01-05", "2026-02-05"]).tz_localize("Asia/Kolkata")
    )
    base = {
        "stock_symbol": "AAA",
        "entry_date": date(2026, 1, 5),
        "latest_date": date(2026, 2, 5),
        "entry_price": 100.0,
        "latest_price": 108.0,
    }
    # Stock +8% but Nifty +10%: a BUY actually lagged the market.
    rows = build_review_rows([{**base, "recommendation": "INCREASE"}, {**base, "recommendation": "SELL"}], bench)
    assert rows[0]["excess_return_pct"] == -2.0 and rows[0]["outcome"] == "miss"
    assert rows[1]["outcome"] == "hit"
    assert rows[0]["outcome_basis"] == "excess_vs_nifty"

    early = build_review_rows(
        [{**base, "recommendation": "HOLD", "latest_date": date(2026, 1, 7), "latest_price": 101.0}], bench
    )
    assert early[0]["outcome"] == "too_early"

    no_bench = build_review_rows([{**base, "recommendation": "INCREASE"}], None)
    assert no_bench[0]["outcome_basis"] == "absolute_return" and no_bench[0]["outcome"] == "hit"
