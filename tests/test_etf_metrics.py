from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.etf_analyzer as etf_analyzer
import backend.main as main
from backend.etf_analyzer import (
    analyze_equity_etfs,
    normalize_nse_etf,
    score_equity_etf,
)
from backend.risk_scoring.beta_calculator import BetaEstimate


def test_nse_etf_normalization_keeps_optional_fields_and_small_cap_metadata():
    result = normalize_nse_etf(
        {
            "symbol": "SMALLBEES",
            "assets": "Nifty Smallcap 250",
            "iNav": "120.5",
            "portfolioTurnoverRatio": "0.42",
            "yieldPct": "1.1",
            "repoRate": "6.5",
        }
    )

    assert result is not None
    assert result["asset_class"] == "Equity"
    assert result["inav"] == 120.5
    assert result["portfolio_turnover_ratio"] == 0.42
    assert result["yield_pct"] == 1.1
    assert result["repo_rate_pct"] == 6.5
    assert etf_analyzer.is_small_cap_etf(result)


def test_small_cap_score_includes_all_six_risk_factors():
    dates = pd.date_range("2025-01-01", periods=130, freq="B")
    market_returns = pd.Series([((index % 9) - 4) / 10_000 for index in range(len(dates))], index=dates)
    daily_returns = pd.Series([((index % 7) - 3) / 1_000 for index in range(len(dates))], index=dates)
    prices = (1 + daily_returns).cumprod() * 100

    result = score_equity_etf(
        prices,
        market_returns,
        20_000_000,
        "Moderate",
        {
            "weighted_average_risk_score": 50,
            "profile_alignment": "within_profile",
            "portfolio_hhi": 0.3,
            "sector_hhi": 0.25,
            "market_cap_hhi": 0.4,
        },
    )

    assert result["status"] == "scored"
    assert result["sortino_1y"] is not None
    assert result["max_drawdown_1y_pct"] is not None
    assert result["tracking_error_1y_pct"] is not None
    assert result["portfolio_hhi"] == 0.3
    assert result["sector_hhi"] == 0.25
    assert result["market_cap_hhi"] == 0.4


def test_composite_etf_risk_score_equal_weights_the_six_factors(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    prices = pd.Series(range(100, 200), index=dates, dtype=float)
    market_returns = pd.Series([0.001] * 100, index=dates)
    monkeypatch.setattr(etf_analyzer, "annualized_volatility", Mock(return_value=20.0))
    monkeypatch.setattr(etf_analyzer, "estimate_beta", Mock(return_value=BetaEstimate(1.0, 100)))
    monkeypatch.setattr(etf_analyzer, "liquidity_flag", Mock(return_value=False))
    monkeypatch.setattr(etf_analyzer, "_sortino_ratio", Mock(return_value=1.0))
    monkeypatch.setattr(etf_analyzer, "_max_drawdown_pct", Mock(return_value=20.0))
    monkeypatch.setattr(etf_analyzer, "_tracking_error_pct", Mock(return_value=7.0))

    result = score_equity_etf(prices, market_returns, 20_000_000, "Moderate", {})

    assert sum(etf_analyzer.ETF_RISK_WEIGHTS.values()) == pytest.approx(1.0)
    assert result["composite_score"] == 45.0


def test_analysis_only_scores_equity_small_cap_etfs(monkeypatch):
    monkeypatch.setattr(
        etf_analyzer,
        "fetch_etf_market_history",
        Mock(return_value=pd.DataFrame({"Close": [100.0], "Volume": [1000]})),
    )
    monkeypatch.setattr(etf_analyzer, "median_daily_turnover", Mock(return_value=1_000_000))
    monkeypatch.setattr(
        etf_analyzer,
        "score_equity_etf",
        Mock(return_value={"status": "scored", "composite_score": 20}),
    )
    records = [
        {"symbol": "NIFTY50", "underlying": "Nifty 50", "asset_class": "Equity"},
        {"symbol": "SMALL250", "underlying": "Nifty Smallcap 250", "asset_class": "Equity"},
        {"symbol": "GOLDETF", "underlying": "Gold", "asset_class": "Commodity"},
    ]

    results = analyze_equity_etfs(
        records,
        pd.Series([1.0, 1.01], index=pd.date_range("2025-01-01", periods=2)),
        "Moderate",
        {},
    )

    assert [item["symbol"] for item in results] == ["SMALL250"]


def test_etf_run_poll_query_uses_psycopg_safe_small_cap_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "fetch_one", Mock(return_value={"id": 17}))

    def fake_fetch_all(query, params=()):
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)

    with TestClient(main.app) as client:
        response = client.get("/etfs/runs/17")

    assert response.status_code == 200
    assert response.json()["recommendations"] == []
    assert captured["params"] == (17,)
    assert "POSITION('smallcap' IN LOWER(e.underlying)) > 0" in captured["query"]
    assert "LIKE '%" not in captured["query"]
