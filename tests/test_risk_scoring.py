import json
from math import isclose

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main
from backend.risk_questionnaire import calculate_risk_profile
from backend.risk_scoring.composite_risk_score import (
    composite_risk_label,
    composite_risk_score,
)
from backend.risk_scoring.market_cap_classifier import classify_market_cap_rank


def test_calculate_risk_profile_assigns_expected_tier():
    answers = {
        "income_stability": 2,
        "emergency_fund": 3,
        "time_horizon": 4,
        "market_volatility_tolerance": 2,
        "loss_aversion": 3,
    }

    result = calculate_risk_profile(answers)

    assert result["capacity_score"] == 9
    assert result["tolerance_score"] == 5
    assert result["total_score"] == 14
    assert result["risk_tier"] == "Balanced"


def test_calculate_risk_profile_accepts_phase_question_dimensions():
    answers = {
        "income_stability": 3,
        "investment_experience": 4,
        "time_horizon": 4,
        "drawdown_reaction": 2,
        "liquidity_needs": 2,
    }

    result = calculate_risk_profile(answers)

    assert result["capacity_score"] == 9
    assert result["tolerance_score"] == 6
    assert result["total_score"] == 15
    assert result["risk_tier"] == "Balanced"


def test_risk_questionnaire_endpoint_validates_and_persists_frontend_payload(monkeypatch):
    writes = []
    monkeypatch.setattr(main, "execute", lambda query, params=(): writes.append(params) or 1)
    client = TestClient(main.app)
    answers = {
        "income_stability": 3,
        "investment_experience": 4,
        "time_horizon": 4,
        "drawdown_reaction": 2,
        "liquidity_needs": 2,
    }

    response = client.post("/risk/questionnaire", json=answers)
    invalid_response = client.post("/risk/questionnaire", json={**answers, "time_horizon": 6})

    assert response.status_code == 200
    assert response.json()["data"]["risk_tier"] == "Balanced"
    assert json.loads(writes[0][4]) == answers
    assert invalid_response.status_code == 422


def test_latest_risk_questionnaire_returns_saved_profile(monkeypatch):
    profile = {
        "total_score": 15,
        "risk_tier": "Balanced",
        "capacity_score": 11,
        "tolerance_score": 4,
        "answers": {"income_stability": 3},
    }
    monkeypatch.setattr(main, "fetch_one", lambda query, params=(): profile)

    response = TestClient(main.app).get("/risk/questionnaire/latest")

    assert response.status_code == 200
    assert response.json() == {"data": profile}


def test_composite_risk_score_and_market_cap_bucket():
    assert classify_market_cap_rank(80) == "Large Cap"

    score = composite_risk_score(15, 0.7, "Large Cap", False)
    assert isclose(score, 20.0)
    assert composite_risk_label(score) == "Low"

    high_score = composite_risk_score(60, 1.8, "Small Cap", True)
    assert isclose(high_score, 80.0)
    assert composite_risk_label(high_score) == "Very High"


def test_metadata_import_assigns_official_rank_bucket(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "ensure_schema", lambda: None)
    monkeypatch.setattr(main, "execute_many", lambda query, params: captured.update(query=query, params=list(params)) or 1)

    response = TestClient(main.app).post(
        "/stocks/metadata/import",
        json=[{"symbol": "tcs", "market_cap_rank": 80, "sector": "IT", "trade_to_trade_flag": False}],
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "updated": 1}
    assert captured["params"][0][0] == "TCS"
    assert captured["params"][0][4] == "Large Cap"


def test_risk_scoring_passes_trade_to_trade_flag(monkeypatch):
    stock = {
        "stock_id": 9,
        "symbol": "TCS",
        "market_cap_rank": 80,
        "market_cap_tier": None,
        "sector": "IT",
        "trade_to_trade_flag": True,
    }
    writes = []
    dates = pd.date_range("2026-09-21", periods=90, freq="D")
    monkeypatch.setattr(main, "start_agent_run", lambda run_type: 5)
    monkeypatch.setattr(main, "finish_agent_run", lambda run_id, status, notes: None)
    monkeypatch.setattr(main, "ensure_schema", lambda: None)
    monkeypatch.setattr(main, "fetch_all", lambda query, params=(): [stock])
    monkeypatch.setattr(
        main,
        "fetch_nifty_50_prices",
        lambda days: pd.Series([100 + i * 0.5 for i in range(90)], index=dates),
    )
    monkeypatch.setattr(
        main,
        "fetch_stock_prices_from_db",
        lambda stock_id: (
            pd.Series([50 + i for i in range(90)], index=dates),
            pd.Series([10 + i for i in range(90)], index=dates),
        ),
    )
    monkeypatch.setattr(main, "execute", lambda query, params=(): writes.append((query, params)) or 1)

    response = TestClient(main.app).post("/risk/score-stocks")

    assert response.status_code == 200
    assert response.json()["run_id"] == 5
    score_params = next(params for query, params in writes if "INSERT INTO risk_scores" in query)
    assert score_params[3] is True
    assert score_params[4] == "Large Cap"
