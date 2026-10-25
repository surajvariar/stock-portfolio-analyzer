import json
import sys
from types import ModuleType

import pandas as pd
from fastapi.testclient import TestClient

import llm_client
import main
from etf_analyzer import (
    infer_etf_asset_class,
    normalize_nse_etf,
    recommend_etf,
    score_equity_etf,
)
from portfolio_aggregation import compute_portfolio_summary
from recommendation_agent import (
    build_ollama_prompt,
    choose_recommendation,
    generate_ollama_recommendation,
)


def test_compute_portfolio_summary_uses_weighted_average_and_hhi():
    holdings = [
        {"symbol": "A", "quantity": 10, "risk_score": 20},
        {"symbol": "B", "quantity": 20, "risk_score": 50},
        {"symbol": "C", "quantity": 30, "risk_score": 80},
    ]

    result = compute_portfolio_summary(holdings)

    assert result["weighted_average_risk_score"] == 60.0
    assert result["portfolio_hhi"] == 0.3889
    assert result["risk_label"] == "High"
    assert "position_value" not in holdings[0]


def test_compute_portfolio_summary_uses_market_value_and_group_concentration():
    holdings = [
        {"quantity": 10, "position_value": 1000, "risk_score": 20, "sector": "Tech", "market_cap_tier": "Large Cap", "risk_tier": "Moderate"},
        {"quantity": 1, "position_value": 9000, "risk_score": 80, "sector": "Banking", "market_cap_tier": "Large Cap", "risk_tier": "Moderate"},
        {"quantity": 40, "position_value": 1000, "risk_score": 50, "sector": "Tech", "market_cap_tier": "Small Cap", "risk_tier": "Moderate"},
    ]

    result = compute_portfolio_summary(holdings)

    assert result["weighted_average_risk_score"] == 71.82
    assert result["sector_hhi"] == 0.7025
    assert result["market_cap_hhi"] == 0.8347
    assert result["profile_alignment"] == "above_profile"


def test_group_hhi_is_unavailable_when_classification_is_missing():
    result = compute_portfolio_summary(
        [
            {"position_value": 1000, "risk_score": 30, "sector": "Tech", "market_cap_tier": "Large Cap"},
            {"position_value": 1000, "risk_score": 60, "sector": None, "market_cap_tier": "Small Cap"},
        ]
    )

    assert result["sector_hhi"] == 1.0
    assert result["market_cap_hhi"] == 0.5


def test_choose_recommendation_respects_risk_tier_and_score():
    assert choose_recommendation(20, "Conservative") == "INCREASE"
    assert choose_recommendation(75, "Conservative") == "SELL"
    assert choose_recommendation(80, "Aggressive") == "HOLD"


def test_nse_etf_normalization_infers_asset_class_and_parses_market_values():
    result = normalize_nse_etf(
        {"symbol": "NIFTYBEES", "assets": "Nifty 50", "ltP": "258.5", "nav": "257.9", "trdVal": "100000000"}
    )

    assert result == {
        "symbol": "NIFTYBEES",
        "underlying": "Nifty 50",
        "asset_class": "Equity",
        "last_price": 258.5,
        "nav": 257.9,
        "traded_value": 100000000.0,
        "volume": None,
        "change_30d_pct": None,
        "change_365d_pct": None,
        "inav": None,
        "portfolio_turnover_ratio": None,
        "yield_pct": None,
        "repo_rate_pct": None,
    }
    assert infer_etf_asset_class("Gold") == "Commodity"
    assert infer_etf_asset_class("Nifty 1D Rate Index") == "Debt"


def test_etf_buy_recommendation_requires_profile_and_portfolio_fit():
    action, _ = recommend_etf(
        35,
        "Moderate",
        portfolio_risk_score=50,
        profile_alignment="within_profile",
        is_illiquid=False,
    )
    assert action == "BUY"

    action, reason = recommend_etf(
        35,
        "Moderate",
        portfolio_risk_score=50,
        profile_alignment="above_profile",
        is_illiquid=False,
    )
    assert action == "WATCH"
    assert "already above" in reason


def test_equity_etf_score_uses_volatility_beta_and_liquidity():
    dates = pd.date_range("2025-01-01", periods=130, freq="B")
    market_returns = pd.Series([((index % 9) - 4) / 10000 for index in range(len(dates))], index=dates)
    prices = (1 + market_returns).cumprod() * 100

    result = score_equity_etf(
        prices,
        market_returns,
        20_000_000,
        "Moderate",
        {"weighted_average_risk_score": 50, "profile_alignment": "within_profile"},
    )

    assert result["status"] == "scored"
    assert result["beta_1y"] == 1.0
    assert result["liquidity_flag"] is False
    assert result["sortino_1y"] is not None
    assert result["max_drawdown_1y_pct"] is not None
    assert result["tracking_error_1y_pct"] is not None
    assert result["recommendation"] == "BUY"


def test_etf_analysis_requires_a_saved_risk_profile(monkeypatch):
    monkeypatch.setattr(main, "ensure_schema", lambda: None)
    monkeypatch.setattr(main, "fetch_one", lambda query, params=(): None)

    with TestClient(main.app) as client:
        response = client.post("/etfs/analyze")

    assert response.status_code == 409
    assert "risk questionnaire" in response.json()["detail"]


def test_build_ollama_prompt_restricts_llm_to_supplied_values():
    prompt = build_ollama_prompt(
        "TCS",
        {
            "composite_score": 72,
            "volatility_1y": 24.6,
            "beta_1y": 1.1,
            "sentiment_score": 68,
            "risk_tier": "Moderate",
        },
        "Moderate",
    )

    assert "Use only the data below" in prompt
    assert "composite_score: 72" in prompt
    assert "Do not calculate" in prompt


def test_cloud_ollama_uses_bearer_auth(monkeypatch):
    captured = {}

    def fake_chat_ollama(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_client, "ChatOllama", fake_chat_ollama)
    monkeypatch.setenv("OLLAMA_MODE", "cloud")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-cloud-key")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b")

    llm_client.create_chat_ollama()

    assert captured["base_url"] == "https://ollama.com"
    assert captured["model"] == "gemma4:31b"
    assert captured["client_kwargs"]["headers"]["Authorization"] == "Bearer test-cloud-key"


def test_local_ollama_is_an_explicit_mode(monkeypatch):
    captured = {}

    def fake_chat_ollama(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_client, "ChatOllama", fake_chat_ollama)
    monkeypatch.setenv("OLLAMA_MODE", "local")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    llm_client.create_chat_ollama()

    assert captured["base_url"] == "http://127.0.0.1:11434"
    assert "client_kwargs" not in captured


def test_langfuse_config_requires_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    assert llm_client.langfuse_callback_config("recommendation-TCS") is None


def test_langfuse_config_sets_trace_metadata(monkeypatch):
    callback = object()
    langfuse_module = ModuleType("langfuse.langchain")
    langfuse_module.CallbackHandler = lambda: callback
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langfuse_module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    config = llm_client.langfuse_callback_config("recommendation-TCS", session_id="41")

    assert config == {
        "callbacks": [callback],
        "run_name": "recommendation-TCS",
        "metadata": {
            "langfuse_tags": ["stock-portfolio-analyzer", "recommendation-TCS"],
            "langfuse_session_id": "41",
        },
    }


def test_recommendation_generation_uses_validated_action_and_python_snapshot(monkeypatch):
    from recommendation_agent import RecommendationOutput

    class FakeStructuredModel:
        def invoke(self, messages, config=None):
            return RecommendationOutput(recommendation="HOLD", reasoning="Mixed supplied indicators.")

    class FakeChatModel:
        def with_structured_output(self, schema, method):
            assert method == "json_mode"
            assert schema is RecommendationOutput
            return FakeStructuredModel()

    monkeypatch.setattr("recommendation_agent.create_chat_ollama", lambda temperature: FakeChatModel())
    monkeypatch.setattr("recommendation_agent.langfuse_callback_config", lambda *args, **kwargs: None)
    metrics = {"composite_score": 52.0, "market_sentiment_label": "Neutral"}

    result = generate_ollama_recommendation("TCS", metrics, "Moderate")

    assert result["recommendation"] == "HOLD"
    assert result["inputs_snapshot"] is metrics
    assert result["model_status"] == "ollama"


def test_recommendation_generation_passes_langfuse_config(monkeypatch):
    from types import SimpleNamespace

    captured = {}

    class FakeStructuredModel:
        def invoke(self, messages, config=None):
            captured["config"] = config
            return SimpleNamespace(recommendation="HOLD", reasoning="Mixed supplied indicators.")

    class FakeChatModel:
        def with_structured_output(self, schema, method):
            return FakeStructuredModel()

    expected_config = {"callbacks": [object()], "run_name": "recommendation-TCS"}
    monkeypatch.setattr("recommendation_agent.create_chat_ollama", lambda temperature: FakeChatModel())
    monkeypatch.setattr("recommendation_agent.langfuse_callback_config", lambda *args, **kwargs: expected_config)

    generate_ollama_recommendation("TCS", {"composite_score": 52}, "Moderate", trace_session_id="41")

    assert captured["config"] is expected_config


def test_recommendation_run_persists_exact_inputs_and_result(monkeypatch):
    writes = []

    class FakeConn:
        def execute(self, query, params=()):
            writes.append((query, params))
            return type("Result", (), {"rowcount": 1})()

    from contextlib import contextmanager

    @contextmanager
    def fake_transaction():
        yield FakeConn()

    def fetch_one(query, params=()):
        if "SELECT risk_tier" in query:
            return {"risk_tier": "Moderate"}
        if "FROM sentiment_scores" in query:
            return None
        if "INSERT INTO agent_runs" in query:
            return {"id": 41}
        raise AssertionError(query)

    monkeypatch.setattr(main, "get_transaction", fake_transaction)
    monkeypatch.setattr(main, "fetch_one", fetch_one)
    monkeypatch.setattr(
        main,
        "fetch_all",
        lambda query, params=(): [
            {
                "stock_id": 7,
                "quantity": 10,
                "avg_cost": 95.0,
                "last_price": 100,
                "symbol": "EXAMPLE",
                "sector": "Technology",
                "volatility_1y": 22.0,
                "beta_1y": 1.1,
                "liquidity_flag": False,
                "market_cap_tier": "Large Cap",
                "composite_score": 45.0,
                "risk_label": "Moderate",
            }
        ],
    )
    monkeypatch.setattr(main, "execute", lambda query, params=(): writes.append((query, params)) or 1)
    monkeypatch.setattr(
        main,
        "generate_ollama_recommendation",
        lambda symbol, metrics, risk_tier, *, trace_session_id=None: {
            "recommendation": "HOLD",
            "reasoning": "Risk is moderate.",
            "inputs_snapshot": metrics,
            "model_status": "ollama",
        },
    )
    monkeypatch.setattr(
        main,
        "fetch_recent_sentiment",
        lambda *args, **kwargs: {"score": 55.0, "label": "Neutral", "age_days": 1},
    )

    response = TestClient(main.app).post("/recommendations/run")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["run_id"] == 41 and body["generated"] == 1
    recommendation_write = next(params for query, params in writes if "INSERT INTO recommendations" in query)
    recommendation = body["recommendations"][0]
    assert recommendation["recommendation"] == "HOLD"
    assert recommendation["model_status"] == "ollama"
    assert recommendation["inputs_snapshot"]["risk_tier"] == "Moderate"
    assert recommendation["inputs_snapshot"]["composite_score"] == 45.0
    saved_snapshot = json.loads(recommendation_write[4])
    assert saved_snapshot["risk_tier"] == "Moderate"
    assert saved_snapshot["composite_score"] == 45.0
    assert recommendation_write[2] == "HOLD"
