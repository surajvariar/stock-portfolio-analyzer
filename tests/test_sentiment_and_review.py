
from fastapi.testclient import TestClient

import backend.main as main
from backend.review_harness import build_review_rows
from backend.sentiment import compute_market_sentiment, summarize_headlines
from backend.sentiment.market_data import fetch_fii_dii_flows
from backend.sentiment.news_sentiment import (
    HeadlineAssessment,
    summarize_news_with_ollama,
)


def test_compute_market_sentiment_is_bullish_when_market_is_strong():
    result = compute_market_sentiment(index_return=0.04, vix_value=12.0, fii_net_flow=1000.0, dii_net_flow=500.0)

    assert result["label"] == "Bullish"
    assert result["score"] >= 70


def test_summarize_headlines_returns_positive_summary():
    result = summarize_headlines([
        "Markets rise as earnings surprise boosts sentiment",
        "Strong buying sentiment continues across sectors",
    ])

    assert result["label"] == "Positive"
    assert result["score"] >= 60


def test_build_review_rows_tracks_inputs_snapshot():
    rows = build_review_rows([
        {
            "stock_symbol": "TCS",
            "recommendation": "BUY",
            "reasoning": "Momentum is strong.",
            "inputs_snapshot": {"score": 76, "risk_tier": "Moderate"},
        }
    ])

    assert rows[0]["stock_symbol"] == "TCS"
    assert rows[0]["recommendation"] == "BUY"
    assert rows[0]["inputs_snapshot"]["risk_tier"] == "Moderate"


def test_build_review_rows_calculates_forward_price_return():
    rows = build_review_rows(
        [
            {
                "stock_symbol": "TCS",
                "recommendation": "HOLD",
                "entry_price": 100,
                "latest_price": 112.5,
            }
        ]
    )

    assert rows[0]["forward_price_return_pct"] == 12.5
    assert rows[0]["review_status"] == "awaiting_price_data"


def test_fetch_fii_dii_flows_parses_verified_nse_fields(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"category": "DII", "date": "25-Sep-2026", "netValue": "2838.17"},
                {"category": "FII/FPI", "date": "25-Sep-2026", "netValue": "-3693.93"},
                {"category": "DII", "date": "24-Sep-2026", "netValue": "100.0"},
            ]

    class FakeSession:
        headers = {}

        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("backend.sentiment.market_data._nse_session", lambda: FakeSession())

    result = fetch_fii_dii_flows()

    assert result == {
        "fii_net_flow": -3693.93,
        "dii_net_flow": 2838.17,
        "flow_date": "2026-09-25",
    }


def test_market_sentiment_refresh_persists_quant_and_news_scores(monkeypatch):
    saved = []
    monkeypatch.setattr(main, "start_agent_run", lambda run_type: 6)
    monkeypatch.setattr(main, "finish_agent_run", lambda run_id, status, notes: None)
    monkeypatch.setattr(
        main,
        "fetch_market_inputs",
        lambda: {
            "index_return": 0.03,
            "vix_value": 12.0,
            "fii_net_flow": 1000.0,
            "dii_net_flow": 500.0,
            "above_sma_50": True,
            "above_sma_200": True,
        },
    )
    monkeypatch.setattr(main, "fetch_news_headlines", lambda query: [{"title": "Market gains"}])
    monkeypatch.setattr(
        main,
        "summarize_news_with_ollama",
        lambda headlines: {"label": "Positive", "score": 75.0, "summary": "Headlines are positive."},
    )
    monkeypatch.setattr(main, "execute", lambda query, params=(): saved.append(params) or 1)

    response = TestClient(main.app).post("/sentiment/refresh", json={"scope_type": "MARKET"})

    assert response.status_code == 200
    result = response.json()
    assert result["run_id"] == 6
    assert result["scope_value"] == "INDIA"
    assert result["composite_sentiment"] == 82.29
    assert result["label"] == "Positive"
    assert len(saved) == 1 and saved[0][0:2] == ("MARKET", "INDIA")


def test_ollama_invalid_news_label_uses_deterministic_label(monkeypatch):
    class FakeStructuredModel:
        def invoke(self, prompt):
            return {"parsed": None, "parsing_error": ValueError("Invalid category")}

    class FakeChatModel:
        def with_structured_output(self, schema, method, include_raw):
            assert method == "json_mode"
            assert include_raw is True
            assert schema is HeadlineAssessment
            return FakeStructuredModel()

    monkeypatch.setattr("backend.sentiment.news_sentiment.create_chat_ollama", lambda temperature: FakeChatModel())
    monkeypatch.setattr("backend.sentiment.news_sentiment.langfuse_callback_config", lambda trace_name: None)

    result = summarize_news_with_ollama(
        [{"title": "Shares rise with strong gains", "content": "Buying activity increased."}]
    )

    assert result["label"] == "Positive"
    assert result["score"] == 65.0
    assert result["model_status"] == "deterministic_fallback"


def test_ollama_structured_news_label_maps_to_fixed_score(monkeypatch):
    class FakeStructuredModel:
        def invoke(self, prompt):
            return {
                "parsed": HeadlineAssessment(label="Positive", summary="Buying interest strengthened."),
                "parsing_error": None,
            }

    class FakeChatModel:
        def with_structured_output(self, schema, method, include_raw):
            assert schema is HeadlineAssessment
            assert method == "json_mode" and include_raw is True
            return FakeStructuredModel()

    monkeypatch.setattr("backend.sentiment.news_sentiment.create_chat_ollama", lambda temperature: FakeChatModel())
    monkeypatch.setattr("backend.sentiment.news_sentiment.langfuse_callback_config", lambda trace_name: None)

    result = summarize_news_with_ollama([{"title": "Headline", "content": "Snippet"}])

    assert result == {
        "label": "Positive",
        "score": 75.0,
        "summary": "Buying interest strengthened.",
        "model_status": "ollama_structured",
    }


def test_news_sentiment_passes_langfuse_config(monkeypatch):
    captured = {}

    class FakeStructuredModel:
        def invoke(self, prompt, config=None):
            captured["config"] = config
            return {
                "parsed": HeadlineAssessment(label="Neutral", summary="Headlines are mixed."),
                "parsing_error": None,
            }

    class FakeChatModel:
        def with_structured_output(self, schema, method, include_raw):
            return FakeStructuredModel()

    expected_config = {"callbacks": [object()], "run_name": "news-sentiment"}
    monkeypatch.setattr("backend.sentiment.news_sentiment.create_chat_ollama", lambda temperature: FakeChatModel())
    monkeypatch.setattr("backend.sentiment.news_sentiment.langfuse_callback_config", lambda trace_name: expected_config)

    result = summarize_news_with_ollama([{"title": "Headline", "content": "Snippet"}])

    assert captured["config"] is expected_config
    assert result["model_status"] == "ollama_structured"
