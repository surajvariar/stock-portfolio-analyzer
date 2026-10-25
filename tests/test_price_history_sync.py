from types import SimpleNamespace

from fastapi.testclient import TestClient

import backend.main as main
import backend.price_history_sync as price_history_sync


def test_incremental_sync_backfills_stocks_without_price_history(monkeypatch):
    backfills = []
    monkeypatch.setattr(
        price_history_sync,
        "get_stock_rows",
        lambda: [{"stock_id": 7, "symbol": "EXAMPLE"}],
    )
    monkeypatch.setattr(price_history_sync, "get_last_date_for_stock", lambda stock_id: None)
    monkeypatch.setattr(
        price_history_sync,
        "backfill_stock",
        lambda symbol, stock_id, days=1095, exchange=None: backfills.append((symbol, stock_id, days, exchange)) or 12,
    )

    result = price_history_sync.sync_all_prices(backfill=False)

    assert backfills == [("EXAMPLE", 7, 1095, None)]
    assert result == {
        "status": "ok",
        "records": [{"stock_id": 7, "symbol": "EXAMPLE", "status": "ok", "inserted": 12}],
    }


def test_backfill_batches_daily_bar_upserts(monkeypatch):
    import pandas as pd

    rows = pd.DataFrame(
        [
            {"date": pd.Timestamp("2025-01-01").date(), "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"date": pd.Timestamp("2025-01-02").date(), "open": 10, "high": 12, "low": 10, "close": 11, "volume": 150},
        ]
    )
    batches = []
    monkeypatch.setattr(price_history_sync, "fetch_market_data", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        price_history_sync,
        "execute_many",
        lambda query, params: batches.append((query, list(params))) or len(params),
    )

    result = price_history_sync.backfill_stock("EXAMPLE", 7)

    assert result == 2
    assert len(batches) == 1
    assert len(batches[0][1]) == 2
    assert all(params[0] == 7 for params in batches[0][1])


def test_price_sync_reports_one_stock_failure_and_continues(monkeypatch):
    monkeypatch.setattr(
        price_history_sync,
        "get_stock_rows",
        lambda: [
            {"stock_id": 7, "symbol": "BROKEN"},
            {"stock_id": 8, "symbol": "WORKING"},
        ],
    )
    monkeypatch.setattr(price_history_sync, "get_last_date_for_stock", lambda stock_id: None)

    def backfill(symbol, stock_id, days=1095, exchange=None):
        if symbol == "BROKEN":
            raise RuntimeError("market data unavailable")
        return 12

    monkeypatch.setattr(price_history_sync, "backfill_stock", backfill)

    result = price_history_sync.sync_all_prices(backfill=True)

    assert result["status"] == "partial"
    assert result["records"] == [
        {"stock_id": 7, "symbol": "BROKEN", "status": "error", "error": "market data unavailable"},
        {"stock_id": 8, "symbol": "WORKING", "status": "ok", "inserted": 12},
    ]


def test_initial_sync_reports_empty_provider_result_as_no_data(monkeypatch):
    monkeypatch.setattr(
        price_history_sync,
        "get_stock_rows",
        lambda: [{"stock_id": 7, "symbol": "UNSUPPORTED"}],
    )
    monkeypatch.setattr(price_history_sync, "get_last_date_for_stock", lambda stock_id: None)
    monkeypatch.setattr(price_history_sync, "backfill_stock", lambda symbol, stock_id, days=1095, exchange=None: 0)

    result = price_history_sync.sync_all_prices(backfill=False)

    assert result == {
        "status": "partial",
        "records": [
            {
                "stock_id": 7,
                "symbol": "UNSUPPORTED",
                "status": "no_data",
                "inserted": 0,
                "reason": "Market data provider returned no price bars.",
            }
        ],
    }


def test_holdings_sync_runs_incremental_price_sync(monkeypatch):
    sync_calls = []
    monkeypatch.setattr(main, "get_authenticated_kite", lambda: object())
    monkeypatch.setattr(
        main,
        "fetch_equity_holdings",
        lambda kite: [
            SimpleNamespace(
                symbol="EXAMPLE",
                isin="TESTISIN",
                exchange="NSE",
                quantity=2,
                average_price=100,
                last_price=110,
            )
        ],
    )
    monkeypatch.setattr(main, "fetch_one", lambda query, params=(): {"id": 7})
    monkeypatch.setattr(main, "execute", lambda query, params=(): 1)

    def sync_prices(backfill):
        sync_calls.append(backfill)
        return {
            "status": "ok",
            "records": [{"stock_id": 7, "symbol": "EXAMPLE", "status": "ok", "inserted": 12}],
        }

    monkeypatch.setattr(main, "sync_all_prices", sync_prices)

    response = TestClient(main.app).post("/sync/holdings")

    assert response.status_code == 200
    assert sync_calls == [False]
    assert response.json()["price_history"] == {
        "status": "ok",
        "stocks_processed": 1,
        "bars_upserted": 12,
        "rebuilt_after_split_or_bonus": [],
        "issues": [],
    }