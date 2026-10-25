import argparse
import logging
import math
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from backend.db import execute, execute_many, fetch_all, fetch_one
from backend.market_hours import drop_incomplete_bars

logger = logging.getLogger(__name__)

UPSERT_DAILY_BAR_SQL = """
    INSERT INTO price_history (stock_id, date, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (stock_id, date) DO UPDATE
    SET open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
"""

# Exchange series suffixes Kite appends to some tradingsymbols (e.g. "XYZ-BE")
# that Yahoo does not use.
SERIES_SUFFIXES = ("-BE", "-BZ", "-SM", "-ST", "-IL", "-BL")
# If a freshly fetched close differs from the stored close for the same date by
# more than this, the provider has re-based history (split / bonus), so the
# stored series is on a different price basis and must be rebuilt.
REBASE_TOLERANCE = 0.01


def normalize_symbol(symbol: str, exchange: str | None = None) -> str:
    cleaned = symbol.strip().upper()
    if cleaned.endswith((".NS", ".BO")):
        return cleaned
    for suffix in SERIES_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    yahoo_suffix = ".BO" if (exchange or "").strip().upper() == "BSE" else ".NS"
    return f"{cleaned}{yahoo_suffix}"


def fetch_market_data(
    symbol: str,
    days: int | None = None,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    interval: str = "1d",
    exchange: str | None = None,
) -> pd.DataFrame:
    """Fetch COMPLETED daily OHLCV bars for a symbol.

    Pass either `days` (fetch the trailing N days from today) or an
    explicit `start`/`end`. We always call yfinance with start/end rather
    than its `period=` kwarg: period only accepts a fixed set of tokens
    (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max), so building a
    custom value like "3y" or "30d" raises YFInvalidPeriodError instead
    of returning data.

    Today's bar is dropped until the session has finished, so a partial
    intraday close/volume is never stored as if it were a final bar.
    """
    ticker = yf.Ticker(normalize_symbol(symbol, exchange))
    if end is None:
        end = datetime.now()
    if start is None:
        start = end - timedelta(days=days if days is not None else 1095)
    data = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
    if data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = [str(col).lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["date", "close"])
    df["volume"] = df["volume"].fillna(0)
    df = drop_incomplete_bars(df)
    return df.copy()


def get_stock_rows() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT h.stock_id, s.symbol, s.exchange
        FROM holdings h
        JOIN stocks s ON s.id = h.stock_id
        ORDER BY s.symbol
        """
    )


def get_last_date_for_stock(stock_id: int) -> date | None:
    row = fetch_one(
        "SELECT MAX(date) AS last_date FROM price_history WHERE stock_id = %s",
        (stock_id,),
    )
    if not row or row["last_date"] is None:
        return None
    return row["last_date"]


def get_stored_closes(stock_id: int, since: date) -> dict[date, float]:
    rows = fetch_all(
        "SELECT date, close FROM price_history WHERE stock_id = %s AND date >= %s",
        (stock_id, since),
    )
    return {row["date"]: float(row["close"]) for row in rows if row["close"] is not None}


def prices_rebased(stored: dict[date, float], fresh_rows: list[dict[str, Any]]) -> bool:
    """True if any overlapping date disagrees with what we stored by more than
    REBASE_TOLERANCE - the signature of a split/bonus adjustment."""
    for row in fresh_rows:
        old = stored.get(row["date"])
        new = row.get("close")
        if old is None or old <= 0 or new is None:
            continue
        if abs(float(new) / old - 1) > REBASE_TOLERANCE:
            return True
    return False


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _bar_params(stock_id: int, row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        stock_id,
        row["date"],
        _clean(row.get("open")),
        _clean(row.get("high")),
        _clean(row.get("low")),
        _clean(row.get("close")),
        _clean(row.get("volume")),
    )


def upsert_daily_bar(stock_id: int, row: dict[str, Any]) -> None:
    execute(UPSERT_DAILY_BAR_SQL, _bar_params(stock_id, row))


def upsert_daily_bars(stock_id: int, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return execute_many(UPSERT_DAILY_BAR_SQL, [_bar_params(stock_id, row) for row in rows])


def backfill_stock(symbol: str, stock_id: int, days: int = 1095, exchange: str | None = None) -> int:
    df = fetch_market_data(symbol, days=days, interval="1d", exchange=exchange)
    if df.empty:
        return 0

    return upsert_daily_bars(stock_id, df.to_dict(orient="records"))


def incremental_sync_stock(
    symbol: str,
    stock_id: int,
    lookback_days: int = 30,
    exchange: str | None = None,
) -> tuple[int, bool]:
    """Refresh recent bars. Returns (bars_written, rebased).

    The window deliberately overlaps already-stored dates: upserting them
    repairs any bar that was stored early, and comparing overlapping closes
    detects a split/bonus re-basing, in which case the whole history is rebuilt
    so the series stays on one consistent price basis.
    """
    last_date = get_last_date_for_stock(stock_id)
    if last_date is None:
        return backfill_stock(symbol, stock_id, days=1095, exchange=exchange), False

    start_dt = last_date - timedelta(days=max(lookback_days, 1))
    df = fetch_market_data(symbol, start=start_dt, interval="1d", exchange=exchange)
    if df.empty:
        return 0, False

    records = df.to_dict(orient="records")
    stored = get_stored_closes(stock_id, start_dt)
    if prices_rebased(stored, records):
        logger.warning("Price basis changed for %s (split/bonus?); rebuilding history.", symbol)
        return backfill_stock(symbol, stock_id, days=1095, exchange=exchange), True

    return upsert_daily_bars(stock_id, records), False


def sync_all_prices(backfill: bool = True, days: int = 1095, lookback_days: int = 30) -> dict[str, Any]:
    records = []
    for row in get_stock_rows():
        stock_id = row["stock_id"]
        symbol = row["symbol"]
        exchange = row.get("exchange")
        try:
            had_history = get_last_date_for_stock(stock_id) is not None
            rebased = False
            if backfill:
                inserted = backfill_stock(symbol, stock_id, days=days, exchange=exchange)
            else:
                inserted, rebased = incremental_sync_stock(
                    symbol, stock_id, lookback_days=lookback_days, exchange=exchange
                )
            if inserted == 0 and (backfill or not had_history):
                records.append(
                    {
                        "stock_id": stock_id,
                        "symbol": symbol,
                        "status": "no_data",
                        "inserted": 0,
                        "reason": "Market data provider returned no price bars.",
                    }
                )
            else:
                record = {"stock_id": stock_id, "symbol": symbol, "status": "ok", "inserted": inserted}
                if rebased:
                    record["rebased"] = True
                records.append(record)
        except Exception as exc:
            records.append({"stock_id": stock_id, "symbol": symbol, "status": "error", "error": str(exc)})

    status = "partial" if any(record["status"] != "ok" for record in records) else "ok"
    return {"status": status, "records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill or refresh daily OHLCV data for holdings.")
    parser.add_argument("--backfill", action="store_true", help="Backfill 3 years of daily OHLCV data.")
    parser.add_argument("--days", type=int, default=1095, help="Backfill length in days when --backfill is used.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Days of already-stored bars to re-fetch and verify in incremental mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backfill:
        result = sync_all_prices(backfill=True, days=args.days)
    else:
        result = sync_all_prices(backfill=False, lookback_days=args.lookback_days)
    print(result)


if __name__ == "__main__":
    main()
