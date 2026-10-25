import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from functools import wraps
from typing import Any, Literal

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from db import (
    ensure_schema,
    execute,
    execute_many,
    fetch_all,
    fetch_one,
    finish_agent_run,
    get_latest_kite_access_token,
    get_transaction,
    mark_interrupted_runs,
    recent_fii_dii_flows,
    save_fii_dii_flow,
    save_kite_access_token,
    start_agent_run,
)
from etf_analyzer import (
    ETF_HISTORY_DAYS,
    PLAIN_NOT_ENOUGH_HISTORY,
    analyze_equity_etfs,
    fetch_nse_etf_universe,
    is_small_cap_etf,
)
from frontend_redirect import (
    RETURN_COOKIE,
    RETURN_COOKIE_MAX_AGE,
    build_redirect_url,
    safe_return_url,
)
from kite_client import (
    fetch_equity_holdings,
    generate_access_token,
    get_authenticated_kite,
    get_login_url,
)
from market_hours import drop_incomplete_series
from plain_language import build_explanation, compose
from portfolio_aggregation import compute_portfolio_summary, group_weights_pct
from price_history_sync import sync_all_prices
from recommendation_agent import generate_ollama_recommendation
from review_harness import build_review_rows
from risk_questionnaire import calculate_risk_profile
from risk_scoring.beta_calculator import estimate_beta
from risk_scoring.composite_risk_score import composite_risk_label, composite_risk_score
from risk_scoring.liquidity_flagger import liquidity_flag as compute_liquidity_flag
from risk_scoring.liquidity_flagger import median_daily_turnover
from risk_scoring.market_cap_classifier import classify_market_cap_rank
from risk_scoring.volatility_calculator import (
    MIN_RETURN_OBSERVATIONS,
    annualized_volatility,
)
from sentiment.market_data import fetch_market_inputs
from sentiment.news_sentiment import fetch_news_headlines, summarize_news_with_ollama
from sentiment.quant_sentiment import (
    composite_sentiment_label,
    compute_market_sentiment,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Sentiment older than this is not fed to recommendations.
MARKET_SENTIMENT_MAX_AGE_DAYS = 3
STOCK_SENTIMENT_MAX_AGE_DAYS = 7
# NSE exposes only the latest day's FII/DII flow, so a rolling window is built
# from the days we have stored.
FLOW_WINDOW_DAYS = 5
MIN_FLOW_DAYS = 3
RECOMMENDATION_MAX_WORKERS = max(1, int(os.getenv("RECOMMENDATION_MAX_WORKERS", "4")))

# Only one recommendation run at a time (they are slow and write a run record).
_recommendation_lock = threading.Lock()
_etf_analysis_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_schema()
        mark_interrupted_runs()
    except Exception as exc:  # noqa: BLE001 - the app should still start if Postgres is briefly down
        logger.warning("Startup schema/run cleanup skipped: %s", exc)
    yield


app = FastAPI(title="Stock Portfolio Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Note: For production, replace "*" with your frontend's specific URL (e.g., ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def track_agent_run(run_type: str):
    def decorate(endpoint):
        @wraps(endpoint)
        def tracked_endpoint(*args, **kwargs):
            run_id = start_agent_run(run_type)
            try:
                result = endpoint(*args, **kwargs)
            except Exception as exc:
                finish_agent_run(run_id, "FAILED", type(exc).__name__)
                raise

            status = "PARTIAL" if isinstance(result, dict) and result.get("status") == "partial" else "COMPLETED"
            finish_agent_run(run_id, status, f"{endpoint.__name__} completed with status {status.lower()}.")
            if isinstance(result, dict):
                result["run_id"] = run_id
            return result

        return tracked_endpoint

    return decorate


class StockMetadataInput(BaseModel):
    symbol: str = Field(min_length=1)
    market_cap_rank: int | None = Field(default=None, ge=1)
    sector: str | None = None
    trade_to_trade_flag: bool | None = None
    isin: str | None = None
    exchange: str | None = None


class SentimentRefreshRequest(BaseModel):
    scope_type: Literal["MARKET", "STOCK", "SECTOR"] = "MARKET"
    scope_value: str | None = None


class RiskQuestionnaireRequest(BaseModel):
    income_stability: int = Field(ge=1, le=5)
    investment_experience: int = Field(ge=1, le=5)
    time_horizon: int = Field(ge=1, le=5)
    drawdown_reaction: int = Field(ge=1, le=5)
    liquidity_needs: int = Field(ge=1, le=5)


def fetch_nifty_50_prices(days: int = 1095) -> pd.Series:
    """Benchmark series for beta and the review harness. Fetched live, not
    from `price_history` - that table is scoped to the user's actual holdings,
    and the index itself isn't a held position.

    Uses start/end dates rather than yfinance's `period=` kwarg: period
    only accepts a fixed set of tokens (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y,
    10y, ytd, max), so a custom value like "1095d" raises
    YFInvalidPeriodError instead of returning data.

    The index is normalized to tz-naive calendar dates (yfinance returns
    Asia/Kolkata-aware timestamps; the stock series read from Postgres is
    naive, and the two never join otherwise). The in-progress bar for today
    is dropped so beta only sees completed sessions.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    data = yf.Ticker("^NSEI").history(start=start, end=end, interval="1d", auto_adjust=False)
    if data.empty:
        return pd.Series(dtype=float)
    df = data.reset_index()
    df.columns = [str(col).lower() for col in df.columns]
    dates = pd.to_datetime(df["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    closes = pd.Series(df["close"].to_numpy(), index=dates.dt.normalize()).dropna()
    return drop_incomplete_series(closes)


def fetch_stock_prices_from_db(stock_id: int) -> tuple[pd.Series, pd.Series]:
    """Reads from `price_history` (populated by price_history_sync.py).
    Not a live yfinance call - symbol normalization (.NS/.BO suffix) already
    happened once, at sync time, so it doesn't need to be repeated here."""
    rows = fetch_all(
        "SELECT date, close, volume FROM price_history WHERE stock_id = %s AND close IS NOT NULL ORDER BY date",
        (stock_id,),
    )
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    idx = pd.to_datetime([r["date"] for r in rows])
    closes = pd.Series([float(r["close"]) for r in rows], index=idx)
    volumes = pd.Series([float(r["volume"] or 0) for r in rows], index=idx)
    return closes, volumes


def _age_days(moment: datetime) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (datetime.now(UTC) - moment).total_seconds() / 86400


def fetch_recent_sentiment(scope_type: str, scope_value: str, max_age_days: int) -> dict[str, Any] | None:
    """Latest stored sentiment for a scope, or None if there is none or it is
    older than `max_age_days` (stale sentiment must not drive today's advice)."""
    row = fetch_one(
        """
        SELECT composite_sentiment, label, created_at
        FROM sentiment_scores
        WHERE scope_type = %s AND scope_value = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (scope_type, scope_value),
    )
    if not row:
        return None
    age = _age_days(row["created_at"])
    if age > max_age_days:
        return None
    return {
        "score": float(row["composite_sentiment"]),
        "label": row["label"],
        "age_days": round(age, 1),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/kite/login-url")
def auth_kite_login_url() -> dict[str, str]:
    return {"login_url": get_login_url()}


@app.get("/login")
@app.get("/auth/kite/login")
def auth_kite_login_redirect(return_to: str | None = None) -> RedirectResponse:
    """Send the browser to Kite. `return_to` is the dashboard page the user came from;
    it is validated and remembered in a short-lived cookie so the callback can send
    them back there instead of leaving them on a JSON page."""
    response = RedirectResponse(get_login_url())
    target = safe_return_url(return_to)
    if target:
        response.set_cookie(
            RETURN_COOKIE, target, max_age=RETURN_COOKIE_MAX_AGE, httponly=True, samesite="lax"
        )
    return response


def _finish_kite_login(request: Request, ok: bool, message: str, error_status: int = 400) -> Response:
    """Return to the dashboard only when this app initiated the Kite login."""
    target = safe_return_url(request.cookies.get(RETURN_COOKIE))
    if target:
        response = RedirectResponse(
            build_redirect_url(target, "success" if ok else "error", None if ok else message),
            status_code=303,
        )
        response.delete_cookie(RETURN_COOKIE)
        return response
    if ok:
        return JSONResponse(content={"status": "success", "message": message})
    return JSONResponse(status_code=error_status, content={"error": message})


@app.get("/callback", response_model=None)
@app.get("/auth/kite/callback", response_model=None)
def auth_kite_callback(request: Request) -> Response:
    request_token = request.query_params.get("request_token")
    if not request_token:
        # Kite sends status=error / no token when the user cancels or login fails.
        return _finish_kite_login(
            request,
            ok=False,
            message="No request_token received. The login may have been cancelled, or the Redirect URL in Kite is wrong.",
        )

    try:
        access_token = generate_access_token(request_token)
    except Exception as exc:  # noqa: BLE001 - surface provider-specific Kite auth failures to the browser
        return _finish_kite_login(request, ok=False, message=str(exc))

    save_kite_access_token(access_token)
    return _finish_kite_login(request, ok=True, message="Logged in and session saved. Now call /sync/holdings")


@app.get("/stocks")
def list_stocks() -> list[dict[str, Any]]:
    return fetch_all("SELECT * FROM stocks ORDER BY symbol")


@app.get("/stocks/metadata/status")
def stock_metadata_status() -> dict[str, Any]:
    ensure_schema()
    return fetch_one(
        """
        SELECT COUNT(*) AS total_stocks,
               COUNT(*) FILTER (WHERE market_cap_rank IS NOT NULL) AS ranked_stocks,
               COUNT(*) FILTER (WHERE sector IS NOT NULL) AS sector_mapped_stocks,
               COUNT(*) FILTER (WHERE trade_to_trade_flag) AS trade_to_trade_stocks
        FROM stocks
        """
    ) or {}


@app.post("/stocks/metadata/import")
def import_stock_metadata(records: list[StockMetadataInput]) -> dict[str, Any]:
    ensure_schema()
    if not records:
        raise HTTPException(status_code=422, detail="Provide at least one stock metadata record.")

    # `%s::boolean` gives Postgres a type for the NULL test; without the cast an
    # untyped None parameter fails with "could not determine data type".
    sql = """
        INSERT INTO stocks (symbol, isin, exchange, market_cap_rank, market_cap_tier, sector, trade_to_trade_flag)
        VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s::boolean, FALSE))
        ON CONFLICT (symbol) DO UPDATE SET
            isin = COALESCE(EXCLUDED.isin, stocks.isin),
            exchange = COALESCE(EXCLUDED.exchange, stocks.exchange),
            market_cap_rank = COALESCE(EXCLUDED.market_cap_rank, stocks.market_cap_rank),
            market_cap_tier = CASE
                WHEN EXCLUDED.market_cap_rank IS NOT NULL THEN EXCLUDED.market_cap_tier
                ELSE stocks.market_cap_tier
            END,
            sector = COALESCE(EXCLUDED.sector, stocks.sector),
            trade_to_trade_flag = CASE
                WHEN %s::boolean IS NULL THEN stocks.trade_to_trade_flag
                ELSE EXCLUDED.trade_to_trade_flag
            END,
            updated_at = NOW()
    """
    params = []
    for record in records:
        symbol = record.symbol.strip().upper()
        if not symbol:
            raise HTTPException(status_code=422, detail="Stock symbols cannot be blank.")
        rank = record.market_cap_rank
        params.append(
            (
                symbol,
                record.isin,
                record.exchange,
                rank,
                classify_market_cap_rank(rank) if rank is not None else None,
                record.sector,
                record.trade_to_trade_flag,
                record.trade_to_trade_flag,
            )
        )

    execute_many(sql, params)
    return {"status": "ok", "updated": len(params)}


def aggregate_holdings(holdings: list[Any]) -> list[dict[str, Any]]:
    """Merge Kite rows by symbol. The same symbol can be held on both NSE and
    BSE; holdings are keyed per stock, so those rows must be summed (with a
    quantity-weighted average cost), not left to overwrite each other.
    Zero-quantity rows are not positions and are dropped."""
    merged: dict[str, dict[str, Any]] = {}
    for h in holdings:
        quantity = float(h.quantity or 0)
        if quantity <= 0:
            continue
        entry = merged.get(h.symbol)
        if entry is None:
            merged[h.symbol] = {
                "symbol": h.symbol,
                "isin": h.isin,
                "exchange": h.exchange,
                "quantity": quantity,
                "cost": quantity * float(h.average_price or 0),
                "last_price": float(h.last_price or 0),
            }
            continue
        entry["quantity"] += quantity
        entry["cost"] += quantity * float(h.average_price or 0)
        if h.exchange == "NSE":  # prefer the NSE listing's exchange and price
            entry["exchange"] = "NSE"
            entry["last_price"] = float(h.last_price or 0)
        entry["isin"] = entry["isin"] or h.isin
    return [
        {
            "symbol": e["symbol"],
            "isin": e["isin"],
            "exchange": e["exchange"],
            "quantity": e["quantity"],
            "avg_cost": e["cost"] / e["quantity"],
            "last_price": e["last_price"],
        }
        for e in merged.values()
    ]


@app.post("/sync/holdings", response_model=None)
def sync_holdings() -> dict[str, Any] | JSONResponse:
    kite = get_authenticated_kite()
    if not kite:
        return JSONResponse(status_code=401, content={"error": "Not logged in. Visit /login first."})

    try:
        raw_holdings = fetch_equity_holdings(kite)
    except Exception as exc:  # pragma: no cover - runtime integration case
        raise HTTPException(status_code=502, detail=f"Kite fetch failed: {exc}") from exc

    if not raw_holdings:
        # An empty answer is ambiguous (API hiccup vs. everything sold), so the
        # stored holdings are deliberately left untouched.
        return {"status": "ok", "synced": 0, "message": "No holdings returned by Kite; stored holdings left unchanged."}

    ensure_schema()
    holdings = aggregate_holdings(raw_holdings)

    # One transaction: either the whole snapshot lands or none of it does.
    with get_transaction() as conn:
        synced_ids: list[int] = []
        for h in holdings:
            stock = conn.execute(
                """
                INSERT INTO stocks (symbol, isin, exchange) VALUES (%s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    isin = COALESCE(NULLIF(EXCLUDED.isin, ''), stocks.isin),
                    exchange = EXCLUDED.exchange,
                    updated_at = NOW()
                RETURNING id
                """,
                (h["symbol"], h["isin"], h["exchange"]),
            ).fetchone()
            stock_id = stock["id"]
            synced_ids.append(stock_id)

            conn.execute(
                """
                INSERT INTO holdings (stock_id, quantity, avg_cost, last_price, synced_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (stock_id) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    avg_cost = EXCLUDED.avg_cost,
                    last_price = EXCLUDED.last_price,
                    synced_at = NOW()
                """,
                (stock_id, h["quantity"], h["avg_cost"], h["last_price"]),
            )
            conn.execute(
                "INSERT INTO holdings_history (stock_id, quantity, avg_cost, last_price, synced_at) VALUES (%s, %s, %s, %s, NOW())",
                (stock_id, h["quantity"], h["avg_cost"], h["last_price"]),
            )

        # Positions that are no longer in the Kite snapshot have been sold.
        # holdings_history keeps their past, so only the current snapshot is pruned.
        removed = [
            row["symbol"]
            for row in conn.execute(
                """
                DELETE FROM holdings h
                USING stocks s
                WHERE s.id = h.stock_id AND NOT (h.stock_id = ANY(%s::int[]))
                RETURNING s.symbol
                """,
                (synced_ids,),
            ).fetchall()
        ]

    price_sync = sync_all_prices(backfill=False)
    price_records = price_sync["records"]
    return {
        "status": "ok",
        "synced": len(holdings),
        "removed_sold_positions": removed,
        "price_history": {
            "status": price_sync["status"],
            "stocks_processed": len(price_records),
            "bars_upserted": sum(record.get("inserted", 0) for record in price_records),
            "rebuilt_after_split_or_bonus": [r["symbol"] for r in price_records if r.get("rebased")],
            "issues": [record for record in price_records if record["status"] != "ok"],
        },
    }


@app.get("/holdings")
def get_holdings_db() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT h.*, s.symbol, s.exchange, s.isin, s.sector
        FROM holdings h
        JOIN stocks s ON s.id = h.stock_id
        ORDER BY s.symbol
        """
    )


@app.post("/risk/questionnaire")
def save_risk_questionnaire(payload: RiskQuestionnaireRequest) -> dict[str, Any]:
    answers = payload.model_dump()
    profile = calculate_risk_profile(answers)
    execute(
        """
        INSERT INTO risk_profile (total_score, risk_tier, capacity_score, tolerance_score, answers)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            profile["total_score"],
            profile["risk_tier"],
            profile["capacity_score"],
            profile["tolerance_score"],
            json.dumps(answers),
        ),
    )
    return {"status": "ok", "data": profile}


@app.get("/risk/questionnaire/latest")
def latest_risk_questionnaire() -> dict[str, Any]:
    profile = fetch_one(
        """
        SELECT total_score, risk_tier, capacity_score, tolerance_score, answers, created_at
        FROM risk_profile
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    return {"data": profile}


def resolve_flows(market_inputs: dict[str, Any]) -> tuple[float | None, float | None, int]:
    """FII/DII net flow for the sentiment score: a rolling sum over the recent
    trading days we have stored when there are enough, else the latest day alone.
    Returns (fii, dii, days_in_window)."""
    fii = market_inputs.get("fii_net_flow")
    dii = market_inputs.get("dii_net_flow")
    flow_date = market_inputs.get("flow_date")
    if fii is None or dii is None or not flow_date:
        return None, None, 0

    try:
        latest = date.fromisoformat(flow_date)
        save_fii_dii_flow(latest, fii, dii)
        recent = [
            row
            for row in recent_fii_dii_flows(FLOW_WINDOW_DAYS)
            if (latest - row["flow_date"]).days <= 9  # ignore stale rows from long gaps
        ]
    except Exception as exc:  # noqa: BLE001 - history is an enhancement, the single day still works
        logger.warning("FII/DII history unavailable (%s); using the latest day only.", type(exc).__name__)
        return fii, dii, 1

    if len(recent) >= MIN_FLOW_DAYS:
        return (
            sum(float(row["fii_net"]) for row in recent),
            sum(float(row["dii_net"]) for row in recent),
            len(recent),
        )
    return fii, dii, 1


@app.post("/sentiment/refresh")
@track_agent_run("SENTIMENT")
def refresh_sentiment(payload: SentimentRefreshRequest) -> dict[str, Any]:
    scope_type = payload.scope_type
    scope_value = payload.scope_value.strip() if payload.scope_value else None
    if scope_type != "MARKET" and not scope_value:
        raise HTTPException(status_code=422, detail="scope_value is required for STOCK and SECTOR sentiment.")
    if scope_type == "STOCK":
        scope_value = scope_value.upper()

    quantitative: dict[str, Any] | None = None
    quant_score: float | None = None
    if scope_type == "MARKET":
        try:
            market_inputs = fetch_market_inputs()
            fii_flow, dii_flow, flow_window_days = resolve_flows(market_inputs)
            quantitative = compute_market_sentiment(
                index_return=market_inputs["index_return"],
                vix_value=market_inputs["vix_value"],
                fii_net_flow=fii_flow,
                dii_net_flow=dii_flow,
                above_sma_50=market_inputs["above_sma_50"],
                above_sma_200=market_inputs["above_sma_200"],
            )
            quantitative["flow_window_days"] = flow_window_days
            quantitative["warnings"] = market_inputs.get("warnings", [])
            quant_score = float(quantitative["score"])
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Market sentiment input fetch failed: {exc}") from exc
        scope_value = "INDIA"
        query = "Indian stock market Nifty Sensex NSE market news"
    elif scope_type == "STOCK":
        query = f"{scope_value} NSE India stock company news"
    else:
        query = f"Indian stock market {scope_value} sector news"

    try:
        headlines = fetch_news_headlines(query)
        qualitative = summarize_news_with_ollama(headlines)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"News sentiment fetch failed: {exc}") from exc

    qual_score = float(qualitative["score"])
    composite = qual_score if quant_score is None else 0.7 * quant_score + 0.3 * qual_score
    label = composite_sentiment_label(composite)
    source_summary = json.dumps(
        {"quantitative": quantitative, "qualitative": qualitative, "headlines": headlines},
        ensure_ascii=True,
    )
    execute(
        """
        INSERT INTO sentiment_scores
            (scope_type, scope_value, quant_score, qual_score, composite_sentiment, label, source_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (scope_type, scope_value, quant_score, qual_score, round(composite, 2), label, source_summary),
    )
    return {
        "status": "ok",
        "scope_type": scope_type,
        "scope_value": scope_value,
        "quantitative": quantitative,
        "qualitative": qualitative,
        "composite_sentiment": round(composite, 2),
        "label": label,
        "headline_count": len(headlines),
    }


@app.get("/sentiment/latest")
def latest_sentiment() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT DISTINCT ON (scope_type, scope_value)
               scope_type, scope_value, quant_score, qual_score,
               composite_sentiment, label, source_summary, created_at
        FROM sentiment_scores
        ORDER BY scope_type, scope_value, created_at DESC
        """
    )


@app.post("/risk/score-stocks")
@track_agent_run("RISK_SCORING")
def score_stocks() -> dict[str, Any]:
    ensure_schema()
    stock_rows = fetch_all(
        """
        SELECT h.stock_id, s.symbol, s.market_cap_rank, s.market_cap_tier,
               s.sector, s.trade_to_trade_flag
        FROM holdings h
        JOIN stocks s ON s.id = h.stock_id
        ORDER BY s.symbol
        """
    )
    if not stock_rows:
        return {"status": "ok", "records": []}

    try:
        market_prices = fetch_nifty_50_prices(days=1095)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Nifty 50 benchmark fetch failed: {exc}") from exc
    if market_prices.empty:
        raise HTTPException(
            status_code=502,
            detail="Nifty 50 benchmark prices are unavailable, so beta cannot be computed. Try again shortly.",
        )
    market_returns = market_prices.pct_change().dropna()

    records: list[dict[str, Any]] = []
    for stock in stock_rows:
        symbol = stock["symbol"]
        close_prices, volumes = fetch_stock_prices_from_db(stock["stock_id"])

        def skip(reason: str) -> None:
            # A previous score must not linger and be mistaken for a current one.
            execute("DELETE FROM risk_scores WHERE stock_id = %s", (stock["stock_id"],))
            records.append(
                {"stock_id": stock["stock_id"], "symbol": symbol, "status": "skipped", "reason": reason}
            )

        if close_prices.empty:
            skip("No price history in DB yet - run /sync/holdings (or price_history_sync.py) first.")
            continue

        # Volatility calculator expects raw prices (it differences internally) -
        # pass close_prices directly, not pre-differenced returns.
        volatility = annualized_volatility(close_prices)
        if volatility is None:
            skip(
                f"Only {max(len(close_prices) - 1, 0)} daily returns available; "
                f"at least {MIN_RETURN_OBSERVATIONS} are needed for a reliable risk score."
            )
            continue

        # Beta needs date-aligned return series; estimate_beta normalizes both
        # indexes to plain dates so tz-aware and tz-naive series still line up.
        stock_returns = close_prices.pct_change().dropna()
        beta_estimate = estimate_beta(stock_returns, market_returns)
        if beta_estimate.beta is None:
            skip(
                f"Only {beta_estimate.observations} overlapping days with the Nifty 50; "
                "not enough to estimate beta."
            )
            continue
        beta = beta_estimate.beta

        market_cap_tier = stock.get("market_cap_tier") or classify_market_cap_rank(stock.get("market_cap_rank"))
        turnover = median_daily_turnover(close_prices, volumes)
        flag = compute_liquidity_flag(turnover, stock.get("trade_to_trade_flag", False))
        composite_score = composite_risk_score(volatility, beta, market_cap_tier, flag)
        risk_label = composite_risk_label(composite_score)
        history_days = len(close_prices)

        execute(
            "UPDATE stocks SET market_cap_tier = %s, updated_at = NOW() WHERE id = %s",
            (market_cap_tier, stock["stock_id"]),
        )

        execute(
            """
            INSERT INTO risk_scores (stock_id, volatility_1y, beta_1y, liquidity_flag, market_cap_tier,
                                     composite_score, risk_label, history_days, beta_observations,
                                     median_daily_turnover)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stock_id) DO UPDATE SET
                volatility_1y = EXCLUDED.volatility_1y,
                beta_1y = EXCLUDED.beta_1y,
                liquidity_flag = EXCLUDED.liquidity_flag,
                market_cap_tier = EXCLUDED.market_cap_tier,
                composite_score = EXCLUDED.composite_score,
                risk_label = EXCLUDED.risk_label,
                history_days = EXCLUDED.history_days,
                beta_observations = EXCLUDED.beta_observations,
                median_daily_turnover = EXCLUDED.median_daily_turnover,
                computed_at = NOW()
            """,
            (
                stock["stock_id"],
                round(volatility, 4),
                round(beta, 4),
                flag,
                market_cap_tier,
                round(composite_score, 2),
                risk_label,
                history_days,
                beta_estimate.observations,
                round(turnover, 2) if turnover is not None else None,
            ),
        )

        records.append(
            {
                "stock_id": stock["stock_id"],
                "symbol": symbol,
                "volatility_1y": round(volatility, 4),
                "beta_1y": round(beta, 4),
                "beta_observations": beta_estimate.observations,
                "liquidity_flag": flag,
                "median_daily_turnover": round(turnover, 2) if turnover is not None else None,
                "market_cap_tier": market_cap_tier,
                "composite_score": round(composite_score, 2),
                "risk_label": risk_label,
            }
        )

    # "partial" lets track_agent_run record the run as PARTIAL, not COMPLETED.
    status = "partial" if any(r.get("status") == "skipped" for r in records) else "ok"
    return {"status": status, "records": records}


@app.get("/risk/scores")
def get_risk_scores() -> list[dict[str, Any]]:
    ensure_schema()
    return fetch_all(
        """
        SELECT s.symbol, s.sector, s.market_cap_rank, rs.market_cap_tier,
               rs.volatility_1y, rs.beta_1y, rs.liquidity_flag,
               rs.composite_score, rs.risk_label, rs.computed_at,
               rs.history_days, rs.beta_observations, rs.median_daily_turnover
        FROM risk_scores rs
        JOIN stocks s ON s.id = rs.stock_id
        ORDER BY s.symbol
        """
    )


@app.get("/portfolio/summary")
def get_portfolio_summary() -> dict[str, Any]:
    rows = fetch_all(
        """
        SELECT s.symbol, s.sector, rs.market_cap_tier,
               h.quantity, h.last_price,
               rs.composite_score AS risk_score
        FROM holdings h
        JOIN stocks s ON s.id = h.stock_id
        LEFT JOIN risk_scores rs ON rs.stock_id = h.stock_id
        ORDER BY s.symbol
        """
    )
    profile = fetch_one("SELECT risk_tier FROM risk_profile ORDER BY created_at DESC LIMIT 1")
    holdings = [
        {
            **row,
            "position_value": float(row["quantity"] or 0) * float(row["last_price"] or 0),
            "risk_tier": profile["risk_tier"] if profile else None,
        }
        for row in rows
    ]
    return compute_portfolio_summary(holdings)


def _prepare_recommendation_run() -> dict[str, Any]:
    """Validate preconditions, gather every input in Python, and open the run
    record. Raises HTTPException (and creates no run) if prerequisites are missing."""
    ensure_schema()
    profile = fetch_one("SELECT risk_tier FROM risk_profile ORDER BY created_at DESC LIMIT 1")
    if not profile:
        raise HTTPException(status_code=409, detail="Complete the risk questionnaire before generating recommendations.")
    risk_tier = profile["risk_tier"]

    rows = fetch_all(
        """
        SELECT h.stock_id, h.quantity, h.avg_cost, h.last_price, s.symbol, s.sector,
               rs.volatility_1y, rs.beta_1y, rs.liquidity_flag,
               rs.market_cap_tier, rs.composite_score, rs.risk_label
        FROM holdings h
        JOIN stocks s ON s.id = h.stock_id
        LEFT JOIN risk_scores rs ON rs.stock_id = h.stock_id
        ORDER BY s.symbol
        """
    )
    eligible = [row for row in rows if row["composite_score"] is not None]
    if not eligible:
        raise HTTPException(status_code=409, detail="Run stock risk scoring before generating recommendations.")

    sized = [
        {
            "symbol": row["symbol"],
            "sector": row["sector"],
            "market_cap_tier": row["market_cap_tier"],
            "position_value": float(row["quantity"] or 0) * float(row["last_price"] or 0),
            "risk_score": float(row["composite_score"]) if row["composite_score"] is not None else None,
            "risk_tier": risk_tier,
        }
        for row in rows
    ]
    portfolio_summary = compute_portfolio_summary(sized)
    total_value = portfolio_summary["total_value"] or 0.0
    sector_weights = group_weights_pct(sized, "sector")

    market_sentiment = fetch_recent_sentiment("MARKET", "INDIA", MARKET_SENTIMENT_MAX_AGE_DAYS)

    jobs: list[dict[str, Any]] = []
    for row, item in zip(rows, sized):
        if row["composite_score"] is None:
            continue
        avg_cost = float(row["avg_cost"] or 0)
        last_price = float(row["last_price"] or 0)
        stock_sentiment = fetch_recent_sentiment("STOCK", row["symbol"], STOCK_SENTIMENT_MAX_AGE_DAYS)
        metrics = {
            "risk_tier": risk_tier,
            "composite_score": float(row["composite_score"]),
            "risk_label": row["risk_label"],
            "volatility_1y": float(row["volatility_1y"]) if row["volatility_1y"] is not None else None,
            "beta_1y": float(row["beta_1y"]) if row["beta_1y"] is not None else None,
            "liquidity_flag": row["liquidity_flag"],
            "market_cap_tier": row["market_cap_tier"],
            "sector": row["sector"],
            "position_weight_pct": round(item["position_value"] / total_value * 100, 2) if total_value else None,
            "sector_weight_pct": sector_weights.get(str(row["sector"])) if row["sector"] else None,
            "unrealised_pnl_pct": round((last_price / avg_cost - 1) * 100, 2) if avg_cost > 0 else None,
            "market_sentiment_score": market_sentiment["score"] if market_sentiment else None,
            "market_sentiment_label": market_sentiment["label"] if market_sentiment else None,
            "market_sentiment_age_days": market_sentiment["age_days"] if market_sentiment else None,
            "stock_sentiment_score": stock_sentiment["score"] if stock_sentiment else None,
            "stock_sentiment_label": stock_sentiment["label"] if stock_sentiment else None,
            "portfolio_risk_score": portfolio_summary["weighted_average_risk_score"],
            "portfolio_risk_coverage_pct": portfolio_summary["coverage_pct"],
            "profile_alignment": portfolio_summary["profile_alignment"],
        }
        jobs.append({"stock_id": row["stock_id"], "symbol": row["symbol"], "metrics": metrics})

    run = fetch_one(
        "INSERT INTO agent_runs (run_type, status) VALUES ('RECOMMENDATION', 'RUNNING') RETURNING id"
    )
    if not run:
        raise HTTPException(status_code=500, detail="Could not create recommendation run.")

    return {
        "run_id": run["id"],
        "risk_tier": risk_tier,
        "jobs": jobs,
        "skipped": [row["symbol"] for row in rows if row["composite_score"] is None],
        "market_sentiment_used": market_sentiment is not None,
    }


def _execute_recommendation_run(plan: dict[str, Any]) -> dict[str, Any]:
    run_id = plan["run_id"]
    risk_tier = plan["risk_tier"]
    jobs = plan["jobs"]
    skipped = plan["skipped"]

    try:
        # Model calls are independent and slow; run a few at once. Results come
        # back in input order. generate_ollama_recommendation never raises - it
        # falls back to the deterministic policy.
        with ThreadPoolExecutor(max_workers=RECOMMENDATION_MAX_WORKERS) as pool:
            results = list(
                pool.map(
                    lambda job: generate_ollama_recommendation(
                        job["symbol"], job["metrics"], risk_tier, trace_session_id=str(run_id)
                    ),
                    jobs,
                )
            )

        fallback_count = sum(1 for r in results if r["model_status"] == "deterministic_fallback")
        override_count = sum(1 for r in results if r["model_status"] == "ollama_overridden")
        status = "PARTIAL" if skipped else "COMPLETED"
        notes = json.dumps(
            {
                "skipped_symbols": skipped,
                "deterministic_fallbacks": fallback_count,
                "policy_overrides": override_count,
                "market_sentiment_used": plan["market_sentiment_used"],
            }
        )

        recommendations: list[dict[str, Any]] = []
        with get_transaction() as conn:
            for job, result in zip(jobs, results):
                conn.execute(
                    """
                    INSERT INTO recommendations
                        (stock_id, run_id, recommendation, reasoning, inputs_snapshot,
                         model_status, model_name, llm_recommendation, policy_recommendation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job["stock_id"],
                        run_id,
                        result["recommendation"],
                        result["reasoning"],
                        json.dumps({**job["metrics"], "explanation": result.get("explanation")}),
                        result["model_status"],
                        result.get("model_name"),
                        result.get("llm_recommendation"),
                        result.get("policy_recommendation"),
                    ),
                )
                recommendations.append(
                    {
                        "symbol": job["symbol"],
                        "recommendation": result["recommendation"],
                        "reasoning": result["reasoning"],
                        "model_status": result["model_status"],
                        "llm_recommendation": result.get("llm_recommendation"),
                        "policy_recommendation": result.get("policy_recommendation"),
                        "explanation": result.get("explanation"),
                        "inputs_snapshot": job["metrics"],
                    }
                )
            conn.execute(
                "UPDATE agent_runs SET status = %s, finished_at = NOW(), notes = %s WHERE id = %s",
                (status, notes, run_id),
            )
    except Exception as exc:
        execute(
            "UPDATE agent_runs SET status = 'FAILED', finished_at = NOW(), notes = %s WHERE id = %s",
            (str(exc), run_id),
        )
        raise HTTPException(status_code=502, detail=f"Recommendation run failed: {exc}") from exc

    return {
        "status": status.lower(),
        "run_id": run_id,
        "generated": len(recommendations),
        "skipped_symbols": skipped,
        "deterministic_fallbacks": fallback_count,
        "policy_overrides": override_count,
        "recommendations": recommendations,
    }


def _run_in_background(plan: dict[str, Any]) -> None:
    try:
        _execute_recommendation_run(plan)
    except Exception:
        logger.exception("Background recommendation run %s failed", plan["run_id"])
    finally:
        _recommendation_lock.release()


@app.post("/recommendations/run", response_model=None)
def run_recommendations(background_tasks: BackgroundTasks, background: bool = False) -> dict[str, Any]:
    """Generate recommendations for every scored holding.

    Synchronous by default (the response contains the recommendations). With
    `?background=true` it returns immediately with the run id and keeps working;
    poll GET /recommendations/runs/{run_id} for status.
    """
    if not _recommendation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A recommendation run is already in progress.")

    handed_off = False
    try:
        plan = _prepare_recommendation_run()
        if background:
            background_tasks.add_task(_run_in_background, plan)
            handed_off = True  # the background task now owns (and will release) the lock
            return {"status": "running", "run_id": plan["run_id"], "queued": len(plan["jobs"])}
        return _execute_recommendation_run(plan)
    finally:
        if not handed_off:
            _recommendation_lock.release()


@app.get("/recommendations")
def list_recommendations() -> list[dict[str, Any]]:
    ensure_schema()
    return fetch_all(
        """
        SELECT r.id, r.run_id, s.symbol, r.recommendation, r.reasoning,
               r.inputs_snapshot, r.created_at,
               r.model_status, r.model_name, r.llm_recommendation, r.policy_recommendation
        FROM recommendations r
        JOIN stocks s ON s.id = r.stock_id
        ORDER BY r.created_at DESC, s.symbol
        """
    )


@app.get("/recommendations/runs")
def list_recommendation_runs() -> list[dict[str, Any]]:
    return fetch_all(
        "SELECT * FROM agent_runs WHERE run_type = 'RECOMMENDATION' ORDER BY started_at DESC"
    )


@app.get("/recommendations/runs/{run_id}")
def get_recommendation_run(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM agent_runs WHERE id = %s AND run_type = 'RECOMMENDATION'", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Recommendation run not found.")
    return run


@app.get("/recommendations/review")
def review_recommendations() -> list[dict[str, Any]]:
    # The entry date is derived in IST (the market's calendar), not the database
    # session's timezone, so a late-evening IST run does not land on the wrong day.
    recommendations = fetch_all(
        """
        SELECT s.symbol AS stock_symbol, r.recommendation, r.reasoning,
               r.inputs_snapshot, r.created_at AS recommended_at,
               entry.date AS entry_date, entry.close AS entry_price,
               latest.date AS latest_date, latest.close AS latest_price
        FROM recommendations r
        JOIN stocks s ON s.id = r.stock_id
        LEFT JOIN LATERAL (
            SELECT date, close
            FROM price_history
            WHERE stock_id = r.stock_id
              AND date > (r.created_at AT TIME ZONE 'Asia/Kolkata')::date
            ORDER BY date
            LIMIT 1
        ) entry ON TRUE
        LEFT JOIN LATERAL (
            SELECT date, close
            FROM price_history
            WHERE stock_id = r.stock_id
            ORDER BY date DESC
            LIMIT 1
        ) latest ON TRUE
        ORDER BY r.created_at DESC, s.symbol
        """
    )
    try:
        benchmark = fetch_nifty_50_prices(days=1095)
    except Exception as exc:  # noqa: BLE001 - the review is still useful without the benchmark
        logger.warning("Benchmark unavailable for review (%s); reporting absolute returns only.", exc)
        benchmark = None
    return build_review_rows(recommendations, benchmark_prices=benchmark)


def _prepare_etf_analysis() -> dict[str, Any]:
    ensure_schema()
    profile = fetch_one("SELECT risk_tier FROM risk_profile ORDER BY created_at DESC LIMIT 1")
    if not profile:
        raise HTTPException(status_code=409, detail="Complete the risk questionnaire before analyzing ETFs.")

    try:
        records, source_timestamp = fetch_nse_etf_universe()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"NSE ETF universe fetch failed: {type(exc).__name__}.") from exc
    if not records:
        raise HTTPException(status_code=502, detail="NSE returned no ETF listings.")

    try:
        market_prices = fetch_nifty_50_prices(days=ETF_HISTORY_DAYS)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Nifty 50 benchmark fetch failed: {type(exc).__name__}.") from exc
    if market_prices.empty:
        raise HTTPException(status_code=502, detail="Nifty 50 benchmark prices are unavailable.")

    execute_many(
        """
        INSERT INTO etfs (symbol, underlying, asset_class, last_price, nav, inav, traded_value,
                          volume, portfolio_turnover_ratio, yield_pct, repo_rate_pct,
                          change_30d_pct, change_365d_pct, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (symbol) DO UPDATE SET
            underlying = EXCLUDED.underlying,
            asset_class = EXCLUDED.asset_class,
            last_price = EXCLUDED.last_price,
            nav = EXCLUDED.nav,
            inav = EXCLUDED.inav,
            traded_value = EXCLUDED.traded_value,
            volume = EXCLUDED.volume,
            portfolio_turnover_ratio = EXCLUDED.portfolio_turnover_ratio,
            yield_pct = EXCLUDED.yield_pct,
            repo_rate_pct = EXCLUDED.repo_rate_pct,
            change_30d_pct = EXCLUDED.change_30d_pct,
            change_365d_pct = EXCLUDED.change_365d_pct,
            updated_at = NOW()
        """,
        [
            (
                item["symbol"], item["underlying"], item["asset_class"], item["last_price"],
                item["nav"], item["inav"], item["traded_value"], item["volume"],
                item["portfolio_turnover_ratio"], item["yield_pct"], item["repo_rate_pct"],
                item["change_30d_pct"], item["change_365d_pct"],
            )
            for item in records
        ],
    )
    run = fetch_one("INSERT INTO agent_runs (run_type, status) VALUES ('ETF_ANALYSIS', 'RUNNING') RETURNING id")
    if not run:
        raise HTTPException(status_code=500, detail="Could not create ETF analysis run.")

    market_sentiment = fetch_recent_sentiment("MARKET", "INDIA", MARKET_SENTIMENT_MAX_AGE_DAYS)

    return {
        "run_id": run["id"],
        "records": records,
        "market_prices": market_prices,
        "risk_tier": profile["risk_tier"],
        "portfolio_summary": get_portfolio_summary(),
        "source_timestamp": source_timestamp,
        "market_sentiment_label": market_sentiment["label"] if market_sentiment else None,
    }


def _execute_etf_analysis(plan: dict[str, Any]) -> dict[str, Any]:
    run_id = plan["run_id"]
    records = plan["records"]
    small_cap_symbols = [
        item["symbol"] for item in records
        if item["asset_class"] == "Equity" and is_small_cap_etf(item)
    ]
    try:
        results = analyze_equity_etfs(
            records,
            plan["market_prices"],
            plan["risk_tier"],
            plan["portfolio_summary"],
            market_sentiment_label=plan.get("market_sentiment_label"),
        )
        scored = [item for item in results if item.get("status") == "scored"]
        skipped = [item for item in results if item.get("status") != "scored"]
        status = "PARTIAL" if skipped else "COMPLETED"
        notes = json.dumps(
            {
                "source_timestamp": plan["source_timestamp"],
                "universe_count": len(records),
                "small_cap_etf_count": len(small_cap_symbols),
                "scored_count": len(scored),
                "skipped": [
                    {"symbol": item["symbol"], "reason": item.get("reason", "No score available.")}
                    for item in skipped
                ],
            }
        )

        with get_transaction() as conn:
            for symbol in small_cap_symbols:
                conn.execute("DELETE FROM etf_risk_scores WHERE symbol = %s", (symbol,))
            for item in scored:
                conn.execute(
                    """
                    INSERT INTO etf_risk_scores
                        (symbol, volatility_1y, beta_1y, sortino_1y, max_drawdown_1y_pct,
                         tracking_error_1y_pct, liquidity_flag, composite_score, risk_label,
                         history_days, beta_observations, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                        volatility_1y = EXCLUDED.volatility_1y,
                        beta_1y = EXCLUDED.beta_1y,
                        sortino_1y = EXCLUDED.sortino_1y,
                        max_drawdown_1y_pct = EXCLUDED.max_drawdown_1y_pct,
                        tracking_error_1y_pct = EXCLUDED.tracking_error_1y_pct,
                        liquidity_flag = EXCLUDED.liquidity_flag,
                        composite_score = EXCLUDED.composite_score,
                        risk_label = EXCLUDED.risk_label,
                        history_days = EXCLUDED.history_days,
                        beta_observations = EXCLUDED.beta_observations,
                        computed_at = NOW()
                    """,
                    (
                        item["symbol"], item["volatility_1y"], item["beta_1y"], item["sortino_1y"],
                        item["max_drawdown_1y_pct"], item["tracking_error_1y_pct"], item["liquidity_flag"],
                        item["composite_score"], item["risk_label"], item["history_days"],
                        item["beta_observations"],
                    ),
                )
                snapshot = {
                    "risk_tier": plan["risk_tier"],
                    "composite_score": item["composite_score"],
                    "volatility_1y": item["volatility_1y"],
                    "beta_1y": item["beta_1y"],
                    "sortino_1y": item["sortino_1y"],
                    "max_drawdown_1y_pct": item["max_drawdown_1y_pct"],
                    "tracking_error_1y_pct": item["tracking_error_1y_pct"],
                    "liquidity_flag": item["liquidity_flag"],
                    "traded_value": item["traded_value"],
                    "median_daily_turnover": item["median_daily_turnover"],
                    "portfolio_risk_score": plan["portfolio_summary"]["weighted_average_risk_score"],
                    "portfolio_hhi": item.get("portfolio_hhi"),
                    "sector_hhi": item.get("sector_hhi"),
                    "market_cap_hhi": item.get("market_cap_hhi"),
                    "profile_alignment": plan["portfolio_summary"]["profile_alignment"],
                    "market_sentiment_label": item.get("market_sentiment_label"),
                    "portfolio_effect": item.get("portfolio_effect"),
                    "inav": item.get("inav"),
                    "inav_premium_discount_pct": item.get("inav_premium_discount_pct"),
                    "portfolio_turnover_ratio": item.get("portfolio_turnover_ratio"),
                    "yield_pct": item.get("yield_pct"),
                    "repo_rate_pct": item.get("repo_rate_pct"),
                    "yield_repo_gap_pct": item.get("yield_repo_gap_pct"),
                    "explanation": item.get("explanation"),
                    "model_status": item.get("model_status"),
                }
                conn.execute(
                    """
                    INSERT INTO etf_recommendations
                        (symbol, run_id, recommendation, reasoning, inputs_snapshot)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (item["symbol"], run_id, item["recommendation"], item["reasoning"], json.dumps(snapshot)),
                )
            for item in skipped:
                snapshot = {
                    "risk_tier": plan["risk_tier"],
                    "status": "skipped",
                    "reason": item.get("reason"),
                    "portfolio_risk_score": plan["portfolio_summary"]["weighted_average_risk_score"],
                    "portfolio_hhi": plan["portfolio_summary"].get("portfolio_hhi"),
                    "sector_hhi": plan["portfolio_summary"].get("sector_hhi"),
                    "market_cap_hhi": plan["portfolio_summary"].get("market_cap_hhi"),
                    "profile_alignment": plan["portfolio_summary"]["profile_alignment"],
                }
                unrated = build_explanation(
                    "UNRATED", item["symbol"], {}, held=False,
                    block_reason=item.get("plain_reason") or PLAIN_NOT_ENOUGH_HISTORY,
                )
                snapshot["explanation"] = unrated
                conn.execute(
                    """
                    INSERT INTO etf_recommendations
                        (symbol, run_id, recommendation, reasoning, inputs_snapshot)
                    VALUES (%s, %s, 'WATCH', %s, %s)
                    """,
                    (item["symbol"], run_id, compose(unrated), json.dumps(snapshot)),
                )
            conn.execute(
                "UPDATE agent_runs SET status = %s, finished_at = NOW(), notes = %s WHERE id = %s",
                (status, notes, run_id),
            )
    except Exception as exc:
        execute(
            "UPDATE agent_runs SET status = 'FAILED', finished_at = NOW(), notes = %s WHERE id = %s",
            (type(exc).__name__, run_id),
        )
        raise HTTPException(status_code=502, detail=f"ETF analysis run failed: {type(exc).__name__}.") from exc

    return {
        "status": status.lower(),
        "run_id": run_id,
        "universe_count": len(records),
        "small_cap_etf_count": len(small_cap_symbols),
        "scored": len(scored),
        "skipped": len(skipped),
        "recommendations": results,
    }


def _run_etf_analysis_in_background(plan: dict[str, Any]) -> None:
    try:
        _execute_etf_analysis(plan)
    except Exception:
        logger.exception("ETF analysis run %s failed", plan["run_id"])
    finally:
        _etf_analysis_lock.release()


@app.post("/etfs/analyze", response_model=None)
def run_etf_analysis(background_tasks: BackgroundTasks, background: bool = False) -> dict[str, Any]:
    """Analyze small-cap equity ETFs; use background=true for asynchronous analysis."""
    if not _etf_analysis_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ETF analysis run is already in progress.")

    handed_off = False
    try:
        plan = _prepare_etf_analysis()
        if background:
            background_tasks.add_task(_run_etf_analysis_in_background, plan)
            handed_off = True
            queued = sum(
                1 for item in plan["records"]
                if item["asset_class"] == "Equity" and is_small_cap_etf(item)
            )
            return {"status": "running", "run_id": plan["run_id"], "queued": queued}
        return _execute_etf_analysis(plan)
    finally:
        if not handed_off:
            _etf_analysis_lock.release()


@app.get("/etfs")
def list_etfs() -> list[dict[str, Any]]:
    ensure_schema()
    return fetch_all(
        """
        SELECT e.symbol, e.underlying, e.asset_class, e.last_price, e.nav, e.inav,
               e.traded_value, e.volume, e.portfolio_turnover_ratio, e.yield_pct, e.repo_rate_pct,
               e.change_30d_pct, e.change_365d_pct,
               e.updated_at, rs.volatility_1y, rs.beta_1y, rs.sortino_1y,
               rs.max_drawdown_1y_pct, rs.tracking_error_1y_pct, rs.liquidity_flag,
             rs.composite_score, rs.risk_label, rs.history_days, rs.computed_at,
             latest.recommendation, latest.reasoning, latest.explanation,
             latest.inputs_snapshot ->> 'portfolio_hhi' AS portfolio_hhi,
             latest.inputs_snapshot ->> 'sector_hhi' AS sector_hhi,
             latest.inputs_snapshot ->> 'market_cap_hhi' AS market_cap_hhi,
             CASE WHEN e.yield_pct IS NOT NULL AND e.repo_rate_pct IS NOT NULL
                  THEN e.yield_pct - e.repo_rate_pct
                  ELSE NULLIF(latest.inputs_snapshot ->> 'yield_repo_gap_pct', '')::NUMERIC
             END AS yield_repo_gap_pct,
             CASE WHEN e.inav > 0 AND e.last_price IS NOT NULL
                  THEN (e.last_price / e.inav - 1) * 100 END AS inav_premium_discount_pct
        FROM etfs e
        LEFT JOIN etf_risk_scores rs ON rs.symbol = e.symbol
         LEFT JOIN LATERAL (
             SELECT recommendation, reasoning, inputs_snapshot,
                    inputs_snapshot -> 'explanation' AS explanation
             FROM etf_recommendations
             WHERE symbol = e.symbol
             ORDER BY created_at DESC, id DESC
             LIMIT 1
         ) latest ON TRUE
         WHERE e.asset_class = 'Equity'
           AND (
               POSITION('smallcap' IN LOWER(e.underlying)) > 0
               OR POSITION('small cap' IN LOWER(e.underlying)) > 0
           )
         ORDER BY CASE latest.recommendation WHEN 'BUY' THEN 0 WHEN 'WATCH' THEN 1 WHEN 'AVOID' THEN 2 ELSE 3 END,
               rs.composite_score, e.symbol
        """
    )


@app.get("/etfs/runs/{run_id}")
def get_etf_analysis_run(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM agent_runs WHERE id = %s AND run_type = 'ETF_ANALYSIS'", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="ETF analysis run not found.")
    run["recommendations"] = fetch_all(
        """
        SELECT e.symbol, e.underlying, e.asset_class, e.last_price, e.nav, e.inav,
               e.traded_value, e.portfolio_turnover_ratio, e.yield_pct, e.repo_rate_pct,
               r.recommendation, r.reasoning, r.inputs_snapshot, r.created_at,
               rs.composite_score, rs.risk_label, rs.volatility_1y, rs.beta_1y,
               rs.sortino_1y, rs.max_drawdown_1y_pct, rs.tracking_error_1y_pct, rs.liquidity_flag,
               r.inputs_snapshot ->> 'portfolio_hhi' AS portfolio_hhi,
               r.inputs_snapshot ->> 'sector_hhi' AS sector_hhi,
               r.inputs_snapshot ->> 'market_cap_hhi' AS market_cap_hhi,
               r.inputs_snapshot ->> 'yield_repo_gap_pct' AS yield_repo_gap_pct,
               CASE WHEN e.inav > 0 AND e.last_price IS NOT NULL
                    THEN (e.last_price / e.inav - 1) * 100 END AS inav_premium_discount_pct
        FROM etf_recommendations r
        JOIN etfs e ON e.symbol = r.symbol
        LEFT JOIN etf_risk_scores rs ON rs.symbol = r.symbol
        WHERE r.run_id = %s
          AND e.asset_class = 'Equity'
          AND (
              POSITION('smallcap' IN LOWER(e.underlying)) > 0
              OR POSITION('small cap' IN LOWER(e.underlying)) > 0
          )
        ORDER BY CASE r.recommendation WHEN 'BUY' THEN 0 WHEN 'WATCH' THEN 1 ELSE 2 END,
                 rs.composite_score, e.symbol
        """,
        (run_id,),
    )
    return run


@app.get("/debug/env")
def debug_env() -> dict[str, str]:
    # The Kite access token lives in Postgres (kite_sessions), not in the environment.
    try:
        token_status = "stored in database" if get_latest_kite_access_token() else "missing"
    except Exception:  # noqa: BLE001
        token_status = "database unavailable"
    return {
        "KITE_API_KEY": "set" if os.getenv("KITE_API_KEY") else "missing",
        "KITE_API_SECRET": "set" if os.getenv("KITE_API_SECRET") else "missing",
        "KITE_ACCESS_TOKEN": token_status,
    }
