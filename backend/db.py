import logging
import os
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5434/stock_portfolio_analyzer",
)


@contextmanager
def get_db_connection():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_transaction():
    """One connection, one transaction: commits on a clean exit, rolls back on
    any exception. Use for multi-statement writes that must succeed together."""
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with get_db_connection() as conn:
        return conn.execute(query, params).fetchone()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        return conn.execute(query, params).fetchall()


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_db_connection() as conn:
        result = conn.execute(query, params)
        return result.rowcount


def execute_many(query: str, params: Iterable[tuple[Any, ...]]) -> int:
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn, conn.cursor() as cursor:
        cursor.executemany(query, params)
        return cursor.rowcount


# --- schema ---------------------------------------------------------------
# Idempotent, additive migrations. Run once per process (at startup, or lazily
# on first use) instead of on every request.
KITE_SESSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS kite_sessions (
        id SERIAL PRIMARY KEY,
        access_token TEXT NOT NULL,
        generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""

SCHEMA_STATEMENTS = (
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS trade_to_trade_flag BOOLEAN NOT NULL DEFAULT FALSE",
    # How much data a risk score was based on, so thin scores are visible.
    "ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS history_days INTEGER",
    "ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS beta_observations INTEGER",
    "ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS median_daily_turnover NUMERIC",
    # Which engine produced each recommendation, for a real audit trail.
    "ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS model_status TEXT",
    "ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS model_name TEXT",
    "ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS llm_recommendation TEXT",
    "ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS policy_recommendation TEXT",
    KITE_SESSIONS_DDL,
    """
    CREATE TABLE IF NOT EXISTS etfs (
        symbol VARCHAR(50) PRIMARY KEY,
        underlying VARCHAR(200),
        asset_class VARCHAR(30) NOT NULL DEFAULT 'Equity',
        last_price NUMERIC(18, 6),
        nav NUMERIC(18, 6),
        inav NUMERIC(18, 6),
        traded_value NUMERIC(24, 2),
        volume NUMERIC(24, 2),
        portfolio_turnover_ratio NUMERIC(12, 4),
        yield_pct NUMERIC(12, 4),
        repo_rate_pct NUMERIC(12, 4),
        change_30d_pct NUMERIC(12, 4),
        change_365d_pct NUMERIC(12, 4),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "ALTER TABLE etfs ADD COLUMN IF NOT EXISTS inav NUMERIC(18, 6)",
    "ALTER TABLE etfs ADD COLUMN IF NOT EXISTS portfolio_turnover_ratio NUMERIC(12, 4)",
    "ALTER TABLE etfs ADD COLUMN IF NOT EXISTS yield_pct NUMERIC(12, 4)",
    "ALTER TABLE etfs ADD COLUMN IF NOT EXISTS repo_rate_pct NUMERIC(12, 4)",
    """
    CREATE TABLE IF NOT EXISTS etf_risk_scores (
        symbol VARCHAR(50) PRIMARY KEY REFERENCES etfs(symbol) ON DELETE CASCADE,
        volatility_1y NUMERIC(10, 4),
        beta_1y NUMERIC(10, 4),
        sortino_1y NUMERIC(12, 4),
        max_drawdown_1y_pct NUMERIC(12, 4),
        tracking_error_1y_pct NUMERIC(12, 4),
        liquidity_flag BOOLEAN,
        composite_score NUMERIC(5, 2),
        risk_label VARCHAR(30),
        history_days INTEGER,
        beta_observations INTEGER,
        computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "ALTER TABLE etf_risk_scores ADD COLUMN IF NOT EXISTS sortino_1y NUMERIC(12, 4)",
    "ALTER TABLE etf_risk_scores ADD COLUMN IF NOT EXISTS max_drawdown_1y_pct NUMERIC(12, 4)",
    "ALTER TABLE etf_risk_scores ADD COLUMN IF NOT EXISTS tracking_error_1y_pct NUMERIC(12, 4)",
    """
    CREATE TABLE IF NOT EXISTS etf_recommendations (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(50) NOT NULL REFERENCES etfs(symbol) ON DELETE CASCADE,
        run_id INTEGER REFERENCES agent_runs(id) ON DELETE SET NULL,
        recommendation VARCHAR(20) NOT NULL,
        reasoning TEXT NOT NULL,
        inputs_snapshot JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # NSE only exposes the latest day's FII/DII flow, so history is accumulated
    # here to allow a multi-day rolling view.
    """
    CREATE TABLE IF NOT EXISTS fii_dii_flows (
        flow_date DATE PRIMARY KEY,
        fii_net NUMERIC NOT NULL,
        dii_net NUMERIC NOT NULL,
        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
)

_schema_lock = threading.Lock()
_schema_ready = False


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        for statement in SCHEMA_STATEMENTS:
            execute(statement)
        _schema_ready = True


# Backwards-compatible name used by older call sites.
ensure_stock_metadata_schema = ensure_schema


def mark_interrupted_runs() -> int:
    """Runs left in RUNNING by a crash/restart would otherwise stay RUNNING forever."""
    return execute(
        """
        UPDATE agent_runs
        SET status = 'FAILED', finished_at = NOW(),
            notes = 'Interrupted: the process stopped before this run finished.'
        WHERE status = 'RUNNING'
        """
    )


# --- kite session -----------------------------------------------------------
def save_kite_access_token(token: str) -> None:
    # Login must not depend on any other table existing, so only this table is ensured.
    execute(KITE_SESSIONS_DDL)
    execute("INSERT INTO kite_sessions (access_token) VALUES (%s)", (token,))


def get_latest_kite_access_token() -> str | None:
    execute(KITE_SESSIONS_DDL)
    row = fetch_one(
        "SELECT access_token FROM kite_sessions ORDER BY generated_at DESC, id DESC LIMIT 1"
    )
    return row["access_token"] if row else None


# --- FII/DII history ----------------------------------------------------------
def save_fii_dii_flow(flow_date: date, fii_net: float, dii_net: float) -> None:
    ensure_schema()
    execute(
        """
        INSERT INTO fii_dii_flows (flow_date, fii_net, dii_net)
        VALUES (%s, %s, %s)
        ON CONFLICT (flow_date) DO UPDATE SET
            fii_net = EXCLUDED.fii_net,
            dii_net = EXCLUDED.dii_net,
            fetched_at = NOW()
        """,
        (flow_date, fii_net, dii_net),
    )


def recent_fii_dii_flows(limit: int = 5) -> list[dict[str, Any]]:
    ensure_schema()
    return fetch_all(
        "SELECT flow_date, fii_net, dii_net FROM fii_dii_flows ORDER BY flow_date DESC LIMIT %s",
        (limit,),
    )


# --- agent runs ---------------------------------------------------------------
def start_agent_run(run_type: str) -> int:
    row = fetch_one(
        "INSERT INTO agent_runs (run_type, status) VALUES (%s, 'RUNNING') RETURNING id",
        (run_type,),
    )
    if row is None:
        raise RuntimeError(f"Could not create {run_type} agent run.")
    return row["id"]


def finish_agent_run(run_id: int, status: str, notes: str) -> None:
    execute(
        "UPDATE agent_runs SET status = %s, finished_at = NOW(), notes = %s WHERE id = %s",
        (status, notes, run_id),
    )
