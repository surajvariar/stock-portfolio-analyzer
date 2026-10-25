CREATE TABLE IF NOT EXISTS kite_sessions (
    id SERIAL PRIMARY KEY,
    access_token TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL UNIQUE,
    isin VARCHAR(50),
    exchange VARCHAR(20),
    market_cap_rank INTEGER,
    market_cap_tier VARCHAR(30),
    sector VARCHAR(100),
    trade_to_trade_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE stocks ADD COLUMN IF NOT EXISTS trade_to_trade_flag BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS holdings (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL UNIQUE REFERENCES stocks(id) ON DELETE CASCADE,
    quantity NUMERIC(18, 6) NOT NULL,
    avg_cost NUMERIC(18, 6) NOT NULL,
    last_price NUMERIC(18, 6),
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS holdings_history (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    quantity NUMERIC(18, 6) NOT NULL,
    avg_cost NUMERIC(18, 6) NOT NULL,
    last_price NUMERIC(18, 6),
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open NUMERIC(18, 6),
    high NUMERIC(18, 6),
    low NUMERIC(18, 6),
    close NUMERIC(18, 6),
    volume BIGINT,
    UNIQUE(stock_id, date)
);

CREATE TABLE IF NOT EXISTS risk_profile (
    id SERIAL PRIMARY KEY,
    total_score INTEGER,
    risk_tier VARCHAR(30),
    capacity_score INTEGER,
    tolerance_score INTEGER,
    answers JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    volatility_1y NUMERIC(10, 4),
    beta_1y NUMERIC(10, 4),
    liquidity_flag BOOLEAN,
    market_cap_tier VARCHAR(30),
    composite_score NUMERIC(5, 2),
    risk_label VARCHAR(30),
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(stock_id)
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL PRIMARY KEY,
    scope_type VARCHAR(20) NOT NULL,
    scope_value VARCHAR(100) NOT NULL,
    quant_score NUMERIC(5, 2),
    qual_score NUMERIC(5, 2),
    composite_sentiment NUMERIC(5, 2),
    label VARCHAR(30),
    source_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id SERIAL PRIMARY KEY,
    run_type VARCHAR(30) NOT NULL,
    status VARCHAR(30) NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES agent_runs(id) ON DELETE SET NULL,
    recommendation VARCHAR(20),
    reasoning TEXT,
    inputs_snapshot JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


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
);

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
);

CREATE TABLE IF NOT EXISTS etf_recommendations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL REFERENCES etfs(symbol) ON DELETE CASCADE,
    run_id INTEGER REFERENCES agent_runs(id) ON DELETE SET NULL,
    recommendation VARCHAR(20) NOT NULL,
    reasoning TEXT NOT NULL,
    inputs_snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
