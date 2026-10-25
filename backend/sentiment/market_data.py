from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
import yfinance as yf

from backend.market_hours import drop_incomplete_series

NSE_HOME_URL = "https://www.nseindia.com"
NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

_NSE_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
}


def _nse_session() -> requests.Session:
    """NSE's API endpoints sit behind bot protection that rejects
    requests with no prior cookies - a bare `requests.get()` straight to
    an /api/ path typically 401s/403s. Visiting a normal page first seeds
    the session with the cookies the API call needs."""
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    session.get(NSE_HOME_URL, timeout=20)
    return session


def fetch_fii_dii_flows() -> dict[str, Any]:
    session = _nse_session()
    response = session.get(NSE_FII_DII_URL, timeout=20)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list) or not records:
        raise ValueError("NSE returned no FII/DII flow records.")

    latest_date = max(
        datetime.strptime(row["date"], "%d-%b-%Y").date()  # noqa: DTZ007 - NSE supplies trading dates, not timestamps
        for row in records
    )
    latest_rows = [
        row
        for row in records
        if datetime.strptime(row["date"], "%d-%b-%Y").date() == latest_date  # noqa: DTZ007
    ]
    flows: dict[str, float] = {}
    for row in latest_rows:
        category = row.get("category")
        if category in {"FII/FPI", "DII"}:
            flows["fii_net_flow" if category == "FII/FPI" else "dii_net_flow"] = float(
                str(row["netValue"]).replace(",", "")
            )

    if set(flows) != {"fii_net_flow", "dii_net_flow"}:
        raise ValueError("NSE response did not include both FII/FPI and DII records.")
    return {**flows, "flow_date": latest_date.isoformat()}


def _nifty_signals() -> dict[str, Any]:
    index = yf.Ticker("^NSEI").history(period="1y", interval="1d", auto_adjust=False)
    if index.empty or "Close" not in index:
        raise ValueError("No Nifty 50 price data was returned.")

    # Use completed sessions only: a live intraday print would make the 20-day
    # return and the SMA comparisons depend on what time of day the refresh ran.
    closes = drop_incomplete_series(index["Close"].dropna())
    index_return = float(closes.pct_change(20).iloc[-1]) if len(closes) > 20 else None
    above_sma_50 = None
    above_sma_200 = None
    if len(closes) >= 50:
        above_sma_50 = bool(closes.iloc[-1] >= closes.rolling(50).mean().iloc[-1])
    if len(closes) >= 200:
        above_sma_200 = bool(closes.iloc[-1] >= closes.rolling(200).mean().iloc[-1])
    return {"index_return": index_return, "above_sma_50": above_sma_50, "above_sma_200": above_sma_200}


def _vix_value() -> float:
    vix_history = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d", auto_adjust=False)
    closes = vix_history["Close"].dropna() if not vix_history.empty else vix_history
    if len(closes) == 0:
        raise ValueError("No India VIX data was returned.")
    return float(closes.iloc[-1])


def fetch_market_inputs() -> dict[str, Any]:
    """Collect every quantitative market input, tolerating partial failure.

    Each source (Nifty, VIX, NSE flows) is fetched independently. A failing
    source becomes None plus an entry in `warnings`, and the sentiment score is
    computed from what is available. Only if NOTHING could be fetched is an
    error raised.
    """
    warnings: list[str] = []
    inputs: dict[str, Any] = {
        "index_return": None,
        "above_sma_50": None,
        "above_sma_200": None,
        "vix_value": None,
        "fii_net_flow": None,
        "dii_net_flow": None,
        "flow_date": None,
    }

    for label, fetch in (
        ("Nifty 50 data", _nifty_signals),
        ("India VIX", lambda: {"vix_value": _vix_value()}),
        ("NSE FII/DII flows", fetch_fii_dii_flows),
    ):
        try:
            inputs.update(fetch())
        except Exception as exc:  # noqa: BLE001 - one broken source must not sink the others
            warnings.append(f"{label} unavailable: {type(exc).__name__}: {exc}")

    signal_keys = ("index_return", "above_sma_50", "above_sma_200", "vix_value", "fii_net_flow")
    if all(inputs[key] is None for key in signal_keys):
        raise ValueError("No market inputs could be fetched. " + " | ".join(warnings))

    inputs["warnings"] = warnings
    return inputs
