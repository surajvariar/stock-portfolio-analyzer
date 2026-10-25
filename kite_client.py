"""
Kite Connect client for EQUITY holdings.

Note: this is simpler than FundPulse's MF holdings client. Kite Connect
exposes equity holdings natively via kite.holdings() - there's no
JS-rendering / scraping workaround needed here, unlike the AMFI/TRI
issues that came up on the mutual fund side.

Auth flow:
1. Set KITE_API_KEY / KITE_API_SECRET in .env and the app's Redirect URL to
   http://localhost:8000/auth/kite/callback.
2. Open /login once per trading day. The callback exchanges the request_token
   for an access_token (valid for the trading day) and stores it in the
   `kite_sessions` Postgres table - NOT in .env.
3. get_authenticated_kite() reads the latest stored token.
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from kiteconnect import KiteConnect

from db import get_latest_kite_access_token

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EquityHolding:
    symbol: str
    isin: str
    exchange: str
    quantity: float
    average_price: float
    last_price: float


def get_authenticated_kite() -> KiteConnect | None:
    access_token = get_latest_kite_access_token()
    if not access_token:
        return None
    api_key = os.environ["KITE_API_KEY"]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def get_login_url() -> str:
    """Build the Kite login URL; configure the callback URL in the Kite console."""
    api_key = os.environ["KITE_API_KEY"]
    kite = KiteConnect(api_key=api_key)
    return kite.login_url()


def generate_access_token(request_token: str) -> str:
    """Exchange the request_token for a fresh access_token. The caller is
    responsible for persisting it, because Kite tokens expire daily."""
    api_key = os.environ["KITE_API_KEY"]
    api_secret = os.environ["KITE_API_SECRET"]
    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    return session_data["access_token"]


def fetch_equity_holdings(kite: KiteConnect) -> list[EquityHolding]:
    """Pulls current equity holdings directly - no pagination, no
    date-range params; Kite returns the full current holdings snapshot
    in one call. The same symbol can appear once per exchange (NSE and BSE);
    the caller is responsible for merging those."""
    raw_holdings = kite.holdings()

    holdings = []
    for h in raw_holdings:
        holdings.append(
            EquityHolding(
                symbol=h["tradingsymbol"],
                isin=h.get("isin", ""),
                exchange=h.get("exchange", "NSE"),
                quantity=h["quantity"],
                average_price=h["average_price"],
                last_price=h["last_price"],
            )
        )
    logger.info("Fetched %d equity holdings from Kite", len(holdings))
    return holdings
