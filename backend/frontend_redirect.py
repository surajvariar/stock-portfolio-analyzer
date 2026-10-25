"""Helpers for sending the browser back to the dashboard after the Kite login.

The Kite redirect URL is fixed in the Kite console and points at this API, so
after sign-in the browser lands on the API, not on the dashboard. To bring the
user back, the dashboard passes `return_to` to /login; it is kept in a short
cookie and used by the callback. Because that value comes from a query string it
is validated here so the callback can never be used as an open redirect.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

RETURN_COOKIE = "kite_return_to"
RETURN_COOKIE_MAX_AGE = 600  # seconds: long enough to finish a Kite login
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
MAX_MESSAGE_LENGTH = 200


def default_frontend_url() -> str | None:
    """Optional fixed dashboard URL (FRONTEND_URL in .env), used when no return_to was given."""
    value = os.getenv("FRONTEND_URL", "").strip()
    return value or None


def _origin(parts) -> str:
    host = parts.hostname or ""
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port else "")


def safe_return_url(candidate: str | None) -> str | None:
    """Return a normalised URL (query and fragment removed), or None if it is not safe.

    Accepted: http(s) URLs on this machine (localhost / 127.0.0.1 / ::1, any port)
    or on the same origin as FRONTEND_URL. Everything else is rejected.
    """
    if not candidate:
        return None
    candidate = candidate.strip()
    # Backslashes and control characters are read differently by browsers and parsers.
    if "\\" in candidate or any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        return None
    try:
        parts = urlsplit(candidate)
        port = parts.port  # raises ValueError on a bad port
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    if "@" in parts.netloc:  # user:pass@host tricks
        return None

    rebuilt_origin = _origin(parts)
    allowed = parts.hostname.lower() in LOCAL_HOSTS
    configured = default_frontend_url()
    if not allowed and configured:
        try:
            allowed = _origin(urlsplit(configured)) == rebuilt_origin
        except ValueError:
            allowed = False
    if not allowed:
        return None

    del port
    return f"{rebuilt_origin}{parts.path or '/'}"


def build_redirect_url(base: str, status: str, message: str | None = None) -> str:
    """Add ?kite=success|error (and a short message) to the dashboard URL.

    The access token is never put in the URL.
    """
    parts = urlsplit(base)
    query = dict(parse_qsl(parts.query))
    query["kite"] = status
    if message:
        query["message"] = message[:MAX_MESSAGE_LENGTH]
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", urlencode(query), ""))
