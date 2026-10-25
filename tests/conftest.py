"""Let the pure-logic tests run without Postgres, LangChain, yfinance, etc.

If an optional heavy dependency is not installed, a MagicMock stands in for it
(only for modules the tests never actually call into).
"""

import importlib
import sys
from unittest.mock import MagicMock

for name in (
    "psycopg",
    "psycopg.rows",
    "dotenv",
    "yfinance",
    "requests",
    "langchain_core",
    "langchain_core.messages",
    "langchain_ollama",
    "langfuse",
    "tavily",
    "backend.llm_client",
    "backend.db",
):
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = MagicMock()
