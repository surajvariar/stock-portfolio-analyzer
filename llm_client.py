from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


def langfuse_callback_config(
    trace_name: str,
    *,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    load_dotenv()
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    try:
        from langfuse.langchain import CallbackHandler

        metadata: dict[str, Any] = {
            "langfuse_tags": ["stock-portfolio-analyzer", trace_name],
        }
        if session_id:
            metadata["langfuse_session_id"] = session_id
        return {
            "callbacks": [CallbackHandler()],
            "run_name": trace_name,
            "metadata": metadata,
        }
    except Exception as exc:  # noqa: BLE001 - tracing must not interrupt model use
        logger.warning("Langfuse tracing unavailable (%s).", type(exc).__name__)
        return None


def create_chat_ollama(temperature: float = 0) -> ChatOllama:
    load_dotenv()
    api_key = os.getenv("OLLAMA_API_KEY")
    mode = os.getenv("OLLAMA_MODE", "cloud" if api_key else "local").strip().lower()
    model = os.getenv("OLLAMA_MODEL") or ("gemma4:31b" if mode == "cloud" else "llama3.1")

    if mode == "cloud":
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is required when OLLAMA_MODE=cloud.")
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com"),
            temperature=temperature,
            client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
        )

    if mode != "local":
        raise ValueError("OLLAMA_MODE must be either 'cloud' or 'local'.")

    return ChatOllama(
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )