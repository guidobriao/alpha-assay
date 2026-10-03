from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    OpenAI = None  # type: ignore[assignment]

from app.core.config import settings
from app.tools.network import sanitize_proxy_env

logger = logging.getLogger(__name__)

# Module-level telemetry collector.
# Set to a list before a pipeline run to collect API call records.
_telemetry_sink: Optional[list] = None


def enable_telemetry(sink: list) -> None:
    global _telemetry_sink
    _telemetry_sink = sink


def disable_telemetry() -> None:
    global _telemetry_sink
    _telemetry_sink = None


def telemetry_records() -> list:
    if _telemetry_sink is None:
        return []
    return list(_telemetry_sink)


def _get_client() -> Any | None:
    if OpenAI is None:
        logger.debug("openai package is not installed, skipping LLM call")
        return None
    if not settings.openai_api_key:
        return None
    sanitize_proxy_env()
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    try:
        return OpenAI(**kwargs)
    except Exception as e:
        logger.warning("OpenAI client initialization failed: %s", e)
        return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    max_tokens: int = 2048,
    purpose: str = "general",
) -> str | None:
    """Call the configured OpenAI-compatible LLM.

    Returns the assistant message content string, or None if the API is
    not configured or the call fails.
    """
    client = _get_client()
    if client is None:
        logger.debug("OPENAI_API_KEY not set, skipping LLM call")
        return None

    kwargs = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    ok = False
    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        logger.debug("LLM response: %s", content[:200] if content else "")
        ok = True
        return content
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None
    finally:
        elapsed = int((time.time() - t0) * 1000)
        if _telemetry_sink is not None:
            _telemetry_sink.append({
                "provider": "llm",
                "purpose": purpose,
                "success": ok,
                "duration_ms": elapsed,
            })


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    purpose: str = "general",
) -> dict | None:
    """Call LLM expecting a JSON object response.

    Returns parsed dict or None on failure.
    """
    raw = call_llm(
        system_prompt, user_prompt,
        json_mode=True, max_tokens=max_tokens,
        purpose=purpose,
    )
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON")
        return None
