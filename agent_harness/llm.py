"""Minimal OpenAI-compatible chat-completions client built on httpx.

No SDK dependency: the tool-call wire format is stable across OpenAI-compatible
servers (OpenAI, vLLM, Ollama, llama.cpp, ...). Reasoning fields from thinking
models are preserved as-is.

Transient failures (429 / 5xx / network) are retried with exponential backoff,
jitter and Retry-After support — sponsored endpoints rate-limit under load.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable

import httpx


class LLMError(RuntimeError):
    pass


# Transient statuses worth retrying. 400/401/403/404 are permanent.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _as_int(value) -> int:
    """Coerce a token count to a non-negative int, defaulting to 0."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if 0 <= n <= 2**53 - 1 else 0


class LLM:
    def __init__(self, config, usage_callback: Callable[[dict], None] | None = None):
        self.cfg = config
        self.url = config.base_url.rstrip("/") + "/chat/completions"
        self.usage_callback = usage_callback
        self.last_usage: dict = {}
        self.retries_used: int = 0
        self._abort_event = threading.Event()
        self.client = httpx.Client(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _backoff(self, resp: httpx.Response | None, attempt: int) -> float:
        """Respect Retry-After when present, else capped exponential + jitter."""
        if resp is not None:
            raw = resp.headers.get("retry-after")
            if raw:
                try:
                    return min(max(float(raw), 0.0), self.cfg.retry_max_delay)
                except ValueError:
                    pass
        delay = self.cfg.retry_base_delay * (2**attempt)
        delay = min(delay, self.cfg.retry_max_delay)
        return delay * (0.5 + random.random() * 0.5)  # 50-100% jitter

    def _wait(self, delay: float) -> bool:
        """Interruptible wait. Returns True if aborted during the wait."""
        return self._abort_event.wait(delay)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat request, return the assistant message dict.

        The returned message may contain ``content`` and/or ``tool_calls``.
        """
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        max_retries = getattr(self.cfg, "max_retries", 0)
        resp: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            if self._abort_event.is_set():
                raise LLMError("aborted")
            try:
                resp = self.client.post(self.url, json=payload)
            except httpx.HTTPError as exc:
                if self._abort_event.is_set():
                    raise LLMError("aborted") from exc
                if attempt < max_retries:
                    self.retries_used += 1
                    if self._wait(self._backoff(None, attempt)):
                        raise LLMError("aborted") from exc
                    continue
                raise LLMError(f"request failed: {exc}") from exc

            if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                self.retries_used += 1
                if self._wait(self._backoff(resp, attempt)):
                    raise LLMError(f"aborted during retry (HTTP {resp.status_code})")
                continue
            break

        assert resp is not None  # loop always either returns/raises or sets resp
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            body = resp.json()
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"bad response: {resp.text[:500]}") from exc

        # Record token usage if the server reported it. Never let a malformed
        # usage object break an otherwise-valid model reply.
        raw = body.get("usage") or {}
        self.last_usage = {
            "model": self.cfg.model,
            "input_tokens": _as_int(raw.get("prompt_tokens")),
            "output_tokens": _as_int(raw.get("completion_tokens")),
            "cache_read_tokens": _as_int(
                (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
            ),
        }
        if self.usage_callback and any(
            self.last_usage[k] for k in ("input_tokens", "output_tokens", "cache_read_tokens")
        ):
            self.usage_callback(self.last_usage)
        return message

    def abort(self) -> None:
        """Stop an in-flight request/retry wait (called from another thread)."""
        self._abort_event.set()
        try:
            self.client.close()
        except Exception:
            pass

    def close(self) -> None:
        self._abort_event.set()
        self.client.close()

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
