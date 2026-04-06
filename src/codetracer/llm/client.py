"""OpenAI-compatible LLM client with categorized retry logic.

Retry strategy:
- Auth / config errors -> fail immediately
- Rate-limit (429)     -> exponential backoff, respects Retry-After header
- Transient (5xx, connection) -> exponential backoff
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMClientConfig(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model_name: str | None = None
    model_kwargs: dict[str, Any] = {}


def _is_retryable(error: BaseException) -> bool:
    """Decide whether *error* warrants a retry."""
    if isinstance(error, KeyboardInterrupt):
        return False
    if isinstance(error, (AuthenticationError, ValueError)):
        return False
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code >= 500 or error.status_code == 408 or error.status_code == 409
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, (OSError, ConnectionError, TimeoutError)):
        return True
    return False


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    category = "transient"
    wait_extra = 0.0
    if isinstance(exc, RateLimitError):
        category = "rate_limit"
        retry_after = getattr(exc, "headers", None)
        if retry_after and hasattr(retry_after, "get"):
            raw = retry_after.get("retry-after")
            if raw:
                try:
                    wait_extra = float(raw)
                except (ValueError, TypeError):
                    pass
    elif isinstance(exc, APIStatusError):
        category = f"http_{exc.status_code}"

    attempt = retry_state.attempt_number
    logger.warning(
        "LLM retry attempt %d (%s): %s%s",
        attempt,
        category,
        str(exc)[:200],
        f" (will wait extra {wait_extra:.1f}s)" if wait_extra else "",
    )

    if wait_extra > 0:
        time.sleep(wait_extra)


class LLMClient:
    def __init__(self, **kwargs: Any) -> None:
        self.config = LLMClientConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._openai: OpenAI | None = None
        self._model_name: str | None = self.config.model_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def _ensure_client(self) -> OpenAI:
        """Lazily create the OpenAI client on first actual use."""
        if self._openai is not None:
            return self._openai

        api_base = (
            self.config.api_base or os.getenv("CODETRACER_API_BASE") or os.getenv("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        if not api_base:
            raise ValueError("Missing api_base. Set llm.api_base config or CODETRACER_API_BASE env var.")

        api_key = self.config.api_key or os.getenv("CODETRACER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError("Missing api_key. Set llm.api_key config or CODETRACER_API_KEY env var.")

        self._openai = OpenAI(api_key=api_key, base_url=api_base)
        return self._openai

    def _detect_model_name(self) -> str:
        client = self._ensure_client()
        ids = [m.id for m in client.models.list().data if getattr(m, "id", None)]
        if not ids:
            raise ValueError("No models returned by server; set llm.model_name explicitly.")
        return sorted(ids)[0]

    def _ensure_model_name(self) -> None:
        if self._model_name is None:
            self._model_name = self._detect_model_name()

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("CODETRACER_RETRY_ATTEMPTS", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        before_sleep=_log_retry,
        retry=retry_if_exception(_is_retryable),
    )
    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self._ensure_model_name()
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._model_name,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            **(self.config.model_kwargs | kwargs),
        )
        usage = resp.usage
        token_info: dict[str, Any] = {}
        if usage:
            token_info = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            self.total_prompt_tokens += token_info.get("prompt_tokens") or 0
            self.total_completion_tokens += token_info.get("completion_tokens") or 0

        self.n_calls += 1
        return {
            "content": (resp.choices[0].message.content or "") if resp.choices else "",
            "usage": token_info,
        }
