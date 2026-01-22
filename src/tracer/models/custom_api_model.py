import logging
import os
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tracer.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("custom_api_model")


class CustomAPIModelConfig(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model_name: Optional[str] = None
    model_kwargs: dict[str, Any] = {}
    cost_tracking: str = "ignore_errors"


class CustomAPIError(Exception):
    pass


class CustomAPIAuthenticationError(Exception):
    pass


class CustomAPIRateLimitError(Exception):
    pass


class CustomAPIModel:
    def __init__(self, **kwargs):
        self.config = CustomAPIModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self._api_base = (self.config.api_base or os.getenv("TRACER_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._api_base:
            raise ValueError("Missing api_base (set model.api_base or TRACER_API_BASE or OPENAI_BASE_URL)")

        self._api_key = self.config.api_key or os.getenv("TRACER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not self._api_key:
            raise ValueError("Missing api_key (set model.api_key or TRACER_API_KEY or OPENAI_API_KEY)")

        self._client = OpenAI(api_key=self._api_key, base_url=self._api_base)
        self._model_name = self.config.model_name or self._detect_model_name()

    def _detect_model_name(self) -> str:
        models = self._client.models.list()
        ids = [m.id for m in models.data if getattr(m, "id", None)]
        if not ids:
            raise ValueError("No models returned by server; set model.model_name explicitly")
        return sorted(ids)[0]

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                CustomAPIAuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore[arg-type]
                **(self.config.model_kwargs | kwargs),
            )
            return resp
        except Exception as e:
            msg = str(e).lower()
            if "401" in msg or "unauthorized" in msg:
                raise CustomAPIAuthenticationError("Authentication failed. Please check your API key.") from e
            if "429" in msg or "rate limit" in msg:
                raise CustomAPIRateLimitError("Rate limit exceeded") from e
            raise CustomAPIError(f"Request failed: {e}") from e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query([{"role": msg["role"], "content": msg["content"]} for msg in messages], **kwargs)
        
        self.n_calls += 1
        cost = 0.0
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": (response.choices[0].message.content or "") if response.choices else "",
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

