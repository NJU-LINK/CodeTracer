"""Model selection for tracer.

We intentionally expose a single OpenAI-compatible calling interface to avoid
vendor-specific wrappers. Users configure only base_url + api_key (and optionally
model_name; otherwise it is auto-detected via /v1/models).
"""

from __future__ import annotations

from tracer import Model
from tracer.models.custom_api_model import CustomAPIModel
from tracer.models.stats import GLOBAL_MODEL_STATS


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    config = dict(config or {})
    if input_model_name:
        config["model_name"] = input_model_name
    return CustomAPIModel(**config)


__all__ = ["GLOBAL_MODEL_STATS", "get_model", "CustomAPIModel"]
