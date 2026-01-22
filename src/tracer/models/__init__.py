"""Model selection for tracer.

We intentionally expose a single OpenAI-compatible calling interface to avoid
vendor-specific wrappers. Users configure only base_url + api_key (and optionally
model_name; otherwise it is auto-detected via /v1/models).
"""

from __future__ import annotations

import os
import threading

from tracer import Model
from tracer.models.custom_api_model import CustomAPIModel


class GlobalModelStats:
    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.cost_limit = float(os.getenv("TRACER_GLOBAL_COST_LIMIT") or os.getenv("MSWEA_GLOBAL_COST_LIMIT") or "0")
        self.call_limit = int(os.getenv("TRACER_GLOBAL_CALL_LIMIT") or os.getenv("MSWEA_GLOBAL_CALL_LIMIT") or "0")

    def add(self, cost: float) -> None:
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.cost_limit < self._cost or 0 < self.call_limit < self._n_calls:
            raise RuntimeError(f"Global cost/call limit exceeded: ${self._cost:.4f} / {self._n_calls}")

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    config = dict(config or {})
    if input_model_name:
        config["model_name"] = input_model_name
    return CustomAPIModel(**config)
