"""Cost tracking module.

Per-model pricing, per-phase cost/token accounting, and budget enforcement.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelCosts:
    """Per-million-token pricing for a model."""

    input_per_mtok: float = 3.0
    output_per_mtok: float = 15.0


_MODEL_COSTS: dict[str, ModelCosts] = {
    "gpt-4o": ModelCosts(2.5, 10.0),
    "gpt-4o-mini": ModelCosts(0.15, 0.60),
    "gpt-4-turbo*": ModelCosts(10.0, 30.0),
    "gpt-4-*": ModelCosts(30.0, 60.0),
    "gpt-3.5-turbo*": ModelCosts(0.5, 1.5),
    "o1-*": ModelCosts(15.0, 60.0),
    "o3-*": ModelCosts(10.0, 40.0),
    "claude-3-5-haiku*": ModelCosts(0.8, 4.0),
    "claude-3-5-sonnet*": ModelCosts(3.0, 15.0),
    "claude-3-7-sonnet*": ModelCosts(3.0, 15.0),
    "claude-sonnet-4*": ModelCosts(3.0, 15.0),
    "claude-haiku-4*": ModelCosts(1.0, 5.0),
    "claude-opus-4*": ModelCosts(15.0, 75.0),
    "deepseek-*": ModelCosts(0.27, 1.10),
    "qwen*": ModelCosts(0.5, 2.0),
}

_DEFAULT_COST = ModelCosts(5.0, 25.0)


def _lookup_model_cost(model: str) -> ModelCosts:
    model_lower = model.lower()
    for pattern, costs in _MODEL_COSTS.items():
        if fnmatch.fnmatch(model_lower, pattern):
            return costs
    return _DEFAULT_COST


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = _lookup_model_cost(model)
    return (
        (input_tokens / 1_000_000) * costs.input_per_mtok
        + (output_tokens / 1_000_000) * costs.output_per_mtok
    )


@dataclass
class PhaseCost:
    """Accumulated cost for one pipeline phase."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    n_calls: int = 0
    duration_s: float = 0.0


@dataclass
class CostTracker:
    """Tracks cost across phases and enforces budget limits."""

    budget_limit_usd: float = 3.0
    warning_pct: float = 80.0
    _phases: dict[str, PhaseCost] = field(default_factory=dict)
    _total_cost: float = 0.0
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0
    _n_calls: int = 0
    _warning_emitted: bool = False

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        phase: str = "trace",
        duration_s: float = 0.0,
    ) -> float:
        """Record usage and return the incremental USD cost."""
        cost = calculate_cost(model, input_tokens, output_tokens)
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._n_calls += 1

        pc = self._phases.setdefault(phase, PhaseCost())
        pc.input_tokens += input_tokens
        pc.output_tokens += output_tokens
        pc.cost_usd += cost
        pc.n_calls += 1
        pc.duration_s += duration_s
        return cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_limit_usd - self._total_cost)

    @property
    def budget_used_pct(self) -> float:
        if self.budget_limit_usd <= 0:
            return 0.0
        return min(100.0, (self._total_cost / self.budget_limit_usd) * 100.0)

    def is_over_budget(self) -> bool:
        return self.budget_limit_usd > 0 and self._total_cost >= self.budget_limit_usd

    def should_warn(self) -> bool:
        if self._warning_emitted:
            return False
        if self.budget_used_pct >= self.warning_pct:
            self._warning_emitted = True
            return True
        return False

    def get_phase_costs(self) -> dict[str, PhaseCost]:
        return dict(self._phases)

    def format_summary(self) -> str:
        lines = [
            f"Total cost:   ${self._total_cost:.4f} / ${self.budget_limit_usd:.2f}",
            f"Total tokens: {self._total_input_tokens:,} input, {self._total_output_tokens:,} output",
            f"API calls:    {self._n_calls}",
        ]
        if self._phases:
            lines.append("Per-phase breakdown:")
            for name, pc in self._phases.items():
                lines.append(
                    f"  {name:12s}: ${pc.cost_usd:.4f}  "
                    f"({pc.input_tokens:,}+{pc.output_tokens:,} tok, "
                    f"{pc.n_calls} calls, {pc.duration_s:.1f}s)"
                )
        return "\n".join(lines)

    def budget_warning_message(self) -> str:
        return (
            f"Budget warning: you have used ${self._total_cost:.4f} of "
            f"${self.budget_limit_usd:.2f} ({self.budget_used_pct:.0f}%). "
            f"Remaining: ${self.budget_remaining:.4f}. "
            "Please prioritize outputting your final conclusions."
        )
