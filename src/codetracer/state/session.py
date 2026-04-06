"""Centralized session state store.

A minimal reactive store with getState / setState / subscribe. All
components (BaseAgent, LLMClient, CostTracker, CompactManager, REPL
handlers) read/write through this store so state is observable and
consistent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


@dataclass
class SessionState:
    """Immutable-ish snapshot of the current session."""

    mode: Literal["benchmark", "interactive"] = "benchmark"
    profile: str = "tracebench"
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    step_count: int = 0
    budget_limit_usd: float = 3.0
    phase: str = "idle"
    abort_requested: bool = False
    compact_count: int = 0
    last_model: str = ""

    run_dir: Path | None = None
    traj: Any = None
    skill: Any = None
    fmt_name: str = ""
    analysis: Any = None
    chat_messages: list[dict[str, str]] = field(default_factory=list)
    llm: Any = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def budget_remaining_usd(self) -> float:
        return max(0.0, self.budget_limit_usd - self.total_cost_usd)

    @property
    def budget_used_pct(self) -> float:
        if self.budget_limit_usd <= 0:
            return 0.0
        return min(100.0, (self.total_cost_usd / self.budget_limit_usd) * 100.0)

    def copy(self, **overrides: Any) -> SessionState:
        """Return a shallow copy with optional field overrides."""
        import dataclasses
        vals = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if "messages" not in overrides:
            vals["messages"] = list(self.messages)
        if "chat_messages" not in overrides:
            vals["chat_messages"] = list(self.chat_messages)
        vals.update(overrides)
        return SessionState(**vals)


Listener = Callable[[], None]
Updater = Callable[[SessionState], SessionState]


class SessionStore:
    """Thread-safe reactive state container.

    Usage::

        store = SessionStore()
        unsub = store.subscribe(lambda: print("changed"))
        store.set_state(lambda s: s.copy(step_count=s.step_count + 1))
        state = store.get_state()
    """

    def __init__(self, initial: SessionState | None = None) -> None:
        self._state = initial or SessionState()
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()

    def get_state(self) -> SessionState:
        return self._state

    def set_state(self, updater: Updater) -> None:
        with self._lock:
            prev = self._state
            next_state = updater(prev)
            if next_state is prev:
                return
            self._state = next_state
        for listener in list(self._listeners):
            listener()

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register *listener*; returns an unsubscribe callable."""
        self._listeners.append(listener)

        def _unsub() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsub

    def reset(self, state: SessionState | None = None) -> None:
        """Reset to a fresh state (for tests or new sessions)."""
        with self._lock:
            self._state = state or SessionState()
