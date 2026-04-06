"""Lifecycle hooks for CodeTracer plugin integration.

External frameworks can subscribe to events emitted during the
CodeTracer pipeline (trajectory loaded, error found, replay started, etc.).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

Event = str
Callback = Callable[..., Any]

# Well-known event names -- pipeline lifecycle
TRAJ_LOADED = "traj_loaded"
ANALYSIS_COMPLETE = "analysis_complete"
ERROR_FOUND = "error_found"
REPLAY_START = "replay_start"
REPLAY_STEP = "replay_step"
REPLAY_COMPLETE = "replay_complete"
REPLAY_DIVERGENCE = "replay_divergence"
CHECKPOINT_SAVED = "checkpoint_saved"

# Agent loop events
STEP_START = "step_start"
STEP_COMPLETE = "step_complete"
LLM_CALL_START = "llm_call_start"
LLM_CALL_COMPLETE = "llm_call_complete"
COMPACT_TRIGGERED = "compact_triggered"
BUDGET_WARNING = "budget_warning"
BUDGET_EXCEEDED = "budget_exceeded"

# Memory events
MEMORY_EXTRACTION = "memory_extraction"

# State events
STATE_CHANGED = "state_changed"


class HookManager:
    """Registry of event callbacks; singleton-friendly but instantiable."""

    def __init__(self) -> None:
        self._hooks: dict[Event, list[Callback]] = defaultdict(list)

    def on(self, event: Event, callback: Callback) -> None:
        """Subscribe *callback* to *event*."""
        self._hooks[event].append(callback)

    def off(self, event: Event, callback: Callback) -> None:
        """Unsubscribe *callback* from *event*."""
        try:
            self._hooks[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event: Event, **kwargs: Any) -> None:
        """Fire all callbacks registered for *event*."""
        for cb in self._hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception:
                logger.exception("Hook callback failed for event %s", event)

    def clear(self, event: Event | None = None) -> None:
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)


# Module-level default instance for convenience
default_hooks = HookManager()
