"""Adapter wrapping BaseRunner into the Executor interface that BaseAgent expects.

The adapter shares infrastructure with an isolated execution context, so
BaseAgent can execute commands via a replay runner without modification.
"""

from __future__ import annotations

from typing import Any

from codetracer.agents.executor import Executor
from codetracer.replay.runner import BaseRunner


class RunnerExecutorAdapter(Executor):
    """Wraps a ``BaseRunner`` to fulfil the ``Executor.run(cmd) -> dict`` contract."""

    def __init__(self, runner: BaseRunner) -> None:
        self._runner = runner

    def run(self, cmd: str) -> dict[str, Any]:
        result = self._runner.execute(cmd)
        return {"output": result.output, "returncode": result.returncode}
