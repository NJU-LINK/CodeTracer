"""Trace agent: iterative diagnosis loop using BaseAgent + ContextAssembler.

Now supports output profiles, cost tracking, compact, memory, and hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codetracer.agents.base import AgentEvent, BaseAgent
from codetracer.agents.compact import CompactManager
from codetracer.agents.context import ContextAssembler
from codetracer.agents.executor import Executor
from codetracer.models.task import TaskContext
from codetracer.services.cost_tracker import CostTracker
from codetracer.state.output_profile import OutputProfile
from codetracer.llm.client import LLMClient
from codetracer.plugins.hooks import HookManager
from codetracer.skills.loader import Skill


class TraceAgent:
    """High-level trace agent that wires context assembly and the base agent loop."""

    def __init__(
        self,
        llm: LLMClient,
        assembler: ContextAssembler,
        run_dir: Path,
        output_path: Path,
        config: dict[str, Any],
        artifacts_dir: Path | None = None,
        *,
        hooks: HookManager | None = None,
        cost_tracker: CostTracker | None = None,
        compact_manager: CompactManager | None = None,
        profile: OutputProfile | None = None,
    ) -> None:
        self._llm = llm
        self._assembler = assembler
        self._run_dir = run_dir
        self._artifacts_dir = artifacts_dir
        self._output_path = output_path
        self._config = config
        self._profile = profile

        trace_cfg = config.get("trace", {})
        env_cfg = config.get("environment", {}).get("env", {"PAGER": "cat", "MANPAGER": "cat"})
        executor = Executor(
            work_dir=artifacts_dir or run_dir,
            timeout=int(trace_cfg.get("timeout", 60)),
            extra_env=env_cfg,
        )
        self._agent = BaseAgent(
            llm,
            executor,
            trace_cfg,
            hooks=hooks,
            cost_tracker=cost_tracker,
            compact_manager=compact_manager,
        )

    def run(
        self,
        skill: Skill | None,
        task_ctx: TaskContext | None = None,
        memory_text: str = "",
        budget_context: str = "",
    ) -> str:
        messages = self._assembler.build_trace_messages(
            self._run_dir,
            skill,
            task_ctx=task_ctx,
            artifacts_dir=self._artifacts_dir,
            profile=self._profile,
            memory_text=memory_text,
            budget_context=budget_context,
        )
        return self._agent.run(messages)

    def run_iter(
        self,
        skill: Skill | None,
        task_ctx: TaskContext | None = None,
        memory_text: str = "",
        budget_context: str = "",
    ):
        """Generator variant that yields AgentEvent objects."""
        messages = self._assembler.build_trace_messages(
            self._run_dir,
            skill,
            task_ctx=task_ctx,
            artifacts_dir=self._artifacts_dir,
            profile=self._profile,
            memory_text=memory_text,
            budget_context=budget_context,
        )
        return self._agent.run_iter(messages)

    def save_trajectory(self, path: Path) -> None:
        self._agent.save_trajectory(path)
