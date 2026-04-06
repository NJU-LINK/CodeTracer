"""ReplayEngine: orchestrates checkpoint creation, environment setup,
command replay, error-analysis injection, and agent hand-off.

Supports two modes:
- auto       : TraceAgent picks the first incorrect step, replay from there
- interactive: user specifies the target step_id

Integrates file-state verification, hooks, and progress events.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console

from codetracer.models import (
    ErrorAnalysis,
    NormalizedTrajectory,
    ReplayResult,
    ReplayStatus,
    StepCheckpoint,
)
from codetracer.plugins.hooks import (
    REPLAY_COMPLETE,
    REPLAY_DIVERGENCE,
    REPLAY_START,
    REPLAY_STEP,
    HookManager,
    default_hooks,
)
from codetracer.services.file_state import FileStateTracker
from codetracer.replay.checkpoint import CheckpointManager
from codetracer.replay.context_inject import ContextInjector
from codetracer.replay.runner import BaseRunner, ReplayRunner

logger = logging.getLogger(__name__)
console = Console(highlight=False)


class ReplayEngine:
    """Top-level orchestrator for the replay flow."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        checkpoints_dir: Path | None = None,
        hooks: HookManager | None = None,
        cost_tracker: Any | None = None,
        compact_manager: Any | None = None,
    ) -> None:
        cfg = config or {}
        self._config = cfg
        self._replay_cfg = cfg.get("replay", {})
        self._checkpoint_mgr = CheckpointManager(checkpoints_dir)
        self._hooks = hooks or default_hooks
        self._file_tracker = FileStateTracker()
        self._cost_tracker = cost_tracker
        self._compact_manager = compact_manager
        sys_tpl = self._replay_cfg.get("system_template")
        bp_tpl = self._replay_cfg.get("breakpoint_template")
        if sys_tpl and bp_tpl:
            self._injector = ContextInjector(system_template=sys_tpl, breakpoint_template=bp_tpl)
        else:
            self._injector = ContextInjector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay_auto(
        self,
        traj: NormalizedTrajectory,
        analysis: ErrorAnalysis,
        *,
        task_dir: Path | None = None,
        env_config: dict | None = None,
        llm: Any = None,
        task_ctx: Any | None = None,
        **kwargs: Any,
    ) -> ReplayResult:
        """Auto mode: find the first incorrect step and replay from there."""
        target = analysis.first_incorrect_step_id
        if target is None:
            console.print("[yellow]No incorrect steps found in analysis; nothing to replay.[/yellow]")
            return ReplayResult(status=ReplayStatus.SUCCESS, steps_replayed=0)

        console.print(f"[bold]Auto-replay: targeting step {target}[/bold]")
        return self.replay_to_step(
            traj, target, analysis,
            task_dir=task_dir, env_config=env_config, llm=llm, task_ctx=task_ctx,
        )

    def replay_interactive(
        self,
        traj: NormalizedTrajectory,
        target_step_id: int,
        analysis: ErrorAnalysis | None = None,
        *,
        task_dir: Path | None = None,
        env_config: dict | None = None,
        llm: Any = None,
        task_ctx: Any | None = None,
        **kwargs: Any,
    ) -> ReplayResult:
        """Interactive mode: user specifies the target step."""
        console.print(f"[bold]Interactive replay: targeting step {target_step_id}[/bold]")
        return self.replay_to_step(
            traj, target_step_id, analysis,
            task_dir=task_dir, env_config=env_config, llm=llm, task_ctx=task_ctx,
        )

    def replay_to_step(
        self,
        traj: NormalizedTrajectory,
        target_step_id: int,
        analysis: ErrorAnalysis | None = None,
        *,
        task_dir: Path | None = None,
        env_config: dict | None = None,
        llm: Any = None,
        task_ctx: Any | None = None,
    ) -> ReplayResult:
        """Core replay flow: build checkpoint -> setup env -> replay -> inject -> hand off."""
        env_config = env_config or {}
        self._hooks.emit(REPLAY_START, target_step_id=target_step_id)

        checkpoint = self._checkpoint_mgr.build(
            traj, target_step_id, analysis,
            env_config=env_config,
            file_tracker=self._file_tracker,
        )
        self._checkpoint_mgr.save(checkpoint)
        console.print(
            f"  Checkpoint: {len(checkpoint.replayed_steps)} steps to replay "
            f"before step {target_step_id}"
        )

        runner = self._create_runner(task_dir, env_config)

        try:
            console.print("  Setting up environment...")
            runner.setup()

            actions = [s.action for s in checkpoint.replayed_steps if s.action.strip()]
            step_ids = [s.step_id for s in checkpoint.replayed_steps if s.action.strip()]
            work_dir = self._resolve_work_dir(env_config)
            console.print(f"  Replaying {len(actions)} commands...")
            results = runner.execute_steps(
                actions,
                file_tracker=self._file_tracker,
                work_dir=work_dir,
                step_ids=step_ids,
            )

            failed = sum(1 for r in results if r.returncode != 0)
            if failed:
                console.print(f"  [yellow]{failed} commands returned non-zero (may be expected)[/yellow]")

            for i, s in enumerate(checkpoint.replayed_steps):
                self._hooks.emit(REPLAY_STEP, step_id=s.step_id, index=i)

            if step_ids and len(step_ids) >= 2:
                diffs = self._file_tracker.diff(step_ids[0], step_ids[-1])
                if diffs:
                    console.print(f"  [dim]File changes across replay: {len(diffs)} file(s) modified/added/removed[/dim]")
                    for d in diffs[:10]:
                        console.print(f"    {d.change_type}: {d.path}")
                    self._hooks.emit(REPLAY_DIVERGENCE, diffs=[d.to_dict() for d in diffs])

            messages = self._injector.build_messages(traj, checkpoint, task_ctx=task_ctx)
            console.print(f"  Context built with error analysis at step {target_step_id}")

            agent_output = ""
            if llm is not None:
                agent_output = self._run_replay_agent(llm, runner, messages)
            else:
                console.print("  [dim]No LLM provided; environment is ready for manual interaction.[/dim]")

            self._hooks.emit(REPLAY_COMPLETE, status="success", steps_replayed=len(actions))
            return ReplayResult(
                status=ReplayStatus.SUCCESS,
                checkpoint=checkpoint,
                agent_output=agent_output,
                steps_replayed=len(actions),
            )
        except Exception as exc:
            logger.exception("Replay failed")
            self._hooks.emit(REPLAY_COMPLETE, status="failed", error=str(exc))
            return ReplayResult(
                status=ReplayStatus.FAILED,
                checkpoint=checkpoint,
                agent_output=str(exc),
                steps_replayed=0,
            )
        finally:
            runner.teardown()

    # ------------------------------------------------------------------
    # Runner creation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_work_dir(env_config: dict) -> Path | None:
        wd = env_config.get("work_dir")
        if wd:
            return Path(wd)
        return None

    @staticmethod
    def _create_runner(task_dir: Path | None, env_config: dict) -> BaseRunner:
        if task_dir is not None:
            return ReplayRunner.create_from_provider(task_dir, env_config)
        return ReplayRunner.create_from_config(env_config)

    # ------------------------------------------------------------------
    # Agent loop after replay (forked-agent pattern)
    # ------------------------------------------------------------------

    def _run_replay_agent(
        self,
        llm: Any,
        runner: BaseRunner,
        messages: list[dict[str, str]],
    ) -> str:
        """Create a forked BaseAgent that reuses full infrastructure.

        The replay agent shares config and cost tracking while keeping its
        mutable execution state isolated.
        """
        from codetracer.agents.base import BaseAgent
        from codetracer.replay.executor_adapter import RunnerExecutorAdapter

        adapter = RunnerExecutorAdapter(runner)

        replay_config = dict(self._config.get("trace", {}))
        replay_config["step_limit"] = int(self._replay_cfg.get("max_replay_steps", 30))

        agent = BaseAgent(
            llm=llm,
            executor=adapter,
            config=replay_config,
            hooks=self._hooks,
            cost_tracker=self._cost_tracker,
            compact_manager=self._compact_manager,
        )

        return agent.run(messages)
