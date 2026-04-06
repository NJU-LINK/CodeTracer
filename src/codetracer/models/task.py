"""TaskContext: task metadata dataclass + BenchPool-driven factory.

All bench-specific behavior is delegated to BenchProvider instances.
TaskContext itself is a plain data holder with thin delegation methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from codetracer.benches.provider import BenchPool, LoadedBenchProvider

logger = logging.getLogger(__name__)

_pool: BenchPool | None = None


def _get_pool() -> BenchPool:
    global _pool
    if _pool is None:
        _pool = BenchPool()
    return _pool


@dataclass
class TaskContext:
    """Task metadata + provider reference for sandbox creation and exploration."""

    bench_type: str
    task_name: str
    task_dir: Path
    sandbox_dir: Path | None = field(default=None, repr=False)
    repo_url: str | None = field(default=None, repr=False)
    base_commit: str | None = field(default=None, repr=False)
    problem_statement: str | None = field(default=None, repr=False)

    _provider: LoadedBenchProvider | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, task_dir: Path, pool: BenchPool | None = None) -> TaskContext:
        """Auto-detect bench format via BenchPool and build a TaskContext."""
        task_dir = task_dir.resolve()
        bp = (pool or _get_pool()).detect(task_dir)
        if bp is None:
            raise FileNotFoundError(
                f"No bench provider can handle {task_dir} — "
                "register a provider or place a BENCH.yaml + provider.py in the benches directory"
            )

        ctx = bp.provider.load_context(task_dir)
        return cls(
            bench_type=ctx.get("bench_type", bp.bench_name),
            task_name=ctx.get("task_name", task_dir.name),
            task_dir=task_dir,
            repo_url=ctx.get("repo_url"),
            base_commit=ctx.get("base_commit"),
            problem_statement=ctx.get("problem_statement"),
            _provider=bp,
        )

    def prepare_sandbox(self, target_parent: Path) -> Path:
        """Delegate sandbox creation to the bench provider. Idempotent."""
        sandbox = target_parent / "task_context"
        if sandbox.exists():
            self.sandbox_dir = sandbox
            return sandbox

        if self._provider is None:
            raise RuntimeError("Cannot prepare sandbox without a bench provider")
        self._provider.provider.prepare_sandbox(self.task_dir, sandbox)
        self.sandbox_dir = sandbox
        return sandbox

    def exploration_instructions(self, sandbox: Path) -> str:
        """Delegate exploration instructions to the bench provider."""
        if self._provider is not None:
            return self._provider.provider.exploration_instructions(sandbox)
        sd = str(sandbox)
        return (
            f"Required exploration commands (execute in order, one per response):\n"
            f"1. ls {sd}/\n"
            f"2. Read key documentation and configuration files you find\n"
            f"3. Explore source directories to understand the codebase structure"
        )
