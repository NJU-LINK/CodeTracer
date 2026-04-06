"""BenchProvider for terminal-bench tasks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from codetracer.benches.provider import BenchProvider


class TerminalBenchProvider(BenchProvider):

    def __init__(self, config: dict | None = None) -> None:
        self._cfg = config or {}

    def configure(self, config: dict) -> None:
        self._cfg = config

    def name(self) -> str:
        return "terminal_bench"

    def can_handle(self, task_dir: Path) -> bool:
        return (task_dir / "task.yaml").exists()

    def load_context(self, task_dir: Path) -> dict[str, Any]:
        return {
            "bench_type": "terminal-bench",
            "task_name": task_dir.name,
        }

    def forbidden_files(self) -> set[str]:
        forbidden = self._cfg.get("forbidden_files", ["run-tests.sh", "solution.sh"])
        return set(forbidden)

    def exploration_instructions(self, sandbox: Path) -> str:
        sd = str(sandbox)
        cmds = self._cfg.get("exploration_commands")
        if cmds:
            lines = [f"{i+1}. {c.format(sandbox=sd)}" for i, c in enumerate(cmds)]
            return "Required exploration commands (execute in order, one per response):\n" + "\n".join(lines)
        return (
            f"Required exploration commands (execute in order, one per response):\n"
            f"1. ls {sd}/\n"
            f"2. cat {sd}/task.yaml\n"
            f"3. cat {sd}/Dockerfile\n"
            f"4. ls {sd}/tests/\n"
            f"5. cat each test file found in step 4\n"
            f"6. Check for supporting subdirectories (e.g. task-deps/) —\n"
            f"   ls and cat relevant files inside them"
        )

    def prepare_sandbox(self, task_dir: Path, sandbox: Path) -> None:
        forbidden = self.forbidden_files()
        shutil.copytree(
            task_dir, sandbox,
            ignore=lambda _d, names: [n for n in names if n in forbidden],
        )

    def runner_type(self) -> str:
        return "docker"

    def create_runner_config(self, task_dir: Path, ctx: dict[str, Any]) -> dict[str, Any]:
        compose = task_dir / "docker-compose.yml"
        if not compose.exists():
            compose = task_dir / "docker-compose.yaml"
        return {
            "docker_compose_path": str(compose),
            "container_name": task_dir.name,
            "timeout": 120,
        }


provider = TerminalBenchProvider()
