"""ReplayRunner: execute trajectory commands inside a target environment.

Supports three backend types:
- docker  : terminal-bench Docker containers via docker-compose
- git     : swe-bench style git-clone + checkout
- local   : plain subprocess in a working directory

The factory ``ReplayRunner.create()`` now delegates to BenchProvider when
available, falling back to runner_type string matching for compatibility.

Includes command partitioning for concurrent execution of read-only
commands.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CommandBatch:
    """Group of commands that can be executed together."""

    commands: list[str] = field(default_factory=list)
    is_concurrent_safe: bool = False


def partition_replay_commands(
    commands: list[str],
    classifier: Any | None = None,
) -> list[CommandBatch]:
    """Partition commands into concurrent-safe (read-only) and serial batches.

    Uses *classifier* (a ``ClassificationStore`` instance) as the single
    source of truth for read/write classification.
    """
    if not commands:
        return []

    if classifier is None:
        from codetracer.services.classification import ClassificationStore
        classifier = ClassificationStore()

    batches: list[CommandBatch] = []
    for cmd in commands:
        safe = classifier.is_read_only(cmd)
        if batches and batches[-1].is_concurrent_safe == safe:
            batches[-1].commands.append(cmd)
        else:
            batches.append(CommandBatch(commands=[cmd], is_concurrent_safe=safe))
    return batches


class RunnerResult:
    __slots__ = ("output", "returncode")

    def __init__(self, output: str, returncode: int) -> None:
        self.output = output
        self.returncode = returncode


class BaseRunner(ABC):
    """Interface every replay backend must implement."""

    @abstractmethod
    def setup(self) -> None:
        """Prepare the execution environment (start container, clone repo, etc.)."""

    @abstractmethod
    def execute(self, command: str) -> RunnerResult:
        """Run a single shell command and return its output."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources."""

    def execute_steps(
        self,
        actions: list[str],
        max_workers: int = 5,
        *,
        file_tracker: Any | None = None,
        work_dir: Path | None = None,
        step_ids: list[int] | None = None,
        classifier: Any | None = None,
    ) -> list[RunnerResult]:
        """Execute *actions*, batching consecutive read-only commands for parallelism.

        If *file_tracker* and *work_dir* are provided, takes a file-state
        snapshot after each batch (keyed by the last step_id in the batch).
        """
        batches = partition_replay_commands(actions, classifier=classifier)
        results: list[RunnerResult] = []
        cmd_index = 0
        for batch in batches:
            if batch.is_concurrent_safe and len(batch.commands) > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as pool:
                    futures = [
                        pool.submit(self.execute, cmd) for cmd in batch.commands
                    ]
                    for future in futures:
                        results.append(future.result())
            else:
                for cmd in batch.commands:
                    results.append(self.execute(cmd))

            if file_tracker is not None and work_dir is not None:
                batch_end = cmd_index + len(batch.commands) - 1
                sid = step_ids[batch_end] if step_ids and batch_end < len(step_ids) else batch_end
                try:
                    file_tracker.snapshot_directory(sid, work_dir)
                except Exception:
                    logger.debug("File state snapshot failed for step %s", sid, exc_info=True)

            cmd_index += len(batch.commands)
        return results


class LocalRunner(BaseRunner):
    """Run commands in a local working directory via subprocess."""

    def __init__(self, work_dir: Path, timeout: int = 120, env: dict[str, str] | None = None) -> None:
        self._work_dir = work_dir
        self._timeout = timeout
        self._env = os.environ | (env or {"PAGER": "cat", "MANPAGER": "cat"})

    def setup(self) -> None:
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, command: str) -> RunnerResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self._work_dir),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
                env=self._env,
            )
            return RunnerResult(proc.stdout + proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as e:
            out = (e.output or b"").decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
            return RunnerResult(out + "\n[TIMEOUT]", -1)

    def teardown(self) -> None:
        pass


class GitRunner(BaseRunner):
    """Clone a repo and checkout a base commit (swe-bench style)."""

    def __init__(self, repo_url: str, base_commit: str, work_dir: Path, timeout: int = 120) -> None:
        self._repo_url = repo_url
        self._base_commit = base_commit
        self._work_dir = work_dir
        self._timeout = timeout
        self._local: LocalRunner | None = None

    def setup(self) -> None:
        if not self._work_dir.exists():
            logger.info("Cloning %s -> %s", self._repo_url, self._work_dir)
            subprocess.run(
                ["git", "clone", "--quiet", self._repo_url, str(self._work_dir)],
                check=True, timeout=300,
            )
        subprocess.run(
            ["git", "-C", str(self._work_dir), "reset", "--hard", self._base_commit],
            check=True, timeout=60,
        )
        self._local = LocalRunner(self._work_dir, timeout=self._timeout)
        self._local.setup()

    def execute(self, command: str) -> RunnerResult:
        if self._local is None:
            raise RuntimeError("GitRunner.setup() not called")
        return self._local.execute(command)

    def teardown(self) -> None:
        pass


class DockerRunner(BaseRunner):
    """Run commands inside a terminal-bench Docker container.

    Requires the ``docker`` Python package and a running Docker daemon.
    Uses docker-compose to build/start the task container, then executes
    commands via ``docker exec``.
    """

    def __init__(self, docker_compose_path: Path, container_name: str, timeout: int = 120) -> None:
        self._compose_path = docker_compose_path
        self._container_name = container_name
        self._timeout = timeout
        self._container: Any = None

    def setup(self) -> None:
        try:
            import docker as docker_lib
        except ImportError:
            raise ImportError("Install the 'docker' package to use DockerRunner: pip install docker")

        compose_dir = self._compose_path.parent
        subprocess.run(
            ["docker", "compose", "-f", str(self._compose_path), "up", "-d", "--build"],
            cwd=str(compose_dir), check=True, timeout=600,
        )
        client = docker_lib.from_env()
        self._container = client.containers.get(self._container_name)
        logger.info("Docker container %s is running", self._container_name)

    def execute(self, command: str) -> RunnerResult:
        if self._container is None:
            raise RuntimeError("DockerRunner.setup() not called")
        exec_result = self._container.exec_run(["bash", "-c", command], demux=True)
        stdout = (exec_result.output[0] or b"").decode("utf-8", errors="replace") if exec_result.output else ""
        stderr = (exec_result.output[1] or b"").decode("utf-8", errors="replace") if exec_result.output and len(exec_result.output) > 1 else ""
        return RunnerResult(stdout + stderr, exec_result.exit_code or 0)

    def teardown(self) -> None:
        if self._compose_path.exists():
            subprocess.run(
                ["docker", "compose", "-f", str(self._compose_path), "down"],
                cwd=str(self._compose_path.parent), timeout=120,
            )
            logger.info("Docker environment torn down")


_RUNNER_TYPE_MAP: dict[str, type[BaseRunner]] = {
    "docker": DockerRunner,
    "git": GitRunner,
    "local": LocalRunner,
}


class ReplayRunner:
    """Factory that selects the right backend.

    Preferred path: provide a BenchProvider via ``create_from_provider()``.
    Legacy path: ``create()`` with bench_type string (kept for backward compat).
    """

    @staticmethod
    def create_from_provider(task_dir: Path, ctx: dict[str, Any]) -> BaseRunner:
        """Create a runner using the registered BenchProvider for *task_dir*."""
        from codetracer.benches.provider import BenchPool

        pool = BenchPool()
        bp = pool.detect(task_dir)
        if bp is not None:
            env_config = bp.provider.create_runner_config(task_dir, ctx)
            rt = bp.provider.runner_type()
            return _build_runner(rt, env_config)

        return _build_runner("local", {"work_dir": str(task_dir), "timeout": 120})

    @staticmethod
    def create(bench_type: str, env_config: dict) -> BaseRunner:
        """Legacy factory: select backend by bench_type string."""
        rt = _resolve_runner_type(bench_type)
        return _build_runner(rt, env_config)

    @staticmethod
    def create_from_config(env_config: dict) -> BaseRunner:
        """Create a runner directly from *env_config* (auto-detect type)."""
        if "docker_compose_path" in env_config:
            return _build_runner("docker", env_config)
        if "repo_url" in env_config:
            return _build_runner("git", env_config)
        return _build_runner("local", env_config)


def _resolve_runner_type(bench_type: str) -> str:
    """Map a bench_type string to a canonical runner type."""
    try:
        from codetracer.benches.provider import BenchPool
        pool = BenchPool()
        bp = pool.get(bench_type) or pool.get(bench_type.replace("-", "_"))
        if bp is not None:
            return bp.provider.runner_type()
    except Exception:
        pass

    if bench_type in ("docker", "terminal-bench"):
        return "docker"
    if bench_type == "git" or bench_type.startswith("swe-"):
        return "git"
    return "local"


def _build_runner(runner_type: str, env_config: dict) -> BaseRunner:
    if runner_type == "docker":
        return DockerRunner(
            docker_compose_path=Path(env_config["docker_compose_path"]),
            container_name=env_config["container_name"],
            timeout=int(env_config.get("timeout", 120)),
        )
    if runner_type == "git":
        return GitRunner(
            repo_url=env_config["repo_url"],
            base_commit=env_config["base_commit"],
            work_dir=Path(env_config["work_dir"]),
            timeout=int(env_config.get("timeout", 120)),
        )
    return LocalRunner(
        work_dir=Path(env_config.get("work_dir", ".")),
        timeout=int(env_config.get("timeout", 120)),
    )
