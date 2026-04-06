"""Shell command executor used by agent loops."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


class Executor:
    """Runs shell commands in a working directory with timeout handling."""

    def __init__(self, work_dir: Path, timeout: int = 60, extra_env: dict[str, str] | None = None) -> None:
        self._work_dir = work_dir
        self._timeout = timeout
        self._env = os.environ | (extra_env or {"PAGER": "cat", "MANPAGER": "cat"})

    def run(self, cmd: str) -> dict[str, Any]:
        """Execute *cmd* and return {output, returncode}.  Raises TimeoutError on timeout."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self._work_dir),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
                env=self._env,
            )
            return {"output": result.stdout + result.stderr, "returncode": result.returncode}
        except subprocess.TimeoutExpired as e:
            out = (e.output or b"").decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
            raise TimeoutError(out) from e
