"""Shared utilities for LLM-driven code generation (skills, bench providers).

Extracted from SkillGenerator and BenchGenerator to eliminate duplication.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


def list_dir(directory: Path, max_files: int = 200) -> str:
    """List files in *directory* (excluding .git), returning up to *max_files* lines."""
    try:
        result = subprocess.run(
            ["find", ".", "-not", "-path", "./.git/*", "-type", "f"],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = result.stdout.splitlines()[:max_files]
        return "\n".join(lines)
    except Exception as e:
        return f"(listing failed: {e})"


def sample_files(
    directory: Path,
    listing: str,
    max_files: int = 3,
    max_size: int = 500_000,
    max_preview_lines: int = 60,
) -> str:
    """Return previews of representative files from *directory*."""
    candidates: list[Path] = []
    for line in listing.splitlines():
        p = (directory / line.lstrip("./")).resolve()
        if p.is_file() and p.stat().st_size < max_size:
            candidates.append(p)
        if len(candidates) >= max_files * 2:
            break

    def _score(p: Path) -> int:
        name = p.name.lower()
        if name in ("results.json", "response.json", "response.txt", "prompt.txt"):
            return 0
        if p.suffix in (".json", ".jsonl", ".txt", ".log"):
            return 1
        return 2

    top = sorted(candidates, key=_score)[:max_files]
    parts: list[str] = []
    for p in top:
        rel = str(p.relative_to(directory))
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[:max_preview_lines]
            preview = "\n".join(lines)
        except Exception:
            preview = "(unreadable)"
        parts.append(f"=== {rel} ===\n{preview}")
    return "\n\n".join(parts) or "(no files sampled)"


def extract_code_block(content: str, lang: str) -> str:
    """Extract the first fenced code block for *lang* from LLM *content*."""
    pattern = rf"```{lang}\s*\n(.*?)```"
    m = re.search(pattern, content, re.DOTALL)
    if m:
        return m.group(1).strip()
    if lang == "python" and "class " in content:
        m2 = re.search(r"```\s*\n(.*?)```", content, re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return ""


def validate_in_subprocess(script: str, timeout: int = 30) -> str | None:
    """Run *script* in an isolated subprocess; return None on success, error string otherwise."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip() == "OK":
            return None
        err = result.stderr.strip() or result.stdout.strip()
        return err or "unknown validation error"
    except subprocess.TimeoutExpired:
        return "validation timed out"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
