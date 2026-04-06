"""Cross-trajectory memory system.

Maintains per-agent-type TRACER.md files that accumulate common failure
patterns and agent-specific quirks across multiple trajectory analyses.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_FILENAME_TEMPLATE = "{agent_type}.md"
_MAX_MEMORY_SIZE = 50_000  # characters


def _default_memory_dir() -> Path:
    try:
        from platformdirs import user_data_dir
        return Path(user_data_dir("codetracer")) / "memory"
    except ImportError:
        return Path.home() / ".codetracer" / "memory"


def _sanitize_agent_type(agent_type: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", agent_type.lower().strip())


def load_memory(
    agent_type: str,
    memory_dir: Path | None = None,
) -> str:
    """Load the TRACER.md memory file for *agent_type*.

    Returns the file contents or an empty string if none exists.
    """
    d = memory_dir or _default_memory_dir()
    safe = _sanitize_agent_type(agent_type)
    if not safe:
        return ""

    path = d / _MEMORY_FILENAME_TEMPLATE.format(agent_type=safe)
    if not path.exists():
        return ""

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_MEMORY_SIZE:
            content = content[-_MAX_MEMORY_SIZE:]
        return content
    except OSError:
        logger.debug("Could not read memory file %s", path, exc_info=True)
        return ""


def update_memory(
    agent_type: str,
    analysis_summary: str,
    failure_patterns: list[str] | None = None,
    memory_dir: Path | None = None,
) -> Path:
    """Append analysis insights to the TRACER.md for *agent_type*.

    Writes a timestamped entry with the summary and any identified
    failure patterns.
    """
    d = memory_dir or _default_memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_agent_type(agent_type)
    if not safe:
        safe = "unknown"

    path = d / _MEMORY_FILENAME_TEMPLATE.format(agent_type=safe)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n## Analysis {timestamp}\n"]

    if analysis_summary:
        lines.append(analysis_summary.strip())
        lines.append("")

    if failure_patterns:
        lines.append("### Common Failure Patterns")
        for pat in failure_patterns:
            lines.append(f"- {pat}")
        lines.append("")

    entry = "\n".join(lines) + "\n"

    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")

    if not existing:
        header = f"# TRACER Memory: {agent_type}\n\nAccumulated failure patterns and analysis insights.\n"
        existing = header

    combined = existing + entry

    if len(combined) > _MAX_MEMORY_SIZE:
        combined = combined[-_MAX_MEMORY_SIZE:]

    path.write_text(combined, encoding="utf-8")
    logger.info("Updated memory for agent_type=%s -> %s", agent_type, path)
    return path


def auto_extract_memory(
    agent_type: str,
    labels_path: Path,
    analysis_summary: str = "",
    memory_dir: Path | None = None,
) -> Path | None:
    """One-shot post-analysis memory extraction.

    Reads labels JSON, extracts failure patterns, and persists them.
    Returns the updated memory path, or None if nothing to record.
    """
    import json as _json

    if not labels_path.exists():
        return None

    try:
        raw = _json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Cannot read labels at %s", labels_path, exc_info=True)
        return None

    patterns = extract_failure_patterns(
        [{"verdict": "incorrect", "reasoning": stage.get("reasoning", "")}
         for stage in raw if isinstance(stage, dict)
         and stage.get("incorrect_step_ids")]
    )

    if not analysis_summary and not patterns:
        return None

    return update_memory(agent_type, analysis_summary, patterns, memory_dir)


def extract_failure_patterns(labels: list[dict[str, Any]]) -> list[str]:
    """Extract short failure pattern strings from analysis labels."""
    patterns: list[str] = []
    for label in labels:
        verdict = label.get("verdict", "")
        reasoning = label.get("reasoning", "")
        if verdict == "incorrect" and reasoning:
            short = reasoning[:200].replace("\n", " ").strip()
            patterns.append(short)
    return patterns
