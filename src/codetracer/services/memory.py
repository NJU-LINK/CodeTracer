"""Cross-trajectory memory system with online extraction.

Maintains per-agent-type TRACER.md files that accumulate common failure
patterns and agent-specific quirks across multiple trajectory analyses.

Online memory: ``OnlineMemoryExtractor`` runs mid-analysis to extract
useful patterns from the conversation so far and persist them, so that
subsequent steps (and future analyses) benefit from accumulated experience.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_FILENAME_TEMPLATE = "{agent_type}.md"
_MAX_MEMORY_SIZE = 50_000  # characters

_ONLINE_EXTRACTION_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

You are reviewing an ongoing trajectory analysis conversation. Extract any
reusable insights that would help analyze FUTURE trajectories of this agent type.

Focus on:
- Recurring failure patterns (e.g., "this agent type often fails to run tests after edits")
- Agent-specific quirks (e.g., "uses non-standard file layouts", "logs are in XML not JSON")
- Effective investigation strategies that worked well
- Common pitfalls to avoid during analysis

Do NOT extract:
- Task-specific details (specific file names, variable names from the task under analysis)
- Observations that only apply to this particular trajectory
- Anything already covered in the existing memory below

Existing memory:
{existing_memory}

Return a concise bulleted list of NEW insights only. If nothing new is worth
recording, return exactly: NO_NEW_INSIGHTS
"""


class OnlineMemoryExtractor:
    """Extracts memory mid-analysis, inspired by Claude Code's SessionMemory.

    Runs asynchronously after every ``step_interval`` agent steps (or when
    estimated token usage exceeds ``token_threshold``).  Fire-and-forget:
    does not block the main agent loop.
    """

    def __init__(
        self,
        agent_type: str,
        llm: Any,
        memory_dir: Path | None = None,
        step_interval: int = 8,
        token_threshold: int = 30_000,
    ) -> None:
        self._agent_type = agent_type
        self._llm = llm
        self._memory_dir = memory_dir or _default_memory_dir()
        self._step_interval = step_interval
        self._token_threshold = token_threshold
        self._last_extraction_step = 0
        self._last_extraction_tokens = 0
        self._lock = threading.Lock()
        self._running = False

    def should_extract(self, step: int, total_tokens: int) -> bool:
        """Check whether extraction should be triggered."""
        step_due = (step - self._last_extraction_step) >= self._step_interval
        token_due = (total_tokens - self._last_extraction_tokens) >= self._token_threshold
        return (step_due or token_due) and not self._running

    def extract_async(
        self,
        messages: list[dict[str, Any]],
        step: int,
        total_tokens: int,
    ) -> None:
        """Launch extraction in a background thread (fire-and-forget)."""
        if self._running:
            return
        self._running = True
        self._last_extraction_step = step
        self._last_extraction_tokens = total_tokens
        t = threading.Thread(
            target=self._do_extract,
            args=(list(messages), step),
            daemon=True,
        )
        t.start()

    def _do_extract(self, messages: list[dict[str, Any]], step: int) -> None:
        """Background extraction: query LLM for insights, persist to memory."""
        try:
            existing = load_memory(self._agent_type, self._memory_dir)
            prompt = _ONLINE_EXTRACTION_PROMPT.format(
                existing_memory=existing or "(no prior memory)"
            )

            # Build a condensed view of the conversation for extraction
            condensed = self._condense_messages(messages)
            extraction_messages = [
                {"role": "user", "content": prompt + "\n\nConversation so far:\n" + condensed},
            ]

            resp = self._llm.query(extraction_messages)
            content = resp.get("content", "").strip()

            if content and content != "NO_NEW_INSIGHTS":
                update_memory(
                    self._agent_type,
                    f"[Online extraction at step {step}]\n{content}",
                    memory_dir=self._memory_dir,
                )
                logger.info("Online memory extraction at step %d: saved new insights", step)
            else:
                logger.debug("Online memory extraction at step %d: no new insights", step)
        except Exception:
            logger.warning("Online memory extraction failed at step %d", step, exc_info=True)
        finally:
            self._running = False

    @staticmethod
    def _condense_messages(messages: list[dict[str, Any]], max_chars: int = 30_000) -> str:
        """Build a condensed view of messages for extraction.

        Keeps the system message in full, then samples assistant/user
        messages from the tail of the conversation, trimming to fit
        within *max_chars*.
        """
        parts: list[str] = []
        total = 0

        # Always include system messages
        for m in messages:
            if m.get("role") == "system":
                text = m["content"][:3000]
                parts.append(f"[system] {text}")
                total += len(text)

        # Include recent messages (most relevant for pattern extraction)
        for m in reversed(messages):
            if m.get("role") == "system":
                continue
            role = m.get("role", "?")
            text = m.get("content", "")
            # Trim individual messages to 2000 chars
            if len(text) > 2000:
                text = text[:1000] + "\n...[trimmed]...\n" + text[-1000:]
            entry = f"[{role}] {text}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)

        return "\n---\n".join(parts)


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
