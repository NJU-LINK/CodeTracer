"""Compact mechanism for context window management.

When the conversation history approaches the model's context window limit,
the CompactManager summarises earlier messages so analysis can continue.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from codetracer.llm.token_estimation import estimate_tokens

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_BUFFER_TOKENS = 13_000
_MAX_CONSECUTIVE_FAILURES = 3

COMPACT_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

Your task is to create a detailed summary of the conversation so far, \
paying close attention to the analysis work done and evidence gathered.

Before providing your final summary, wrap your analysis in <analysis> \
tags to organize your thoughts. Then provide the summary in <summary> tags.

Your summary should include the following sections:

1. Primary Request and Intent: What trajectory is being analysed and why.
2. Key Technical Concepts: Technologies, frameworks, and agent patterns observed.
3. Files and Code Sections: Files examined, modified, or referenced with key snippets.
4. Errors and Fixes: Errors encountered during analysis and how they were resolved.
5. Evidence Gathered: Steps inspected, labels assigned so far, reasoning chains.
6. Analysis Progress: What has been completed and what remains.
7. Pending Tasks: Outstanding analysis work.
8. Current Work: Precisely what was being worked on immediately before this summary.
9. Next Step: The immediate next action to take.

Respond ONLY with the <analysis> and <summary> blocks. No tool calls.
"""


def _format_compact_summary(raw: str) -> str:
    """Strip the <analysis> scratchpad and extract <summary> content."""
    cleaned = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw)
    match = re.search(r"<summary>([\s\S]*?)</summary>", cleaned)
    if match:
        return f"[Compact Summary]\n{match.group(1).strip()}"
    return f"[Compact Summary]\n{cleaned.strip()}"


class CompactManager:
    """Manages context-window compaction for agent conversations."""

    def __init__(
        self,
        context_window: int = _DEFAULT_CONTEXT_WINDOW,
        buffer_tokens: int = _DEFAULT_BUFFER_TOKENS,
        max_failures: int = _MAX_CONSECUTIVE_FAILURES,
        enabled: bool = True,
    ) -> None:
        self._threshold = max(0, context_window - buffer_tokens)
        self._max_failures = max_failures
        self._consecutive_failures = 0
        self._compact_count = 0
        self.enabled = enabled

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def compact_count(self) -> int:
        return self._compact_count

    def should_compact(self, messages: list[dict[str, Any]]) -> bool:
        if not self.enabled:
            return False
        if self._consecutive_failures >= self._max_failures:
            return False
        tokens = estimate_tokens(messages)
        return tokens >= self._threshold

    def compact(
        self,
        messages: list[dict[str, Any]],
        llm: Any,
    ) -> list[dict[str, Any]]:
        """Summarize *messages* and return a shorter replacement list.

        Keeps the original system message and replaces everything else
        with a single user message containing the summary, followed by
        an assistant acknowledgement.
        """
        try:
            summary_resp = llm.query(
                messages + [{"role": "user", "content": COMPACT_PROMPT}],
            )
            summary_text = _format_compact_summary(summary_resp.get("content", ""))

            system_msgs = [m for m in messages if m.get("role") == "system"]
            compacted: list[dict[str, Any]] = list(system_msgs) + [
                {"role": "user", "content": summary_text},
                {
                    "role": "assistant",
                    "content": (
                        "Understood. I have reviewed the summary of our prior analysis. "
                        "I will continue from where we left off."
                    ),
                },
            ]
            self._consecutive_failures = 0
            self._compact_count += 1
            logger.info(
                "Compacted %d messages -> %d (compact #%d)",
                len(messages),
                len(compacted),
                self._compact_count,
            )
            return compacted

        except Exception:
            self._consecutive_failures += 1
            logger.warning(
                "Compact failed (%d/%d consecutive failures)",
                self._consecutive_failures,
                self._max_failures,
                exc_info=True,
            )
            return messages
