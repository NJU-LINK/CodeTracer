"""Session persistence for interactive mode.

Append-only JSONL transcript recording with state snapshot for resume.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionPersistence:
    """Manages session transcript persistence and restore."""

    def __init__(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._transcript_path = session_dir / "transcript.jsonl"
        self._state_path = session_dir / "session_state.json"

    @property
    def transcript_path(self) -> Path:
        return self._transcript_path

    def ensure_dir(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def record_message(self, message: dict[str, Any]) -> None:
        """Append a single message to the transcript."""
        self.ensure_dir()
        entry = {
            "timestamp": time.time(),
            "type": "message",
            **message,
        }
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record_messages(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple messages to the transcript."""
        if not messages:
            return
        self.ensure_dir()
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            for msg in messages:
                entry = {
                    "timestamp": time.time(),
                    "type": "message",
                    **msg,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Record a non-message event (compact, error, etc.)."""
        self.ensure_dir()
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            **(data or {}),
        }
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_state(self, state: dict[str, Any]) -> None:
        """Snapshot full state for resume."""
        self.ensure_dir()
        state_with_meta = {
            "saved_at": time.time(),
            **state,
        }
        self._state_path.write_text(
            json.dumps(state_with_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def restore_state(self) -> dict[str, Any] | None:
        """Load the last saved state, or None if unavailable."""
        if not self._state_path.exists():
            return None
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug("Could not restore session state from %s", self._state_path, exc_info=True)
            return None

    def restore_messages(self) -> list[dict[str, Any]]:
        """Load all messages from the transcript."""
        if not self._transcript_path.exists():
            return []
        messages: list[dict[str, Any]] = []
        try:
            with open(self._transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("type") == "message":
                        messages.append(entry)
        except (json.JSONDecodeError, OSError):
            logger.debug("Could not restore transcript from %s", self._transcript_path, exc_info=True)
        return messages

    def has_session(self) -> bool:
        """Check whether a previous session exists to resume."""
        return self._transcript_path.exists() and self._transcript_path.stat().st_size > 0
