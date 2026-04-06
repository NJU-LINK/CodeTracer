"""ClassificationStore: self-evolving command classification with persistent storage.

Uses regex as the default, stores learned patterns to JSONL, and falls back
to LLM for commands that don't match any known pattern.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = Path(user_config_dir("codetracer")) / "classifications.jsonl"

_DEFAULT_CHANGE_RE = re.compile(
    r"\b(write|echo\s+.+>|tee\s|cat\s+>|cp\s|mv\s|mkdir|touch|rm\s|sed\s+-i|"
    r"patch|install|pip\s+install|npm\s+install|apt|yum|make|cmake|"
    r"git\s+(add|commit|checkout|merge|rebase|cherry-pick)|"
    r"python\s+.*\.py|bash\s+.*\.sh|\.\/|chmod|chown|ln\s+-s|"
    r"open\s*\(.*['\"]w['\"]|\.write\(|json\.dump)",
    re.IGNORECASE,
)

_DEFAULT_EXPLORE_RE = re.compile(
    r"\b(ls|find|cat\s+(?!>)|head|tail|grep|rg|wc|diff|less|more|"
    r"python\s+-c\s+'import\s+json|curl|wget|git\s+(log|status|diff|show|branch)|"
    r"echo\s+(?!\s*>)|pwd|which|type|file|stat|du|df|ps|"
    r"test\s+-[fed]|pytest|python\s+-m\s+pytest)",
    re.IGNORECASE,
)

_LLM_PROMPT = (
    "Classify this shell command as either 'change' (modifies filesystem, installs "
    "packages, edits code, writes files) or 'explore' (reads, searches, tests, "
    "inspects without side effects). Respond with ONLY the single word 'change' or 'explore'.\n\n"
    "Command: {action}"
)


class ClassificationStore:
    """Persistent, growing collection of command -> type mappings."""

    def __init__(
        self,
        store_path: Path | None = None,
        change_re: re.Pattern | None = None,
        explore_re: re.Pattern | None = None,
    ) -> None:
        self._store_path = store_path or _DEFAULT_STORE_PATH
        self._change_re = change_re or _DEFAULT_CHANGE_RE
        self._explore_re = explore_re or _DEFAULT_EXPLORE_RE
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            for line in self._store_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = entry.get("key", "")
                cls = entry.get("cls", "")
                if key and cls:
                    self._cache[key] = cls
        except Exception:
            logger.debug("Failed to load classification store", exc_info=True)

    def _persist(self, key: str, cls: str) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "cls": cls}, ensure_ascii=False) + "\n")

    @staticmethod
    def _normalize_key(action: str) -> str:
        return action.strip().splitlines()[0][:200] if action else ""

    def classify(self, action: str) -> str:
        """Classify using regex, then stored patterns. Returns 'change' or 'explore'."""
        if self._change_re.search(action):
            return "change"
        if self._explore_re.search(action):
            return "explore"
        key = self._normalize_key(action)
        if key in self._cache:
            return self._cache[key]
        return "explore"

    def classify_with_llm(self, action: str, llm: Any) -> str:
        """Classify via regex first; on miss, ask the LLM and persist the result."""
        if self._change_re.search(action):
            return "change"
        if self._explore_re.search(action):
            return "explore"

        key = self._normalize_key(action)
        if key in self._cache:
            return self._cache[key]

        try:
            resp = llm.query([
                {"role": "system", "content": "You classify shell commands. Respond with a single word."},
                {"role": "user", "content": _LLM_PROMPT.format(action=action[:500])},
            ])
            result = resp.get("content", "").strip().lower()
            cls = "change" if "change" in result else "explore"
        except Exception:
            logger.debug("LLM classification failed, defaulting to explore", exc_info=True)
            cls = "explore"

        self.store(action, cls)
        return cls

    def store(self, action: str, classification: str) -> None:
        """Manually store a classification for future reuse."""
        key = self._normalize_key(action)
        if not key:
            return
        self._cache[key] = classification
        self._persist(key, classification)

    def is_read_only(self, action: str) -> bool:
        """Convenience: True if the command is classified as 'explore' (read-only)."""
        return self.classify(action) == "explore"

    def add_regex(self, pattern: str, cls: str) -> None:
        """Extend the default regex sets at runtime."""
        if cls == "change":
            self._change_re = re.compile(
                self._change_re.pattern + "|" + pattern, self._change_re.flags
            )
        else:
            self._explore_re = re.compile(
                self._explore_re.pattern + "|" + pattern, self._explore_re.flags
            )
