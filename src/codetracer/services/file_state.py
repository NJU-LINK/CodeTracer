"""File state tracking across trajectory steps.

Tracks file snapshots at each step so replay can verify environment
consistency and trace analysis can produce precise diffs.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FileSnapshot:
    """Immutable snapshot of a single file."""

    path: str
    content_hash: str
    size: int
    timestamp: float

    @classmethod
    def from_path(cls, filepath: str | Path) -> FileSnapshot | None:
        p = Path(filepath)
        if not p.is_file():
            return None
        try:
            data = p.read_bytes()
            return cls(
                path=str(p),
                content_hash=hashlib.sha256(data).hexdigest(),
                size=len(data),
                timestamp=p.stat().st_mtime,
            )
        except OSError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "size": self.size,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FileSnapshot:
        return cls(
            path=d["path"],
            content_hash=d["content_hash"],
            size=d["size"],
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class FileDiff:
    """Represents a change between two snapshots of the same path."""

    path: str
    change_type: str  # "added", "removed", "modified"
    old_hash: str = ""
    new_hash: str = ""
    old_size: int = 0
    new_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "old_size": self.old_size,
            "new_size": self.new_size,
        }


@dataclass
class StepFileState:
    """File state snapshot taken after a specific step."""

    step_id: int
    snapshots: dict[str, FileSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "snapshots": {k: v.to_dict() for k, v in self.snapshots.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepFileState:
        return cls(
            step_id=d["step_id"],
            snapshots={
                k: FileSnapshot.from_dict(v)
                for k, v in d.get("snapshots", {}).items()
            },
        )


class FileStateTracker:
    """LRU-based file state cache tracking changes across trajectory steps."""

    def __init__(self, max_entries: int = 200) -> None:
        self._max_entries = max_entries
        self._step_states: OrderedDict[int, StepFileState] = OrderedDict()

    def snapshot(self, step_id: int, paths: list[str]) -> StepFileState:
        """Take a snapshot of the given *paths* for *step_id*."""
        snaps: dict[str, FileSnapshot] = {}
        for p in paths:
            fs = FileSnapshot.from_path(p)
            if fs is not None:
                snaps[os.path.normpath(p)] = fs

        state = StepFileState(step_id=step_id, snapshots=snaps)
        self._step_states[step_id] = state

        while len(self._step_states) > self._max_entries:
            self._step_states.popitem(last=False)

        return state

    def snapshot_directory(self, step_id: int, directory: str | Path) -> StepFileState:
        """Snapshot all files under *directory*."""
        d = Path(directory)
        paths = [str(f) for f in d.rglob("*") if f.is_file()]
        return self.snapshot(step_id, paths)

    def get_state(self, step_id: int) -> StepFileState | None:
        return self._step_states.get(step_id)

    def diff(self, step_a: int, step_b: int) -> list[FileDiff]:
        """Compute file diffs between two step states."""
        state_a = self._step_states.get(step_a)
        state_b = self._step_states.get(step_b)
        if state_a is None or state_b is None:
            return []

        a_snaps = state_a.snapshots
        b_snaps = state_b.snapshots
        all_paths = set(a_snaps.keys()) | set(b_snaps.keys())
        diffs: list[FileDiff] = []

        for path in sorted(all_paths):
            sa = a_snaps.get(path)
            sb = b_snaps.get(path)
            if sa is None and sb is not None:
                diffs.append(FileDiff(path, "added", new_hash=sb.content_hash, new_size=sb.size))
            elif sa is not None and sb is None:
                diffs.append(FileDiff(path, "removed", old_hash=sa.content_hash, old_size=sa.size))
            elif sa is not None and sb is not None and sa.content_hash != sb.content_hash:
                diffs.append(FileDiff(
                    path, "modified",
                    old_hash=sa.content_hash, new_hash=sb.content_hash,
                    old_size=sa.size, new_size=sb.size,
                ))

        return diffs

    def diff_against_current(self, step_id: int, paths: list[str]) -> list[FileDiff]:
        """Compare stored state at *step_id* against current file system."""
        stored = self._step_states.get(step_id)
        if stored is None:
            return []

        current_snaps: dict[str, FileSnapshot] = {}
        for p in paths:
            fs = FileSnapshot.from_path(p)
            norm = os.path.normpath(p)
            if fs is not None:
                current_snaps[norm] = fs

        all_paths = set(stored.snapshots.keys()) | set(current_snaps.keys())
        diffs: list[FileDiff] = []
        for path in sorted(all_paths):
            old = stored.snapshots.get(path)
            new = current_snaps.get(path)
            if old is None and new is not None:
                diffs.append(FileDiff(path, "added", new_hash=new.content_hash, new_size=new.size))
            elif old is not None and new is None:
                diffs.append(FileDiff(path, "removed", old_hash=old.content_hash, old_size=old.size))
            elif old is not None and new is not None and old.content_hash != new.content_hash:
                diffs.append(FileDiff(
                    path, "modified",
                    old_hash=old.content_hash, new_hash=new.content_hash,
                    old_size=old.size, new_size=new.size,
                ))
        return diffs
