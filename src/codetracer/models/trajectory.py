"""Trajectory data models: steps, file refs, stage ranges."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileRef:
    """Reference to a source location within a raw trajectory file."""

    path: str
    line_start: int
    line_end: int
    content: str

    @classmethod
    def from_dict(cls, d: dict) -> FileRef:
        return cls(
            path=d.get("path", ""),
            line_start=d.get("line_start", 1),
            line_end=d.get("line_end", 1),
            content=d.get("content", ""),
        )

    def to_dict(self) -> dict:
        return {"path": self.path, "line_start": self.line_start, "line_end": self.line_end, "content": self.content}


@dataclass
class StepRecord:
    """One normalized step from a trajectory: an action and its observation."""

    step_id: int
    action: str
    observation: str | None = None
    action_ref: FileRef | None = None
    observation_ref: FileRef | None = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "observation": self.observation,
            "action_ref": self.action_ref.to_dict() if self.action_ref else None,
            "observation_ref": self.observation_ref.to_dict() if self.observation_ref else None,
        }


@dataclass
class NormalizedTrajectory:
    """Fully normalized trajectory ready for tree building and tracing."""

    steps: list[StepRecord]
    task_description: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def write_steps_json(self, path: Path) -> None:
        import json

        path.write_text(
            json.dumps([s.to_dict() for s in self.steps], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class StageRange:
    """Inclusive step span for a single stage."""

    stage: str
    start_step_id: int
    end_step_id: int

    def to_dict(self) -> dict:
        return {"stage": self.stage, "start_step_id": self.start_step_id, "end_step_id": self.end_step_id}
