"""Replay models: checkpoints, results, and status."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from codetracer.models.analysis import ErrorAnalysis
from codetracer.models.trajectory import StepRecord


@dataclass
class StepCheckpoint:
    """Serializable snapshot of trajectory state at a given step.

    Analogous to a git commit -- captures everything needed to restore
    the environment to the state right before *target_step_id* executed.
    """

    target_step_id: int
    replayed_steps: list[StepRecord] = field(default_factory=list)
    error_analysis: ErrorAnalysis | None = None
    env_config: dict = field(default_factory=dict)
    file_states: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_step_id": self.target_step_id,
            "replayed_steps": [s.to_dict() for s in self.replayed_steps],
            "error_analysis": self.error_analysis.to_dict() if self.error_analysis else None,
            "env_config": self.env_config,
            "file_states": self.file_states,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepCheckpoint:
        steps = [
            StepRecord(
                step_id=s["step_id"],
                action=s.get("action", ""),
                observation=s.get("observation"),
            )
            for s in d.get("replayed_steps", [])
        ]
        ea = ErrorAnalysis.from_dict(d["error_analysis"]) if d.get("error_analysis") else None
        return cls(
            target_step_id=d["target_step_id"],
            replayed_steps=steps,
            error_analysis=ea,
            env_config=d.get("env_config", {}),
            file_states=d.get("file_states", []),
        )

    def save(self, path: Path) -> None:
        path.write_text(_json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StepCheckpoint:
        return cls.from_dict(_json.loads(path.read_text(encoding="utf-8")))


class ReplayStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ReplayResult:
    """Outcome of a replay session."""

    status: ReplayStatus
    checkpoint: StepCheckpoint | None = None
    agent_output: str = ""
    steps_replayed: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "agent_output": self.agent_output,
            "steps_replayed": self.steps_replayed,
            "metadata": self.metadata,
        }
