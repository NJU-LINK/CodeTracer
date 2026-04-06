"""Error analysis models: verdicts, labels, and analysis results."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class StepVerdict(str, Enum):
    INCORRECT = "incorrect"
    UNUSEFUL = "unuseful"


@dataclass
class StepLabel:
    """Label for a single step identified during error analysis."""

    step_id: int
    verdict: StepVerdict
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {"step_id": self.step_id, "verdict": self.verdict.value, "reasoning": self.reasoning}

    @classmethod
    def from_dict(cls, d: dict) -> StepLabel:
        return cls(
            step_id=d["step_id"],
            verdict=StepVerdict(d["verdict"]),
            reasoning=d.get("reasoning", ""),
        )


@dataclass
class ErrorAnalysis:
    """Result of trajectory error analysis produced by TraceAgent."""

    traj_id: str
    labels: list[StepLabel] = field(default_factory=list)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def first_incorrect_step_id(self) -> int | None:
        incorrect = [l for l in self.labels if l.verdict == StepVerdict.INCORRECT]
        return min((l.step_id for l in incorrect), default=None)

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "labels": [l.to_dict() for l in self.labels],
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ErrorAnalysis:
        return cls(
            traj_id=d.get("traj_id", ""),
            labels=[StepLabel.from_dict(l) for l in d.get("labels", [])],
            summary=d.get("summary", ""),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(_json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ErrorAnalysis:
        return cls.from_dict(_json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_labels_json(cls, path: Path, traj_id: str) -> ErrorAnalysis:
        """Parse a codetracer_labels.json (stage-level format) into ErrorAnalysis."""
        raw = _json.loads(path.read_text(encoding="utf-8"))
        labels: list[StepLabel] = []
        for stage in raw:
            if not isinstance(stage, dict):
                continue
            reasoning = stage.get("reasoning", "")
            for sid in stage.get("incorrect_step_ids", []):
                labels.append(StepLabel(step_id=sid, verdict=StepVerdict.INCORRECT, reasoning=reasoning))
            for sid in stage.get("unuseful_step_ids", []):
                labels.append(StepLabel(step_id=sid, verdict=StepVerdict.UNUSEFUL, reasoning=reasoning))
        return cls(traj_id=traj_id, labels=labels)
