"""Checkpoint management: build, save, and restore StepCheckpoints.

Optionally records file state snapshots at each step via FileStateTracker.
"""

from __future__ import annotations

import logging
from pathlib import Path

from codetracer.models import (
    ErrorAnalysis,
    NormalizedTrajectory,
    StepCheckpoint,
)
from codetracer.services.file_state import FileStateTracker, StepFileState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Creates and persists StepCheckpoint objects from a trajectory."""

    def __init__(self, checkpoints_dir: Path | None = None) -> None:
        self._dir = checkpoints_dir

    def build(
        self,
        traj: NormalizedTrajectory,
        target_step_id: int,
        analysis: ErrorAnalysis | None = None,
        env_config: dict | None = None,
        file_tracker: FileStateTracker | None = None,
    ) -> StepCheckpoint:
        """Build a checkpoint capturing all steps before *target_step_id*.

        If *file_tracker* is provided, its stored step states are
        serialised into the checkpoint for later verification.
        """
        preceding = [s for s in traj.steps if s.step_id < target_step_id]
        file_states: list[dict] = []
        if file_tracker is not None:
            for step in preceding:
                st = file_tracker.get_state(step.step_id)
                if st is not None:
                    file_states.append(st.to_dict())
        return StepCheckpoint(
            target_step_id=target_step_id,
            replayed_steps=preceding,
            error_analysis=analysis,
            env_config=env_config or {},
            file_states=file_states,
        )

    def save(self, checkpoint: StepCheckpoint, name: str | None = None) -> Path:
        """Persist checkpoint to disk. Returns the written path."""
        if self._dir is None:
            raise ValueError("checkpoints_dir not configured")
        self._dir.mkdir(parents=True, exist_ok=True)
        fname = name or f"checkpoint_step_{checkpoint.target_step_id}.json"
        path = self._dir / fname
        checkpoint.save(path)
        logger.info("Saved checkpoint -> %s", path)
        return path

    def load(self, path: Path) -> StepCheckpoint:
        return StepCheckpoint.load(path)

    def list_checkpoints(self) -> list[Path]:
        if self._dir is None or not self._dir.exists():
            return []
        return sorted(self._dir.glob("checkpoint_*.json"))
