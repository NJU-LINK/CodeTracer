"""Normalizer: detect format -> parse trajectory -> write steps.json / task.md."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rich.console import Console

from codetracer.models import FileRef, NormalizedTrajectory, StepRecord
from codetracer.skills.loader import Skill
from codetracer.skills.pool import SkillPool

console = Console(highlight=False)


def _load_pre_normalized(run_dir: Path) -> NormalizedTrajectory:
    """Load a directory that already contains steps.json (pre-normalized data, not an agent skill)."""
    steps_path = run_dir / "steps.json"
    raw = json.loads(steps_path.read_text(encoding="utf-8", errors="replace"))
    steps: list[StepRecord] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        step_id = entry.get("step_id", len(steps) + 1)
        a_dict = entry.get("action_ref") or {}
        o_dict = entry.get("observation_ref") or {}
        a_ref = FileRef.from_dict(a_dict) if a_dict else None
        o_ref = FileRef.from_dict(o_dict) if o_dict else None
        action = entry.get("action") or (a_ref.content if a_ref else "") or ""
        observation = entry.get("observation", o_ref.content if o_ref else None)
        steps.append(
            StepRecord(
                step_id=step_id,
                action=action,
                observation=observation,
                thinking=entry.get("thinking"),
                parallel_group=entry.get("parallel_group"),
                tool_type=entry.get("tool_type"),
                action_ref=a_ref,
                observation_ref=o_ref,
            )
        )

    task_description = ""
    task_md = run_dir / "task.md"
    if task_md.exists():
        task_description = task_md.read_text(encoding="utf-8", errors="replace")

    return NormalizedTrajectory(
        steps=steps,
        task_description=task_description,
        metadata={"format": "pre_normalized", "run_dir": str(run_dir)},
    )


def _load_step_jsonl_dir(run_dir: Path) -> NormalizedTrajectory:
    """Load a directory with per-step ``step_N.jsonl`` annotation files.

    Each file contains ``command``, ``terminal_result``, ``step_index``, etc.
    This is the annotation format produced by the batch annotation pipeline.
    """
    jsonl_files = sorted(
        run_dir.glob("step_*.jsonl"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    steps: list[StepRecord] = []
    for p in jsonl_files:
        raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        step_id = raw.get("step_index", int(p.stem.split("_", 1)[1]))
        action = raw.get("command", "")
        observation = raw.get("terminal_result", "")
        steps.append(StepRecord(step_id=step_id, action=action, observation=observation))

    task_description = ""
    task_md = run_dir / "task.md"
    if task_md.exists():
        task_description = task_md.read_text(encoding="utf-8", errors="replace")

    return NormalizedTrajectory(
        steps=steps,
        task_description=task_description,
        metadata={"format": "step_jsonl", "run_dir": str(run_dir)},
    )


class Normalizer:
    """Orchestrates format detection and parsing into a NormalizedTrajectory."""

    def __init__(self, pool: SkillPool) -> None:
        self._pool = pool

    def is_pre_normalized(self, run_dir: Path) -> bool:
        """True if run_dir already contains steps.json (no agent skill needed)."""
        return (run_dir / "steps.json").exists()

    def is_step_jsonl_dir(self, run_dir: Path) -> bool:
        """True if run_dir contains step_N.jsonl files (annotation format)."""
        if self.is_pre_normalized(run_dir):
            return False
        return any(run_dir.glob("step_*.jsonl"))

    def detect(self, run_dir: Path, format_override: str | None = None) -> Skill:
        """Return the matching skill for *run_dir*.

        If *format_override* is given (and not "auto"), look it up directly.
        Otherwise use fingerprint-based detection.
        """
        if format_override and format_override != "auto":
            skill = self._pool.get(format_override)
            if skill is None:
                raise ValueError(f"No skill registered for format '{format_override}'")
            return skill

        name = self._pool.detect(run_dir)
        if name is None:
            raise ValueError(f"Unable to detect trajectory format in {run_dir}")
        skill = self._pool.get(name)
        assert skill is not None
        return skill

    def normalize_pre_normalized(
        self, run_dir: Path, output_dir: Path | None = None, *, quiet: bool = False
    ) -> NormalizedTrajectory:
        """Load an already-normalized directory directly (no skill involved)."""
        traj = _load_pre_normalized(run_dir)
        if not quiet:
            console.print(f"Loaded [bold]{len(traj.steps)}[/bold] pre-normalized steps")
        self._write_derived_artifacts(run_dir, traj, output_dir=output_dir, quiet=quiet)
        return traj

    def normalize_step_jsonl(
        self, run_dir: Path, output_dir: Path | None = None, *, quiet: bool = False
    ) -> NormalizedTrajectory:
        """Load a directory with per-step step_N.jsonl annotation files."""
        traj = _load_step_jsonl_dir(run_dir)
        if not quiet:
            console.print(f"Loaded [bold]{len(traj.steps)}[/bold] steps from step_N.jsonl files")

        write_dir = output_dir or run_dir
        steps_json_path = write_dir / "steps.json"
        if not steps_json_path.exists():
            traj.write_steps_json(steps_json_path)
            if not quiet:
                console.print(f"Wrote normalized steps -> [bold]{steps_json_path}[/bold]")

        self._write_derived_artifacts(run_dir, traj, output_dir=output_dir, quiet=quiet)
        return traj

    def normalize(
        self, run_dir: Path, skill: Skill, output_dir: Path | None = None, *, quiet: bool = False
    ) -> NormalizedTrajectory:
        """Parse *run_dir* using *skill* and write derived artifacts."""
        traj = skill.parse(run_dir)
        if not quiet:
            console.print(f"Parsed [bold]{len(traj.steps)}[/bold] steps via [bold green]{skill.name}[/bold green]")

        write_dir = output_dir or run_dir
        steps_json_path = write_dir / "steps.json"
        if not steps_json_path.exists():
            traj.write_steps_json(steps_json_path)
            if not quiet:
                console.print(f"Wrote normalized steps -> [bold]{steps_json_path}[/bold]")

        self._write_derived_artifacts(run_dir, traj, output_dir=output_dir, quiet=quiet)
        return traj

    def _write_derived_artifacts(
        self, run_dir: Path, traj: NormalizedTrajectory, output_dir: Path | None = None, *, quiet: bool = False
    ) -> None:
        write_dir = output_dir or run_dir

        task_md_path = write_dir / "task.md"
        if not task_md_path.exists() and traj.task_description:
            task_md_path.write_text(traj.task_description, encoding="utf-8")

        stage_ranges_path = write_dir / "stage_ranges.json"
        if not stage_ranges_path.exists():
            source = run_dir / "stage_ranges.json"
            if source.exists() and source != stage_ranges_path:
                shutil.copy2(source, stage_ranges_path)
            else:
                n = len(traj.steps)
                stage_ranges_path.write_text(
                    json.dumps([{"stage": "full", "start_step_id": 1, "end_step_id": n}], indent=2),
                    encoding="utf-8",
                )
