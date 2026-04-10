"""Output validation for CodeTracer analysis results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codetracer.models.trajectory import NormalizedTrajectory
from codetracer.state.output_profile import OutputProfile


@dataclass
class ValidationResult:
    """Result of validating an analysis output against its trajectory."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def validate_analysis_output(
    output_path: Path,
    traj: NormalizedTrajectory,
    profile: OutputProfile,
) -> ValidationResult:
    """Validate that *output_path* conforms to the expected profile schema
    and references valid step IDs from *traj*."""
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    if not output_path.exists():
        return ValidationResult(valid=False, errors=[f"Output file not found: {output_path}"])

    raw_text = output_path.read_text(encoding="utf-8", errors="replace")
    if not raw_text.strip():
        return ValidationResult(valid=False, errors=["Output file is empty"])

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return ValidationResult(valid=False, errors=[f"Invalid JSON: {e}"])

    valid_step_ids = {s.step_id for s in traj.steps}

    if profile.name == "tracebench":
        errors.extend(_validate_tracebench(data, valid_step_ids, metrics))
    elif profile.name == "rl_feedback":
        errors.extend(_validate_rl_feedback(data, valid_step_ids, metrics))
    elif profile.name == "detailed":
        errors.extend(_validate_detailed(data, valid_step_ids, metrics))
    else:
        if not isinstance(data, (list, dict)):
            errors.append(f"Expected JSON array or object, got {type(data).__name__}")

    if not errors:
        for entry in (data if isinstance(data, list) else [data]):
            if isinstance(entry, dict):
                reasoning = entry.get("reasoning", "")
                if not reasoning:
                    warnings.append(f"Empty reasoning in entry with step/stage: {entry.get('step_id', entry.get('stage_id', '?'))}")

    metrics["file_size_bytes"] = len(raw_text)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def _validate_tracebench(
    data: Any, valid_ids: set[int], metrics: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, list):
        errors.append(f"Expected JSON array, got {type(data).__name__}")
        return errors

    metrics["stage_count"] = len(data)
    total_incorrect = 0
    total_unuseful = 0

    for i, stage in enumerate(data):
        if not isinstance(stage, dict):
            errors.append(f"Stage {i}: expected object, got {type(stage).__name__}")
            continue
        for key in ("stage_id", "incorrect_step_ids", "unuseful_step_ids", "reasoning"):
            if key not in stage:
                errors.append(f"Stage {i}: missing required key '{key}'")
        for sid in stage.get("incorrect_step_ids", []):
            if sid not in valid_ids:
                errors.append(f"Stage {i}: incorrect step_id {sid} not in trajectory (valid: {min(valid_ids)}-{max(valid_ids)})")
            total_incorrect += 1
        for sid in stage.get("unuseful_step_ids", []):
            if sid not in valid_ids:
                errors.append(f"Stage {i}: unuseful step_id {sid} not in trajectory")
            total_unuseful += 1

    metrics["total_incorrect"] = total_incorrect
    metrics["total_unuseful"] = total_unuseful
    return errors


def _validate_rl_feedback(
    data: Any, valid_ids: set[int], metrics: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, list):
        errors.append(f"Expected JSON array, got {type(data).__name__}")
        return errors

    metrics["entry_count"] = len(data)
    valid_verdicts = {"incorrect", "unuseful", "correct"}

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            errors.append(f"Entry {i}: expected object")
            continue
        for key in ("step_id", "verdict", "deviation_type", "reasoning", "reward_signal"):
            if key not in entry:
                errors.append(f"Entry {i}: missing required key '{key}'")
        sid = entry.get("step_id")
        if sid is not None and sid not in valid_ids:
            errors.append(f"Entry {i}: step_id {sid} not in trajectory")
        verdict = entry.get("verdict", "")
        if verdict not in valid_verdicts:
            errors.append(f"Entry {i}: invalid verdict '{verdict}'")
        reward = entry.get("reward_signal")
        if reward is not None and not (-1.0 <= reward <= 1.0):
            errors.append(f"Entry {i}: reward_signal {reward} out of range [-1.0, 1.0]")

    return errors


def _validate_detailed(
    data: Any, valid_ids: set[int], metrics: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append(f"Expected JSON object, got {type(data).__name__}")
        return errors

    for key in ("root_cause_chain", "critical_decision_points", "correct_strategy", "stage_labels", "summary"):
        if key not in data:
            errors.append(f"Missing required key '{key}'")

    for i, dp in enumerate(data.get("critical_decision_points", [])):
        if isinstance(dp, dict):
            sid = dp.get("step_id")
            if sid is not None and sid not in valid_ids:
                errors.append(f"Decision point {i}: step_id {sid} not in trajectory")

    return errors
