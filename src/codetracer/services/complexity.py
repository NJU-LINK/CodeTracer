"""Trajectory complexity estimation for adaptive analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

from codetracer.models.trajectory import NormalizedTrajectory


@dataclass
class TrajectoryComplexity:
    """Complexity profile of a trajectory for adaptive analysis."""

    step_count: int
    unique_tool_types: int
    stage_count: int
    has_parallel_calls: bool
    has_thinking_blocks: bool
    complexity_tier: str  # "simple", "moderate", "complex"
    adaptive_instructions: str


_TOOL_PREFIX_RE = re.compile(r"^\[(\w+)\]")

_SIMPLE_INSTRUCTIONS = """\
This is a short trajectory ({step_count} steps). Inspect EVERY step individually.
Produce an exhaustive analysis covering each step's correctness."""

_MODERATE_INSTRUCTIONS = """\
This is a moderate-length trajectory ({step_count} steps, {stage_count} stages).
Use the tree index for navigation. Sample at least one step from each stage.
Focus on state-changing actions and their immediate consequences."""

_COMPLEX_INSTRUCTIONS = """\
This is a long trajectory ({step_count} steps, {stage_count} stages, \
{unique_tool_types} distinct tool types).
Strategy: identify critical decision points first by scanning the tree index.
Deep-dive ONLY into steps surrounding deviations (incorrect edits, wrong targets, \
redundant exploration loops). Do not attempt to inspect every step."""


def estimate_complexity(
    traj: NormalizedTrajectory,
    stage_count: int | None = None,
) -> TrajectoryComplexity:
    """Estimate trajectory complexity and produce adaptive instructions."""
    step_count = len(traj.steps)

    tool_types: set[str] = set()
    has_parallel = False
    has_thinking = False
    seen_groups: set[int] = set()

    for step in traj.steps:
        if step.tool_type:
            tool_types.add(step.tool_type)
        else:
            m = _TOOL_PREFIX_RE.match(step.action)
            if m:
                tool_types.add(m.group(1))

        if step.parallel_group is not None:
            if step.parallel_group in seen_groups:
                has_parallel = True
            seen_groups.add(step.parallel_group)

        if step.thinking:
            has_thinking = True

    stages = stage_count or 1
    unique = len(tool_types)

    if step_count <= 20:
        tier = "simple"
        template = _SIMPLE_INSTRUCTIONS
    elif step_count <= 100:
        tier = "moderate"
        template = _MODERATE_INSTRUCTIONS
    else:
        tier = "complex"
        template = _COMPLEX_INSTRUCTIONS

    instructions = template.format(
        step_count=step_count,
        stage_count=stages,
        unique_tool_types=unique,
    )

    if has_parallel:
        instructions += (
            "\nNote: this trajectory contains parallel tool calls "
            "(steps sharing the same parallel_group). Evaluate them as a batch."
        )
    if has_thinking:
        instructions += (
            "\nNote: thinking blocks are available. Use them to understand "
            "the agent's reasoning before judging correctness."
        )

    return TrajectoryComplexity(
        step_count=step_count,
        unique_tool_types=unique,
        stage_count=stages,
        has_parallel_calls=has_parallel,
        has_thinking_blocks=has_thinking,
        complexity_tier=tier,
        adaptive_instructions=instructions,
    )
