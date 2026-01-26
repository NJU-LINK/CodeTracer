from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepIdMapsInput:
    task_md_path: Path
    tree_md_path: Path
    steps_json_path: Path
    stage_ranges_json_path: Path


def load_step_id_maps_input(task_dir: Path) -> StepIdMapsInput:
    task_dir = task_dir.resolve()
    task_md_path = task_dir / "task.md"
    tree_md_path = task_dir / "tree.md"
    steps_json_path = task_dir / "steps.json"
    stage_ranges_json_path = task_dir / "stage_ranges.json"

    if not task_md_path.is_file():
        raise FileNotFoundError(f"Missing task.md in {task_dir}")
    if not tree_md_path.is_file():
        raise FileNotFoundError(f"Missing tree.md in {task_dir}")
    if not steps_json_path.is_file():
        raise FileNotFoundError(f"Missing steps.json in {task_dir}")
    if not stage_ranges_json_path.is_file():
        raise FileNotFoundError(f"Missing stage_ranges.json in {task_dir}")

    return StepIdMapsInput(
        task_md_path=task_md_path,
        tree_md_path=tree_md_path,
        steps_json_path=steps_json_path,
        stage_ranges_json_path=stage_ranges_json_path,
    )


