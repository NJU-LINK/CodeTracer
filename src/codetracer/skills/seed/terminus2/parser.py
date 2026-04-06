"""Terminus2 parser: agent-logs/episode-N/response.txt + prompt.txt."""

from __future__ import annotations

import json
from pathlib import Path

from codetracer.models import FileRef, NormalizedTrajectory, StepRecord


class Terminus2Parser:
    format_id = "terminus2"

    def can_parse(self, run_dir: Path) -> bool:
        agent_logs = run_dir / "agent-logs"
        if not agent_logs.is_dir():
            return False
        for ep in agent_logs.iterdir():
            if ep.is_dir() and ep.name.startswith("episode-") and (ep / "response.txt").exists():
                return True
        return False

    def parse(self, run_dir: Path) -> NormalizedTrajectory:
        traj_dir = _resolve_trial_dir(run_dir)
        steps = _extract_steps(traj_dir, run_dir)
        return NormalizedTrajectory(
            steps=steps,
            task_description=_read_task(run_dir),
            metadata={"format": self.format_id, "run_dir": str(run_dir)},
        )


def _resolve_trial_dir(run_dir: Path) -> Path:
    if not (run_dir / "agent-logs").exists():
        trial_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "results.json").exists())
        if len(trial_dirs) == 1:
            return trial_dirs[0]
    return run_dir


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _file_ref(path: Path, base: Path) -> FileRef | None:
    if not path.exists():
        return None
    content = _safe_read(path)
    lines = content.splitlines()
    rel = str(path.relative_to(base)) if base in path.parents else str(path)
    return FileRef(path=rel, line_start=1, line_end=max(1, len(lines)), content=content)


def _extract_action_text(response_txt: str) -> str:
    try:
        obj = json.loads(response_txt)
        if isinstance(obj, dict):
            cmds = obj.get("commands") or []
            if isinstance(cmds, list) and cmds:
                parts = []
                for c in cmds:
                    if isinstance(c, dict):
                        parts.append(c.get("keystrokes", ""))
                    elif isinstance(c, str):
                        parts.append(c)
                return "\n".join(p for p in parts if p)
            analysis = obj.get("analysis") or obj.get("state_analysis") or ""
            plan = obj.get("plan") or obj.get("explanation") or ""
            return "\n".join(x for x in [analysis, plan] if x)
    except Exception:
        pass
    return response_txt.strip()


def _extract_observation_text(prompt_txt: str) -> str:
    marker = "New Terminal Output:"
    idx = prompt_txt.find(marker)
    if idx != -1:
        return prompt_txt[idx + len(marker) :].strip()
    return prompt_txt.strip()


def _extract_steps(traj_dir: Path, traj_root: Path) -> list[StepRecord]:
    logs = traj_dir / "agent-logs"
    if not logs.is_dir():
        return []

    ep_nums = []
    for p in logs.iterdir():
        if p.is_dir() and p.name.startswith("episode-"):
            try:
                ep_nums.append(int(p.name.split("-", 1)[1]))
            except ValueError:
                pass
    if not ep_nums:
        return []

    max_ep = max(ep_nums)
    steps: list[StepRecord] = []
    for step_id in range(1, max_ep + 2):
        a_path = logs / f"episode-{step_id - 1}" / "response.txt"
        o_path = logs / f"episode-{step_id}" / "prompt.txt"
        if not a_path.exists() and not o_path.exists():
            continue
        a_ref = _file_ref(a_path, traj_root)
        o_ref = _file_ref(o_path, traj_root)
        action = _extract_action_text(_safe_read(a_path)) if a_path.exists() else ""
        observation = _extract_observation_text(_safe_read(o_path)) if o_path.exists() else None
        steps.append(StepRecord(step_id=step_id, action=action, observation=observation, action_ref=a_ref, observation_ref=o_ref))
    return steps


def _read_task(run_dir: Path) -> str:
    results = run_dir / "results.json"
    if results.exists():
        try:
            return json.loads(_safe_read(results)).get("instruction", "")
        except Exception:
            pass
    return ""


parser = Terminus2Parser()
