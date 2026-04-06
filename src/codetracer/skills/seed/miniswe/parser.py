"""MiniSWE parser: agent-logs/mini.traj.json, sessions/agent.log, or commands.txt fallback."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from codetracer.models import FileRef, NormalizedTrajectory, StepRecord


class MinisweParser:
    format_id = "miniswe"

    def can_parse(self, run_dir: Path) -> bool:
        return (
            (run_dir / "agent-logs" / "mini.traj.json").exists()
            or (run_dir / "sessions" / "agent.log").exists()
            or (run_dir / "commands.txt").exists()
        )

    def parse(self, run_dir: Path) -> NormalizedTrajectory:
        steps = _extract_steps(run_dir, run_dir)
        return NormalizedTrajectory(
            steps=steps,
            task_description=_read_task(run_dir),
            metadata={"format": self.format_id, "run_dir": str(run_dir)},
        )


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _file_ref(path: Path, base: Path, line_start: int, line_end: int, lines: list[str]) -> FileRef:
    rel = str(path.relative_to(base)) if base in path.parents else str(path)
    content = "\n".join(lines[line_start - 1 : line_end])
    return FileRef(path=rel, line_start=line_start, line_end=line_end, content=content)


def _find_message_range(lines: list[str], ts_val: Any) -> tuple[int, int] | None:
    needle = f'"timestamp": {ts_val}'
    hit = None
    for i, line in enumerate(lines):
        if needle in line:
            hit = i
            break
    if hit is None:
        return None
    start = hit
    while start > 0 and not lines[start].lstrip().startswith("{"):
        start -= 1
    end = hit
    while end < len(lines) - 1:
        if lines[end].rstrip().endswith("},") or lines[end].rstrip().endswith("}"):
            break
        end += 1
    return start + 1, end + 1


def _steps_from_traj_json(path: Path, traj_root: Path) -> list[StepRecord]:
    txt = _safe_read(path)
    lines = txt.splitlines()
    try:
        data = json.loads(txt)
    except Exception:
        return []
    msgs = data.get("messages")
    if not isinstance(msgs, list):
        return []

    action_idxs = [
        i
        for i, m in enumerate(msgs)
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str) and "```bash" in m["content"]
    ]
    rel = str(path.relative_to(traj_root)) if traj_root in path.parents else str(path)

    out: list[StepRecord] = []
    for step_i, idx in enumerate(action_idxs):
        step_id = step_i + 1
        m = msgs[idx] if isinstance(msgs[idx], dict) else {}
        a_rng = _find_message_range(lines, m.get("timestamp"))
        a_ref = None
        if a_rng:
            ls, le = a_rng
            a_ref = FileRef(path=rel, line_start=ls, line_end=le, content="\n".join(lines[ls - 1 : le]))

        action = _extract_bash_from_content(m.get("content", ""))
        o_ref = None
        observation = None
        for m2 in msgs[idx + 1 :]:
            if not isinstance(m2, dict) or m2.get("role") != "user":
                continue
            c2 = m2.get("content") or ""
            if not isinstance(c2, str) or "<returncode>" not in c2:
                continue
            o_rng = _find_message_range(lines, m2.get("timestamp"))
            if o_rng:
                ls2, le2 = o_rng
                o_ref = FileRef(path=rel, line_start=ls2, line_end=le2, content="\n".join(lines[ls2 - 1 : le2]))
            observation = c2
            break

        out.append(
            StepRecord(step_id=step_id, action=action, observation=observation, action_ref=a_ref, observation_ref=o_ref)
        )
    return out


def _extract_bash_from_content(content: str) -> str:
    m = re.search(r"```bash\s*\n(.*?)\n?```", content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


def _steps_from_agent_log(log_path: Path, traj_root: Path) -> list[StepRecord]:
    txt = _safe_read(log_path)
    lines = txt.splitlines()
    rel = str(log_path.relative_to(traj_root)) if traj_root in log_path.parents else str(log_path)

    def ref_slice(ls: int, le: int) -> FileRef:
        return FileRef(path=rel, line_start=ls, line_end=le, content="\n".join(lines[ls - 1 : le]))

    def find_fence_end(fs: int) -> int | None:
        for j in range(fs + 1, len(lines)):
            if lines[j].strip() == "```":
                return j
        return None

    last_block: tuple[int, int] | None = None
    last_block_closed_at: int | None = None
    last_block_used = True

    out: list[StepRecord] = []
    for idx, line in enumerate(lines):
        if line.strip() == "```bash":
            fe = find_fence_end(idx)
            if fe is None:
                continue
            start = idx
            for b in range(max(0, idx - 6), idx):
                if lines[b].lstrip().startswith("THOUGHT:"):
                    start = b
                    break
            last_block = (start, fe)
            last_block_closed_at = fe
            last_block_used = False
            continue

        if "<returncode>" not in line:
            continue
        if last_block is None or last_block_used or last_block_closed_at is None:
            continue
        if idx - last_block_closed_at > 250:
            last_block_used = True
            continue

        obs_end = idx
        for k in range(idx, len(lines)):
            if "</output>" in lines[k]:
                obs_end = k
                break
            if k > idx and lines[k].strip() == "```bash":
                obs_end = k - 1
                break

        bs, be = last_block
        action_cmd = "\n".join(lines[bs : be + 1])
        a_ref = ref_slice(bs + 1, be + 1)
        o_ref = ref_slice(idx + 1, obs_end + 1)
        observation = "\n".join(lines[idx : obs_end + 1])
        out.append(
            StepRecord(
                step_id=len(out) + 1, action=action_cmd, observation=observation, action_ref=a_ref, observation_ref=o_ref
            )
        )
        last_block_used = True

    return out


def _steps_from_commands_txt(cmd_path: Path, traj_root: Path) -> list[StepRecord]:
    rel = str(cmd_path.relative_to(traj_root)) if traj_root in cmd_path.parents else str(cmd_path)
    lines = _safe_read(cmd_path).splitlines()
    out: list[StepRecord] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        try:
            obj = ast.literal_eval(s)
        except Exception:
            continue
        if not isinstance(obj, list) or not obj or obj[-1] != "Enter":
            continue
        parts = [str(x) for x in obj[:-1] if isinstance(x, (str, int, float))]
        cmd = "".join(parts).strip()
        if not cmd:
            continue
        ref = FileRef(path=rel, line_start=i + 1, line_end=i + 1, content=line)
        out.append(StepRecord(step_id=len(out) + 1, action=cmd, observation=None, action_ref=ref, observation_ref=None))
    return out


def _extract_steps(traj_dir: Path, traj_root: Path) -> list[StepRecord]:
    traj_json = traj_dir / "agent-logs" / "mini.traj.json"
    if traj_json.exists():
        steps = _steps_from_traj_json(traj_json, traj_root)
        if steps:
            return steps

    agent_log = traj_dir / "sessions" / "agent.log"
    if agent_log.exists():
        steps = _steps_from_agent_log(agent_log, traj_root)
        if steps:
            return steps

    cmd_txt = traj_dir / "commands.txt"
    return _steps_from_commands_txt(cmd_txt, traj_root)


def _read_task(run_dir: Path) -> str:
    results = run_dir / "results.json"
    if results.exists():
        try:
            return json.loads(_safe_read(results)).get("instruction", "")
        except Exception:
            pass
    return ""


parser = MinisweParser()
