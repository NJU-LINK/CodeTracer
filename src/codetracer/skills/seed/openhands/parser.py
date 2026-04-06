"""OpenHands parser: sessions/sessions/*/events/*.json (sharded) or sessions/*.json (flat)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from codetracer.models import FileRef, NormalizedTrajectory, StepRecord


class OpenHandsParser:
    format_id = "openhands"

    def can_parse(self, run_dir: Path) -> bool:
        if (run_dir / "sessions" / "sessions").is_dir():
            return True
        sessions_dir = run_dir / "sessions"
        return sessions_dir.is_dir() and any(p.suffix == ".json" for p in sessions_dir.iterdir() if p.is_file())

    def parse(self, run_dir: Path) -> NormalizedTrajectory:
        traj_dir = _resolve_trial_dir(run_dir)
        steps = _extract_steps(traj_dir, run_dir)
        return NormalizedTrajectory(
            steps=steps,
            task_description=_read_task(run_dir),
            metadata={"format": self.format_id, "run_dir": str(run_dir)},
        )


def _resolve_trial_dir(run_dir: Path) -> Path:
    if not (run_dir / "sessions" / "sessions").exists():
        trial_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "results.json").exists())
        if len(trial_dirs) == 1:
            return trial_dirs[0]
    return run_dir


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _file_ref(path: Path, base: Path) -> FileRef:
    content = _safe_read(path)
    lines = content.splitlines()
    rel = str(path.relative_to(base)) if base in path.parents else str(path)
    return FileRef(path=rel, line_start=1, line_end=max(1, len(lines)), content=content)


def _extract_cmd(ev: dict[str, Any]) -> str | None:
    if ev.get("action") == "run_ipython":
        args = ev.get("args") or {}
        code = (isinstance(args, dict) and args.get("code") or "").strip()
        return code or None

    tcm = ev.get("tool_call_metadata") or {}
    if isinstance(tcm, dict):
        args = tcm.get("args") or {}
        if isinstance(args, dict):
            cmd = (args.get("command") or "").strip()
            if cmd:
                return cmd
        mr = tcm.get("model_response") or {}
        if isinstance(mr, dict):
            for choice in mr.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message") or {}
                for tc in isinstance(msg, dict) and msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    arg_str = isinstance(fn, dict) and fn.get("arguments") or ""
                    if not isinstance(arg_str, str):
                        continue
                    try:
                        arg_obj = json.loads(arg_str)
                        cmd = (isinstance(arg_obj, dict) and arg_obj.get("command") or "").strip()
                        if cmd:
                            return cmd
                    except Exception:
                        pass

    msg_txt = ev.get("message") or ""
    if isinstance(msg_txt, str) and msg_txt.startswith("Running command:"):
        cmd = msg_txt[len("Running command:") :].strip()
        return cmd or None

    return None


def _load_events_sharded(sessions_sessions: Path) -> list[dict[str, Any]]:
    session_dirs = sorted(p for p in sessions_sessions.iterdir() if p.is_dir())
    if not session_dirs:
        return []
    events_dir = session_dirs[0] / "events"
    if not events_dir.exists():
        cache_files = sorted(sessions_sessions.glob("*/event_cache/*.json"))
        if not cache_files:
            return []

        def shard_key(p: Path) -> tuple:
            m = re.match(r"^(\d+)-(\d+)\.json$", p.name)
            return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9)

        evs: list[dict[str, Any]] = []
        for sf in sorted(cache_files, key=shard_key):
            try:
                data = json.loads(sf.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if isinstance(data, list):
                evs.extend(e for e in data if isinstance(e, dict))
        evs.sort(key=lambda e: int(e["id"]) if isinstance(e.get("id"), int) else 10**9)
        return evs

    evs: list[dict[str, Any]] = []
    for p in events_dir.iterdir():
        if not (p.is_file() and p.suffix == ".json"):
            continue
        try:
            obj = json.loads(_safe_read(p))
            if isinstance(obj, dict):
                evs.append(obj)
        except Exception:
            pass
    evs.sort(key=lambda e: int(e["id"]) if isinstance(e.get("id"), int) else 10**9)
    return evs


def _load_events_flat(sessions_dir: Path) -> list[dict[str, Any]]:
    json_files = [p for p in sessions_dir.iterdir() if p.is_file() and p.suffix == ".json"]
    if not json_files:
        return []
    main_file = max(json_files, key=lambda p: p.stat().st_size)
    try:
        events = json.loads(main_file.read_text(encoding="utf-8"))
        return events if isinstance(events, list) else []
    except Exception:
        return []


def _extract_steps(traj_dir: Path, traj_root: Path) -> list[StepRecord]:
    sessions_sessions = traj_dir / "sessions" / "sessions"
    if sessions_sessions.is_dir():
        events = _load_events_sharded(sessions_sessions)
    else:
        events = _load_events_flat(traj_dir / "sessions")

    actions: dict[int, dict[str, Any]] = {}
    observations: dict[int, dict[str, Any]] = {}
    events_dir = traj_dir / "sessions" / "sessions"

    for ev in events:
        eid = ev.get("id")
        if not isinstance(eid, int) or eid == 0:
            continue
        if ev.get("action") in ("run", "run_ipython") and _extract_cmd(ev) is not None:
            actions[eid] = ev
        cause = ev.get("cause")
        if isinstance(cause, int) and "observation" in ev:
            observations.setdefault(cause, ev)

    out: list[StepRecord] = []
    for i, action_id in enumerate(sorted(actions)):
        step_id = i + 1
        a_ev = actions[action_id]
        o_ev = observations.get(action_id)
        action = _extract_cmd(a_ev) or ""
        observation = (o_ev.get("content") or "") if o_ev else None

        a_path = events_dir / "sessions" / "events" / f"{action_id}.json" if events_dir.is_dir() else None
        o_path = (
            events_dir / "sessions" / "events" / f"{observations.get(action_id, {}).get('id', '')}.json"
            if o_ev
            else None
        )
        a_ref = (
            _file_ref(a_path, traj_root)
            if a_path and a_path.exists()
            else FileRef(path="<memory>", line_start=1, line_end=1, content=json.dumps(a_ev, ensure_ascii=False))
        )
        o_ref = (
            _file_ref(o_path, traj_root)
            if o_path and o_path.exists()
            else (
                FileRef(path="<memory>", line_start=1, line_end=1, content=json.dumps(o_ev, ensure_ascii=False))
                if o_ev
                else None
            )
        )

        out.append(
            StepRecord(step_id=step_id, action=action, observation=observation, action_ref=a_ref, observation_ref=o_ref)
        )
    return out


def _read_task(run_dir: Path) -> str:
    results = run_dir / "results.json"
    if results.exists():
        try:
            return json.loads(_safe_read(results)).get("instruction", "")
        except Exception:
            pass
    return ""


parser = OpenHandsParser()
