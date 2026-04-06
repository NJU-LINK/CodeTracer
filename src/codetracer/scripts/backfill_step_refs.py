#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


TRAJ_ROOT = Path("/data/terminalbench/traj")


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _rel_to_traj_root(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(TRAJ_ROOT))
    except Exception:
        return str(p.name)


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


@dataclass(frozen=True)
class RefPair:
    action_path: str
    action_line: int
    action_text: str
    observation_path: str
    observation_start: int
    observation_end: int
    observation_text: str


def _openhands_refs(raw_dir: Path) -> list[RefPair] | None:
    ev_dirs = list(raw_dir.glob("sessions/sessions/*/events"))
    if not ev_dirs:
        return None

    ev_dir = ev_dirs[0]
    events = {}
    for p in sorted(ev_dir.glob("*.json"), key=lambda x: int(x.stem)):
        try:
            events[int(p.stem)] = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue

    obs_by_cause: dict[int, tuple[Path, dict]] = {}
    for eid, d in events.items():
        if d.get("observation") == "run" and isinstance(d.get("cause"), int):
            obs_by_cause[int(d["cause"])] = (ev_dir / f"{eid}.json", d)

    pairs: list[RefPair] = []
    for eid, d in sorted(events.items(), key=lambda kv: kv[0]):
        if d.get("action") != "run":
            continue
        args = d.get("args") or {}
        if not isinstance(args, dict):
            continue
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        obs = obs_by_cause.get(eid)
        if obs is None:
            continue
        obs_path, obs_d = obs
        obs_text = obs_d.get("content") or ""
        if not isinstance(obs_text, str):
            obs_text = ""

        pairs.append(
            RefPair(
                action_path=_rel_to_traj_root(ev_dir / f"{eid}.json"),
                action_line=1,
                action_text=cmd,
                observation_path=_rel_to_traj_root(obs_path),
                observation_start=1,
                observation_end=1,
                observation_text=obs_text,
            )
        )

    return pairs


def _commands_txt_lines(raw_dir: Path) -> list[tuple[int, str]] | None:
    p = raw_dir / "commands.txt"
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    cmds: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        try:
            v = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(v, str):
            s = v.replace("\r\n", "\n").replace("\r", "\n")
            s = s[:-1] if s.endswith("\n") else s
            if s and s != "C-d":
                cmds.append((i, s))
    return cmds


def _promptish(line: str) -> bool:
    # Example: root@host:/app# ls -la
    return bool(re.match(r"^[^\s].*#\s", line))


def _terminus_like_refs(raw_dir: Path) -> list[RefPair] | None:
    cmds = _commands_txt_lines(raw_dir)
    if not cmds:
        return None

    log_path = raw_dir / "sessions" / "agent.log"
    if not log_path.exists():
        return None

    raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean_lines = [_strip_ansi(x) for x in raw_lines]

    pairs: list[RefPair] = []
    search_pos = 0
    for cmd_line_no, cmd in cmds:
        found = -1
        for idx in range(search_pos, len(clean_lines)):
            if cmd in clean_lines[idx]:
                found = idx
                break
        if found == -1:
            # Fallback: map observation to the same command line
            pairs.append(
                RefPair(
                    action_path=_rel_to_traj_root(raw_dir / "commands.txt"),
                    action_line=cmd_line_no,
                    action_text=cmd,
                    observation_path=_rel_to_traj_root(log_path),
                    observation_start=max(1, search_pos + 1),
                    observation_end=max(1, search_pos + 1),
                    observation_text="",
                )
            )
            continue

        # Determine observation span until next prompt-like line
        obs_start = found + 2  # 1-based, next line after the command line
        obs_end = min(len(clean_lines), obs_start)
        for j in range(found + 1, len(clean_lines)):
            if _promptish(clean_lines[j]):
                obs_end = j  # 1-based end is line before prompt
                break
            obs_end = j + 1

        text = "\n".join(clean_lines[obs_start - 1 : obs_end])
        if len(text) > 4000:
            text = text[:4000]

        pairs.append(
            RefPair(
                action_path=_rel_to_traj_root(raw_dir / "commands.txt"),
                action_line=cmd_line_no,
                action_text=cmd,
                observation_path=_rel_to_traj_root(log_path),
                observation_start=obs_start,
                observation_end=obs_end,
                observation_text=text,
            )
        )
        search_pos = found + 1

    return pairs


def _pick_ref_pairs(raw_dir: Path) -> list[RefPair] | None:
    pairs = _openhands_refs(raw_dir)
    if pairs:
        return pairs
    pairs = _terminus_like_refs(raw_dir)
    if pairs:
        return pairs
    return None


def _needs_backfill(steps: list[dict]) -> bool:
    for s in steps[:10]:
        if not isinstance(s, dict):
            continue
        if s.get("action_ref") is None or s.get("observation_ref") is None:
            return True
        path = (s.get("action_ref") or {}).get("path", "")
        if isinstance(path, str) and path.startswith("/"):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backfill action_ref/observation_ref for terminal-bench outputs using raw trajectories under /data/terminalbench/traj."
    )
    ap.add_argument(
        "--manifest",
        default="/data/terminalbench/agent_failure_analysis/bench/bench_manifest.verified.jsonl",
        help="bench manifest JSONL",
    )
    ap.add_argument(
        "--output-root",
        default="/data/terminalbench/outputs/codetracer_verified",
        help="output root containing <traj_id>/steps.json",
    )
    ap.add_argument("--limit", type=int, default=0, help="optional limit for how many terminal trajectories to process")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write files")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    output_root = Path(args.output_root)

    changed = 0
    scanned = 0
    skipped_swe = 0
    missing_output = 0
    missing_raw = 0
    unsupported = 0

    with manifest.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ann_rel = entry.get("annotation_relpath", "")
            if not ann_rel.startswith("agent_failure_analysis"):
                skipped_swe += 1
                continue

            traj_id = entry["traj_id"]
            source_rel = entry.get("source_relpath") or ""
            if not source_rel:
                missing_raw += 1
                continue

            out_dir = output_root / traj_id
            steps_json_path = out_dir / "steps.json"
            if not steps_json_path.exists():
                missing_output += 1
                continue

            raw_dir = (TRAJ_ROOT / source_rel).resolve()
            if not raw_dir.exists():
                missing_raw += 1
                continue

            scanned += 1
            if args.limit and scanned > args.limit:
                break

            steps = json.loads(steps_json_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(steps, list) or not steps:
                continue
            if not _needs_backfill(steps):
                continue

            ref_pairs = _pick_ref_pairs(raw_dir)
            if not ref_pairs:
                unsupported += 1
                continue

            # Step indices in these manifests are 1-based.
            for i, s in enumerate(steps, start=1):
                if not isinstance(s, dict):
                    continue
                if i > len(ref_pairs):
                    break
                rp = ref_pairs[i - 1]
                s["action_ref"] = {
                    "path": rp.action_path,
                    "line_start": rp.action_line,
                    "line_end": rp.action_line,
                    "content": rp.action_text,
                }
                s["observation_ref"] = {
                    "path": rp.observation_path,
                    "line_start": rp.observation_start,
                    "line_end": rp.observation_end,
                    "content": rp.observation_text,
                }

            if args.dry_run:
                print(f"WOULD_WRITE {steps_json_path} (raw={raw_dir})")
            else:
                steps_json_path.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"WROTE {steps_json_path} (raw={raw_dir})")
            changed += 1

    print(
        json.dumps(
            {
                "terminal_scanned_with_steps_json": scanned,
                "changed": changed,
                "skipped_non_terminal_entries": skipped_swe,
                "missing_output_steps_json": missing_output,
                "missing_raw_dir_or_source_relpath": missing_raw,
                "unsupported_raw_format": unsupported,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

