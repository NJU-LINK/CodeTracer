#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

INJECTED_DIR_RE = re.compile(
    r"(?:^|/|-)injected|negonly|gtonly|mixed-gt2|gt-?\d+pct|partial-steps|partial-injected",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Counters:
    deleted_injected_dirs: int = 0
    moved_files: int = 0
    moved_dirs: int = 0
    renamed_files: int = 0
    skipped_existing: int = 0
    case_dirs_processed: int = 0


def _print(s: str) -> None:
    try:
        print(s)
    except BrokenPipeError:
        # Allow piping to tools like head without crashing.
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)


def _is_case_root(d: Path) -> bool:
    steps = d / "steps.json"
    if not steps.exists() or steps.is_symlink():
        return False
    return (d / "tree.md").exists() and (d / "stage_ranges.json").exists()


def _safe_mkdir(d: Path, dry_run: bool) -> None:
    if dry_run:
        return
    d.mkdir(parents=True, exist_ok=True)


def _move(src: Path, dst: Path, dry_run: bool, counters: dict[str, int]) -> None:
    if not src.exists():
        return
    if dst.exists():
        counters["skipped_existing"] += 1
        return
    _safe_mkdir(dst.parent, dry_run)
    if dry_run:
        _print(f"MOVE {src} -> {dst}")
        counters["moved_files" if src.is_file() or src.is_symlink() else "moved_dirs"] += 1
        return
    src.rename(dst)
    counters["moved_files" if dst.is_file() or dst.is_symlink() else "moved_dirs"] += 1


def _rename(src: Path, new_name: str, dry_run: bool, counters: dict[str, int]) -> None:
    if not src.exists():
        return
    dst = src.with_name(new_name)
    if dst.exists():
        counters["skipped_existing"] += 1
        return
    if dry_run:
        _print(f"RENAME {src} -> {dst}")
        counters["renamed_files"] += 1
        return
    src.rename(dst)
    counters["renamed_files"] += 1


def delete_injected_dirs(root: Path, dry_run: bool, counters: dict[str, int]) -> None:
    # Delete deepest first to avoid walk issues.
    injected = sorted(
        [p for p in root.rglob("*") if p.is_dir() and INJECTED_DIR_RE.search(str(p.relative_to(root)))],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in injected:
        # Only delete obvious injected result dirs (usually contain mini_tracer_labels.json)
        if not (d / "mini_tracer_labels.json").exists():
            continue
        if dry_run:
            _print(f"DELETE_DIR {d}")
            counters["deleted_injected_dirs"] += 1
            continue
        shutil.rmtree(d, ignore_errors=True)
        counters["deleted_injected_dirs"] += 1


def _ensure_model_dir(case_dir: Path, model: str, dry_run: bool) -> Path:
    d = case_dir / model
    _safe_mkdir(d, dry_run)
    return d


def delete_generated_dirs(root: Path, dry_run: bool) -> None:
    # Remove previously generated symlink-based views.
    for d in sorted([p for p in root.rglob("by_model") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        if dry_run:
            _print(f"DELETE_DIR {d}")
            continue
        shutil.rmtree(d, ignore_errors=True)


def reorganize_case_dir(case_dir: Path, dry_run: bool, counters: dict[str, int]) -> None:
    # 1) GPT artifacts at case root -> gpt-5/ (only if we detect any GPT artifacts)
    has_gpt_logs = (case_dir / "_tracer_logs__gpt-5").is_dir()
    has_root_canonical = (case_dir / "mini_tracer.traj.json").exists() or (case_dir / "mini_tracer_labels.json").exists()
    has_root_misnamed = (case_dir / "mini_tracer__claude-sonnet-4-20250514-thinking.traj.json").exists() or (
        case_dir / "mini_tracer_labels__claude-sonnet-4-20250514-thinking.json"
    ).exists()

    gpt_dir: Path | None = None
    if has_gpt_logs or has_root_canonical or has_root_misnamed:
        gpt_dir = _ensure_model_dir(case_dir, "gpt-5", dry_run)

    # Prefer: if root has gpt logs, move them into gpt-5/
    if gpt_dir is not None:
        for logs in (
            case_dir / "_tracer_logs__gpt-5",
            case_dir / "_tracer_logs__gpt5",
            case_dir / "_tracer_logs__gpt-5.0",
        ):
            if logs.is_dir():
                _move(logs, gpt_dir / logs.name, dry_run, counters)

    # Root canonical filenames
    root_traj = case_dir / "mini_tracer.traj.json"
    root_lab = case_dir / "mini_tracer_labels.json"
    if gpt_dir is not None:
        if root_traj.exists():
            _move(root_traj, gpt_dir / "mini_tracer.traj.json", dry_run, counters)
        if root_lab.exists():
            _move(root_lab, gpt_dir / "mini_tracer_labels.json", dry_run, counters)

    # Root misnamed GPT files (use the names you referenced)
    mis_traj = case_dir / "mini_tracer__claude-sonnet-4-20250514-thinking.traj.json"
    mis_lab = case_dir / "mini_tracer_labels__claude-sonnet-4-20250514-thinking.json"
    # Treat these as GPT only when gpt logs exist (matches your "broken-python" rule)
    if gpt_dir is not None and has_gpt_logs:
        if mis_traj.exists():
            _move(mis_traj, gpt_dir / "mini_tracer.traj.json", dry_run, counters)
        if mis_lab.exists():
            _move(mis_lab, gpt_dir / "mini_tracer_labels.json", dry_run, counters)

    # 2) For each model subdir (claude/deepseek/etc): standardize filenames and move root logs if any
    for sub in [p for p in case_dir.iterdir() if p.is_dir()]:
        name = sub.name
        if name in {"by_model"}:
            continue
        if name == "gpt-5":
            continue
        if INJECTED_DIR_RE.search(name):
            continue

        # treat as model dir only if it looks like one
        if not ((sub / "mini_tracer.traj.json").exists() or any(sub.glob("_tracer_logs__*/run_meta.json"))):
            continue

        # Move logs from root into model dir if present
        root_logs = case_dir / f"_tracer_logs__{name}"
        if root_logs.is_dir():
            _move(root_logs, sub / root_logs.name, dry_run, counters)

        # Normalize within model dir: if label is missing but there is a misnamed label file in subdir, rename it.
        for p in sub.glob("mini_tracer_labels__*.json"):
            _rename(p, "mini_tracer_labels.json", dry_run, counters)
        for p in sub.glob("mini_tracer__*.traj.json"):
            _rename(p, "mini_tracer.traj.json", dry_run, counters)

    # 3) If there are model-suffixed artifacts at root (not GPT), move them into their model folder.
    for p in list(case_dir.glob("mini_tracer__*.traj.json")):
        # Skip the special GPT-misnamed file unless we are *not* treating it as GPT.
        if p.name == "mini_tracer__claude-sonnet-4-20250514-thinking.traj.json" and has_gpt_logs:
            continue
        model = p.name.removeprefix("mini_tracer__").removesuffix(".traj.json")
        mdir = _ensure_model_dir(case_dir, model, dry_run)
        _move(p, mdir / "mini_tracer.traj.json", dry_run, counters)

    for p in list(case_dir.glob("mini_tracer_labels__*.json")):
        if p.name == "mini_tracer_labels__claude-sonnet-4-20250514-thinking.json" and has_gpt_logs:
            continue
        model = p.name.removeprefix("mini_tracer_labels__").removesuffix(".json")
        mdir = _ensure_model_dir(case_dir, model, dry_run)
        _move(p, mdir / "mini_tracer_labels.json", dry_run, counters)

    # Move root tracer logs into matching model dirs if they exist.
    for logs in sorted(case_dir.glob("_tracer_logs__*")):
        if not logs.is_dir():
            continue
        lname = logs.name.removeprefix("_tracer_logs__")
        if lname in {"gpt-5", "gpt5", "gpt-5.0"}:
            continue
        mdir = case_dir / lname
        if mdir.is_dir():
            _move(logs, mdir / logs.name, dry_run, counters)

    counters["case_dirs_processed"] += 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete injected result directories and reorganize mini-tracer outputs in-place by model."
    )
    ap.add_argument(
        "--root",
        default="/data/terminalbench/agent_failure_analysis/bench/step_id_maps",
        help="Root step_id_maps directory",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without modifying the filesystem")
    ap.add_argument("--skip-delete", action="store_true", help="Skip deleting injected directories")
    ap.add_argument("--skip-reorg", action="store_true", help="Skip reorganizing case directories")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of case dirs processed (for testing)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    counters: dict[str, int] = {
        "deleted_injected_dirs": 0,
        "moved_files": 0,
        "moved_dirs": 0,
        "renamed_files": 0,
        "skipped_existing": 0,
        "case_dirs_processed": 0,
    }

    # Clean up previously generated views (symlink-based).
    delete_generated_dirs(root, args.dry_run)

    if not args.skip_delete:
        delete_injected_dirs(root, args.dry_run, counters)

    if not args.skip_reorg:
        processed = 0
        for d in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: str(p)):
            if not _is_case_root(d):
                continue
            processed += 1
            if args.limit and processed > args.limit:
                break
            reorganize_case_dir(d, args.dry_run, counters)

    print(json.dumps(counters, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

