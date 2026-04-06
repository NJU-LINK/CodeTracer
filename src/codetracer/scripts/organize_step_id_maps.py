#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

INJECTED_DIR_RE = re.compile(r"(?:^|-)injected|negonly|gtonly|gt-\\d+pct|mixed-gt2", re.IGNORECASE)


@dataclass(frozen=True)
class ModelArtifacts:
    model: str
    traj_path: Path | None
    label_path: Path | None
    logs_dir: Path | None


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rel_symlink(target: Path, source: Path, dry_run: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    rel = os.path.relpath(str(source), start=str(target.parent))
    if dry_run:
        print(f"LINK {target} -> {rel}")
        return
    target.symlink_to(rel)


def _copy_json(target: Path, source: Path, dry_run: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if dry_run:
        print(f"COPY {target} <- {source}")
        return
    target.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def _find_logs_dir(case_dir: Path, model: str) -> Path | None:
    # Logs can be at root: _tracer_logs__<model>/
    root_logs = case_dir / f"_tracer_logs__{model}"
    if root_logs.is_dir():
        return root_logs
    # Or inside model subdir: <model>/_tracer_logs__<model>/
    sub_logs = case_dir / model / f"_tracer_logs__{model}"
    if sub_logs.is_dir():
        return sub_logs
    return None


def _is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if INJECTED_DIR_RE.search(p.name):
        return False
    # Heuristic: model dirs commonly contain mini_tracer.traj.json or tracer logs
    if (p / "mini_tracer.traj.json").exists():
        return True
    if (p / f"_tracer_logs__{p.name}" / "run_meta.json").exists():
        return True
    return False


def _infer_model_from_run_meta(run_meta: Path) -> str | None:
    try:
        data = json.loads(run_meta.read_text(encoding="utf-8", errors="replace"))
        m = data.get("model")
        return m if isinstance(m, str) and m else None
    except Exception:
        return None


def _detect_gpt_root_artifacts(case_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    GPT artifacts for some cases live at the case root.

    Patterns observed:
    - mini_tracer.traj.json + mini_tracer_labels.json (treat as GPT)
    - mini_tracer__claude-sonnet-4-20250514-thinking.traj.json + mini_tracer_labels__claude-sonnet-4-20250514-thinking.json
      with a root _tracer_logs__gpt-5 directory (treat as GPT-5)
    """
    # Preferred: if gpt-5 logs exist, use the suffix files (even if misnamed)
    gpt_logs = case_dir / "_tracer_logs__gpt-5"
    if gpt_logs.is_dir():
        traj = case_dir / "mini_tracer__claude-sonnet-4-20250514-thinking.traj.json"
        lab = case_dir / "mini_tracer_labels__claude-sonnet-4-20250514-thinking.json"
        if traj.exists() or lab.exists():
            return (traj if traj.exists() else None, lab if lab.exists() else None, gpt_logs)

    # Fallback: root mini_tracer.* files
    traj = case_dir / "mini_tracer.traj.json"
    lab = case_dir / "mini_tracer_labels.json"
    if traj.exists() or lab.exists():
        return (traj if traj.exists() else None, lab if lab.exists() else None, None)

    return (None, None, None)


def _collect_case_models(case_dir: Path) -> list[ModelArtifacts]:
    models: list[ModelArtifacts] = []

    # Model subdirs
    for sub in sorted([p for p in case_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not _is_model_dir(sub):
            continue
        model = sub.name
        traj = sub / "mini_tracer.traj.json"
        lab = sub / "mini_tracer_labels.json"
        logs = _find_logs_dir(case_dir, model)
        models.append(
            ModelArtifacts(
                model=model,
                traj_path=traj if traj.exists() else None,
                label_path=lab if lab.exists() else None,
                logs_dir=logs,
            )
        )

    # GPT at root (special handling)
    g_traj, g_lab, g_logs = _detect_gpt_root_artifacts(case_dir)
    if g_traj or g_lab or g_logs:
        models.append(ModelArtifacts(model="gpt-5", traj_path=g_traj, label_path=g_lab, logs_dir=g_logs))

    # De-dup by model name (prefer entries with a traj)
    dedup: dict[str, ModelArtifacts] = {}
    for m in models:
        cur = dedup.get(m.model)
        if cur is None:
            dedup[m.model] = m
            continue
        if cur.traj_path is None and m.traj_path is not None:
            dedup[m.model] = m
            continue
        if cur.label_path is None and m.label_path is not None:
            dedup[m.model] = m
            continue

    return [dedup[k] for k in sorted(dedup)]


def _is_case_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    steps = d / "steps.json"
    # Model subfolders often symlink steps.json/task.md/etc back to the case root.
    # Only treat directories with a real steps.json as case roots.
    if not steps.exists() or steps.is_symlink():
        return False
    return (d / "tree.md").exists() and (d / "stage_ranges.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description="Organize step_id_maps mini-tracer outputs into by_model/<model>/{traj,labels}.")
    ap.add_argument("--root", required=True, help="Root directory to scan (e.g. .../step_id_maps/miniswe/...)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    ap.add_argument("--copy", action="store_true", help="Copy JSON files instead of symlinks")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of case dirs processed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    processed = 0
    changed = 0
    skipped = 0

    for d in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: str(p)):
        if not _is_case_dir(d):
            continue
        processed += 1
        if args.limit and processed > args.limit:
            break

        models = _collect_case_models(d)
        if not models:
            skipped += 1
            continue

        out_base = d / "by_model"
        if not args.dry_run:
            _safe_mkdir(out_base)

        for m in models:
            out_dir = out_base / m.model
            if not args.dry_run:
                _safe_mkdir(out_dir)

            if m.traj_path and m.traj_path.exists():
                if args.copy:
                    _copy_json(out_dir / "traj.json", m.traj_path, args.dry_run)
                else:
                    _rel_symlink(out_dir / "traj.json", m.traj_path, args.dry_run)
                changed += 1

            if m.label_path and m.label_path.exists():
                if args.copy:
                    _copy_json(out_dir / "labels.json", m.label_path, args.dry_run)
                else:
                    _rel_symlink(out_dir / "labels.json", m.label_path, args.dry_run)
                changed += 1

            if m.logs_dir and m.logs_dir.exists():
                logs_out = out_dir / "logs"
                if args.copy:
                    # shallow: link/copy run_meta.json and tracer stdout/stderr if present
                    for name in ("run_meta.json", "tracer.stdout.txt", "tracer.stderr.txt"):
                        src = m.logs_dir / name
                        if src.exists():
                            _copy_json(logs_out / name, src, args.dry_run)
                else:
                    _rel_symlink(logs_out, m.logs_dir, args.dry_run)

        # no destructive operations; injected dirs are left untouched

    print(
        json.dumps(
            {"processed_case_dirs": processed, "changed_actions": changed, "skipped_no_models": skipped},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

