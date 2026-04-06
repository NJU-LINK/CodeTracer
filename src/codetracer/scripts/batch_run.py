"""Batch run entry point for CodeTracer.

Usage:
    python -m codetracer.scripts.batch_run --manifest MANIFEST --output OUTPUT [OPTIONS]

Runs ``codetracer analyze`` in parallel across a JSONL manifest of trajectories.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_PREFIX_MAP = {
    "merged_cleaned_step25": "/data/terminalbench/merged_cleaned_step25(1)",
    "merged_cleaned_step20_three_waves": "/data/terminalbench/merged_cleaned_step20_three_waves(1)",
    "agent_failure_analysis": "/data/terminalbench",
}


def _resolve_ann_path(ann_rel: str) -> str | None:
    for prefix, base in _PREFIX_MAP.items():
        if ann_rel.startswith(prefix):
            return os.path.join(base, ann_rel)
    return None


def _run_one(
    item: dict[str, Any],
    codetracer_bin: str,
    model: str,
    api_base: str,
    api_key: str,
    cost_limit: float,
    output_root: str,
    tasks_root: str,
    dry_run: bool,
) -> tuple[str, bool, str]:
    traj_id = item["traj_id"]
    run_dir = item["run_dir"]
    bench_type = item.get("bench_type", "terminal-bench")

    item_path = Path(output_root) / "_work" / f"{traj_id}.json"
    item_path.parent.mkdir(parents=True, exist_ok=True)
    item_path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")

    cmd = [
        codetracer_bin, "analyze", run_dir,
        "--model", model,
        "--api-base", api_base,
        "--api-key", api_key,
        "--cost-limit", str(cost_limit),
        "--output-dir", output_root,
        "--traj-id", traj_id,
        "--annotation-tree",
        "--traj-annotation-path", str(item_path),
        "--skip-discovery",
        "--skip-sandbox",
    ]
    if bench_type == "terminal-bench":
        cmd += ["--tasks-root", tasks_root]
    else:
        cmd += ["--task-dir", run_dir]
    if dry_run:
        cmd += ["--dry-run"]

    log_path = Path(output_root) / "_logs" / f"{traj_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    return traj_id, ok, str(log_path)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="CodeTracer batch run")
    p.add_argument("--manifest", required=True, help="JSONL manifest path")
    p.add_argument("--output", required=True, help="Output root directory")
    p.add_argument("--model", default=os.environ.get("CODETRACER_MODEL", "claude-sonnet-4-20250514"))
    p.add_argument("--api-base", default=os.environ.get("CODETRACER_API_BASE", ""))
    p.add_argument("--api-key", default=os.environ.get("CODETRACER_API_KEY", ""))
    p.add_argument("--tasks-root", default="/data/terminalbench/terminal-bench/tasks")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--cost-limit", type=float, default=3.0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    codetracer_bin = os.environ.get("CODETRACER_BIN", "codetracer")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    skipped = 0
    missing = 0

    with open(args.manifest) as f:
        for line in f:
            entry = json.loads(line.strip())
            traj_id = entry["traj_id"]
            ann_rel = entry["annotation_relpath"]
            run_dir = _resolve_ann_path(ann_rel)
            if run_dir is None or not os.path.isdir(run_dir):
                missing += 1
                continue
            if args.resume:
                traj_path = os.path.join(args.output, traj_id, "codetracer_labels.traj.json")
                if os.path.exists(traj_path):
                    skipped += 1
                    continue
            items.append({"traj_id": traj_id, "run_dir": run_dir, "annotation": entry})

    print(f"Work items: {len(items)} to process, {skipped} already done, {missing} missing")
    if not items:
        print("Nothing to do.")
        return

    ok_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                _run_one, item, codetracer_bin, args.model, args.api_base,
                args.api_key, args.cost_limit, args.output, args.tasks_root, args.dry_run,
            ): item["traj_id"]
            for item in items
        }
        for future in as_completed(futures):
            traj_id, ok, log_path = future.result()
            if ok:
                ok_count += 1
                print(f"[OK]   {traj_id}")
            else:
                fail_count += 1
                print(f"[FAIL] {traj_id} -- see {log_path}")

    print(f"\nDone. OK: {ok_count}, FAIL: {fail_count}")


if __name__ == "__main__":
    main()
