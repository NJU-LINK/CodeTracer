"""CodeTracer development testing tool.

Iteratively runs CodeTracer on sample trajectories and validates output quality.
Provides a feedback loop for improving analysis prompts and parsers.

Usage:
    codetracer-dev --data-dir data/traj/claude_code/claude_code_coder --model gpt-5
    codetracer-dev --validate-only --data-dir data/traj/claude_code/claude_code_coder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console(highlight=False)


def _discover_trajectories(data_dir: Path, max_samples: int, config: dict[str, Any]) -> list[Path]:
    """Find trajectory directories under *data_dir* using deep recursive discovery."""
    from codetracer.discovery.explorer import discover_trajectory_dirs

    candidates = discover_trajectory_dirs(data_dir, config)

    if max_samples > 0:
        candidates = candidates[:max_samples]
    return candidates


def _validate_existing_output(
    run_dir: Path, profile_name: str, config: dict[str, Any],
) -> dict[str, Any]:
    """Validate an existing output file without running LLM analysis."""
    from codetracer.discovery.explorer import detect_or_generate_skill
    from codetracer.llm.client import LLMClient
    from codetracer.query.normalizer import Normalizer
    from codetracer.services.validation import validate_analysis_output
    from codetracer.skills.pool import SkillPool
    from codetracer.state.output_profile import load_profile

    pool = SkillPool()
    normalizer = Normalizer(pool)
    profile = load_profile(profile_name)

    output_path = run_dir / profile.output_file
    if not output_path.exists():
        return {"valid": False, "errors": [f"No output file: {output_path}"], "warnings": [], "metrics": {}}

    llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
    llm = LLMClient(**llm_cfg)
    try:
        _, traj = detect_or_generate_skill(run_dir, normalizer, pool, llm, config)
    except ValueError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "metrics": {}}

    result = validate_analysis_output(output_path, traj, profile)
    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "metrics": result.metrics,
    }


def _run_analysis(
    run_dir: Path,
    model: str,
    api_base: str | None,
    api_key: str | None,
    profile_name: str,
    cost_limit: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run full CodeTracer analysis on a single trajectory."""
    from codetracer.agents.context import ContextAssembler
    from codetracer.agents.trace_agent import TraceAgent
    from codetracer.llm.client import LLMClient
    from codetracer.query.normalizer import Normalizer
    from codetracer.query.tree_builder import TreeBuilder
    from codetracer.services.complexity import estimate_complexity
    from codetracer.services.cost_tracker import CostTracker
    from codetracer.services.validation import validate_analysis_output
    from codetracer.skills.pool import SkillPool
    from codetracer.state.output_profile import load_profile

    llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
    llm_cfg["model_name"] = model
    if api_base:
        llm_cfg["api_base"] = api_base
    if api_key:
        llm_cfg["api_key"] = api_key
    config.setdefault("trace", {})["cost_limit"] = cost_limit

    llm = LLMClient(**llm_cfg)
    pool = SkillPool()
    normalizer = Normalizer(pool)
    profile = load_profile(profile_name, config)
    cost_tracker = CostTracker(budget_limit_usd=cost_limit)

    # 1. Normalize via unified detect-or-generate
    from codetracer.discovery.explorer import detect_or_generate_skill
    try:
        skill, traj = detect_or_generate_skill(run_dir, normalizer, pool, llm, config)
    except ValueError as exc:
        return {"valid": False, "errors": [str(exc)], "cost": 0.0}

    # 2. Build tree
    tree_path = run_dir / "tree.md"
    if not tree_path.exists():
        builder = TreeBuilder()
        tree_md = builder.build(traj)
        tree_path.write_text(tree_md, encoding="utf-8")

    # 3. Analyze
    complexity = estimate_complexity(traj)
    traj_meta = {
        "complexity_tier": complexity.complexity_tier,
        "adaptive_instructions": complexity.adaptive_instructions,
        "step_count": complexity.step_count,
        "unique_tool_types": complexity.unique_tool_types,
    }

    fmt_name = skill.name if skill else "pre_normalized"

    output_path = run_dir / profile.output_file
    assembler = ContextAssembler(config, pool)
    agent = TraceAgent(
        llm, assembler, run_dir, output_path, config,
        cost_tracker=cost_tracker,
        profile=profile,
        agent_type=fmt_name,
    )

    t0 = time.time()
    try:
        agent.run(skill, traj_metadata=traj_meta)
    except Exception as e:
        return {"valid": False, "errors": [f"Analysis failed: {e}"], "cost": cost_tracker.total_cost}
    duration = time.time() - t0

    # 4. Validate
    vr = validate_analysis_output(output_path, traj, profile)
    return {
        "valid": vr.valid,
        "errors": vr.errors,
        "warnings": vr.warnings,
        "metrics": {**vr.metrics, "duration_s": round(duration, 1), "cost_usd": round(cost_tracker.total_cost, 4)},
        "cost": cost_tracker.total_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeTracer development testing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/traj"), help="Root of trajectory data")
    parser.add_argument("--profile", default="detailed", help="Output profile to test")
    parser.add_argument("--max-samples", type=int, default=3, help="Max trajectories to test (0 = all)")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--cost-limit", type=float, default=2.0, help="Max cost per trajectory (USD)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing outputs")
    parser.add_argument("-c", "--config", type=Path, default=None, help="Config YAML override")
    args = parser.parse_args()

    if not args.data_dir.exists():
        console.print(f"[red]Data directory not found: {args.data_dir}[/red]")
        sys.exit(1)

    from codetracer.query.config import load_config
    config = load_config(args.config)

    trajs = _discover_trajectories(args.data_dir, args.max_samples, config)
    if not trajs:
        console.print(f"[yellow]No trajectories found in {args.data_dir}[/yellow]")
        sys.exit(0)

    console.print(f"Found [bold]{len(trajs)}[/bold] trajectories in {args.data_dir}")

    results: list[dict[str, Any]] = []
    total_cost = 0.0

    for i, traj_dir in enumerate(trajs, 1):
        label = traj_dir.name[:50]
        console.print(f"\n[bold cyan][{i}/{len(trajs)}][/bold cyan] {label}")

        if args.validate_only:
            r = _validate_existing_output(traj_dir, args.profile, config)
        else:
            model = args.model or config.get("llm", {}).get("model_name")
            if not model:
                console.print("[red]--model is required (or set llm.model_name in config)[/red]")
                sys.exit(1)
            r = _run_analysis(
                traj_dir, model, args.api_base, args.api_key,
                args.profile, args.cost_limit, config,
            )
            total_cost += r.get("cost", 0.0)

        r["traj_dir"] = str(traj_dir)
        r["traj_name"] = label
        results.append(r)

        status = "[green]PASS[/green]" if r["valid"] else "[red]FAIL[/red]"
        console.print(f"  {status}")
        for err in r.get("errors", []):
            console.print(f"    [red]ERROR:[/red] {err}")
        for warn in r.get("warnings", []):
            console.print(f"    [yellow]WARN:[/yellow] {warn}")

    # Summary table
    console.print("\n[bold]== Summary ==[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Trajectory", max_width=40)
    table.add_column("Status")
    table.add_column("Errors")
    table.add_column("Warnings")
    if not args.validate_only:
        table.add_column("Cost ($)")
        table.add_column("Duration (s)")

    for r in results:
        status = "PASS" if r["valid"] else "FAIL"
        row = [
            r["traj_name"],
            f"[green]{status}[/green]" if r["valid"] else f"[red]{status}[/red]",
            str(len(r.get("errors", []))),
            str(len(r.get("warnings", []))),
        ]
        if not args.validate_only:
            metrics = r.get("metrics", {})
            row.append(f"{metrics.get('cost_usd', 0):.4f}")
            row.append(f"{metrics.get('duration_s', 0):.1f}")
        table.add_row(*row)

    console.print(table)

    passed = sum(1 for r in results if r["valid"])
    console.print(f"\n[bold]{passed}/{len(results)}[/bold] passed")
    if not args.validate_only:
        console.print(f"Total cost: [bold]${total_cost:.4f}[/bold]")

    # Write summary JSON
    summary_path = args.data_dir / "codetracer_dev_report.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Report -> [bold]{summary_path}[/bold]")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
