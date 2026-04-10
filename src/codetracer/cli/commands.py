"""CodeTracer CLI: command-group entry point for analyze, replay, inspect, and interactive modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(rich_markup_mode="rich", add_completion=False, invoke_without_command=True, help="CodeTracer: trajectory diagnosis & replay")
console = Console(highlight=False)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    """Launch interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from codetracer.cli.repl import interactive_repl
        interactive_repl()


# ===================================================================
# analyze  (the original main command)
# ===================================================================

@app.command(help="Run CodeTracer analysis on a raw or pre-normalized agent run directory.")
def analyze(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    format_override: str = typer.Option("auto", "--format", "-f", help="Force format or 'auto' for detection"),
    model: str | None = typer.Option(None, "-m", "--model", help="LLM model name"),
    api_base: str | None = typer.Option(None, "--api-base", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key"),
    config_path: Path | None = typer.Option(None, "-c", "--config", help="Path to YAML config override"),
    cost_limit: float = typer.Option(3.0, "-l", "--cost-limit", help="Max LLM spend (USD)"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output labels JSON path"),
    skip_discovery: bool = typer.Option(False, "--skip-discovery", help="Fail if format is unknown"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Detect + parse only, skip tracing"),
    use_llm_tree: bool = typer.Option(False, "--llm-tree", help="Use LLM for richer tree classification"),
    task_dir: Path | None = typer.Option(None, "--task-dir", "-t", help="Path to terminal-bench task folder"),
    tasks_root: Path | None = typer.Option(None, "--tasks-root", help="Root of terminal-bench tasks for auto-resolution"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Root output directory"),
    traj_id: str | None = typer.Option(None, "--traj-id", help="Unique ID for this run"),
    annotation_tree: bool = typer.Option(False, "--annotation-tree", help="Build tree from annotation labels"),
    traj_annotation: str | None = typer.Option(None, "--traj-annotation", help="JSON string of manifest entry"),
    traj_annotation_path: Path | None = typer.Option(None, "--traj-annotation-path", help="Path to annotation JSON"),
    annotation_base: Path | None = typer.Option(None, "--annotation-base", help="Root dir for annotation_relpath"),
    skip_sandbox: bool = typer.Option(False, "--skip-sandbox", help="Skip sandbox creation"),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Output profile: detailed (default), tracebench, or rl_feedback"),
    mode: str = typer.Option("benchmark", "--mode", help="Mode: benchmark (one-shot) or interactive (REPL)"),
) -> Any:
    if mode == "interactive":
        from codetracer.cli.repl import interactive_repl
        interactive_repl(
            run_dir.resolve(),
            config_path=config_path,
            profile=profile or "detailed",
        )
        return None

    from platformdirs import user_config_dir

    from codetracer.agents.context import ContextAssembler
    from codetracer.agents.trace_agent import TraceAgent
    from codetracer.discovery.explorer import detect_or_generate_skill
    from codetracer.llm.client import LLMClient
    from codetracer.models.task import TaskContext
    from codetracer.query.config import load_config
    from codetracer.query.normalizer import Normalizer
    from codetracer.query.tree_builder import TreeBuilder
    from codetracer.skills.pool import SkillPool

    run_dir = run_dir.resolve()

    work_dir: Path | None = None
    if output_dir:
        if not traj_id:
            raise typer.BadParameter("--traj-id is required when --output-dir is specified")
        work_dir = Path(output_dir) / traj_id
        work_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Output directory: [bold]{work_dir}[/bold]")

    task_ctx: TaskContext | None = None
    if task_dir:
        task_ctx = TaskContext.load(task_dir)
        console.print(f"Loaded task context: [bold]{task_ctx.task_name}[/bold]")
    elif tasks_root:
        task_ctx = _resolve_task_ctx(tasks_root, run_dir)
        if task_ctx:
            console.print(f"Auto-resolved task context: [bold]{task_ctx.task_name}[/bold]")
        else:
            console.print("[dim]--tasks-root given but no matching task folder found[/dim]")

    if task_ctx is not None and not skip_sandbox:
        sandbox_parent = work_dir or run_dir
        sandbox = task_ctx.prepare_sandbox(sandbox_parent)
        console.print(f"Task sandbox: [bold]{sandbox}[/bold]")

    config = load_config(config_path)
    llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
    if model:
        llm_cfg["model_name"] = model
    if api_base:
        llm_cfg["api_base"] = api_base
    if api_key:
        llm_cfg["api_key"] = api_key
    if cost_limit:
        config.setdefault("trace", {})["cost_limit"] = cost_limit

    llm = LLMClient(**llm_cfg)

    user_skill_dir = Path(user_config_dir("codetracer")) / "skills"
    user_skill_dir.mkdir(parents=True, exist_ok=True)
    pool = SkillPool(user_dir=user_skill_dir)

    normalizer = Normalizer(pool)
    skill = None

    if skip_discovery:
        # No auto-generation; fail on unknown format
        if normalizer.is_pre_normalized(run_dir):
            console.print("Pre-normalized directory (steps.json found)")
            traj = normalizer.normalize_pre_normalized(run_dir, output_dir=work_dir)
        elif normalizer.is_step_jsonl_dir(run_dir):
            console.print("Annotation directory (step_N.jsonl files found)")
            traj = normalizer.normalize_step_jsonl(run_dir, output_dir=work_dir)
        else:
            skill = normalizer.detect(run_dir, format_override)
            console.print(f"Detected format: [bold green]{skill.name}[/bold green]")
            traj = normalizer.normalize(run_dir, skill, output_dir=work_dir)
    else:
        try:
            skill, traj = detect_or_generate_skill(
                run_dir, normalizer, pool, llm, config,
                user_skill_dir=user_skill_dir,
                format_override=format_override,
            )
            if skill:
                console.print(f"Detected format: [bold green]{skill.name}[/bold green]")
            else:
                console.print("Pre-normalized / annotation directory")
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if dry_run:
        console.print("[yellow]--dry-run: stopping before trace.[/yellow]")
        return None

    write_base = work_dir or run_dir
    tree_md_path = write_base / "tree.md"
    if not tree_md_path.exists():
        builder = TreeBuilder(llm if use_llm_tree else None, config=config.get("tree", {}))
        if annotation_tree:
            if not traj_annotation and not traj_annotation_path:
                raise typer.BadParameter("--traj-annotation or --traj-annotation-path required with --annotation-tree")
            if traj_annotation_path:
                loaded = json.loads(Path(traj_annotation_path).read_text(encoding="utf-8", errors="replace"))
                manifest_entry = loaded["annotation"] if isinstance(loaded, dict) and isinstance(loaded.get("annotation"), dict) else loaded
            else:
                manifest_entry = json.loads(traj_annotation or "{}")
            tree_md = builder.build_from_annotation(traj, manifest_entry, run_dir=run_dir)
        elif use_llm_tree:
            tree_md = builder.build_with_llm(traj)
        else:
            tree_md = builder.build(traj)
        tree_md_path.write_text(tree_md, encoding="utf-8")
        console.print(f"Built tree index -> [bold]{tree_md_path}[/bold]")

    from codetracer.plugins.hooks import MEMORY_EXTRACTION, default_hooks
    from codetracer.services.complexity import estimate_complexity
    from codetracer.services.cost_tracker import CostTracker
    from codetracer.services.memory import auto_extract_memory, load_memory
    from codetracer.state.output_profile import get_default_profile_name, load_profile

    profile_name = profile or get_default_profile_name(config)
    out_profile = load_profile(profile_name, config)

    fmt_name = skill.name if skill else "pre_normalized"
    cost_tracker = CostTracker(budget_limit_usd=cost_limit)
    memory_text = load_memory(fmt_name)

    complexity = estimate_complexity(traj)
    traj_meta = {
        "complexity_tier": complexity.complexity_tier,
        "adaptive_instructions": complexity.adaptive_instructions,
        "step_count": complexity.step_count,
        "unique_tool_types": complexity.unique_tool_types,
    }

    output_path = output or write_base / out_profile.output_file
    assembler = ContextAssembler(config, pool)
    agent = TraceAgent(
        llm, assembler, run_dir, output_path, config,
        artifacts_dir=work_dir,
        cost_tracker=cost_tracker,
        profile=out_profile,
        agent_type=fmt_name,
    )
    result = agent.run(skill, task_ctx=task_ctx, memory_text=memory_text, traj_metadata=traj_meta)
    console.print(f"Trace finished: [bold]{result}[/bold]")
    console.print(cost_tracker.format_summary())

    traj_path = output_path.parent / (output_path.stem + ".traj.json")
    agent.save_trajectory(traj_path)
    console.print(f"Labels     -> [bold green]{output_path}[/bold green]")
    console.print(f"Trajectory -> [bold green]{traj_path}[/bold green]")

    auto_extract_memory(fmt_name, output_path, analysis_summary=result)
    default_hooks.emit(MEMORY_EXTRACTION, agent_type=fmt_name)

    return None


# ===================================================================
# replay
# ===================================================================

@app.command(help="Replay a trajectory to a specific step with error analysis injected.")
def replay(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    step: int | None = typer.Option(None, "--step", "-s", help="Target step_id to replay to (interactive mode)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-detect first incorrect step and replay"),
    analysis_path: Path | None = typer.Option(None, "--analysis", "-a", help="Path to ErrorAnalysis JSON"),
    bench_type: str = typer.Option("local", "--bench-type", "-b", help="Environment type: local, docker, git"),
    docker_compose: Path | None = typer.Option(None, "--docker-compose", help="Path to docker-compose.yaml"),
    container_name: str | None = typer.Option(None, "--container-name", help="Docker container name"),
    repo_url: str | None = typer.Option(None, "--repo-url", help="Git repo URL (swe-bench)"),
    base_commit: str | None = typer.Option(None, "--base-commit", help="Git base commit (swe-bench)"),
    model: str | None = typer.Option(None, "-m", "--model", help="LLM model name for replay agent"),
    api_base: str | None = typer.Option(None, "--api-base", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key"),
    config_path: Path | None = typer.Option(None, "-c", "--config", help="Path to YAML config override"),
    no_agent: bool = typer.Option(False, "--no-agent", help="Only replay steps, don't start agent loop"),
) -> Any:
    from codetracer.llm.client import LLMClient
    from codetracer.models import ErrorAnalysis
    from codetracer.query.config import load_config
    from codetracer.query.normalizer import Normalizer
    from codetracer.replay.engine import ReplayEngine
    from codetracer.skills.pool import SkillPool

    run_dir = run_dir.resolve()

    if not auto and step is None:
        raise typer.BadParameter("Provide --step <N> or --auto")

    config = load_config(config_path)

    pool = SkillPool()
    normalizer = Normalizer(pool)

    if normalizer.is_pre_normalized(run_dir):
        traj = normalizer.normalize_pre_normalized(run_dir)
    elif normalizer.is_step_jsonl_dir(run_dir):
        traj = normalizer.normalize_step_jsonl(run_dir)
    else:
        skill = normalizer.detect(run_dir)
        traj = normalizer.normalize(run_dir, skill)

    analysis = None
    if analysis_path:
        analysis = ErrorAnalysis.load(analysis_path)
    elif (run_dir / "codetracer_labels.json").exists():
        analysis = ErrorAnalysis.from_labels_json(run_dir / "codetracer_labels.json", run_dir.name)

    env_config: dict[str, Any] = {"work_dir": str(run_dir), "timeout": 120}
    if docker_compose:
        env_config["docker_compose_path"] = str(docker_compose)
    if container_name:
        env_config["container_name"] = container_name
    if repo_url:
        env_config["repo_url"] = repo_url
    if base_commit:
        env_config["base_commit"] = base_commit

    llm = None
    if not no_agent:
        llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
        if model:
            llm_cfg["model_name"] = model
        if api_base:
            llm_cfg["api_base"] = api_base
        if api_key:
            llm_cfg["api_key"] = api_key
        llm = LLMClient(**llm_cfg)

    engine = ReplayEngine(config=config, checkpoints_dir=run_dir / ".checkpoints")

    if auto:
        if analysis is None:
            raise typer.BadParameter("--auto requires error analysis (provide --analysis or run analyze first)")
        result = engine.replay_auto(traj, analysis, bench_type=bench_type, env_config=env_config, llm=llm)
    else:
        result = engine.replay_interactive(traj, step, analysis, bench_type=bench_type, env_config=env_config, llm=llm)

    console.print(f"\nReplay result: [bold]{result.status.value}[/bold] ({result.steps_replayed} steps replayed)")
    return None


# ===================================================================
# inspect
# ===================================================================

@app.command(help="Inspect a specific step or range from a trajectory.")
def inspect(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    step: int | None = typer.Option(None, "--step", "-s", help="Step ID to inspect"),
    range_start: int | None = typer.Option(None, "--from", help="Start of step range"),
    range_end: int | None = typer.Option(None, "--to", help="End of step range"),
    show_tree: bool = typer.Option(False, "--tree", help="Show the tree index"),
) -> Any:
    from codetracer.query.normalizer import Normalizer
    from codetracer.skills.pool import SkillPool

    run_dir = run_dir.resolve()
    pool = SkillPool()
    normalizer = Normalizer(pool)

    if normalizer.is_pre_normalized(run_dir):
        traj = normalizer.normalize_pre_normalized(run_dir)
    elif normalizer.is_step_jsonl_dir(run_dir):
        traj = normalizer.normalize_step_jsonl(run_dir)
    else:
        skill = normalizer.detect(run_dir)
        traj = normalizer.normalize(run_dir, skill)

    if show_tree:
        tree_path = run_dir / "tree.md"
        if tree_path.exists():
            console.print(tree_path.read_text(encoding="utf-8"))
        else:
            from codetracer.query.tree_builder import TreeBuilder
            console.print(TreeBuilder().build(traj))
        return None

    if step is not None:
        matches = [s for s in traj.steps if s.step_id == step]
        if not matches:
            console.print(f"[red]Step {step} not found (range: {traj.steps[0].step_id}-{traj.steps[-1].step_id})[/red]")
            return None
        for s in matches:
            _print_step(s)
        return None

    start = range_start or traj.steps[0].step_id
    end = range_end or traj.steps[-1].step_id
    for s in traj.steps:
        if start <= s.step_id <= end:
            _print_step(s)
    return None


# ===================================================================
# interactive
# ===================================================================

@app.command(help="Enter interactive REPL with menu-driven actions and LLM chat.")
def interactive(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    model: str | None = typer.Option(None, "-m", "--model", help="LLM model name"),
    api_base: str | None = typer.Option(None, "--api-base", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key"),
    config_path: Path | None = typer.Option(None, "-c", "--config", help="Path to YAML config override"),
    resume: bool = typer.Option(False, "--resume", help="Resume a previous interactive session"),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Output profile: detailed (default), tracebench, or rl_feedback"),
) -> Any:
    from codetracer.cli.repl import interactive_repl
    interactive_repl(
        run_dir.resolve(),
        model=model, api_base=api_base, api_key=api_key,
        config_path=config_path, resume=resume, profile=profile,
    )


# ===================================================================
# run  (one-click auto-detect + analyze + replay)
# ===================================================================

@app.command(help="One-click: auto-detect format, analyze trajectory, report errors, and optionally replay.")
def run(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    model: str | None = typer.Option(None, "-m", "--model", help="LLM model name"),
    api_base: str | None = typer.Option(None, "--api-base", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key"),
    config_path: Path | None = typer.Option(None, "-c", "--config", help="Path to YAML config override"),
    cost_limit: float = typer.Option(3.0, "-l", "--cost-limit", help="Max LLM spend (USD)"),
    auto_replay: bool = typer.Option(False, "--replay", "-r", help="Auto-replay from the first incorrect step"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output labels JSON path"),
    skip_discovery: bool = typer.Option(False, "--skip-discovery", help="Fail if format is unknown"),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Output profile: detailed (default), tracebench, or rl_feedback"),
) -> Any:
    from platformdirs import user_config_dir

    from codetracer.agents.context import ContextAssembler
    from codetracer.agents.trace_agent import TraceAgent
    from codetracer.discovery.explorer import detect_or_generate_skill
    from codetracer.llm.client import LLMClient
    from codetracer.query.config import load_config
    from codetracer.query.normalizer import Normalizer
    from codetracer.query.tree_builder import TreeBuilder
    from codetracer.skills.pool import SkillPool

    run_dir = run_dir.resolve()
    config = load_config(config_path)

    llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
    if model:
        llm_cfg["model_name"] = model
    if api_base:
        llm_cfg["api_base"] = api_base
    if api_key:
        llm_cfg["api_key"] = api_key
    if cost_limit:
        config.setdefault("trace", {})["cost_limit"] = cost_limit

    llm = LLMClient(**llm_cfg)
    user_skill_dir = Path(user_config_dir("codetracer")) / "skills"
    user_skill_dir.mkdir(parents=True, exist_ok=True)
    pool = SkillPool(user_dir=user_skill_dir)
    normalizer = Normalizer(pool)

    # 1. Normalize
    console.print("[bold]Step 1/4: Detecting format and normalizing...[/bold]")
    skill = None
    if skip_discovery:
        if normalizer.is_pre_normalized(run_dir):
            traj = normalizer.normalize_pre_normalized(run_dir)
            console.print("  Pre-normalized directory (steps.json found)")
        elif normalizer.is_step_jsonl_dir(run_dir):
            traj = normalizer.normalize_step_jsonl(run_dir)
            console.print("  Annotation directory (step_N.jsonl files found)")
        else:
            skill = normalizer.detect(run_dir)
            console.print(f"  Detected format: [bold green]{skill.name}[/bold green]")
            traj = normalizer.normalize(run_dir, skill)
    else:
        try:
            skill, traj = detect_or_generate_skill(
                run_dir, normalizer, pool, llm, config,
                user_skill_dir=user_skill_dir,
            )
            if skill:
                console.print(f"  Detected format: [bold green]{skill.name}[/bold green]")
            else:
                console.print("  Pre-normalized / annotation directory")
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    # 2. Build tree
    console.print("[bold]Step 2/4: Building tree index...[/bold]")
    tree_path = run_dir / "tree.md"
    if not tree_path.exists():
        builder = TreeBuilder()
        tree_md = builder.build(traj)
        tree_path.write_text(tree_md, encoding="utf-8")
    console.print(f"  Tree -> {tree_path}")

    # 3. Analyze
    from codetracer.plugins.hooks import MEMORY_EXTRACTION, default_hooks
    from codetracer.services.complexity import estimate_complexity
    from codetracer.services.cost_tracker import CostTracker
    from codetracer.services.memory import auto_extract_memory, load_memory
    from codetracer.state.output_profile import get_default_profile_name, load_profile

    profile_name = profile or get_default_profile_name(config)
    out_profile = load_profile(profile_name, config)
    fmt_name = skill.name if skill else "pre_normalized"
    cost_tracker = CostTracker(budget_limit_usd=cost_limit)
    memory_text = load_memory(fmt_name)

    complexity = estimate_complexity(traj)
    traj_meta = {
        "complexity_tier": complexity.complexity_tier,
        "adaptive_instructions": complexity.adaptive_instructions,
        "step_count": complexity.step_count,
        "unique_tool_types": complexity.unique_tool_types,
    }

    console.print("[bold]Step 3/4: Running error analysis...[/bold]")
    output_path = output or run_dir / out_profile.output_file
    assembler = ContextAssembler(config, pool)
    agent = TraceAgent(
        llm, assembler, run_dir, output_path, config,
        cost_tracker=cost_tracker,
        profile=out_profile,
        agent_type=fmt_name,
    )
    result = agent.run(skill, memory_text=memory_text, traj_metadata=traj_meta)

    traj_path = output_path.parent / (output_path.stem + ".traj.json")
    agent.save_trajectory(traj_path)
    console.print(f"  Labels     -> [bold green]{output_path}[/bold green]")
    console.print(f"  Trajectory -> [bold green]{traj_path}[/bold green]")
    console.print(cost_tracker.format_summary())

    auto_extract_memory(fmt_name, output_path, analysis_summary=result)
    default_hooks.emit(MEMORY_EXTRACTION, agent_type=fmt_name)

    # 4. Show errors + optional replay
    console.print("[bold]Step 4/4: Results[/bold]")
    from codetracer.models import ErrorAnalysis as _EA
    analysis = _EA.from_labels_json(output_path, run_dir.name)
    if not analysis.labels:
        console.print("  [green]No errors found -- trajectory looks clean.[/green]")
    else:
        for label in analysis.labels:
            icon = "[red]incorrect[/red]" if label.verdict.value == "incorrect" else "[yellow]unuseful[/yellow]"
            console.print(f"  Step {label.step_id} [{icon}]: {label.reasoning[:100]}")

        if auto_replay:
            target = analysis.first_incorrect_step_id
            if target is not None:
                console.print(f"\n[bold]Auto-replaying from step {target}...[/bold]")
                from codetracer.replay.engine import ReplayEngine
                engine = ReplayEngine(config=config, checkpoints_dir=run_dir / ".checkpoints")
                replay_result = engine.replay_auto(
                    traj, analysis,
                    env_config={"work_dir": str(run_dir)},
                    llm=llm,
                )
                console.print(f"Replay: [bold]{replay_result.status.value}[/bold] ({replay_result.steps_replayed} steps)")

    return None


# ===================================================================
# Helpers
# ===================================================================

def _print_step(s: Any) -> None:
    console.print(f"\n[bold]Step {s.step_id}[/bold]")
    console.print(f"  Action: {(s.action or '')[:200]}")
    if s.observation:
        console.print(f"  Observation: {s.observation[:200]}")


def _resolve_task_ctx(tasks_root: Path, run_dir: Path) -> Any:
    from codetracer.models.task import TaskContext

    candidates = [run_dir.name]
    for parent in run_dir.parents:
        candidates.append(parent.name)
        if parent == run_dir.parent.parent:
            break

    for name in candidates:
        task_path = tasks_root / name
        if not task_path.is_dir():
            continue
        if (task_path / "task.yaml").exists() or (task_path / "summary.json").exists():
            return TaskContext.load(task_path)
    return None


