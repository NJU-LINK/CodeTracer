#!/usr/bin/env python3
"""
Run tracer on an existing agent run output directory.

The tracer provides only the core mini-swe-agent execution loop. The LLM must
explore the directory, decide step segmentation, map steps to (relative path,
line range), and write labeling outputs.
"""

from __future__ import annotations

import traceback
import json
from pathlib import Path
from typing import Any

import typer
import yaml
from jinja2 import StrictUndefined, Template
from rich.console import Console

from tracer.agents.interactive import InteractiveAgent
from tracer.config import builtin_config_dir, get_config_path
from tracer.environments import get_environment
from tracer.models import get_model
from tracer.run.config import configure_if_first_time
from tracer.run.utils.save import save_traj
from tracer.step_id_maps.load import load_step_id_maps_input
from tracer.utils.log import logger
from tracer.tracebench import extract_tracebench_record

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

DEFAULT_CONFIG = builtin_config_dir / "tracer.yaml"

def _prepare_environment_config(env_cfg: dict[str, Any] | None, *, run_dir: Path) -> dict[str, Any]:
    cfg = dict(env_cfg or {})
    env_type = cfg.get("environment_class") or "local"
    if env_type == "docker":
        container_dir = str(cfg.get("cwd") or "/work")
        cfg["cwd"] = container_dir
        run_args = list(cfg.get("run_args") or ["--rm"])
        mount = f"{run_dir}:{container_dir}"
        if mount not in run_args:
            run_args += ["-v", mount]
        cfg["run_args"] = run_args
        return cfg

    cfg["cwd"] = str(run_dir)
    return cfg


@app.command(help=__doc__)
def main(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    input_format: str = typer.Option("step_id_maps", "--input-format", help="Input format: step_id_maps or tracebench"),
    stage_starts: str = typer.Option(
        "", "--stage-starts", help="Comma-separated 1-indexed stage start step_ids (ordered), e.g. '1,8,23,38'"
    ),
    traj_id: str | None = typer.Option(None, "--traj-id", help="Tracebench traj_id for record selection/download"),
    split: str = typer.Option("full", "--split", help="Tracebench split: full or verified"),
    download_tracebench: bool = typer.Option(
        False, "--download-tracebench/--no-download-tracebench", help="Download manifest (and artifact if present) from HF"
    ),
    tracebench_repo: str = typer.Option("Schwerli/Tracebench", "--tracebench-repo", help="HF dataset repo id"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only resolve/inject Tracebench record and exit"),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output trajectory file for the tracer run (default: <run_dir>/mini_tracer.traj.json)",
    ),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    yolo: bool = typer.Option(True, "-y", "--yolo", help="Execute LM commands immediately (recommended)"),
) -> Any:
    configure_if_first_time()

    run_dir = run_dir.resolve()
    if output is None:
        output = run_dir / "mini_tracer.traj.json"

    config_path = get_config_path(config_spec)
    console.print(f"Loading tracer config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())

    task_prompt = ""
    tracer_cfg = config.get("tracer", {}) if isinstance(config, dict) else {}
    if isinstance(tracer_cfg, dict):
        task_prompt = tracer_cfg.get("task_prompt") if isinstance(tracer_cfg.get("task_prompt"), str) else ""
    if not task_prompt.strip():
        raise typer.BadParameter("Missing tracer.task_prompt in config")

    if input_format not in ("step_id_maps", "tracebench"):
        raise typer.BadParameter("--input-format must be 'step_id_maps' or 'tracebench'")

    stage_starts_list: list[int] = []
    if stage_starts.strip():
        try:
            stage_starts_list = [int(x.strip()) for x in stage_starts.split(",") if x.strip()]
        except ValueError as e:
            raise typer.BadParameter("--stage-starts must be a comma-separated list of integers") from e

    record = None

    if input_format == "step_id_maps":
        si = load_step_id_maps_input(run_dir)
        # stage_ranges.json is the source of truth for stages. stage_starts_list is optional (back-compat).
        stage_ranges = json.loads(si.stage_ranges_json_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(stage_ranges, list) or not all(isinstance(x, dict) for x in stage_ranges):
            raise typer.BadParameter("Invalid stage_ranges.json: expected a JSON list of objects")
        if not stage_starts_list:
            starts = []
            for s in stage_ranges:
                v = s.get("start_step_id")
                if isinstance(v, int):
                    starts.append(v)
            stage_starts_list = starts

        if dry_run:
            console.print("Loaded step_id_maps input from directory:")
            console.print(f"- task.md: {si.task_md_path}")
            console.print(f"- tree.md: {si.tree_md_path}")
            console.print(f"- steps.json: {si.steps_json_path}")
            console.print(f"- stage_ranges.json: {si.stage_ranges_json_path}")
            console.print(f"- stage_starts (derived): {stage_starts_list}")
            return None
    else:
        if split not in ("full", "verified"):
            raise typer.BadParameter("--split must be 'full' or 'verified'")
        record = extract_tracebench_record(
            run_dir=run_dir,
            traj_id=traj_id,
            split=split,  # type: ignore[arg-type]
            download_tracebench=download_tracebench,
            tracebench_repo=tracebench_repo,
        )
        if dry_run:
            console.print(f"Tracebench record loaded: [bold green]{record.traj_id}[/bold green] (source={record.source})")
            if record.downloaded_artifact is not None:
                console.print(f"Downloaded artifact: [bold green]{record.downloaded_artifact}[/bold green]")
            return None

    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit

    task = ""

    model = get_model(model_name, config.get("model", {}))
    env_cfg = _prepare_environment_config(config.get("environment", {}), run_dir=run_dir)
    env = get_environment(env_cfg, default_type="local")

    agent_config = config.get("agent", {})
    if yolo:
        agent_config = agent_config | {"mode": "yolo"}
    agent_config = agent_config | {"confirm_exit": False}

    agent = InteractiveAgent(model, env, **agent_config)

    exit_status, result, extra_info = None, None, None
    try:
        template_vars: dict[str, Any] = {
            "run_dir": str(run_dir),
            "stage_starts": stage_starts_list,
            "task_md_content": si.task_md_path.read_text(encoding="utf-8", errors="replace") if input_format == "step_id_maps" else "",
            "tree_md_content": si.tree_md_path.read_text(encoding="utf-8", errors="replace") if input_format == "step_id_maps" else "",
            "stage_ranges_json_content": si.stage_ranges_json_path.read_text(encoding="utf-8", errors="replace") if input_format == "step_id_maps" else "",
            "tracebench_source": record.source if record is not None else "",
            "tracebench_record_json": record.record_json() if record is not None else "",
        }
        # tracer.task_prompt itself may contain Jinja placeholders (e.g. {{ task_md_content }}).
        # instance_template injects tracer_task_prompt as a raw string, so we must render it here.
        template_vars["tracer_task_prompt"] = Template(task_prompt, undefined=StrictUndefined).render(**template_vars)
        exit_status, result = agent.run(task, **template_vars)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running tracer: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
        console.print(f"Saved tracer trajectory to [bold green]'{output}'[/bold green]")

    return agent


