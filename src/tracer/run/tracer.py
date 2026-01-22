#!/usr/bin/env python3
"""
Run tracer on an existing agent run output directory.

The tracer provides only the core mini-swe-agent execution loop. The LLM must
explore the directory, decide step segmentation, map steps to (relative path,
line range), and write labeling outputs.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from tracer.agents.interactive import InteractiveAgent
from tracer.config import builtin_config_dir, get_config_path
from tracer.environments.local import LocalEnvironment
from tracer.models import get_model
from tracer.run.config import configure_if_first_time
from tracer.run.utils.save import save_traj
from tracer.utils.log import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

DEFAULT_CONFIG = builtin_config_dir / "extra" / "tracer.yaml"


def _build_task(task_prompt: str, run_dir: Path) -> str:
    return (
        task_prompt.rstrip()
        + "\n\n"
        + "Run directory (absolute path):\n"
        + f"{run_dir}\n\n"
        + "Write outputs into this directory:\n"
        + "- mini_tracer_labels.jsonl\n"
        + "- mini_tracer_summary.json\n"
    )


@app.command(help=__doc__)
def main(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
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

    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit

    task = _build_task(task_prompt, run_dir)

    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**(config.get("environment", {}) | {"cwd": str(run_dir)}))

    agent_config = config.get("agent", {})
    if yolo:
        agent_config = agent_config | {"mode": "yolo"}
    agent_config = agent_config | {"confirm_exit": False}

    agent = InteractiveAgent(model, env, **agent_config)

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(task)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running tracer: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
        console.print(f"Saved tracer trajectory to [bold green]'{output}'[/bold green]")

    return agent


