"""ContextAssembler: multi-layer prompt composition.

System prompts are composed from independent layers:
  1. Base system instructions (from config template)
  2. Output profile finalize instructions
  3. Cross-trajectory memory
  4. Budget awareness context
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codetracer.utils.template import render_template as _render_tpl

from codetracer.models.task import TaskContext
from codetracer.skills.loader import Skill
from codetracer.skills.pool import SkillPool


class ContextAssembler:
    """Composes LLM messages from config templates, skill docs, and run data."""

    def __init__(self, config: dict[str, Any], skill_pool: SkillPool) -> None:
        self._config = config
        self._pool = skill_pool

    def build_trace_messages(
        self,
        run_dir: Path,
        skill: Skill | None,
        task_ctx: TaskContext | None = None,
        artifacts_dir: Path | None = None,
        profile: Any | None = None,
        memory_text: str = "",
        budget_context: str = "",
    ) -> list[dict[str, str]]:
        """Build [system, user] messages for the trace (diagnosis) agent.

        Uses layered prompt composition: base system + profile + memory + budget.
        """
        trace_cfg = self._config.get("trace", {})
        read_dir = artifacts_dir or run_dir

        task_md = _read_if_exists(read_dir / "task.md")
        tree_md = _read_if_exists(read_dir / "tree.md")
        stage_ranges_json = _read_if_exists(read_dir / "stage_ranges.json") or "[]"

        sandbox = task_ctx.sandbox_dir if task_ctx else None
        template_vars: dict[str, Any] = {
            "task_description": task_md,
            "tree_md": tree_md,
            "stage_ranges_json": stage_ranges_json,
            "work_dir": str(read_dir),
            "skill_doc": skill.doc if skill else "(pre-normalized trajectory)",
            "skill_name": skill.name if skill else "pre_normalized",
            "task_dir": str(sandbox) if sandbox else "",
            "task_name": task_ctx.task_name if task_ctx else "",
            "exploration_instructions": (
                task_ctx.exploration_instructions(sandbox) if task_ctx and sandbox else ""
            ),
            "problem_statement": task_ctx.problem_statement or "" if task_ctx else "",
        }

        layers = [
            self._render_base_system(trace_cfg, template_vars),
            self._render_profile_instructions(profile),
            self._render_memory_layer(memory_text),
            self._render_budget_layer(budget_context),
        ]
        system = "\n\n".join(layer for layer in layers if layer)

        instance = _render(trace_cfg.get("instance_template", ""), **template_vars)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instance},
        ]

    def build_discovery_messages(
        self, run_dir: Path, listing: str, samples: str
    ) -> list[dict[str, str]]:
        """Build [system, user] messages for the skill generator agent."""
        disc_cfg = self._config.get("discovery", {})

        template_vars: dict[str, Any] = {
            "run_dir": str(run_dir),
            "listing": listing,
            "samples": samples,
            "skill_index": self._pool.skill_index(),
        }

        system = _render(disc_cfg.get("system_template", ""), **template_vars)
        instance = _render(disc_cfg.get("instance_template", ""), **template_vars)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instance},
        ]

    # ------------------------------------------------------------------
    # Prompt layers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_base_system(
        trace_cfg: dict[str, Any], template_vars: dict[str, Any]
    ) -> str:
        return _render(trace_cfg.get("system_template", ""), **template_vars)

    @staticmethod
    def _render_profile_instructions(profile: Any | None) -> str:
        if profile is None:
            return ""
        instruction = getattr(profile, "finalize_instruction", "")
        if not instruction:
            return ""
        return f"== Output Format ==\n{instruction}"

    @staticmethod
    def _render_memory_layer(memory_text: str) -> str:
        if not memory_text:
            return ""
        return f"== Prior Knowledge ==\n{memory_text}"

    @staticmethod
    def _render_budget_layer(budget_context: str) -> str:
        if not budget_context:
            return ""
        return f"== Budget ==\n{budget_context}"


def _read_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _render(template: str, **kwargs: Any) -> str:
    return _render_tpl(template, **kwargs)
