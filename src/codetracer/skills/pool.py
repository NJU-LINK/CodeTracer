"""Skill pool: unified registry over seed and user skill directories."""

from __future__ import annotations

import logging
from pathlib import Path

from codetracer.skills.loader import Skill, load_skill

logger = logging.getLogger(__name__)

_SEED_DIR = Path(__file__).parent / "seed"


class SkillPool:
    """Manages all available skills (seed + user-generated)."""

    def __init__(self, seed_dir: Path = _SEED_DIR, user_dir: Path | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._scan(seed_dir)
        if user_dir is not None:
            self._scan(user_dir)

    def _scan(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "SKILL.md").exists():
                continue
            try:
                skill = load_skill(child)
                self._skills[skill.name] = skill
            except Exception as exc:
                logger.warning("Failed to load skill from %s: %s", child, exc)

    def detect(self, run_dir: Path) -> str | None:
        """Return the name of the highest-priority skill that can parse *run_dir*.

        Uses a two-pass approach:
        1. Fingerprint-filtered match (fast glob pre-filter + can_parse)
        2. Fallback: try can_parse on all skills (catches layouts where
           fingerprints don't match but the parser still knows the format)
        """
        by_priority = sorted(self._skills.values(), key=lambda s: s.priority, reverse=True)
        for skill in by_priority:
            if skill.matches(run_dir):
                return skill.name
        for skill in by_priority:
            try:
                if skill.parser_module.parser.can_parse(run_dir):
                    return skill.name
            except Exception:
                continue
        return None

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def skill_index(self) -> str:
        """Compact markdown index of all skills for LLM context injection."""
        if not self._skills:
            return "(no skills loaded)"
        lines: list[str] = ["# Available Skills", ""]
        for skill in self._skills.values():
            fps = ", ".join(f"`{f}`" for f in skill.fingerprints)
            lines.append(f"- **{skill.name}**: {skill.description.strip()}")
            if fps:
                lines.append(f"  Fingerprints: {fps}")
        return "\n".join(lines)
