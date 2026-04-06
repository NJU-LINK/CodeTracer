"""Skill loader: reads SKILL.md frontmatter + imports parser.py from a skill directory."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from codetracer.models import NormalizedTrajectory


@dataclass
class Skill:
    """A loaded skill: metadata from SKILL.md frontmatter, full doc body, and parser module."""

    name: str
    description: str
    fingerprints: list[str]
    metadata: dict[str, Any]
    skill_dir: Path
    doc: str
    parser_module: ModuleType
    priority: int = 0

    def matches(self, run_dir: Path) -> bool:
        """Two-level check: fast fingerprint pre-filter, then authoritative can_parse()."""
        if self.fingerprints and not any(list(run_dir.glob(fp)) for fp in self.fingerprints):
            return False
        return self.parser_module.parser.can_parse(run_dir)

    def parse(self, run_dir: Path) -> NormalizedTrajectory:
        return self.parser_module.parser.parse(run_dir)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md into YAML frontmatter dict and markdown body."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    end = stripped.find("---", 3)
    if end == -1:
        return {}, text
    raw_yaml = stripped[3:end]
    body = stripped[end + 3:].lstrip("\n")
    fm: dict[str, Any] = yaml.safe_load(raw_yaml) or {}
    return fm, body


def load_skill(skill_dir: Path) -> Skill:
    """Load a single skill from its directory (must contain SKILL.md and parser.py)."""
    skill_md_path = skill_dir / "SKILL.md"
    parser_py_path = skill_dir / "parser.py"

    if not skill_md_path.exists():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")
    if not parser_py_path.exists():
        raise FileNotFoundError(f"Missing parser.py in {skill_dir}")

    text = skill_md_path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)

    module_name = f"codetracer_skills.{skill_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, parser_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load parser.py from {skill_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "parser"):
        raise AttributeError(f"parser.py in {skill_dir} must define a module-level `parser` object")

    return Skill(
        name=fm.get("name", skill_dir.name),
        description=fm.get("description", ""),
        fingerprints=fm.get("fingerprints", []),
        metadata=fm.get("metadata", {}),
        skill_dir=skill_dir,
        doc=body,
        parser_module=module,
        priority=int(fm.get("priority", 0)),
    )
