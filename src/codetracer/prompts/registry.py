"""TemplateRegistry: discovers and serves SkillTemplate instances by agent name."""

from __future__ import annotations

import logging
from pathlib import Path

from codetracer.prompts.base_template import SkillTemplate

logger = logging.getLogger(__name__)

_PRESETS_DIR = Path(__file__).parent / "presets"


class TemplateRegistry:
    """Manages agent-specific prompt templates (presets + user-supplied)."""

    def __init__(self, presets_dir: Path = _PRESETS_DIR, user_dir: Path | None = None) -> None:
        self._templates: dict[str, SkillTemplate] = {}
        self._scan(presets_dir)
        if user_dir is not None:
            self._scan(user_dir)

    def _scan(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for p in sorted(directory.glob("*.yaml")):
            try:
                tpl = SkillTemplate.from_yaml(p)
                self._templates[tpl.name] = tpl
            except Exception:
                logger.warning("Failed to load template from %s", p, exc_info=True)

    def get(self, name: str) -> SkillTemplate | None:
        return self._templates.get(name)

    def list_templates(self) -> list[str]:
        return sorted(self._templates.keys())

    def register(self, template: SkillTemplate) -> None:
        self._templates[template.name] = template


default_template_registry = TemplateRegistry()
