"""Shared Jinja2 render utility.

Eliminates the duplicated ``Template(tpl, undefined=StrictUndefined).render()``
pattern found across ContextAssembler, BaseAgent, ContextInjector, and
SkillTemplate.
"""

from __future__ import annotations

from typing import Any

from jinja2 import StrictUndefined, Template


def render_template(template: str, **kwargs: Any) -> str:
    """Render a Jinja2 template string with strict undefined variables."""
    if not template:
        return ""
    return Template(template, undefined=StrictUndefined).render(**kwargs)
