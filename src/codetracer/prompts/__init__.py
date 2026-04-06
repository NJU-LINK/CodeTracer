"""CodeTracer prompt templates -- base templates and registry."""

from codetracer.prompts.base_template import SkillTemplate
from codetracer.prompts.registry import TemplateRegistry, default_template_registry

__all__ = ["SkillTemplate", "TemplateRegistry", "default_template_registry"]
