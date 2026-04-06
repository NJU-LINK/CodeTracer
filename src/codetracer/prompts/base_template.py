"""SkillTemplate: base class for agent-specific prompt templates.

Each template knows how to format CodeTracer error analysis and replay
instructions for a particular closed-source or open-source agent's
conversation style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from codetracer.utils.template import render_template


@dataclass
class SkillTemplate:
    """A loaded template describing how to inject CodeTracer context into a specific agent."""

    name: str
    agent: str
    description: str = ""
    error_format: str = ""
    replay_instructions: str = ""
    context_injection: str = ""
    metadata: dict = field(default_factory=dict)

    def render_error_block(self, analysis_dict: dict[str, Any]) -> str:
        """Render the error analysis for this agent's prompt style."""
        return render_template(self.error_format, analysis=analysis_dict)

    def render_replay_instructions(self, **kwargs: Any) -> str:
        return render_template(self.replay_instructions, **kwargs)

    def render_context_injection(self, **kwargs: Any) -> str:
        return render_template(self.context_injection, **kwargs)

    @classmethod
    def from_yaml(cls, path: Path) -> SkillTemplate:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            name=raw.get("name", path.stem),
            agent=raw.get("agent", ""),
            description=raw.get("description", ""),
            error_format=raw.get("error_format", ""),
            replay_instructions=raw.get("replay_instructions", ""),
            context_injection=raw.get("context_injection", ""),
            metadata=raw.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "agent": self.agent,
            "description": self.description,
            "error_format": self.error_format,
            "replay_instructions": self.replay_instructions,
            "context_injection": self.context_injection,
            "metadata": self.metadata,
        }
