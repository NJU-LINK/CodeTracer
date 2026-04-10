"""Output profile system.

Profiles define how the trace agent formats its final output.
``tracebench`` produces the existing ``codetracer_labels.json`` format;
``detailed`` produces a richer root-cause analysis document.
Profiles are loaded from the ``output.profiles`` section of the config YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputProfile:
    """Definition of one output profile."""

    name: str
    schema_ref: str = ""
    finalize_instruction: str = ""
    output_file: str = "codetracer_labels.json"
    json_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> OutputProfile:
        return cls(
            name=name,
            schema_ref=d.get("schema_ref", ""),
            finalize_instruction=d.get("finalize_instruction", ""),
            output_file=d.get("output_file", "codetracer_labels.json"),
            json_schema=d.get("json_schema", {}),
        )


_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "tracebench": {
        "schema_ref": "tracebench_labels",
        "finalize_instruction": (
            "Output the final labels as codetracer_labels.json. "
            "Each entry is a JSON object with keys: stage_id (int), "
            "incorrect_step_ids (list[int]), unuseful_step_ids (list[int]), "
            "reasoning (str). Wrap all entries in a JSON array."
        ),
        "output_file": "codetracer_labels.json",
        "json_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage_id": {"type": "integer"},
                    "incorrect_step_ids": {"type": "array", "items": {"type": "integer"}},
                    "unuseful_step_ids": {"type": "array", "items": {"type": "integer"}},
                    "reasoning": {"type": "string"},
                },
                "required": ["stage_id", "incorrect_step_ids", "unuseful_step_ids", "reasoning"],
            },
        },
    },
    "detailed": {
        "schema_ref": "deep_analysis",
        "finalize_instruction": (
            "Output a comprehensive root cause analysis as codetracer_analysis.json. "
            "Include: root_cause_chain (list of step descriptions from final failure "
            "back to initial error), critical_decision_points (list of {step_id, decision, "
            "should_have}), correct_strategy (str), stage_labels (same format as tracebench), "
            "and summary (str)."
        ),
        "output_file": "codetracer_analysis.json",
        "json_schema": {
            "type": "object",
            "properties": {
                "root_cause_chain": {"type": "array", "items": {"type": "string"}},
                "critical_decision_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {"type": "integer"},
                            "decision": {"type": "string"},
                            "should_have": {"type": "string"},
                        },
                    },
                },
                "correct_strategy": {"type": "string"},
                "stage_labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage_id": {"type": "integer"},
                            "incorrect_step_ids": {"type": "array", "items": {"type": "integer"}},
                            "unuseful_step_ids": {"type": "array", "items": {"type": "integer"}},
                            "reasoning": {"type": "string"},
                        },
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["root_cause_chain", "critical_decision_points", "correct_strategy", "stage_labels", "summary"],
        },
    },
    "rl_feedback": {
        "schema_ref": "rl_feedback",
        "finalize_instruction": (
            "Output codetracer_rl_feedback.json with per-step deviation analysis. "
            "Each entry must include: step_id (int), verdict (incorrect|unuseful|correct), "
            "deviation_type (wrong_tool|wrong_target|redundant|premature_conclusion|"
            "missing_exploration|none), correct_alternative (str, what the agent should "
            "have done), impact_severity (critical|moderate|minor|none), reasoning (str), "
            "reward_signal (float, -1.0 to 1.0). Include ALL steps, not just erroneous "
            "ones. Correct steps should have verdict=correct, deviation_type=none, "
            "reward_signal between 0.0 and 1.0."
        ),
        "output_file": "codetracer_rl_feedback.json",
        "json_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["incorrect", "unuseful", "correct"]},
                    "deviation_type": {"type": "string"},
                    "correct_alternative": {"type": "string"},
                    "impact_severity": {"type": "string", "enum": ["critical", "moderate", "minor", "none"]},
                    "reasoning": {"type": "string"},
                    "reward_signal": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                },
                "required": ["step_id", "verdict", "deviation_type", "reasoning", "reward_signal"],
            },
        },
    },
}


def load_profile(name: str, config: dict[str, Any] | None = None) -> OutputProfile:
    """Load an output profile by *name*.

    Looks first in the config ``output.profiles`` section, then falls
    back to the built-in profiles.
    """
    if config:
        user_profiles = config.get("output", {}).get("profiles", {})
        if name in user_profiles:
            return OutputProfile.from_dict(name, user_profiles[name])

    if name in _BUILTIN_PROFILES:
        return OutputProfile.from_dict(name, _BUILTIN_PROFILES[name])

    raise ValueError(
        f"Unknown output profile '{name}'. "
        f"Available: {list(_BUILTIN_PROFILES.keys())}"
    )


def get_default_profile_name(config: dict[str, Any] | None = None) -> str:
    if config:
        return config.get("output", {}).get("default_profile", "detailed")
    return "detailed"
