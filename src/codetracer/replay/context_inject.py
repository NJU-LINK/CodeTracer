"""ContextInjector: compose LLM messages with error analysis at a breakpoint step."""

from __future__ import annotations

from typing import Any

from codetracer.models import ErrorAnalysis, NormalizedTrajectory, StepCheckpoint, StepVerdict
from codetracer.utils.template import render_template

_DEFAULT_BREAKPOINT_TEMPLATE = """\
You are resuming an agent trajectory that was previously analyzed for errors.

== Trajectory Context ==
Task: {{ task_description }}
{% if problem_statement %}
== Problem Statement ==
{{ problem_statement }}
{% endif %}
{% if exploration_instructions %}
== Exploration Instructions ==
{{ exploration_instructions }}
{% endif %}

Steps executed so far (1 to {{ preceding_count }}):
{% for s in preceding_steps -%}
[Step {{ s.step_id }}] {{ s.action[:120] }}
{% endfor %}

== Error Analysis at Step {{ target_step_id }} ==
{{ analysis_summary }}

{% for label in error_labels -%}
- Step {{ label.step_id }} ({{ label.verdict }}): {{ label.reasoning }}
{% endfor %}

== Instructions ==
The trajectory went wrong at step {{ target_step_id }}.
You are now at that exact point. The environment has been restored to
the state right before step {{ target_step_id }} was executed.

Review the error analysis above and continue from here with a corrected approach.
Do NOT repeat the commands from steps 1-{{ preceding_count }}; they have already
been replayed for you.
"""

_DEFAULT_SYSTEM_TEMPLATE = """\
You are CodeTracer Replay Agent. You are resuming a failed agent run at a
specific breakpoint where errors were detected. Your job is to continue
the task using a corrected strategy informed by the error analysis provided.

Respond with exactly ONE bash code block per message:
```bash
<single command>
```
"""


class ContextInjector:
    """Builds LLM message lists with error analysis injected at a breakpoint."""

    def __init__(
        self,
        system_template: str = _DEFAULT_SYSTEM_TEMPLATE,
        breakpoint_template: str = _DEFAULT_BREAKPOINT_TEMPLATE,
    ) -> None:
        self._system_tpl = system_template
        self._breakpoint_tpl = breakpoint_template

    def build_messages(
        self,
        traj: NormalizedTrajectory,
        checkpoint: StepCheckpoint,
        extra_vars: dict[str, Any] | None = None,
        task_ctx: Any | None = None,
    ) -> list[dict[str, str]]:
        """Build [system, user] messages for the replay agent.

        The user message contains:
        - summary of preceding steps
        - full error analysis at the breakpoint
        - instructions to continue from the breakpoint
        - optional TaskContext (problem statement, exploration instructions)
        """
        analysis = checkpoint.error_analysis
        preceding = checkpoint.replayed_steps

        tpl_vars: dict[str, Any] = {
            "task_description": traj.task_description or "(no task description)",
            "preceding_count": len(preceding),
            "preceding_steps": [{"step_id": s.step_id, "action": s.action} for s in preceding],
            "target_step_id": checkpoint.target_step_id,
            "analysis_summary": analysis.summary if analysis else "",
            "error_labels": [l.to_dict() for l in analysis.labels] if analysis else [],
            "problem_statement": "",
            "exploration_instructions": "",
        }

        if task_ctx is not None:
            tpl_vars["problem_statement"] = getattr(task_ctx, "problem_statement", "") or ""
            tpl_vars["exploration_instructions"] = getattr(task_ctx, "exploration_instructions", "") or ""

        if extra_vars:
            tpl_vars.update(extra_vars)

        system = self._render(self._system_tpl, **tpl_vars)
        user = self._render(self._breakpoint_tpl, **tpl_vars)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def build_analysis_context(self, analysis: ErrorAnalysis) -> str:
        """Render a compact error-analysis block suitable for injection into
        an existing agent conversation (plugin / template use-case)."""
        lines = [f"== CodeTracer Error Analysis (traj: {analysis.traj_id}) =="]
        if analysis.summary:
            lines.append(analysis.summary)
        for label in analysis.labels:
            lines.append(f"- Step {label.step_id} [{label.verdict.value}]: {label.reasoning}")
        return "\n".join(lines)

    @staticmethod
    def _render(template: str, **kwargs: Any) -> str:
        return render_template(template, **kwargs)
