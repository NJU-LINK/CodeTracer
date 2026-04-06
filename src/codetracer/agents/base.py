"""Base agent loop with progress events, compact, cost tracking, and structured finalize.

The core loop is exposed as a generator ``run_iter()`` that yields typed
``AgentEvent`` objects.  ``run()`` is a blocking wrapper for backward
compatibility.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from rich.console import Console

from codetracer.agents.compact import CompactManager
from codetracer.agents.executor import Executor
from codetracer.services.cost_tracker import CostTracker
from codetracer.llm.client import LLMClient
from codetracer.plugins.hooks import HookManager, default_hooks

console = Console(highlight=False)
logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"```bash\s*\n([\s\S]*?)\n?```")
_FINAL_SIGNAL = "TRACER_FINAL_OUTPUT"


class FormatError(Exception):
    pass


class Submitted(Exception):
    pass


class LimitsExceeded(Exception):
    pass


@dataclass
class AgentEvent:
    """Typed event yielded by the agent loop."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Shared interactive agent loop.

    ``run_iter()`` yields ``AgentEvent`` objects so callers (REPL, batch
    dispatcher) can observe progress, enforce timeouts, or cancel.
    ``run()`` is the legacy blocking wrapper.
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: Executor,
        config: dict[str, Any],
        *,
        hooks: HookManager | None = None,
        cost_tracker: CostTracker | None = None,
        compact_manager: CompactManager | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._cost_limit = float(config.get("cost_limit", 3.0))
        self._step_limit = int(config.get("step_limit", 0))
        self._action_regex = (
            re.compile(config["action_regex"]) if "action_regex" in config else _ACTION_RE
        )
        self._observation_template: str = config.get(
            "observation_template", _DEFAULT_OBS_TEMPLATE
        )
        self._format_error_template: str = config.get(
            "format_error_template", _DEFAULT_FMT_ERR
        )
        self._timeout_template: str = config.get(
            "timeout_template", _DEFAULT_TIMEOUT
        )
        self._messages: list[dict[str, Any]] = []
        self._abort = threading.Event()

        self._hooks = hooks or default_hooks
        self._cost_tracker = cost_tracker or CostTracker(budget_limit_usd=self._cost_limit)
        self._compact = compact_manager

        ctx_cfg = config.get("context_window", {})
        if self._compact is None and ctx_cfg.get("auto_compact", False):
            self._compact = CompactManager(
                context_window=int(ctx_cfg.get("context_window_size", 128_000)),
                buffer_tokens=int(ctx_cfg.get("buffer_tokens", 13_000)),
                max_failures=int(ctx_cfg.get("max_consecutive_failures", 3)),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, messages: list[dict[str, str]]) -> str:
        """Blocking wrapper around ``run_iter``."""
        result = ""
        for event in self.run_iter(messages):
            if event.type == "complete":
                result = event.data.get("output", "")
        return result

    def run_iter(
        self, messages: list[dict[str, str]]
    ) -> Generator[AgentEvent, None, str]:
        """Generator that drives the agent loop, yielding events."""
        self._messages = [
            {"role": m["role"], "content": m["content"], "timestamp": time.time()}
            for m in messages
        ]
        result = ""
        while not self._abort.is_set():
            try:
                for event in self._step_iter():
                    yield event
            except FormatError as e:
                self._add("user", str(e))
                yield AgentEvent("format_error", {"error": str(e)})
            except Submitted as e:
                result = str(e)
                yield AgentEvent("complete", {"output": result})
                return result
            except LimitsExceeded:
                result = "limits_exceeded"
                yield AgentEvent("limits_exceeded", {"cost": self._cost_tracker.total_cost})
                return result

        result = "aborted"
        yield AgentEvent("aborted", {})
        return result

    def abort(self) -> None:
        """Request graceful cancellation of the agent loop."""
        self._abort.set()

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def _step_iter(self) -> Generator[AgentEvent, None, None]:
        """Single agent step, yielding events."""
        if self._step_limit > 0 and self._llm.n_calls >= self._step_limit:
            raise LimitsExceeded()

        if self._cost_tracker.is_over_budget():
            raise LimitsExceeded()

        step_num = self._llm.n_calls + 1
        self._hooks.emit("step_start", step=step_num)
        yield AgentEvent("step_start", {"step": step_num})

        if self._compact and self._compact.should_compact(self._messages):
            self._hooks.emit("compact_triggered", step=step_num)
            yield AgentEvent("compact", {"step": step_num})
            self._messages = self._compact.compact(self._messages, self._llm)

        if self._cost_tracker.should_warn():
            warning = self._cost_tracker.budget_warning_message()
            self._add("user", warning)
            self._hooks.emit("budget_warning", message=warning)
            yield AgentEvent("budget_warning", {"message": warning})

        self._hooks.emit("llm_call_start", step=step_num)
        t0 = time.time()
        resp = self._llm.query(
            [{"role": m["role"], "content": m["content"]} for m in self._messages]
        )
        duration = time.time() - t0

        usage = resp.get("usage", {})
        model = self._llm.model_name or "unknown"
        self._cost_tracker.add_usage(
            model,
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            phase="trace",
            duration_s=duration,
        )
        self._hooks.emit(
            "llm_call_complete",
            step=step_num,
            cost=self._cost_tracker.total_cost,
            duration=duration,
        )

        self._add("assistant", resp["content"], usage=usage)
        preview = resp["content"][:500] + ("..." if len(resp["content"]) > 500 else "")
        console.print(f"\n[bold cyan]Step {step_num}[/bold cyan]")
        console.print(preview)
        yield AgentEvent("llm_response", {"step": step_num, "preview": preview})

        actions = self._action_regex.findall(resp["content"])
        if len(actions) != 1:
            raise FormatError(
                self._render(self._format_error_template, actions=actions)
            )

        cmd = actions[0].strip()
        yield AgentEvent("tool_exec", {"step": step_num, "command": cmd})
        try:
            output = self._executor.run(cmd)
        except TimeoutError as tout_output:
            raise FormatError(
                self._render(
                    self._timeout_template,
                    action={"action": cmd},
                    output=str(tout_output),
                )
            )

        observation = self._render(self._observation_template, output=output)
        self._add("user", observation)

        first_line = output.get("output", "").lstrip().splitlines()
        if first_line and first_line[0].strip() == _FINAL_SIGNAL:
            raw_output = "".join(
                output["output"].lstrip().splitlines(keepends=True)[1:]
            )
            self._hooks.emit("step_complete", step=step_num, final=True)
            yield AgentEvent("step_complete", {"step": step_num, "final": True})
            raise Submitted(raw_output)

        self._hooks.emit("step_complete", step=step_num, final=False)
        yield AgentEvent("step_complete", {"step": step_num, "final": False})

    # ------------------------------------------------------------------
    # Structured finalize
    # ------------------------------------------------------------------

    def finalize_structured(
        self,
        raw_output: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Optionally re-format *raw_output* using response_format json_object.

        If *json_schema* is provided, makes one final LLM call asking
        the model to produce valid JSON matching the schema.  If the raw
        output already parses as valid JSON, returns it directly.
        """
        if not json_schema:
            return raw_output

        try:
            json.loads(raw_output)
            return raw_output
        except (json.JSONDecodeError, ValueError):
            pass

        prompt = (
            "Convert the following analysis output into valid JSON matching "
            f"this schema:\n{json.dumps(json_schema, indent=2)}\n\n"
            f"Raw output:\n{raw_output}\n\n"
            "Return ONLY the JSON, no explanation."
        )
        try:
            resp = self._llm.query(
                [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = resp.get("content", "")
            json.loads(content)
            return content
        except Exception:
            logger.warning("Structured finalize failed; returning raw output")
            return raw_output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add(
        self, role: str, content: str, usage: dict[str, Any] | None = None
    ) -> None:
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if usage:
            entry["usage"] = usage
        self._messages.append(entry)

    @staticmethod
    def _render(template: str, **kwargs: Any) -> str:
        from codetracer.utils.template import render_template
        return render_template(template, **kwargs)

    def save_trajectory(self, path: Path) -> None:
        data = {
            "messages": self._messages,
            "n_calls": self._llm.n_calls,
            "total_prompt_tokens": self._llm.total_prompt_tokens,
            "total_completion_tokens": self._llm.total_completion_tokens,
            "cost_summary": self._cost_tracker.format_summary(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


_DEFAULT_OBS_TEMPLATE = """\
<returncode>{{ output.returncode }}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>Output too long.</warning>
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided>{{ output.output | length - 10000 }} chars elided</elided>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
"""

_DEFAULT_FMT_ERR = (
    "Please provide EXACTLY ONE bash code block. Found {{ actions | length }} blocks."
)

_DEFAULT_TIMEOUT = """\
Command timed out: {{ action['action'] }}
{% if output | length < 10000 %}Output: {{ output }}{% else %}Output (truncated): {{ output[:3000] }}{% endif %}
"""
