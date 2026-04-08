"""Claude Code Cast parser: extract steps from sessions/claude_code.log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codetracer.models import FileRef, NormalizedTrajectory, StepRecord


class ClaudeCodeCastParser:
    format_id = "claude_code_cast"

    def can_parse(self, run_dir: Path) -> bool:
        trial = _resolve_trial_dir(run_dir)
        log = trial / "sessions" / "claude_code.log"
        return log.exists() and (trial / "results.json").exists()

    def parse(self, run_dir: Path) -> NormalizedTrajectory:
        trial = _resolve_trial_dir(run_dir)
        log_path = trial / "sessions" / "claude_code.log"
        results_path = trial / "results.json"

        messages = _extract_last_full_conversation(log_path)
        steps = _messages_to_steps(messages, trial)

        if not steps:
            commands_path = trial / "commands.txt"
            if commands_path.exists():
                steps = _parse_commands_txt(commands_path)

        task = _read_task(results_path)
        metadata = _read_metadata(results_path)
        metadata["format"] = self.format_id
        metadata["run_dir"] = str(run_dir)

        return NormalizedTrajectory(
            steps=steps,
            task_description=task,
            metadata=metadata,
        )


def _resolve_trial_dir(run_dir: Path) -> Path:
    """Handle nested trial dirs like task_name/task_name.1-of-1.date/."""
    if (run_dir / "sessions" / "claude_code.log").exists():
        return run_dir

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        # Direct child with sessions/
        if (child / "sessions" / "claude_code.log").exists():
            return child
        # Two-level nesting: task_name/trial_dir/
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (grandchild / "sessions" / "claude_code.log").exists():
                return grandchild
    return run_dir


def _extract_last_full_conversation(log_path: Path) -> list[dict[str, Any]]:
    """Find the last LiteLLM request with tools containing Anthropic-format content.

    The log contains pairs of requests: one in Anthropic format (content is
    a list with tool_use blocks) and one in OpenAI format (content is a
    string, tool_use stripped). We want the Anthropic format since it
    preserves tool call structure.
    """
    best_messages: list[dict[str, Any]] = []
    best_count = 0

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if "messages" not in obj or "tools" not in obj:
                continue

            msgs = obj["messages"]
            if not isinstance(msgs, list):
                continue

            if len(msgs) < best_count:
                continue

            if not _has_tool_use_blocks(msgs):
                continue

            best_count = len(msgs)
            best_messages = msgs

    return best_messages


def _has_tool_use_blocks(msgs: list[dict[str, Any]]) -> bool:
    """Check whether any assistant message has list content with tool_use blocks."""
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    return False


def _messages_to_steps(
    messages: list[dict[str, Any]], trial_dir: Path
) -> list[StepRecord]:
    """Convert Claude API messages into StepRecord list.

    Each tool_use block in an assistant message becomes one step.
    The corresponding tool result (matched by tool_call_id) is the observation.
    """
    tool_results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            tid = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        parts.append(c.get("text", str(c)))
                    else:
                        parts.append(str(c))
                content = "\n".join(parts)
            tool_results[tid] = str(content)

    steps: list[StepRecord] = []
    step_id = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue

            step_id += 1
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            tool_id = block.get("id", "")

            action = _format_action(name, inp)
            observation = tool_results.get(tool_id)

            action_ref = FileRef(
                path="<claude_code.log>",
                line_start=step_id,
                line_end=step_id,
                content=json.dumps(block, ensure_ascii=False)[:2000],
            )
            obs_ref = None
            if observation is not None:
                obs_ref = FileRef(
                    path="<claude_code.log>",
                    line_start=step_id,
                    line_end=step_id,
                    content=observation[:2000],
                )

            steps.append(
                StepRecord(
                    step_id=step_id,
                    action=action,
                    observation=observation,
                    action_ref=action_ref,
                    observation_ref=obs_ref,
                )
            )

    return steps


def _format_action(tool_name: str, inp: dict[str, Any]) -> str:
    """Produce a human-readable action string from a tool_use block."""
    if tool_name == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        prefix = f"[Bash] {desc}: " if desc else "[Bash] "
        return prefix + cmd
    if tool_name == "Write":
        return f"[Write] {inp.get('file_path', '')}"
    if tool_name == "Edit":
        return f"[Edit] {inp.get('file_path', '')}"
    if tool_name == "Read":
        return f"[Read] {inp.get('file_path', '')}"
    if tool_name == "Glob":
        return f"[Glob] {inp.get('pattern', '')}"
    if tool_name == "Grep":
        return f"[Grep] {inp.get('pattern', '')} in {inp.get('path', '.')}"
    if tool_name in ("TodoWrite", "TodoRead"):
        return f"[{tool_name}]"
    if tool_name == "WebFetch":
        return f"[WebFetch] {inp.get('url', '')}"
    if tool_name == "Task":
        return f"[Task/{inp.get('subagent_type', '')}] {inp.get('description', '')}"
    if tool_name == "BashOutput":
        return f"[BashOutput] {inp.get('bash_id', '')}"
    return f"[{tool_name}] {json.dumps(inp, ensure_ascii=False)[:200]}"


def _parse_commands_txt(commands_path: Path) -> list[StepRecord]:
    """Fallback: extract steps from commands.txt (tmux send-keys log).

    Each line is a Python repr of a list like ['command text', 'Enter'].
    We extract the actual command strings, ignoring control sequences.
    """
    import ast

    steps: list[StepRecord] = []
    step_id = 0
    skip_prefixes = (
        "asciinema rec", "clear", "source /installed-agent",
        "bash ", "C-d", "C-c",
    )

    try:
        text = commands_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return steps

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = ast.literal_eval(line)
        except Exception:
            continue

        if not isinstance(parts, list) or not parts:
            continue

        cmd = parts[0].strip()
        if not cmd or cmd in ("Enter",) or any(cmd.startswith(p) for p in skip_prefixes):
            continue

        step_id += 1
        steps.append(
            StepRecord(
                step_id=step_id,
                action=f"[tmux] {cmd}",
                observation=None,
                action_ref=FileRef(
                    path=str(commands_path.name),
                    line_start=step_id,
                    line_end=step_id,
                    content=line,
                ),
            )
        )

    return steps


def _read_task(results_path: Path) -> str:
    if not results_path.exists():
        return ""
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        return data.get("instruction", "")
    except Exception:
        return ""


def _read_metadata(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        return {
            "trial_name": data.get("trial_name", ""),
            "task_id": data.get("task_id", ""),
            "is_resolved": data.get("is_resolved"),
            "parser_results": data.get("parser_results", {}),
        }
    except Exception:
        return {}


parser = ClaudeCodeCastParser()
