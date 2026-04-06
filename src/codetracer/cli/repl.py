"""Interactive REPL for CodeTracer with a styled terminal UI.

Slash-command system (type / for an arrow-key selectable menu), free-text LLM
chat, and a welcome banner with a filled woodpecker mascot.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

_VERSION = "0.2.0"

# -- Color palette constants (Rich 256-color) --------------------------------
_C_PRIMARY = "color(75)"       # steel blue  -- commands, prompt, accents
_C_SECONDARY = "color(117)"    # sky blue    -- titles, headers
_C_BORDER = "color(69)"        # medium blue -- panel / box borders
_C_MUTED = "color(153)"        # pale blue   -- secondary text
_C_WARM = "color(215)"         # warm orange -- crest, beak accent

_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/errors",  "Show diagnosed errors (runs analyze if needed)"),
    ("/cause",   "LLM deep-dive into why a specific step failed"),
    ("/replay",  "Auto-replay from the first incorrect step"),
    ("/inspect", "Show action/observation for a step"),
    ("/tree",    "Display the trajectory tree index"),
    ("/export",  "Write codetracer_labels.json"),
    ("/run",     "One-click: analyze + errors + auto-replay"),
    ("/chat",    "Enter conversational chat about the trajectory"),
    ("/status",  "Show current session info"),
    ("/help",    "Show available commands"),
    ("/quit",    "Exit CodeTracer"),
]


# ---------------------------------------------------------------------------
# Session state -- centralized via SessionStore
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smart directory auto-detect & trajectory scanner
# ---------------------------------------------------------------------------

def _detect_format(d: Path, normalizer: Any, pool: Any) -> str | None:
    """Return format name if *d* is a recognized trajectory, else None."""
    if normalizer.is_pre_normalized(d):
        return "pre_normalized"
    if normalizer.is_step_jsonl_dir(d):
        return "step_jsonl"
    return pool.detect(d)


def _auto_detect_directory(
    normalizer: Any,
    pool: Any,
) -> Path | None:
    """Silently scan cwd for the first matching trajectory and return its path."""
    root = Path.cwd()

    fmt = _detect_format(root, normalizer, pool)
    if fmt is not None:
        return root

    with console.status(
        "[dim]Scanning for trajectories...[/]",
        spinner="dots", spinner_style="dim",
    ):
        result = _find_first_trajectory(root, normalizer, pool, max_depth=4)

    if result is not None:
        return result

    console.print(f"\n[dim]No recognized trajectories under {root}[/]")
    try:
        raw = console.input(f"[{_C_MUTED}]Trajectory directory:[/] ").strip()
        if not raw:
            return None
        p = Path(raw).expanduser().resolve()
        if not p.is_dir():
            console.print(f"[red]Not a directory: {p}[/]")
            return None
        return p
    except (EOFError, KeyboardInterrupt):
        return None


def _find_first_trajectory(
    root: Path,
    normalizer: Any,
    pool: Any,
    max_depth: int = 4,
) -> Path | None:
    """Depth-first walk returning the first directory matching a known format."""
    seen: set[Path] = set()

    def _walk(d: Path, depth: int) -> Path | None:
        if depth > max_depth:
            return None
        d = d.resolve()
        if d in seen:
            return None
        seen.add(d)

        if _detect_format(d, normalizer, pool) is not None:
            return d

        try:
            children = sorted(
                (c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")),
                key=lambda p: p.name,
            )
        except PermissionError:
            return None
        for child in children:
            hit = _walk(child, depth + 1)
            if hit is not None:
                return hit
        return None

    return _walk(root, 0)


def _read_input() -> str:
    """Read input char-by-char; triggers slash menu instantly when '/' is the first key."""
    if not sys.stdin.isatty():
        return console.input(f"\n[bold {_C_PRIMARY}]>[/] ")
    try:
        import readchar
    except ImportError:
        return console.input(f"\n[bold {_C_PRIMARY}]>[/] ")

    console.print(f"\n[bold {_C_PRIMARY}]>[/] ", end="")
    buf: list[str] = []
    while True:
        ch = readchar.readkey()
        if ch == "/" and not buf:
            sys.stdout.write("/\n")
            sys.stdout.flush()
            return "/"
        if ch in ("\r", "\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(buf)
        if ch == "\x03":
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        if ch in ("\x7f", "\x08"):
            if buf:
                buf.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if len(ch) == 1 and ch.isprintable():
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interactive_repl(
    run_dir: Path | None = None,
    *,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    config_path: Path | None = None,
    resume: bool = False,
    profile: str | None = None,
) -> None:
    """Enter the interactive REPL. If *run_dir* is None, prompt for it."""
    from platformdirs import user_config_dir

    from codetracer.query.config import load_config
    from codetracer.query.normalizer import Normalizer
    from codetracer.services.session_persistence import SessionPersistence
    from codetracer.skills.pool import SkillPool
    from codetracer.state.session import SessionState, SessionStore

    config = load_config(config_path)
    user_skill_dir = Path(user_config_dir("codetracer")) / "skills"
    user_skill_dir.mkdir(parents=True, exist_ok=True)
    pool = SkillPool(user_dir=user_skill_dir)
    normalizer = Normalizer(pool)

    if run_dir is None:
        run_dir = _auto_detect_directory(normalizer, pool)
        if run_dir is None:
            console.print("[dim]No directory selected. Exiting.[/]")
            return

    persistence = SessionPersistence(run_dir / ".codetracer_session")

    if resume and persistence.has_session():
        console.print(f"[{_C_MUTED}]Resuming previous session...[/]")
        restored = persistence.restore_messages()
        if restored:
            console.print(f"[{_C_MUTED}]  Restored {len(restored)} messages.[/]")
    else:
        restored = []

    llm = _make_llm(config, model=model, api_base=api_base, api_key=api_key)

    skill = None
    traj = None
    with console.status("[dim]Initializing...[/]", spinner="dots", spinner_style="dim"):
        if normalizer.is_pre_normalized(run_dir):
            traj = normalizer.normalize_pre_normalized(run_dir, quiet=True)
        elif normalizer.is_step_jsonl_dir(run_dir):
            traj = normalizer.normalize_step_jsonl(run_dir, quiet=True)
        else:
            try:
                skill = normalizer.detect(run_dir)
            except ValueError:
                pass
            if skill is not None:
                traj = normalizer.normalize(run_dir, skill, quiet=True)

    if traj is None and skill is None:
        traj = _discover_and_normalize(
            run_dir, normalizer, pool, user_skill_dir, llm, config,
        )
        if traj is None:
            return

    fmt_name = skill.name if skill else traj.metadata.get("format", "pre-normalized")

    store = SessionStore(SessionState(
        mode="interactive",
        profile=profile or "detailed",
        run_dir=run_dir,
        traj=traj,
        skill=skill,
        llm=llm,
        config=config,
        fmt_name=fmt_name,
        analysis=_try_load_analysis(run_dir),
    ))

    if restored:
        store.set_state(lambda s: s.copy(
            chat_messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in restored],
        ))

    def _on_state_change() -> None:
        state = store.get_state()
        cur_msgs = state.chat_messages
        if len(cur_msgs) > _on_state_change._prev_count:  # type: ignore[attr-defined]
            new_msgs = cur_msgs[_on_state_change._prev_count:]  # type: ignore[attr-defined]
            persistence.record_messages(new_msgs)
            _on_state_change._prev_count = len(cur_msgs)  # type: ignore[attr-defined]
    _on_state_change._prev_count = len(store.get_state().chat_messages)  # type: ignore[attr-defined]
    store.subscribe(_on_state_change)

    _print_welcome(store)

    last_interrupt = 0.0
    while True:
        try:
            raw = _read_input().strip()

            if not raw:
                continue

            if raw == "/":
                chosen = _interactive_slash_menu()
                if chosen is None:
                    continue
                if chosen == "/quit":
                    console.print("[dim]Goodbye.[/dim]")
                    break
                _dispatch_slash(store, chosen)
                continue

            low = raw.lower()
            if low in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye.[/dim]")
                break

            if low.startswith("/"):
                _dispatch_slash(store, raw)
            else:
                _chat_once(store, raw)

            last_interrupt = 0.0

        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_interrupt < 1.5:
                console.print("\n[dim]Goodbye.[/dim]")
                break
            last_interrupt = now
            console.print(
                f"\n  [dim]Interrupted \u00b7 What should CodeTracer do instead?[/]"
            )
            continue


# ---------------------------------------------------------------------------
# Welcome banner with filled woodpecker mascot
# ---------------------------------------------------------------------------

def _print_welcome(store: Any) -> None:
    s = store.get_state()
    console.print()

    C1 = _C_PRIMARY
    C2 = _C_WARM
    C3 = _C_BORDER
    C4 = _C_MUTED

    mascot_lines = [
        f"   [{C2}]\u2584\u2584[/]",
        f"  [{C2}]\u2588\u2588[/][{C1}]\u2584\u2584[/]",
        f" [{C1}]\u2588\u2588[/][black on {C1}]\u25cf[/][{C1}]\u2588\u2588\u2588[/]",
        f" [{C1}]\u2588\u2588\u2588\u2588\u2588[/][{C2}]\u2580\u2580\u25ba[/]",
        f"  [{C1}]\u2588\u2588[/][{C3}]\u2588\u2588[/][{C1}]\u2588\u2588[/]",
        f" [{C1}]\u2588\u2588[/][{C3}]\u2588\u2588\u2588[/][{C1}]\u2588\u2588[/]",
        f"[{C3}]\u2588\u2588\u2588\u2588\u2588[/][{C1}]\u2588\u2588[/]",
        f"[{C3}]\u2588\u2588\u2588[/] [{C4}]\u2588 \u2588[/]",
        f"    [{C4}]\u2580 \u2580[/]",
    ]
    mascot = Text.from_markup("\n".join(mascot_lines))

    n_errors = 0
    if s.analysis and hasattr(s.analysis, "labels"):
        n_errors = len(s.analysis.labels)
    recent_text = (
        f"[bold red]{n_errors} errors detected[/]" if n_errors
        else "[dim]No recent activity[/]"
    )

    left_lines = [
        "",
        "[bold]Welcome back![/]",
        f"[bold {_C_WARM}]Tips for getting started[/]",
        f"Type [bold {_C_PRIMARY}]/[/] to see commands",
        "Or ask a question in natural language",
    ]
    left = Text.from_markup("\n".join(left_lines))

    right_lines = [
        "",
        "",
        f"[bold {_C_SECONDARY}]Recent activity[/]",
        recent_text,
        "",
    ]
    right = Text.from_markup("\n".join(right_lines))

    cols = Columns([mascot, left, right], padding=(0, 3))

    console.print(Panel(
        cols,
        title=f"[bold {_C_SECONDARY}] CodeTracer v{_VERSION} [/]",
        border_style=_C_BORDER,
        padding=(0, 2),
        box=box.ROUNDED,
    ))

    llm_label = "[bold green]connected[/]" if s.llm else "[dim]not configured[/]"
    error_label = (
        f"[bold red]{n_errors} errors[/]" if n_errors
        else "[green]clean[/]" if s.analysis
        else "[dim]not analyzed[/]"
    )

    console.print(
        f"  [bold {_C_PRIMARY}]{s.fmt_name}[/] "
        f"[dim]\u00b7[/] {s.traj.step_count} steps "
        f"[dim]\u00b7[/] LLM {llm_label} "
        f"[dim]\u00b7[/] {error_label}"
    )
    console.print(f"  [dim]{s.run_dir}[/]")


# ---------------------------------------------------------------------------
# Interactive slash-command menu (readchar-based arrow-key list)
# ---------------------------------------------------------------------------

def _interactive_slash_menu() -> str | None:
    """Arrow-key navigable command list. Bold highlight on current item."""
    if not sys.stdin.isatty():
        return _static_slash_menu_fallback()
    try:
        import readchar
    except ImportError:
        return _static_slash_menu_fallback()

    items = list(_SLASH_COMMANDS)
    idx = 0
    n = len(items)

    def _render(selected: int) -> None:
        lines: list[str] = []
        for i, (cmd, desc) in enumerate(items):
            if i == selected:
                lines.append(f"  [bold {_C_PRIMARY}]\u276f {cmd:10s}[/] [bold]{desc}[/]")
            else:
                lines.append(f"    [{_C_MUTED}]{cmd:10s}[/] [dim]{desc}[/]")
        console.print("\n".join(lines))

    _render(idx)

    while True:
        key = readchar.readkey()
        if key == readchar.key.UP:
            idx = (idx - 1) % n
        elif key == readchar.key.DOWN:
            idx = (idx + 1) % n
        elif key in ("\r", "\n", readchar.key.ENTER):
            return items[idx][0]
        elif key in ("\x1b", "\x03"):
            return None
        else:
            continue
        # move cursor up N lines, clear, and re-render
        sys.stdout.write(f"\033[{n}A\033[J")
        sys.stdout.flush()
        _render(idx)


def _interactive_list(choices: list[tuple[str, str]], title: str = "") -> str | None:
    """Generic arrow-key list picker. choices = [(display, value), ...]. Returns value."""
    if not sys.stdin.isatty():
        return None
    try:
        import readchar
    except ImportError:
        return None

    items = list(choices)
    idx = 0
    n = len(items)
    if n == 0:
        return None

    def _render(selected: int) -> None:
        lines: list[str] = []
        for i, (display, _val) in enumerate(items):
            if i == selected:
                lines.append(f"  [bold {_C_PRIMARY}]\u276f[/] [bold]{display}[/]")
            else:
                lines.append(f"    [{_C_MUTED}]{display}[/]")
        console.print("\n".join(lines))

    if title:
        console.print(f"\n[bold {_C_SECONDARY}]{title}[/]")
    _render(idx)

    while True:
        key = readchar.readkey()
        if key == readchar.key.UP:
            idx = (idx - 1) % n
        elif key == readchar.key.DOWN:
            idx = (idx + 1) % n
        elif key in ("\r", "\n", readchar.key.ENTER):
            return items[idx][1]
        elif key in ("\x1b", "\x03"):
            return None
        else:
            continue
        sys.stdout.write(f"\033[{n}A\033[J")
        sys.stdout.flush()
        _render(idx)


def _static_slash_menu_fallback() -> str | None:
    """Fallback for non-TTY: print a static table."""
    table = Table(show_header=False, box=None, padding=(0, 3), pad_edge=True)
    table.add_column(style=f"bold {_C_PRIMARY}", width=14)
    table.add_column(style="dim")
    for cmd, desc in _SLASH_COMMANDS:
        table.add_row(cmd, desc)
    console.print()
    console.print(table)
    return None


def _dispatch_slash(store: Any, raw: str) -> None:
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    handlers: dict[str, Any] = {
        "/errors": lambda: _action_show_errors(store),
        "/cause": lambda: _action_root_cause(store, arg),
        "/replay": lambda: _action_replay(store),
        "/inspect": lambda: _action_inspect(store, arg),
        "/tree": lambda: _action_tree(store),
        "/export": lambda: _action_export(store),
        "/run": lambda: _action_full_run(store),
        "/chat": lambda: _action_chat(store),
        "/status": lambda: _action_status(store),
        "/help": lambda: _interactive_slash_menu(),
    }

    handler = handlers.get(cmd)
    if handler is None:
        console.print(
            f"[dim]Unknown command: {cmd}. "
            f"Type [bold {_C_PRIMARY}]/[/] to see available commands.[/]"
        )
    else:
        handler()


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _action_show_errors(store: Any) -> None:
    s = store.get_state()
    if s.analysis is None:
        if s.llm is None:
            console.print("[red]No error analysis found and no LLM configured.[/]")
            console.print("[dim]Pass --model/--api-base/--api-key or run 'codetracer analyze' first.[/]")
            return
        console.print("[yellow]No existing analysis. Running analysis now...[/]")
        analysis = _run_analyze(store)
        if analysis is None:
            return
        store.set_state(lambda st: st.copy(analysis=analysis))
        s = store.get_state()

    from codetracer.models import ErrorAnalysis
    assert isinstance(s.analysis, ErrorAnalysis)

    if not s.analysis.labels:
        console.print("[green]No errors found -- trajectory looks clean.[/]")
        return

    table = Table(
        title=f"[bold {_C_SECONDARY}]Error Analysis[/]",
        show_lines=True,
        border_style=_C_BORDER,
    )
    table.add_column("Step", style=_C_PRIMARY, width=6)
    table.add_column("Verdict", width=12)
    table.add_column("Reasoning", max_width=80)
    for label in s.analysis.labels:
        style = "red" if label.verdict.value == "incorrect" else "yellow"
        table.add_row(
            str(label.step_id),
            f"[{style}]{label.verdict.value}[/{style}]",
            label.reasoning[:120],
        )
    console.print(table)


def _action_root_cause(store: Any, arg: str = "") -> None:
    s = store.get_state()
    if s.llm is None:
        console.print("[red]LLM not configured. Pass --model/--api-base/--api-key.[/]")
        return

    sid: int | None = None
    if arg:
        try:
            sid = int(arg)
        except ValueError:
            pass
    if sid is None:
        try:
            sid = int(console.input(f"  [{_C_MUTED}]Step ID to investigate:[/] ").strip())
        except (ValueError, EOFError):
            console.print("[dim]Cancelled.[/]")
            return

    step = next((st for st in s.traj.steps if st.step_id == sid), None)
    if step is None:
        console.print(f"[red]Step {sid} not found.[/]")
        return

    console.print(f"[dim]Analyzing root cause of step {sid}...[/]")

    context_steps = "\n".join(
        f"[Step {st.step_id}] {(st.action or '')[:150]}"
        for st in s.traj.steps
    )
    analysis_info = ""
    if s.analysis and s.analysis.labels:
        analysis_info = "\n".join(
            f"- Step {l.step_id} ({l.verdict.value}): {l.reasoning}"
            for l in s.analysis.labels
        )

    analysis_block = ("Existing error labels:\n" + analysis_info + "\n\n") if analysis_info else ""

    messages = [
        {"role": "system", "content": (
            "You are CodeTracer, a trajectory diagnosis expert. "
            "Analyze the given step from an agent trajectory and explain the root cause of the error. "
            "Be specific about what went wrong and why. Provide actionable suggestions."
        )},
        {"role": "user", "content": (
            f"Trajectory overview ({s.traj.step_count} steps):\n{context_steps}\n\n"
            f"{analysis_block}"
            f"Step {sid} details:\n"
            f"Action: {step.action or '(empty)'}\n"
            f"Observation: {(step.observation or '(none)')[:3000]}\n\n"
            f"Explain the root cause of the error at step {sid}. "
            f"What went wrong? Why? What should the agent have done instead?"
        )},
    ]

    resp = s.llm.query(messages)
    console.print()
    console.print(Panel(
        Markdown(resp["content"]),
        title=f"[bold {_C_SECONDARY}]Root Cause -- Step {sid}[/]",
        border_style=_C_BORDER,
    ))


def _action_replay(store: Any) -> None:
    s = store.get_state()
    from codetracer.models import ErrorAnalysis
    from codetracer.replay.engine import ReplayEngine

    if not isinstance(s.analysis, ErrorAnalysis) or not s.analysis.labels:
        console.print("[red]No error analysis available. Run /errors first.[/]")
        return

    target = s.analysis.first_incorrect_step_id
    if target is None:
        console.print("[green]No incorrect steps to replay from.[/]")
        return

    console.print(f"[bold]Replaying from step {target}...[/]")
    engine = ReplayEngine(config=s.config, checkpoints_dir=s.run_dir / ".checkpoints")
    result = engine.replay_auto(
        s.traj, s.analysis,
        env_config={"work_dir": str(s.run_dir)},
        llm=s.llm,
    )
    console.print(f"\nReplay: [bold]{result.status.value}[/] ({result.steps_replayed} steps)")


def _action_inspect(store: Any, arg: str = "") -> None:
    s = store.get_state()
    sid: int | None = None
    if arg:
        try:
            sid = int(arg)
        except ValueError:
            pass
    if sid is None:
        try:
            sid = int(console.input(f"  [{_C_MUTED}]Step ID:[/] ").strip())
        except (ValueError, EOFError):
            console.print("[dim]Cancelled.[/]")
            return

    step = next((st for st in s.traj.steps if st.step_id == sid), None)
    if step is None:
        rng = f"{s.traj.steps[0].step_id}-{s.traj.steps[-1].step_id}" if s.traj.steps else "?"
        console.print(f"[red]Step {sid} not found (range: {rng}).[/]")
        return

    console.print()
    console.print(Panel(
        f"[bold]Action:[/]\n{step.action or '(empty)'}\n\n"
        f"[bold]Observation:[/]\n{(step.observation or '(none)')[:3000]}",
        title=f"[bold {_C_SECONDARY}]Step {step.step_id}[/]",
        border_style=_C_BORDER,
    ))


def _action_tree(store: Any) -> None:
    s = store.get_state()
    tree_path = s.run_dir / "tree.md"
    if tree_path.exists():
        console.print(tree_path.read_text(encoding="utf-8"))
    else:
        from codetracer.query.tree_builder import TreeBuilder
        tree_md = TreeBuilder().build(s.traj)
        console.print(tree_md)


def _action_export(store: Any) -> None:
    s = store.get_state()
    from codetracer.models import ErrorAnalysis

    if not isinstance(s.analysis, ErrorAnalysis) or not s.analysis.labels:
        console.print("[yellow]No labels to export. Run /errors first.[/]")
        return

    out_path = s.run_dir / "codetracer_labels.json"

    stages: dict[int, dict[str, Any]] = {}
    for label in s.analysis.labels:
        key = label.step_id
        if key not in stages:
            stages[key] = {"stage_id": key, "incorrect_step_ids": [], "unuseful_step_ids": [], "reasoning": ""}
        bucket = "incorrect_step_ids" if label.verdict.value == "incorrect" else "unuseful_step_ids"
        stages[key][bucket].append(label.step_id)
        if label.reasoning and not stages[key]["reasoning"]:
            stages[key]["reasoning"] = label.reasoning

    out_path.write_text(json.dumps(list(stages.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Labels exported to {out_path}[/]")


def _action_full_run(store: Any) -> None:
    s = store.get_state()
    if s.llm is None:
        console.print("[red]LLM not configured. Pass --model/--api-base/--api-key.[/]")
        return

    with console.status("[dim]Analyzing trajectory...[/]", spinner="dots", spinner_style="dim"):
        analysis = _run_analyze(store)
    if analysis is None:
        return
    store.set_state(lambda st: st.copy(analysis=analysis))

    console.print(f"[bold {_C_SECONDARY}]Step 2/3:[/] Error summary")
    _action_show_errors(store)

    s = store.get_state()
    from codetracer.models import ErrorAnalysis
    if not isinstance(s.analysis, ErrorAnalysis) or not s.analysis.labels:
        console.print("[green]No errors -- nothing to replay.[/]")
        return

    target = s.analysis.first_incorrect_step_id
    if target is None:
        return

    with console.status(f"[dim]Replaying from step {target}...[/]", spinner="dots", spinner_style="dim"):
        _action_replay(store)


def _action_chat(store: Any) -> None:
    s = store.get_state()
    if s.llm is None:
        console.print("[red]LLM not configured. Pass --model/--api-base/--api-key.[/]")
        return

    console.print("[dim]Chat mode. Type /back to return.[/]")

    if not s.chat_messages:
        store.set_state(lambda st: st.copy(
            chat_messages=[{"role": "system", "content": _build_chat_system_prompt(store)}],
        ))

    while True:
        try:
            question = console.input(f"\n[bold {_C_SECONDARY}]chat>[/] ").strip()
        except EOFError:
            break

        if not question:
            continue
        if question.lower() in ("/back", "/menu", "/quit", "back", "menu"):
            break

        s = store.get_state()
        new_msgs = list(s.chat_messages) + [{"role": "user", "content": question}]
        store.set_state(lambda st: st.copy(chat_messages=new_msgs))

        resp = s.llm.query(new_msgs)
        content = resp["content"]
        store.set_state(lambda st: st.copy(
            chat_messages=list(st.chat_messages) + [{"role": "assistant", "content": content}],
        ))
        console.print()
        console.print(Markdown(content))


def _action_status(store: Any) -> None:
    s = store.get_state()
    llm_label = "[bold green]connected[/]" if s.llm else "[dim]not configured[/]"
    n_errors = 0
    if s.analysis and hasattr(s.analysis, "labels"):
        n_errors = len(s.analysis.labels)
    error_label = (
        f"[bold red]{n_errors} errors[/]" if n_errors
        else "[green]clean[/]" if s.analysis
        else "[dim]not analyzed[/]"
    )

    console.print(Panel(
        f"[bold]Directory:[/]  {s.run_dir}\n"
        f"[bold]Format:[/]     [{_C_PRIMARY}]{s.fmt_name}[/]\n"
        f"[bold]Steps:[/]      {s.traj.step_count}\n"
        f"[bold]LLM:[/]        {llm_label}\n"
        f"[bold]Analysis:[/]   {error_label}",
        title=f"[bold {_C_SECONDARY}]Session Status[/]",
        border_style=_C_BORDER,
    ))


# ---------------------------------------------------------------------------
# Skill discovery for unknown formats
# ---------------------------------------------------------------------------

def _discover_and_normalize(
    run_dir: Path,
    normalizer: Any,
    pool: Any,
    user_skill_dir: Path,
    llm: Any | None,
    config: dict[str, Any],
) -> Any:
    """Handle unknown trajectory format: show directory listing, invoke SkillGenerator, normalize."""
    from codetracer.utils.llm_generator import list_dir, sample_files
    from codetracer.skills.generator import SkillGenerator

    console.print(f"\n[yellow]Unknown trajectory format in [bold]{run_dir}[/bold][/]")

    listing = list_dir(run_dir)
    listing_lines = listing.splitlines()
    n_files = len(listing_lines)
    preview = "\n".join(f"  {l}" for l in listing_lines[:20])
    if n_files > 20:
        preview += f"\n  [dim]... and {n_files - 20} more files[/]"
    console.print(Panel(
        preview,
        title=f"[bold {_C_SECONDARY}]Directory listing ({n_files} files)[/]",
        border_style=_C_BORDER,
    ))

    samples = sample_files(run_dir, listing)
    if samples and samples != "(no files sampled)":
        sample_lines = samples[:2000]
        if len(samples) > 2000:
            sample_lines += "\n..."
        console.print(Panel(
            sample_lines,
            title=f"[bold {_C_SECONDARY}]Sample file contents[/]",
            border_style=_C_BORDER,
        ))

    if llm is None:
        console.print(
            f"[red]No LLM configured -- cannot auto-generate a parser.[/]\n"
            f"[dim]Pass --model/--api-base/--api-key to enable skill generation,[/]\n"
            f"[dim]or set OPENAI_BASE_URL and OPENAI_API_KEY environment variables.[/]"
        )
        return None

    console.print(f"\n[bold {_C_SECONDARY}]Generating parser via LLM...[/]")
    generator = SkillGenerator(llm, pool, config.get("discovery", {}))
    try:
        with console.status("[dim]LLM is analyzing the directory structure...[/]", spinner="dots", spinner_style="dim"):
            skill = generator.generate(run_dir, user_skill_dir)
        console.print(f"[green]Generated skill:[/] [bold]{skill.name}[/]")
        console.print(f"[dim]Saved to {user_skill_dir / skill.name}[/]")
        traj = normalizer.normalize(run_dir, skill)
        return traj
    except RuntimeError as exc:
        console.print(f"[red]Skill generation failed: {exc}[/]")
        console.print("[dim]You can manually create a skill or provide a pre-normalized directory.[/]")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_once(store: Any, question: str) -> None:
    s = store.get_state()
    if s.llm is None:
        console.print("[red]LLM not configured. Pass --model/--api-base/--api-key to chat.[/]")
        return

    if not s.chat_messages:
        store.set_state(lambda st: st.copy(
            chat_messages=[{"role": "system", "content": _build_chat_system_prompt(store)}],
        ))
        s = store.get_state()

    new_msgs = list(s.chat_messages) + [{"role": "user", "content": question}]
    store.set_state(lambda st: st.copy(chat_messages=new_msgs))

    resp = s.llm.query(new_msgs)
    content = resp["content"]
    store.set_state(lambda st: st.copy(
        chat_messages=list(st.chat_messages) + [{"role": "assistant", "content": content}],
    ))
    console.print()
    console.print(Markdown(content))


def _build_chat_system_prompt(store: Any) -> str:
    s = store.get_state()
    step_summary = "\n".join(
        f"[Step {st.step_id}] {(st.action or '')[:120]}"
        for st in s.traj.steps
    )

    analysis_info = ""
    if s.analysis and hasattr(s.analysis, "labels") and s.analysis.labels:
        analysis_info = "Known errors:\n" + "\n".join(
            f"- Step {l.step_id} ({l.verdict.value}): {l.reasoning}"
            for l in s.analysis.labels
        )

    stages_info = ""
    stage_path = s.run_dir / "stage_ranges.json"
    if stage_path.exists():
        stages_info = f"Stage ranges:\n{stage_path.read_text(encoding='utf-8')[:2000]}"

    return (
        "You are CodeTracer, a trajectory diagnosis assistant. "
        "You help users understand agent trajectory errors, suggest fixes, "
        "and explain what happened at each step.\n\n"
        f"Trajectory: {s.traj.step_count} steps\n"
        f"Task: {s.traj.task_description[:500] if s.traj.task_description else '(unknown)'}\n\n"
        f"Step summary:\n{step_summary}\n\n"
        f"{analysis_info}\n\n"
        f"{stages_info}\n\n"
        "Answer the user's questions about this trajectory. Be concise and specific. "
        "When referencing steps, use the step IDs. "
        "If the user asks about a specific step, include its action and observation in your analysis."
    )


def _make_llm(config: dict, *, model: str | None, api_base: str | None, api_key: str | None) -> Any:
    """Create an LLMClient if credentials are available, otherwise return None."""
    llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
    if model:
        llm_cfg["model_name"] = model
    if api_base:
        llm_cfg["api_base"] = api_base
    if api_key:
        llm_cfg["api_key"] = api_key

    has_base = llm_cfg.get("api_base") or os.getenv("CODETRACER_API_BASE") or os.getenv("OPENAI_BASE_URL")
    has_key = llm_cfg.get("api_key") or os.getenv("CODETRACER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not (has_base and has_key):
        return None

    from codetracer.llm.client import LLMClient
    return LLMClient(**llm_cfg)


def _run_analyze(store: Any) -> Any:
    """Run the TraceAgent analysis pipeline and return an ErrorAnalysis."""
    from platformdirs import user_config_dir

    from codetracer.agents.context import ContextAssembler
    from codetracer.agents.trace_agent import TraceAgent
    from codetracer.query.tree_builder import TreeBuilder
    from codetracer.skills.pool import SkillPool

    s = store.get_state()
    if s.llm is None:
        console.print("[red]LLM not configured.[/]")
        return None

    tree_path = s.run_dir / "tree.md"
    if not tree_path.exists():
        with console.status("[dim]Building tree index...[/]", spinner="dots", spinner_style="dim"):
            builder = TreeBuilder()
            tree_md = builder.build(s.traj)
            tree_path.write_text(tree_md, encoding="utf-8")
        console.print(f"  Built tree index -> {tree_path}")

    user_skill_dir = Path(user_config_dir("codetracer")) / "skills"
    user_skill_dir.mkdir(parents=True, exist_ok=True)
    pool = SkillPool(user_dir=user_skill_dir)

    output_path = s.run_dir / "codetracer_labels.json"
    assembler = ContextAssembler(s.config, pool)
    agent = TraceAgent(s.llm, assembler, s.run_dir, output_path, s.config)

    with console.status("[dim]Running trace agent...[/]", spinner="dots", spinner_style="dim"):
        agent.run(s.skill)

    traj_path = output_path.parent / (output_path.stem + ".traj.json")
    agent.save_trajectory(traj_path)
    console.print(f"  Labels     -> {output_path}")
    console.print(f"  Trajectory -> {traj_path}")

    return _try_load_analysis(s.run_dir)


def _try_load_analysis(run_dir: Path) -> Any:
    labels_path = run_dir / "codetracer_labels.json"
    if not labels_path.exists():
        return None
    try:
        from codetracer.models import ErrorAnalysis
        return ErrorAnalysis.from_labels_json(labels_path, run_dir.name)
    except Exception:
        return None
