"""Welcome banner and woodpecker mascot art for the CodeTracer REPL."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_VERSION = "0.2.0"

C_PRIMARY = "color(75)"       # steel blue  -- commands, prompt, accents
C_SECONDARY = "color(117)"    # sky blue    -- titles, headers
C_BORDER = "color(69)"        # medium blue -- panel / box borders
C_MUTED = "color(153)"        # pale blue   -- secondary text
C_WARM = "color(215)"         # warm orange -- crest, beak accent


def print_welcome(console: Console, store: Any) -> None:
    """Print the welcome banner with the woodpecker mascot."""
    s = store.get_state()
    console.print()

    mascot_lines = [
        f"   [{C_WARM}]\u2584\u2584[/]",
        f"  [{C_WARM}]\u2588\u2588[/][{C_PRIMARY}]\u2584\u2584[/]",
        f" [{C_PRIMARY}]\u2588\u2588[/][black on {C_PRIMARY}]\u25cf[/][{C_PRIMARY}]\u2588\u2588\u2588[/]",
        f" [{C_PRIMARY}]\u2588\u2588\u2588\u2588\u2588[/][{C_WARM}]\u2580\u2580\u25ba[/]",
        f"  [{C_PRIMARY}]\u2588\u2588[/][{C_BORDER}]\u2588\u2588[/][{C_PRIMARY}]\u2588\u2588[/]",
        f" [{C_PRIMARY}]\u2588\u2588[/][{C_BORDER}]\u2588\u2588\u2588[/][{C_PRIMARY}]\u2588\u2588[/]",
        f"[{C_BORDER}]\u2588\u2588\u2588\u2588\u2588[/][{C_PRIMARY}]\u2588\u2588[/]",
        f"[{C_BORDER}]\u2588\u2588\u2588[/] [{C_MUTED}]\u2588 \u2588[/]",
        f"    [{C_MUTED}]\u2580 \u2580[/]",
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
        f"[bold {C_WARM}]Tips for getting started[/]",
        f"Type [bold {C_PRIMARY}]/[/] to see commands",
        "Or ask a question in natural language",
    ]
    left = Text.from_markup("\n".join(left_lines))

    right_lines = [
        "",
        "",
        f"[bold {C_SECONDARY}]Recent activity[/]",
        recent_text,
        "",
    ]
    right = Text.from_markup("\n".join(right_lines))

    cols = Columns([mascot, left, right], padding=(0, 3))

    console.print(Panel(
        cols,
        title=f"[bold {C_SECONDARY}] CodeTracer v{_VERSION} [/]",
        border_style=C_BORDER,
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
        f"  [bold {C_PRIMARY}]{s.fmt_name}[/] "
        f"[dim]\u00b7[/] {s.traj.step_count} steps "
        f"[dim]\u00b7[/] LLM {llm_label} "
        f"[dim]\u00b7[/] {error_label}"
    )
    console.print()
