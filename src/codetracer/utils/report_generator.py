"""Jinja2-based LaTeX report generator for CodeTracer analysis results."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _make_table_rows(rows: list[list[str]]) -> str:
    """Convert a list of row-lists into LaTeX table rows."""
    lines = []
    for row in rows:
        escaped = [_latex_escape(str(cell)) for cell in row]
        lines.append(" & ".join(escaped) + r" \\")
    return "\n".join(lines)


def render_report(
    template_path: Path,
    data: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a LaTeX report from *template_path* using *data*.

    Uses Jinja2 with LaTeX-friendly delimiters: ((( ))), ((* *)), ((= =)).
    Returns the path to the rendered .tex file.
    """
    env = Environment(
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((=",
        comment_end_string="=))",
        autoescape=False,
    )
    env.filters["latex_escape"] = _latex_escape
    env.filters["table_rows"] = _make_table_rows

    template_text = template_path.read_text(encoding="utf-8")
    template = env.from_string(template_text)

    defaults = {
        "generation_date": datetime.now().strftime(r"\today"),
        "total_trajectories": 0,
        "agent_count": 0,
        "dataset_description": "No dataset loaded.",
        "cc_file_count": 0,
        "cc_module_count": 0,
        "cc_tool_count": 0,
        "cc_module_table": "",
        "cc_tool_taxonomy_table": "",
        "agent_summary_table": "",
        "tool_usage_table": "",
        "step_count_table": "",
        "explore_change_table": "",
        "behavioral_comparison": "Analysis not yet performed.",
        "performance_gap_analysis": "Analysis not yet performed.",
        "reward_signal_discussion": "Analysis not yet performed.",
        "findings_and_recommendations": "Analysis not yet performed.",
    }
    defaults.update(data)

    rendered = template.render(**defaults)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def compile_pdf(tex_path: Path, output_dir: Path | None = None) -> Path | None:
    """Compile a .tex file to PDF using pdflatex. Returns PDF path or None on failure."""
    out_dir = output_dir or tex_path.parent
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(out_dir), str(tex_path)],
            capture_output=True,
            timeout=120,
            check=False,
        )
        # Run twice for TOC
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(out_dir), str(tex_path)],
            capture_output=True,
            timeout=120,
            check=False,
        )
        pdf_path = out_dir / tex_path.with_suffix(".pdf").name
        if pdf_path.exists():
            return pdf_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
