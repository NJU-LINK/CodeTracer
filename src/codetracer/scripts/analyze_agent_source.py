"""Analyze agent source code and trajectories to compare industrial vs academic agents.

Three-phase analysis:
  Phase A: Parse Claude Code TypeScript source -> architecture metrics
  Phase B: Parse all trajectories -> tool usage, step count, success rates
  Phase C: Assemble LaTeX report from template

Usage:
    codetracer-analyze-source --data-dir data/traj --reference-dir reference/claude_code_src/src
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console(highlight=False)


# ===================================================================
# Phase A: Claude Code Source Architecture Analysis
# ===================================================================

def _analyze_claude_code_source(reference_dir: Path) -> dict[str, Any]:
    """Parse Claude Code TypeScript source to extract architecture metrics."""
    if not reference_dir.exists():
        return {"error": f"Reference dir not found: {reference_dir}"}

    # Count files per module
    modules: dict[str, int] = {}
    total_files = 0
    ts_files = list(reference_dir.rglob("*.ts")) + list(reference_dir.rglob("*.tsx"))

    for f in ts_files:
        total_files += 1
        rel = f.relative_to(reference_dir)
        module = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        modules[module] = modules.get(module, 0) + 1

    # Discover tools from tools/ directory
    tools_dir = reference_dir / "tools"
    tool_dirs: list[str] = []
    if tools_dir.exists():
        for child in sorted(tools_dir.iterdir()):
            if child.is_dir():
                tool_dirs.append(child.name)

    # Categorize tools
    tool_categories: dict[str, list[str]] = {
        "File Operations": [],
        "Shell & Execution": [],
        "Search & Navigation": [],
        "Agent & Planning": [],
        "Web & External": [],
        "Workspace & Config": [],
        "Task Management": [],
        "Other": [],
    }

    # Map normalized names (lowercase, "tool" suffix stripped) to categories
    _cat_map = {
        "read": "File Operations", "write": "File Operations", "edit": "File Operations",
        "fileread": "File Operations", "filewrite": "File Operations", "fileedit": "File Operations",
        "notebookedit": "File Operations", "notebookread": "File Operations",
        "glob": "Search & Navigation", "grep": "Search & Navigation",
        "lsp": "Search & Navigation", "toolsearch": "Search & Navigation",
        "bash": "Shell & Execution", "powershell": "Shell & Execution", "repl": "Shell & Execution",
        "agent": "Agent & Planning", "enterplanmode": "Agent & Planning",
        "exitplanmode": "Agent & Planning", "todowrite": "Agent & Planning",
        "todoread": "Agent & Planning", "sendmessage": "Agent & Planning",
        "askuserquestion": "Agent & Planning", "brief": "Agent & Planning",
        "websearch": "Web & External", "webfetch": "Web & External",
        "mcp": "Web & External", "listmcpresources": "Web & External",
        "readmcpresource": "Web & External", "mcpauth": "Web & External",
        "enterworktree": "Workspace & Config", "exitworktree": "Workspace & Config",
        "skill": "Workspace & Config", "schedulecron": "Workspace & Config",
        "config": "Workspace & Config", "sleep": "Workspace & Config",
        "remotetrigger": "Workspace & Config", "syntheticoutput": "Workspace & Config",
        "taskcreate": "Task Management", "taskget": "Task Management",
        "tasklist": "Task Management", "taskoutput": "Task Management",
        "taskstop": "Task Management", "taskupdate": "Task Management",
        "teamcreate": "Task Management", "teamdelete": "Task Management",
    }

    for tool in tool_dirs:
        # Normalize: lowercase, strip trailing "tool" suffix
        key = tool.lower()
        if key.endswith("tool"):
            key = key[:-4]
        cat = _cat_map.get(key, "Other")
        tool_categories[cat].append(tool)

    # Build module table rows
    module_rows = []
    for mod, count in sorted(modules.items(), key=lambda x: -x[1]):
        desc = _MODULE_DESCRIPTIONS.get(mod, "")
        module_rows.append(f"{_latex_esc(mod)} & {count} & {_latex_esc(desc)} \\\\")

    # Build tool taxonomy rows
    _cat_desc = {
        "File Operations": "Read, write, edit files and notebooks",
        "Shell & Execution": "Shell commands, REPL, PowerShell",
        "Search & Navigation": "File search, code search, LSP",
        "Agent & Planning": "Sub-agents, planning, user interaction",
        "Web & External": "Web search, fetch, MCP integration",
        "Workspace & Config": "Worktrees, cron, config, skills",
        "Task Management": "Background task lifecycle management",
        "Other": "Uncategorized tools",
    }
    tool_rows = []
    for cat, tools in tool_categories.items():
        if tools:
            tool_names = ", ".join(t.replace("Tool", "") for t in tools[:6])
            if len(tools) > 6:
                tool_names += f", +{len(tools) - 6} more"
            desc = _cat_desc.get(cat, "")
            tool_rows.append(
                f"{_latex_esc(cat)} & {_latex_esc(tool_names)} & "
                f"{_latex_esc(desc)} ({len(tools)}) \\\\"
            )

    return {
        "cc_file_count": total_files,
        "cc_module_count": len(modules),
        "cc_tool_count": len(tool_dirs),
        "cc_module_table": "\n".join(module_rows),
        "cc_tool_taxonomy_table": "\n".join(tool_rows),
        "modules": modules,
        "tool_dirs": tool_dirs,
        "tool_categories": {k: v for k, v in tool_categories.items() if v},
    }


_MODULE_DESCRIPTIONS: dict[str, str] = {
    "tools": "Tool implementations (Bash, Read, Write, Edit, etc.)",
    "services": "Service layer (compaction, MCP, orchestration)",
    "utils": "Utilities (permissions, messages, hooks, config)",
    "constants": "Configuration constants and prompt templates",
    "bootstrap": "Startup and session state initialization",
    "(root)": "Core agent loop, query engine, entry points",
    "components": "UI component rendering (React/Ink)",
    "commands": "CLI command definitions and handlers",
    "hooks": "Pre/post execution hooks (tool, sampling, etc.)",
    "ink": "Terminal UI framework integration layer",
    "bridge": "IPC bridge for IDE extensions",
    "skills": "Skill system (prompt-based workflows)",
    "cli": "CLI entry points and argument parsing",
    "keybindings": "Keyboard shortcut definitions",
    "tasks": "Background task management system",
    "types": "TypeScript type definitions and interfaces",
    "migrations": "Config/state schema migrations",
    "context": "Context assembly and management",
    "memdir": "Memory directory (persistent context)",
    "entrypoints": "Application launch entry points",
    "buddy": "Pair programming / collaboration mode",
    "state": "State management and persistence",
    "vim": "Vim keybinding mode support",
    "remote": "Remote session management",
    "query": "Query engine for search and retrieval",
    "server": "Language server protocol integration",
    "screens": "Full-screen UI modes",
}


def _latex_esc(s: str) -> str:
    """Minimal LaTeX escaping for table cells."""
    for c, r in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(c, r)
    return s


# ===================================================================
# Phase B: Trajectory Pattern Analysis
# ===================================================================

def _analyze_trajectories(data_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse all trajectories and compute aggregate statistics."""
    from codetracer.llm.client import LLMClient
    from codetracer.query.normalizer import Normalizer
    from codetracer.skills.pool import SkillPool

    pool = SkillPool()
    normalizer = Normalizer(pool)

    # Create LLM client for auto-skill generation on unknown formats
    llm = None
    if config:
        llm_cfg: dict[str, Any] = dict(config.get("llm", {}))
        try:
            llm = LLMClient(**llm_cfg)
        except Exception:
            pass

    agent_stats: dict[str, dict[str, Any]] = {}
    global_tool_counter: Counter = Counter()

    for agent_dir in sorted(data_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        agent_name = agent_dir.name
        trajs_parsed = 0
        step_counts: list[int] = []
        success_count = 0
        total_count = 0
        tool_counter: Counter = Counter()
        explore_count = 0
        change_count = 0

        # Find trajectory directories
        traj_dirs = _find_traj_dirs(agent_dir, config)

        for traj_dir in traj_dirs:
            total_count += 1
            try:
                traj = _parse_one_traj(normalizer, traj_dir, pool, llm, config)
            except Exception:
                continue

            trajs_parsed += 1
            step_counts.append(len(traj.steps))

            # Check success
            results_path = traj_dir / "results.json"
            if results_path.exists():
                try:
                    rd = json.loads(results_path.read_text(encoding="utf-8", errors="replace"))
                    if rd.get("is_resolved") is True:
                        success_count += 1
                except Exception:
                    pass

            # Tool usage
            for step in traj.steps:
                tool = step.tool_type or _extract_tool_type(step.action)
                tool_counter[tool] += 1
                global_tool_counter[tool] += 1

                # Rough explore/change classification
                if tool in ("Read", "Glob", "Grep", "WebSearch", "WebFetch", "Inspect"):
                    explore_count += 1
                else:
                    change_count += 1

        if trajs_parsed == 0:
            continue

        agent_stats[agent_name] = {
            "trajectories": trajs_parsed,
            "total_dirs": total_count,
            "step_counts": step_counts,
            "mean_steps": round(statistics.mean(step_counts), 1) if step_counts else 0,
            "median_steps": round(statistics.median(step_counts), 1) if step_counts else 0,
            "max_steps": max(step_counts) if step_counts else 0,
            "std_steps": round(statistics.stdev(step_counts), 1) if len(step_counts) > 1 else 0,
            "success_rate": round(success_count / total_count * 100, 1) if total_count else 0,
            "tool_counter": dict(tool_counter.most_common()),
            "explore_count": explore_count,
            "change_count": change_count,
            "unique_tools": len(tool_counter),
        }

    return {
        "agent_stats": agent_stats,
        "global_tool_counter": dict(global_tool_counter.most_common()),
        "total_trajectories": sum(s["trajectories"] for s in agent_stats.values()),
        "agent_count": len(agent_stats),
    }


def _find_traj_dirs(agent_dir: Path, config: dict[str, Any] | None = None) -> list[Path]:
    """Find trajectory directories under an agent directory using deep recursive discovery."""
    from codetracer.discovery.explorer import discover_trajectory_dirs
    return discover_trajectory_dirs(agent_dir, config)


def _parse_one_traj(normalizer, traj_dir: Path, pool=None, llm=None, config=None):
    """Parse a single trajectory directory with auto-skill generation fallback."""
    if llm is not None and pool is not None and config is not None:
        from codetracer.discovery.explorer import detect_or_generate_skill
        _, traj = detect_or_generate_skill(traj_dir, normalizer, pool, llm, config)
        return traj
    # Fallback (no LLM available)
    if normalizer.is_pre_normalized(traj_dir):
        return normalizer.normalize_pre_normalized(traj_dir, quiet=True)
    if normalizer.is_step_jsonl_dir(traj_dir):
        return normalizer.normalize_step_jsonl(traj_dir, quiet=True)
    skill = normalizer.detect(traj_dir)
    return normalizer.normalize(traj_dir, skill, quiet=True)


_TOOL_RE = re.compile(r"^\[(\w+)")

def _extract_tool_type(action: str) -> str:
    """Extract tool type from formatted action string."""
    m = _TOOL_RE.match(action)
    return m.group(1) if m else "unknown"


# ===================================================================
# Phase C: Report Assembly
# ===================================================================

def _build_report_data(
    source_data: dict[str, Any],
    traj_data: dict[str, Any],
) -> dict[str, Any]:
    """Combine Phase A and B data into template variables."""
    data: dict[str, Any] = {}

    # Source architecture
    data.update({
        k: source_data.get(k, "")
        for k in [
            "cc_file_count", "cc_module_count", "cc_tool_count",
            "cc_module_table", "cc_tool_taxonomy_table",
        ]
    })

    # Trajectory stats
    data["total_trajectories"] = traj_data.get("total_trajectories", 0)
    data["agent_count"] = traj_data.get("agent_count", 0)

    agent_stats = traj_data.get("agent_stats", {})
    total_tools = sum(traj_data.get("global_tool_counter", {}).values()) or 1

    # Agent summary table
    rows = []
    for name, stats in sorted(agent_stats.items()):
        rows.append(
            f"{_latex_esc(name)} & {stats['trajectories']} & "
            f"{stats['mean_steps']} & {stats['success_rate']}\\% & "
            f"{stats['unique_tools']} \\\\"
        )
    data["agent_summary_table"] = "\n".join(rows) if rows else "No data \\\\"

    # Tool usage table
    tool_rows = []
    global_counter = traj_data.get("global_tool_counter", {})
    for tool, count in sorted(global_counter.items(), key=lambda x: -x[1])[:20]:
        pct = round(count / total_tools * 100, 1)
        agents_using = sum(
            1 for s in agent_stats.values() if tool in s.get("tool_counter", {})
        )
        tool_rows.append(f"{_latex_esc(tool)} & {count} & {pct}\\% & {agents_using} \\\\")
    data["tool_usage_table"] = "\n".join(tool_rows) if tool_rows else "No data \\\\"

    # Step count table
    step_rows = []
    for name, stats in sorted(agent_stats.items()):
        step_rows.append(
            f"{_latex_esc(name)} & {stats['mean_steps']} & "
            f"{stats['median_steps']} & {stats['max_steps']} & "
            f"{stats['std_steps']} \\\\"
        )
    data["step_count_table"] = "\n".join(step_rows) if step_rows else "No data \\\\"

    # Explore/change table
    ec_rows = []
    for name, stats in sorted(agent_stats.items()):
        total = stats["explore_count"] + stats["change_count"]
        ratio = round(stats["explore_count"] / max(stats["change_count"], 1), 2)
        eff = round(stats["change_count"] / max(total, 1) * 100, 1)
        ec_rows.append(
            f"{_latex_esc(name)} & {stats['explore_count']} & "
            f"{stats['change_count']} & {ratio} & {eff}\\% \\\\"
        )
    data["explore_change_table"] = "\n".join(ec_rows) if ec_rows else "No data \\\\"

    # Dataset description
    agent_names = list(agent_stats.keys())
    data["dataset_description"] = (
        f"Our dataset contains {data['total_trajectories']} trajectories from "
        f"{data['agent_count']} agent systems: {', '.join(agent_names)}."
    )

    # Textual analysis sections (placeholder - can be filled by LLM in future)
    data["behavioral_comparison"] = _generate_behavioral_comparison(agent_stats)
    data["performance_gap_analysis"] = _generate_performance_gap(agent_stats)
    data["reward_signal_discussion"] = _generate_reward_discussion()
    data["findings_and_recommendations"] = _generate_findings(agent_stats, source_data)

    return data


def _generate_behavioral_comparison(agent_stats: dict[str, Any]) -> str:
    """Generate behavioral comparison text from statistics."""
    lines = []
    for name, stats in sorted(agent_stats.items()):
        total = stats["explore_count"] + stats["change_count"]
        if total == 0:
            continue
        explore_pct = round(stats["explore_count"] / total * 100, 1)
        lines.append(
            f"\\textbf{{{_latex_esc(name)}}}: "
            f"Average {stats['mean_steps']} steps per trajectory, "
            f"{explore_pct}\\% exploration, "
            f"{stats['success_rate']}\\% success rate, "
            f"{stats['unique_tools']} unique tools used."
        )
    return "\n\n".join(lines) if lines else "No trajectory data available for comparison."


def _generate_performance_gap(agent_stats: dict[str, Any]) -> str:
    """Generate performance gap analysis text."""
    if len(agent_stats) < 2:
        return "Insufficient data for cross-agent comparison (need at least 2 agent systems)."

    rates = {name: stats["success_rate"] for name, stats in agent_stats.items()}
    best = max(rates, key=lambda k: rates[k])
    worst = min(rates, key=lambda k: rates[k])

    return (
        f"The highest success rate is {_latex_esc(best)} at {rates[best]}\\%, "
        f"while the lowest is {_latex_esc(worst)} at {rates[worst]}\\%. "
        f"This {rates[best] - rates[worst]:.1f} percentage point gap suggests "
        f"significant architectural and strategic differences between agent systems."
    )


def _generate_reward_discussion() -> str:
    return (
        "For RL-based agent improvement, we propose a multi-dimensional reward signal "
        "that combines: (1) task-level success/failure as a sparse terminal reward, "
        "(2) per-step deviation labels as dense intermediate rewards, and "
        "(3) efficiency penalties for redundant exploration or unnecessarily long trajectories. "
        "The CodeTracer \\texttt{rl\\_feedback} output profile generates per-step reward signals "
        "in the range $[-1.0, 1.0]$, suitable for direct use in policy gradient methods."
    )


def _generate_findings(agent_stats: dict[str, Any], source_data: dict[str, Any]) -> str:
    tool_count = source_data.get("cc_tool_count", 0)
    return (
        f"\\begin{{enumerate}}\n"
        f"  \\item Industrial agents like Claude Code invest heavily in tooling ({tool_count} tools) "
        f"and error recovery, while academic agents focus on a narrow tool set.\n"
        f"  \\item Production agents implement sophisticated context management (compaction, "
        f"budgeting, feature gating) that academic agents typically lack.\n"
        f"  \\item The exploration-to-change ratio is a strong predictor of trajectory quality: "
        f"excessive exploration correlates with lower success rates.\n"
        f"  \\item RL feedback signals derived from trajectory deviation analysis can bridge "
        f"the industrial-academic gap by providing dense training signal.\n"
        f"  \\item Parallel tool execution (available in industrial agents) significantly "
        f"reduces wall-clock time but introduces ordering-sensitivity bugs.\n"
        f"\\end{{enumerate}}"
    )


# ===================================================================
# Phase C-LLM: LLM-powered qualitative analysis
# ===================================================================

def _llm_generate_section(llm_client: Any, section_name: str, context: str) -> str:
    """Use LLM to generate a rich qualitative section for the report."""
    resp = llm_client.query(
        [
            {"role": "system", "content": (
                "You are a research analyst writing sections of an academic LaTeX report "
                "comparing industrial coding agents (like Claude Code) with academic agents. "
                "Write LaTeX-formatted content (no \\section headers, just body text). "
                "Use \\textbf{}, \\textit{}, \\begin{itemize}, etc. as appropriate. "
                "Be specific, analytical, and cite concrete numbers from the data provided. "
                "Escape special LaTeX chars: & % $ # _ { } ~ ^"
            )},
            {"role": "user", "content": f"Write the '{section_name}' section.\n\nData:\n{context}"},
        ],
        max_tokens=2000,
        temperature=0.3,
    )
    return resp["content"]


def _run_llm_analysis(
    report_data: dict[str, Any],
    source_data: dict[str, Any],
    traj_data: dict[str, Any],
) -> dict[str, str]:
    """Generate qualitative report sections using LLM."""
    from codetracer.llm.client import LLMClient
    from codetracer.query.config import load_config

    config = load_config()
    llm_cfg = dict(config.get("llm", {}))
    llm = LLMClient(**llm_cfg)

    agent_stats = traj_data.get("agent_stats", {})
    global_tools = traj_data.get("global_tool_counter", {})

    # Build context summaries for each section
    stats_summary = json.dumps({
        name: {
            "trajectories": s["trajectories"],
            "mean_steps": s["mean_steps"],
            "success_rate": s["success_rate"],
            "unique_tools": s["unique_tools"],
            "explore_count": s["explore_count"],
            "change_count": s["change_count"],
            "top_tools": dict(list(s.get("tool_counter", {}).items())[:10]),
        }
        for name, s in agent_stats.items()
    }, indent=2)

    arch_summary = (
        f"Claude Code: {source_data.get('cc_file_count', 0)} files, "
        f"{source_data.get('cc_module_count', 0)} modules, "
        f"{source_data.get('cc_tool_count', 0)} tools. "
        f"Tool categories: {json.dumps({k: len(v) for k, v in source_data.get('tool_categories', {}).items()})}. "
        f"Key patterns: async generator loop, layered error recovery, concurrency-safe tool partitioning, "
        f"feature-gated code, hook-based extensibility."
    )

    sections = {}

    console.print("  [dim]Generating behavioral comparison...[/dim]")
    sections["behavioral_comparison"] = _llm_generate_section(
        llm, "Behavioral Differences from Trajectories",
        f"Agent trajectory statistics:\n{stats_summary}\n\n"
        f"Global tool usage (top 15): {json.dumps(dict(list(global_tools.items())[:15]))}\n\n"
        f"Architecture: {arch_summary}\n\n"
        "Compare behavioral patterns: tool usage diversity, exploration vs exploitation ratio, "
        "step efficiency, common failure patterns. Discuss how industrial agents differ from "
        "academic agents in their trajectory behavior."
    )

    console.print("  [dim]Generating performance gap analysis...[/dim]")
    sections["performance_gap_analysis"] = _llm_generate_section(
        llm, "Performance Gap Analysis",
        f"Agent statistics:\n{stats_summary}\n\n"
        f"Architecture: {arch_summary}\n\n"
        "Analyze the performance gap between industrial and academic agents. Discuss "
        "what architectural features contribute to these gaps. If only one agent type "
        "is available, compare observed patterns against known academic baselines "
        "(e.g., SWE-agent, Aider, AutoCodeRover typically have 20-40% SWE-bench resolve rates)."
    )

    console.print("  [dim]Generating reward signal discussion...[/dim]")
    sections["reward_signal_discussion"] = _llm_generate_section(
        llm, "Reward Signal Design",
        f"Deviation types: wrong_tool, wrong_target, redundant, premature_conclusion, missing_exploration.\n"
        f"Agent tool usage: {json.dumps(dict(list(global_tools.items())[:10]))}\n"
        f"Agent stats:\n{stats_summary}\n\n"
        "Discuss how to design reward signals for RL-based agent improvement using "
        "CodeTracer's per-step deviation analysis. Cover: sparse vs dense rewards, "
        "reward shaping from tool choice quality, efficiency bonuses, and multi-objective rewards. "
        "The CodeTracer rl_feedback profile outputs per-step reward_signal in [-1.0, 1.0]."
    )

    console.print("  [dim]Generating findings and recommendations...[/dim]")
    sections["findings_and_recommendations"] = _llm_generate_section(
        llm, "Key Findings and Recommendations",
        f"Architecture: {arch_summary}\n\n"
        f"Agent stats:\n{stats_summary}\n\n"
        f"Global tool usage: {json.dumps(dict(list(global_tools.items())[:15]))}\n\n"
        "Synthesize 5-8 concrete findings and actionable recommendations. Cover: "
        "architectural lessons, tool design principles, trajectory optimization strategies, "
        "RL training signal design, and areas for future research. Use \\begin{enumerate} format."
    )

    console.print(f"  [dim]LLM analysis cost: ${llm.cost:.4f}, {llm.n_calls} calls[/dim]")
    return sections


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze agent source code and trajectories, generate LaTeX report",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/traj"), help="Root of trajectory data")
    parser.add_argument("--reference-dir", type=Path, default=Path("reference/claude_code_src/src"), help="Claude Code source directory")
    parser.add_argument("--template", type=Path, default=Path("report/template.tex"), help="LaTeX template")
    parser.add_argument("--output-dir", type=Path, default=Path("report/generated"), help="Output directory")
    parser.add_argument("--compile-pdf", action="store_true", help="Compile LaTeX to PDF")
    parser.add_argument("--skip-source", action="store_true", help="Skip Phase A (source analysis)")
    parser.add_argument("--skip-traj", action="store_true", help="Skip Phase B (trajectory analysis)")
    args = parser.parse_args()

    from codetracer.query.config import load_config
    config = load_config()

    # Phase A
    source_data: dict[str, Any] = {}
    if not args.skip_source:
        console.print("[bold]Phase A: Analyzing Claude Code source architecture...[/bold]")
        source_data = _analyze_claude_code_source(args.reference_dir)
        if "error" in source_data:
            console.print(f"[yellow]Warning: {source_data['error']}[/yellow]")
        else:
            console.print(
                f"  Found {source_data['cc_file_count']} files, "
                f"{source_data['cc_module_count']} modules, "
                f"{source_data['cc_tool_count']} tools"
            )

    # Phase B
    traj_data: dict[str, Any] = {"agent_stats": {}, "global_tool_counter": {}, "total_trajectories": 0, "agent_count": 0}
    if not args.skip_traj:
        console.print("[bold]Phase B: Analyzing trajectories...[/bold]")
        if args.data_dir.exists():
            traj_data = _analyze_trajectories(args.data_dir, config)
            console.print(
                f"  Analyzed {traj_data['total_trajectories']} trajectories "
                f"from {traj_data['agent_count']} agents"
            )
        else:
            console.print(f"[yellow]Data dir not found: {args.data_dir}[/yellow]")

    # Phase C
    console.print("[bold]Phase C: Assembling LaTeX report...[/bold]")
    report_data = _build_report_data(source_data, traj_data)

    # LLM-powered qualitative analysis
    console.print("[bold]Phase C-LLM: Generating qualitative analysis with LLM...[/bold]")
    llm_sections = _run_llm_analysis(report_data, source_data, traj_data)
    report_data.update(llm_sections)

    if not args.template.exists():
        console.print(f"[red]Template not found: {args.template}[/red]")
        sys.exit(1)

    from codetracer.utils.report_generator import compile_pdf, render_report

    output_path = args.output_dir / "agent_analysis_report.tex"
    render_report(args.template, report_data, output_path)
    console.print(f"  Report -> [bold green]{output_path}[/bold green]")

    # Save raw data
    raw_data_path = args.output_dir / "analysis_data.json"
    serializable = {
        "source": {k: v for k, v in source_data.items() if isinstance(v, (str, int, float, list, dict, bool))},
        "trajectories": {
            k: v for k, v in traj_data.items()
            if k != "agent_stats"
        },
        "agent_stats": {
            name: {k: v for k, v in stats.items() if k != "step_counts"}
            for name, stats in traj_data.get("agent_stats", {}).items()
        },
    }
    raw_data_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"  Data   -> [bold]{raw_data_path}[/bold]")

    if args.compile_pdf:
        console.print("  Compiling PDF...")
        pdf = compile_pdf(output_path)
        if pdf:
            console.print(f"  PDF    -> [bold green]{pdf}[/bold green]")
        else:
            console.print("  [yellow]PDF compilation failed (is pdflatex installed?)[/yellow]")

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
