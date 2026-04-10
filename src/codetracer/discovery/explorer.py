"""Deep recursive trajectory discovery with LLM-guided directory analysis.

Three-phase discovery:
  A. Fast recursive marker scan (no LLM)
  B. Skill-based detection sweep (no LLM)
  C. LLM-guided architecture analysis (only when fast scan finds nothing)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from codetracer.llm.client import LLMClient
from codetracer.query.normalizer import Normalizer
from codetracer.skills.generator import SkillGenerator
from codetracer.skills.loader import Skill
from codetracer.skills.pool import SkillPool

logger = logging.getLogger(__name__)

_DEFINITIVE_MARKERS = {
    "steps.json",  # pre-normalized trajectory
}

_TRAJECTORY_GLOB_MARKERS = [
    "step_*.jsonl",
]

_AMBIGUOUS_MARKERS = {
    "results.json",  # could be aggregate or per-task
    "sessions",      # could be at various levels
    "agent-logs",
}

_DEFAULT_SKIP_DIRS = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "env",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
})

_DEFAULT_MAX_DEPTH = 10

_LLM_SYSTEM = """\
You are a directory structure analyst. Given a recursive listing of a directory, \
identify subdirectories that likely contain agent execution trajectories.

Look for patterns such as:
- Log files from agent runs (sessions, episodes, steps)
- Results files (results.json, evaluation output)
- Step-by-step recordings (step_N.jsonl, actions.json)
- API call traces or conversation logs

Return a JSON array of relative directory paths (from the root) that most likely \
contain complete agent trajectory data. Only include directories, not files.
Return [] if nothing looks like a trajectory directory.

Example output:
["runs/task_1", "runs/task_2", "experiment_03/agent_output"]
"""


def discover_trajectory_dirs(
    root: Path,
    config: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
) -> list[Path]:
    """Discover trajectory directories under *root* using three-phase strategy.

    Phase A: Fast recursive marker scan (no LLM).
    Phase B: Skill-based detection (no LLM, only if Phase A found candidates).
    Phase C: LLM-guided architecture analysis (only when Phase A is empty).

    Returns a sorted list of absolute paths to trajectory directories.
    """
    config = config or {}
    disc_cfg = config.get("discovery", {})
    max_depth = disc_cfg.get("max_depth", _DEFAULT_MAX_DEPTH)
    skip_dirs = frozenset(disc_cfg.get("skip_dirs", _DEFAULT_SKIP_DIRS))

    root = root.resolve()

    # Phase A: fast recursive marker scan
    candidates = _marker_scan(root, max_depth, skip_dirs)
    if candidates:
        logger.info("Phase A: found %d trajectory dirs via marker scan", len(candidates))
        return sorted(candidates)

    # Phase C: LLM-guided discovery (only when marker scan found nothing)
    if llm is not None:
        logger.info("Phase A found nothing; invoking LLM-guided directory analysis")
        llm_candidates = _llm_directory_analysis(root, llm, max_depth, skip_dirs)
        if llm_candidates:
            logger.info("Phase C: LLM identified %d candidate dirs", len(llm_candidates))
            return sorted(llm_candidates)

    logger.warning("No trajectory directories found under %s", root)
    return []


def detect_or_generate_skill(
    run_dir: Path,
    normalizer: Normalizer,
    pool: SkillPool,
    llm: LLMClient,
    config: dict[str, Any],
    user_skill_dir: Path | None = None,
    format_override: str | None = None,
) -> tuple[Skill | None, Any]:
    """Unified detect-or-generate: returns ``(skill_or_None, NormalizedTrajectory)``.

    1. Pre-normalized (steps.json) → ``(None, traj)``
    2. Step-JSONL annotations → ``(None, traj)``
    3. Skill detection → ``(skill, traj)``
    4. Skill generation via LLM → ``(generated_skill, traj)``
    5. All failed → raises ``ValueError``
    """
    if normalizer.is_pre_normalized(run_dir):
        return None, normalizer.normalize_pre_normalized(run_dir, quiet=True)

    if normalizer.is_step_jsonl_dir(run_dir):
        return None, normalizer.normalize_step_jsonl(run_dir, quiet=True)

    try:
        skill = normalizer.detect(run_dir, format_override)
        return skill, normalizer.normalize(run_dir, skill, quiet=True)
    except ValueError:
        pass

    # Auto-generate a new skill
    if user_skill_dir is None:
        from platformdirs import user_config_dir
        user_skill_dir = Path(user_config_dir("codetracer")) / "skills"
    user_skill_dir.mkdir(parents=True, exist_ok=True)

    disc_cfg = config.get("discovery", {})
    generator = SkillGenerator(llm, pool, disc_cfg)
    try:
        skill = generator.generate(run_dir, user_skill_dir)
        return skill, normalizer.normalize(run_dir, skill, quiet=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot parse trajectory in {run_dir}: detection failed and "
            f"auto-generation failed ({exc})"
        ) from exc


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _marker_scan(
    root: Path, max_depth: int, skip_dirs: frozenset[str]
) -> list[Path]:
    """Recursively scan for directories containing trajectory markers."""
    found: list[Path] = []
    _walk(root, root, 0, max_depth, skip_dirs, found)
    return found


def _walk(
    current: Path,
    root: Path,
    depth: int,
    max_depth: int,
    skip_dirs: frozenset[str],
    out: list[Path],
) -> None:
    if depth > max_depth:
        return

    try:
        entries = list(current.iterdir())
    except PermissionError:
        return

    entry_names = {e.name for e in entries}
    has_definitive = bool(entry_names & _DEFINITIVE_MARKERS)

    if not has_definitive:
        for pattern in _TRAJECTORY_GLOB_MARKERS:
            if any(current.glob(pattern)):
                has_definitive = True
                break

    has_ambiguous = bool(entry_names & _AMBIGUOUS_MARKERS)
    has_marker = has_definitive or has_ambiguous
    subdirs = [e for e in entries if e.is_dir() and e.name not in skip_dirs]

    if has_marker and not subdirs:
        # Leaf directory with a marker — definitely a trajectory dir
        out.append(current)
        return

    if has_marker and subdirs:
        # Has markers AND subdirectories. Check children first:
        # if children have their own trajectory dirs, prefer those (this directory
        # is likely an aggregate/parent container).
        child_found: list[Path] = []
        for subdir in sorted(subdirs):
            _walk(subdir, root, depth + 1, max_depth, skip_dirs, child_found)
        if child_found:
            out.extend(child_found)
        else:
            out.append(current)
        return

    # No markers — recurse into subdirectories
    for subdir in sorted(subdirs):
        _walk(subdir, root, depth + 1, max_depth, skip_dirs, out)


def _llm_directory_analysis(
    root: Path,
    llm: LLMClient,
    max_depth: int,
    skip_dirs: frozenset[str],
) -> list[Path]:
    """Use LLM to identify trajectory directories from directory listing."""
    listing = _build_dir_listing(root, max_depth, skip_dirs, max_lines=500)
    if not listing.strip():
        return []

    messages = [
        {"role": "system", "content": _LLM_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Directory root: {root}\n\n"
                f"Recursive listing:\n{listing}\n\n"
                "Identify trajectory directories and return a JSON array of relative paths."
            ),
        },
    ]

    try:
        resp = llm.query(messages)
    except Exception:
        logger.warning("LLM directory analysis failed", exc_info=True)
        return []

    content = resp.get("content", "")
    return _parse_llm_paths(root, content)


def _build_dir_listing(
    root: Path,
    max_depth: int,
    skip_dirs: frozenset[str],
    max_lines: int = 500,
) -> str:
    """Build a truncated recursive directory listing."""
    lines: list[str] = []

    def _walk_listing(current: Path, depth: int) -> None:
        if depth > max_depth or len(lines) >= max_lines:
            return
        indent = "  " * depth
        try:
            children = sorted(current.iterdir())
        except PermissionError:
            return
        for child in children:
            if child.name in skip_dirs:
                continue
            if len(lines) >= max_lines:
                lines.append(f"{indent}... (truncated)")
                return
            if child.is_dir():
                lines.append(f"{indent}{child.name}/")
                _walk_listing(child, depth + 1)
            else:
                lines.append(f"{indent}{child.name}")

    _walk_listing(root, 0)
    return "\n".join(lines)


def _parse_llm_paths(root: Path, content: str) -> list[Path]:
    """Extract directory paths from LLM JSON response."""
    # Try to find a JSON array in the response
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        paths = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return []

    validated: list[Path] = []
    for p in paths:
        if not isinstance(p, str):
            continue
        candidate = (root / p).resolve()
        # Security: ensure candidate is under root
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_dir():
            validated.append(candidate)
    return validated
