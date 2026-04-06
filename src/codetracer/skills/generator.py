"""Skill generator: LLM-driven creation of new skills for unknown trajectory formats."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from codetracer.llm.client import LLMClient
from codetracer.skills.loader import Skill, load_skill
from codetracer.skills.pool import SkillPool
from codetracer.utils.llm_generator import (
    extract_code_block,
    list_dir,
    sample_files,
    validate_in_subprocess,
)


class SkillGenerator:
    """Uses an LLM to analyze an unknown run directory and produce a new skill (SKILL.md + parser.py)."""

    def __init__(self, llm: LLMClient, pool: SkillPool, config: dict[str, Any]) -> None:
        self._llm = llm
        self._pool = pool
        self._max_attempts = config.get("max_attempts", 3)
        self._system_template = config.get("system_template", _DEFAULT_SYSTEM)
        self._instance_template = config.get("instance_template", _DEFAULT_INSTANCE)
        self._validation_error_template = config.get("validation_error_template", _DEFAULT_VALIDATION_ERROR)

    def generate(self, run_dir: Path, user_dir: Path) -> Skill:
        """Analyze *run_dir*, generate a skill, save it under *user_dir*, and register it."""
        listing = list_dir(run_dir)
        samples = sample_files(run_dir, listing)
        skill_index = self._pool.skill_index()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_template},
            {
                "role": "user",
                "content": self._instance_template.format(
                    run_dir=str(run_dir),
                    listing=listing,
                    samples=samples,
                    skill_index=skill_index,
                ),
            },
        ]

        for attempt in range(self._max_attempts):
            resp = self._llm.query(messages)
            content = resp["content"]
            messages.append({"role": "assistant", "content": content})

            parser_code = extract_code_block(content, "python")
            skill_md_text = extract_code_block(content, "markdown")
            if not parser_code:
                messages.append({"role": "user", "content": "Please return both a ```markdown SKILL.md and a ```python parser.py code block."})
                continue

            if not skill_md_text:
                format_id = _extract_format_id(parser_code)
                skill_md_text = _synthesize_skill_md(format_id, run_dir)

            format_id = _extract_format_id(parser_code)
            error = _validate_parser_script(parser_code, run_dir)
            if error is None:
                skill_dir = user_dir / format_id
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(skill_md_text, encoding="utf-8")
                (skill_dir / "parser.py").write_text(parser_code, encoding="utf-8")
                skill = load_skill(skill_dir)
                self._pool.register(skill)
                return skill

            if attempt < self._max_attempts - 1:
                messages.append({"role": "user", "content": self._validation_error_template.format(error=error)})

        raise RuntimeError(f"Skill generator failed after {self._max_attempts} attempts for {run_dir}")


def _extract_format_id(code: str) -> str:
    m = re.search(r'format_id\s*=\s*["\']([^"\']+)["\']', code)
    return m.group(1) if m else "generated_unknown"


def _synthesize_skill_md(format_id: str, run_dir: Path) -> str:
    return f"""---
name: {format_id}
description: Auto-generated parser for {run_dir.name}
fingerprints: []
metadata:
  version: "1.0"
  source: generated
---

# {format_id} Trajectory Parser

Auto-generated skill. Inspect and refine as needed.
"""


def _validate_parser_script(code: str, run_dir: Path) -> str | None:
    script = f"""\
import sys, types
sys.path.insert(0, "{Path(__file__).parent.parent.parent.parent}")
from pathlib import Path

mod = types.ModuleType("_val_mod")
exec(compile('''{code}''', "<parser>", "exec"), mod.__dict__)

run_dir = Path(r"{run_dir}")
p = mod.parser
assert isinstance(p.format_id, str) and p.format_id, "format_id must be a non-empty string"
assert isinstance(p.can_parse(run_dir), bool), "can_parse must return bool"
traj = p.parse(run_dir)
assert hasattr(traj, "steps"), "parse() must return NormalizedTrajectory with .steps"
assert isinstance(traj.steps, list), ".steps must be a list"
for s in traj.steps[:3]:
    assert hasattr(s, "step_id"), "StepRecord must have step_id"
    assert hasattr(s, "action"), "StepRecord must have action"
print("OK")
"""
    return validate_in_subprocess(script)


_DEFAULT_SYSTEM = """\
You are a code agent specializing in parsing agent trajectory files.
You will be given the file layout of an unknown run directory and must produce TWO outputs:

1. A SKILL.md file (in a ```markdown block) with YAML frontmatter containing name, description, fingerprints, and metadata fields, followed by documentation of the directory layout and extraction logic.
2. A parser.py file (in a ```python block) implementing the parser.

The parser must follow this interface:

```python
from pathlib import Path
from codetracer.models import FileRef, NormalizedTrajectory, StepRecord

class GeneratedParser:
    format_id = "<unique_format_id>"

    def can_parse(self, run_dir: Path) -> bool: ...
    def parse(self, run_dir: Path) -> NormalizedTrajectory:
        ...
        return NormalizedTrajectory(
            steps=steps,
            task_description=instruction,
            metadata={"format": self.format_id, "run_dir": str(run_dir)},
        )

parser = GeneratedParser()
```

Rules:
- format_id must be a short lowercase slug
- Each StepRecord must have step_id (1-indexed int), action (str), observation (str|None)
- The file must end with `parser = GeneratedParser()`
- parse() must extract task_description from the run data (e.g. results.json \
"instruction" field) and pass it to NormalizedTrajectory
"""

_DEFAULT_INSTANCE = """\
Run directory: {run_dir}

Directory listing (first 200 files):
{listing}

Sample file contents:
{samples}

Existing skills for reference:
{skill_index}

Write a SKILL.md and parser.py for this trajectory format.
"""

_DEFAULT_VALIDATION_ERROR = """\
Your parser raised an error during validation:

{error}

Fix the parser and return corrected ```markdown and ```python blocks.
"""
