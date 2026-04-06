"""BenchProvider: abstract interface for bench-type-specific operations.

Replaces all hardcoded bench-type branching throughout the framework with
a data-driven provider system modeled after the existing SkillPool pattern.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BenchRecord:
    """Normalized record returned by a provider's fetch_record()."""

    repo: str = ""
    base_commit: str = ""
    problem_statement: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class BenchProvider(ABC):
    """Everything CodeTracer needs to know about a specific bench type."""

    def configure(self, config: dict[str, Any]) -> None:
        """Inject BENCH.yaml config after module load. Override to use it."""
        if hasattr(self, "_cfg"):
            self._cfg = config

    @abstractmethod
    def name(self) -> str: ...

    # -- detection --
    @abstractmethod
    def can_handle(self, task_dir: Path) -> bool: ...

    # -- task context --
    @abstractmethod
    def load_context(self, task_dir: Path) -> dict[str, Any]:
        """Return {bench_type, task_name, repo_url, base_commit, problem_statement, ...}."""

    def forbidden_files(self) -> set[str]:
        return set()

    def exploration_instructions(self, sandbox: Path) -> str:
        sd = str(sandbox)
        return (
            f"Required exploration commands (execute in order, one per response):\n"
            f"1. ls {sd}/\n"
            f"2. Read key documentation and configuration files you find\n"
            f"3. Explore source directories to understand the codebase structure"
        )

    # -- data fetching (optional; not all benches have remote datasets) --
    def fetch_record(self, case_id: str, hints: dict) -> BenchRecord:
        return BenchRecord()

    # -- sandbox --
    def prepare_sandbox(self, task_dir: Path, sandbox: Path) -> None:
        """Populate *sandbox* from *task_dir*. Default: copy with forbidden files removed."""
        import shutil
        forbidden = self.forbidden_files()
        shutil.copytree(
            task_dir, sandbox,
            ignore=lambda _d, names: [n for n in names if n in forbidden],
        )

    # -- runner --
    def runner_type(self) -> str:
        return "local"

    def create_runner_config(self, task_dir: Path, ctx: dict[str, Any]) -> dict[str, Any]:
        """Build env_config dict used by ReplayRunner / BaseRunner."""
        return {"work_dir": str(task_dir), "timeout": 120}


# ---------------------------------------------------------------------------
# Provider loaded from BENCH.yaml + provider.py on disk
# ---------------------------------------------------------------------------

@dataclass
class LoadedBenchProvider:
    """Metadata loaded from BENCH.yaml + the provider module."""

    bench_name: str
    description: str
    fingerprints: list[str]
    config: dict[str, Any]
    bench_dir: Path
    provider_module: ModuleType
    priority: int = 0

    @property
    def provider(self) -> BenchProvider:
        return self.provider_module.provider

    def matches(self, task_dir: Path) -> bool:
        if self.fingerprints and not any(list(task_dir.glob(fp)) for fp in self.fingerprints):
            return False
        return self.provider.can_handle(task_dir)


def _parse_bench_yaml(text: str) -> dict[str, Any]:
    return yaml.safe_load(text) or {}


def load_bench(bench_dir: Path) -> LoadedBenchProvider:
    yaml_path = bench_dir / "BENCH.yaml"
    py_path = bench_dir / "provider.py"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing BENCH.yaml in {bench_dir}")
    if not py_path.exists():
        raise FileNotFoundError(f"Missing provider.py in {bench_dir}")

    cfg = _parse_bench_yaml(yaml_path.read_text(encoding="utf-8"))

    module_name = f"codetracer_benches.{bench_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load provider.py from {bench_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "provider"):
        raise AttributeError(f"provider.py in {bench_dir} must define a module-level `provider` object")

    module.provider.configure(cfg)

    return LoadedBenchProvider(
        bench_name=cfg.get("name", bench_dir.name),
        description=cfg.get("description", ""),
        fingerprints=cfg.get("fingerprints", []),
        config=cfg,
        bench_dir=bench_dir,
        provider_module=module,
        priority=int(cfg.get("priority", 0)),
    )


# ---------------------------------------------------------------------------
# BenchPool
# ---------------------------------------------------------------------------

_SEED_DIR = Path(__file__).parent.parent / "benches" / "seed"


class BenchPool:
    """Registry of available BenchProviders (seed + user-generated)."""

    def __init__(self, seed_dir: Path = _SEED_DIR, user_dir: Path | None = None) -> None:
        self._providers: dict[str, LoadedBenchProvider] = {}
        self._scan(seed_dir)
        if user_dir is not None:
            self._scan(user_dir)

    def _scan(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "BENCH.yaml").exists():
                continue
            try:
                loaded = load_bench(child)
                self._providers[loaded.bench_name] = loaded
            except Exception as exc:
                logger.warning("Failed to load bench provider from %s: %s", child, exc)

    def detect(self, task_dir: Path) -> LoadedBenchProvider | None:
        by_priority = sorted(self._providers.values(), key=lambda p: p.priority, reverse=True)
        for lp in by_priority:
            if lp.matches(task_dir):
                return lp
        return None

    def get(self, name: str) -> LoadedBenchProvider | None:
        return self._providers.get(name)

    def register(self, loaded: LoadedBenchProvider) -> None:
        self._providers[loaded.bench_name] = loaded

    def list_providers(self) -> list[str]:
        return sorted(self._providers.keys())


# ---------------------------------------------------------------------------
# BenchGenerator (LLM-driven, parallel to SkillGenerator)
# ---------------------------------------------------------------------------

_GEN_SYSTEM = """\
You are a code agent that generates bench provider modules for CodeTracer.
Given a task directory listing and sample files, produce TWO outputs:

1. A BENCH.yaml (in a ```yaml block) with: name, description, fingerprints, and any
   extra config fields (hf_dataset_id, field_mappings, runner_type, forbidden_files, etc.)

2. A provider.py (in a ```python block) implementing a BenchProvider subclass.

The provider.py must follow this interface:

```python
from pathlib import Path
from typing import Any
from codetracer.benches.provider import BenchProvider, BenchRecord

class GeneratedProvider(BenchProvider):
    def __init__(self, config: dict | None = None):
        self._cfg = config or {}

    def name(self) -> str:
        return "<name>"

    def can_handle(self, task_dir: Path) -> bool: ...
    def load_context(self, task_dir: Path) -> dict[str, Any]: ...
    # override other methods as needed

provider = GeneratedProvider()
```
"""

_GEN_INSTANCE = """\
Task directory: {task_dir}

Directory listing (first 100 files):
{listing}

Sample file contents:
{samples}

Existing bench providers for reference:
{provider_index}

Write a BENCH.yaml and provider.py for this bench type.
"""


class BenchGenerator:
    """Uses an LLM to create new BenchProvider modules for unknown bench formats."""

    def __init__(self, llm: Any, pool: BenchPool, max_attempts: int = 3) -> None:
        self._llm = llm
        self._pool = pool
        self._max_attempts = max_attempts

    def generate(self, task_dir: Path, user_dir: Path) -> LoadedBenchProvider:
        from codetracer.utils.llm_generator import (
            extract_code_block as _extract_block,
            list_dir as _list_dir,
            sample_files as _sample_files,
            validate_in_subprocess,
        )

        listing = _list_dir(task_dir, max_files=100)
        samples = _sample_files(task_dir, listing, max_files=3, max_size=200_000, max_preview_lines=40)
        provider_index = "\n".join(f"- {n}" for n in self._pool.list_providers()) or "(none)"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _GEN_SYSTEM},
            {"role": "user", "content": _GEN_INSTANCE.format(
                task_dir=str(task_dir), listing=listing,
                samples=samples, provider_index=provider_index,
            )},
        ]

        for attempt in range(self._max_attempts):
            resp = self._llm.query(messages)
            content = resp["content"]
            messages.append({"role": "assistant", "content": content})

            provider_code = _extract_block(content, "python")
            yaml_text = _extract_block(content, "yaml")

            if not provider_code:
                messages.append({"role": "user", "content": "Return both a ```yaml BENCH.yaml and a ```python provider.py block."})
                continue

            if not yaml_text:
                yaml_text = f"name: generated_{task_dir.name}\ndescription: Auto-generated\nfingerprints: []\n"

            error = _build_provider_validation(provider_code, task_dir, validate_in_subprocess)
            if error is None:
                bench_name = (yaml.safe_load(yaml_text) or {}).get("name", f"generated_{task_dir.name}")
                out_dir = user_dir / bench_name
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "BENCH.yaml").write_text(yaml_text, encoding="utf-8")
                (out_dir / "provider.py").write_text(provider_code, encoding="utf-8")
                loaded = load_bench(out_dir)
                self._pool.register(loaded)
                return loaded

            if attempt < self._max_attempts - 1:
                messages.append({"role": "user", "content": f"Validation error:\n{error}\n\nFix and return corrected blocks."})

        raise RuntimeError(f"BenchGenerator failed after {self._max_attempts} attempts for {task_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_provider_validation(code: str, task_dir: Path, validate_fn: Any) -> str | None:
    """Build and run validation script for a generated provider."""
    from codetracer.utils.llm_generator import validate_in_subprocess as _validate

    script = f"""\
import sys, types
sys.path.insert(0, "{Path(__file__).parent.parent.parent.parent}")
from pathlib import Path
mod = types.ModuleType("_val")
exec(compile('''{code}''', "<provider>", "exec"), mod.__dict__)
p = mod.provider
assert callable(getattr(p, "can_handle", None)), "provider must have can_handle()"
assert callable(getattr(p, "load_context", None)), "provider must have load_context()"
assert callable(getattr(p, "name", None)), "provider must have name()"
print("OK")
"""
    return _validate(script, timeout=20)
