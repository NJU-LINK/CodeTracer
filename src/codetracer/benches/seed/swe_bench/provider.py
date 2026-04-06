"""BenchProvider for all SWE-bench variants (verified, pro, multi, poly).

Variant-specific config (dataset IDs, field mappings) is loaded from BENCH.yaml
so no hardcoded dataset IDs exist in Python code.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from codetracer.benches.provider import BenchProvider, BenchRecord

logger = logging.getLogger(__name__)

_DATASET_CACHE: dict[str, Any] = {}
_MULTI_INDEX: dict[str, dict[str, Any]] = {}


class SweBenchProvider(BenchProvider):

    def __init__(self, config: dict | None = None) -> None:
        self._cfg = config or {}
        self._variants: dict[str, dict] = self._cfg.get("variants", {})

    def configure(self, config: dict) -> None:
        self._cfg = config
        self._variants = config.get("variants", {})

    def name(self) -> str:
        return "swe_bench"

    def can_handle(self, task_dir: Path) -> bool:
        summary = task_dir / "summary.json"
        if not summary.exists():
            return False
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            bench = data.get("task_meta", {}).get("bench", "")
            return bench in self._variants
        except Exception:
            return False

    def _detect_variant(self, task_dir: Path) -> str:
        summary = task_dir / "summary.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        return data.get("task_meta", {}).get("bench", "")

    def load_context(self, task_dir: Path) -> dict[str, Any]:
        summary = task_dir / "summary.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        meta = data.get("task_meta", {})
        bench = meta.get("bench", "")
        row = meta.get("row", {})
        case_id = task_dir.name

        record = self.fetch_record(case_id, {"bench": bench, "row": row})

        repo_slug = record.repo
        repo_url = f"https://github.com/{repo_slug}" if repo_slug else None

        return {
            "bench_type": f"swe-{bench}",
            "task_name": case_id,
            "repo_url": repo_url,
            "base_commit": record.base_commit,
            "problem_statement": record.problem_statement,
        }

    def fetch_record(self, case_id: str, hints: dict) -> BenchRecord:
        bench = hints.get("bench", "")
        row = hints.get("row", {})
        variant = self._variants.get(bench, {})
        if not variant:
            return BenchRecord()

        fm = variant.get("field_mappings", {})
        if bench == "multi":
            return self._fetch_multi(case_id, row, variant, fm)
        return self._fetch_standard(case_id, row, variant, fm)

    def _fetch_standard(self, case_id: str, row: dict, variant: dict, fm: dict) -> BenchRecord:
        dataset_id = variant.get("hf_dataset_id", "")
        split = variant.get("hf_split", "test")
        ds = _load_hf_dataset(dataset_id, split)

        id_field = fm.get("case_id", "instance_id")
        repo_field = fm.get("repo", "repo")
        commit_field = fm.get("commit", "base_commit")
        problem_field = fm.get("problem", "problem_statement")

        if ds is not None:
            for record in ds:
                if record.get(id_field) == case_id:
                    return BenchRecord(
                        repo=record.get(repo_field, row.get("repo", "")),
                        base_commit=record.get(commit_field, row.get("base_commit", "")),
                        problem_statement=record.get(problem_field, ""),
                    )

        logger.warning("Instance %s not found in %s; using summary.json fallback", case_id, dataset_id)
        return BenchRecord(
            repo=row.get("repo", ""),
            base_commit=row.get("base_commit", ""),
            problem_statement=row.get("hints_text", ""),
        )

    def _fetch_multi(self, case_id: str, row: dict, variant: dict, fm: dict) -> BenchRecord:
        org = row.get("org", "")
        repo_short = row.get("repo", "")
        org_repo_key = f"{org}__{repo_short}"
        dataset_id = variant.get("hf_dataset_id", "")
        lang_dirs = variant.get("lang_dirs", [])

        entry = _load_multi_jsonl(org_repo_key, case_id, dataset_id, lang_dirs)
        if entry is not None:
            repo_full = f"{entry.get('org', org)}/{entry.get('repo', repo_short)}"
            commit_path = fm.get("commit", "base.sha")
            parts = commit_path.split(".")
            val = entry
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else ""
            base_commit = val if isinstance(val, str) else ""
            ps = _build_multi_problem_statement(entry, fm.get("problem", "resolved_issues"))
            return BenchRecord(repo=repo_full, base_commit=base_commit, problem_statement=ps)

        logger.warning("Instance %s not found in Multi-SWE-bench; using fallback", case_id)
        repo_full = f"{org}/{repo_short}" if org and repo_short else row.get("repo", "")
        return BenchRecord(repo=repo_full)

    def exploration_instructions(self, sandbox: Path) -> str:
        sd = str(sandbox)
        cmds = self._cfg.get("exploration_commands")
        if cmds:
            lines = [f"{i+1}. {c.format(sandbox=sd)}" for i, c in enumerate(cmds)]
            return "Required exploration commands (execute in order, one per response):\n" + "\n".join(lines)
        return (
            f"Required exploration commands (execute in order, one per response):\n"
            f"1. ls {sd}/\n"
            f"2. Explore the repo structure to understand the codebase\n"
            f"3. Navigate to the area of code related to the problem statement\n"
            f"4. Read relevant source files to understand the context"
        )

    def prepare_sandbox(self, task_dir: Path, sandbox: Path) -> None:
        """Clone the repo at base_commit into sandbox."""
        ctx = self.load_context(task_dir)
        repo_url = ctx.get("repo_url")
        base_commit = ctx.get("base_commit")
        if not repo_url or not base_commit:
            raise ValueError(f"SWE-bench sandbox requires repo_url and base_commit")

        logger.info("Cloning %s -> %s", repo_url, sandbox)
        subprocess.run(["git", "clone", "--quiet", repo_url, str(sandbox)], check=True, timeout=300)
        subprocess.run(["git", "-C", str(sandbox), "reset", "--hard", base_commit], check=True, timeout=60)
        subprocess.run(["git", "-C", str(sandbox), "remote", "remove", "origin"], check=True, timeout=10)

    def runner_type(self) -> str:
        return "git"

    def create_runner_config(self, task_dir: Path, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "repo_url": ctx.get("repo_url", ""),
            "base_commit": ctx.get("base_commit", ""),
            "work_dir": str(task_dir / "repo"),
            "timeout": 120,
        }


# ---------------------------------------------------------------------------
# Dataset loading helpers (module-level caches)
# ---------------------------------------------------------------------------

def _load_hf_dataset(dataset_id: str, split: str) -> Any:
    cache_key = f"{dataset_id}:{split}"
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    try:
        from datasets import load_dataset
        logger.info("Loading HF dataset %s (split=%s)...", dataset_id, split)
        ds = load_dataset(dataset_id, split=split)
        _DATASET_CACHE[cache_key] = ds
        return ds
    except Exception:
        logger.exception("Failed to load HF dataset %s", dataset_id)
        _DATASET_CACHE[cache_key] = None
        return None


def _load_multi_jsonl(
    org_repo_key: str, case_id: str, repo_id: str, lang_dirs: list[str]
) -> dict[str, Any] | None:
    if org_repo_key in _MULTI_INDEX:
        return _MULTI_INDEX[org_repo_key].get(case_id)

    jsonl_path = _find_multi_jsonl_path(org_repo_key, repo_id, lang_dirs)
    if jsonl_path is None:
        _MULTI_INDEX[org_repo_key] = {}
        return None

    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id, jsonl_path, repo_type="dataset")
        index: dict[str, Any] = {}
        with open(local_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                iid = entry.get("instance_id", "")
                if iid:
                    index[iid] = entry
        _MULTI_INDEX[org_repo_key] = index
        return index.get(case_id)
    except Exception:
        logger.exception("Failed to load Multi-SWE-bench JSONL %s", jsonl_path)
        _MULTI_INDEX[org_repo_key] = {}
        return None


def _find_multi_jsonl_path(org_repo_key: str, repo_id: str, lang_dirs: list[str]) -> str | None:
    try:
        from huggingface_hub import list_repo_tree
        for lang in lang_dirs:
            files = list(list_repo_tree(repo_id, repo_type="dataset", path_in_repo=lang))
            for f in files:
                expected = f"{org_repo_key}_dataset.jsonl"
                if f.path.endswith(expected):
                    return f.path
    except Exception:
        logger.exception("Failed to list Multi-SWE-bench repo tree")
    return None


def _build_multi_problem_statement(entry: dict, problem_field: str) -> str:
    resolved = entry.get(problem_field, [])
    if isinstance(resolved, list) and resolved:
        issue = resolved[0]
        title = issue.get("title", "")
        body = issue.get("body", "")
        if title or body:
            return f"## {title}\n\n{body}" if body else f"## {title}"

    title = entry.get("title", "")
    body = entry.get("body", "")
    if title or body:
        return f"## {title}\n\n{body}" if body else f"## {title}"
    return ""


provider = SweBenchProvider()
