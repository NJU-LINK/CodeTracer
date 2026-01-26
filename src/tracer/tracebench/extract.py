from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import requests


TracebenchSplit = Literal["full", "verified"]


@dataclass(frozen=True)
class TracebenchRecord:
    traj_id: str
    record: dict[str, Any]
    source: Literal["local", "tracebench"]
    artifact_path: str | None = None
    downloaded_artifact: Path | None = None

    def record_json(self) -> str:
        return json.dumps(self.record, indent=2, ensure_ascii=False)


def _iter_jsonl_lines(text_iter: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in text_iter:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _find_record_in_jsonl(path: Path, traj_id: str) -> dict[str, Any] | None:
    for obj in _iter_jsonl_lines(path.read_text(encoding="utf-8", errors="replace").splitlines().__iter__()):
        if obj.get("traj_id") == traj_id:
            return obj
    return None


def _find_local_record(run_dir: Path, traj_id: str | None) -> dict[str, Any] | None:
    candidates = [
        run_dir / "record.json",
        run_dir / "manifest.json",
        run_dir / "meta.json",
        run_dir / "metadata.json",
    ]
    for p in candidates:
        if p.is_file():
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                return obj

    if traj_id is None:
        return None

    for p in sorted(run_dir.glob("*.jsonl")):
        rec = _find_record_in_jsonl(p, traj_id)
        if rec is not None:
            return rec

    for p in sorted(run_dir.rglob("*.jsonl")):
        rec = _find_record_in_jsonl(p, traj_id)
        if rec is not None:
            return rec

    return None


def _hf_resolve_url(repo: str, filename: str) -> str:
    repo = repo.strip()
    if not repo:
        raise ValueError("tracebench_repo must be non-empty")
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


def _download_text(url: str) -> Iterator[str]:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            yield raw


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _find_record_in_tracebench(repo: str, split: TracebenchSplit, traj_id: str) -> dict[str, Any]:
    manifest_name = f"bench_manifest.{split}.jsonl"
    url = _hf_resolve_url(repo, manifest_name)
    for obj in _iter_jsonl_lines(_download_text(url)):
        if obj.get("traj_id") == traj_id:
            return obj
    raise ValueError(f"traj_id not found in {manifest_name}: {traj_id}")


def extract_tracebench_record(
    *,
    run_dir: Path,
    traj_id: str | None,
    split: TracebenchSplit,
    download_tracebench: bool,
    tracebench_repo: str,
) -> TracebenchRecord:
    run_dir = run_dir.resolve()

    if download_tracebench:
        if not traj_id:
            raise ValueError("--traj-id is required when --download-tracebench is set")
        rec = _find_record_in_tracebench(tracebench_repo, split, traj_id)
        artifact_path = rec.get("artifact_path") if isinstance(rec.get("artifact_path"), str) else None
        downloaded = None
        if artifact_path:
            url = _hf_resolve_url(tracebench_repo, artifact_path)
            downloaded = run_dir / "artifact" / Path(artifact_path).name
            if not downloaded.exists():
                _download_file(url, downloaded)
        return TracebenchRecord(
            traj_id=traj_id,
            record=rec,
            source="tracebench",
            artifact_path=artifact_path,
            downloaded_artifact=downloaded,
        )

    rec = _find_local_record(run_dir, traj_id)
    if rec is None:
        raise ValueError(
            "Could not find a local Tracebench record. Provide record.json/manifest.json/meta.json, "
            "or pass --traj-id and include a manifest JSONL in the run directory, "
            "or enable --download-tracebench."
        )
    resolved_traj_id = traj_id or (rec.get("traj_id") if isinstance(rec.get("traj_id"), str) else "")
    if not resolved_traj_id:
        raise ValueError("Local record has no traj_id; pass --traj-id explicitly or include it in record.json")
    return TracebenchRecord(traj_id=resolved_traj_id, record=rec, source="local")


