"""Tree index builder: converts a normalized trajectory into a tree.md navigation index.

Command classification uses ClassificationStore (regex defaults + persistent
JSONL storage + optional LLM fallback) instead of hardcoded regex patterns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codetracer.utils.template import render_template

from codetracer.models import NormalizedTrajectory, StepRecord, TraceTree, TreeNode
from codetracer.services.classification import ClassificationStore

logger = logging.getLogger(__name__)

_ANNOTATION_IMPACT_MAP: dict[str, str] = {
    "state changes": "change",
    "just explore": "explore",
}


@dataclass
class _StepMeta:
    step_id: int
    label: str
    node_type: str  # "change" | "explore"


def _classify_step(step: StepRecord, store: ClassificationStore) -> str:
    return store.classify(step.action or "")


def _derive_label(step: StepRecord) -> str:
    action = (step.action or "").strip()
    first_line = action.splitlines()[0] if action else ""
    return first_line[:60] if first_line else f"step {step.step_id}"


def _build_parent_pointers(metas: list[_StepMeta]) -> list[int]:
    parents: list[int] = []
    for i, m in enumerate(metas):
        if i == 0:
            parents.append(-1)
        elif m.node_type == "change":
            parents.append(i - 1)
        else:
            parents.append(parents[i - 1])
    return parents


def _build_tree_nodes(metas: list[_StepMeta], parents: list[int]) -> tuple[list[TreeNode], list[TreeNode]]:
    nodes = [TreeNode(step_id=m.step_id, label=m.label, node_type=m.node_type) for m in metas]
    root_children: list[TreeNode] = []
    for i, p in enumerate(parents):
        if p == -1:
            root_children.append(nodes[i])
        else:
            nodes[p].children.append(nodes[i])
    return root_children, nodes


def _load_step_annotations(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Read ``llm_analysis.command_impact`` from each ``step_N.jsonl`` in *run_dir*."""
    result: dict[int, dict[str, Any]] = {}
    for p in sorted(run_dir.glob("step_*.jsonl")):
        try:
            stem = p.stem
            step_id = int(stem.split("_", 1)[1])
            raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            analysis = raw.get("llm_analysis", {})
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            impact = analysis.get("command_impact", {})
            result[step_id] = {
                "assessment": impact.get("assessment", ""),
                "impact_type": impact.get("impact_type", ""),
                "stage_label": analysis.get("stage_label", ""),
            }
        except Exception:
            logger.debug("Skipping annotation file %s", p, exc_info=True)
    return result


def _annotation_to_node_type(ann: dict[str, Any]) -> str:
    """Map per-step annotation fields to tree node_type using configurable mapping."""
    impact = ann.get("impact_type", "")
    return _ANNOTATION_IMPACT_MAP.get(impact, "explore")


_DEFAULT_SYSTEM = "You are a trajectory analysis assistant. Respond only with the requested JSON."
_DEFAULT_INSTANCE = (
    "Classify each step as 'change' (modifies state: writes files, installs packages, "
    "edits code) or 'explore' (reads, searches, tests without state change). "
    "Also give a short 3-6 word label per step.\n\n"
    "Steps:\n{{ steps_json }}\n\n"
    'Return JSON array: [{"step_id": N, "node_type": "change"|"explore", "label": "..."}]'
)


class TreeBuilder:
    def __init__(self, llm=None, config: dict[str, Any] | None = None) -> None:
        self._llm = llm
        cfg = config or {}
        self._system_template = cfg.get("system_template", _DEFAULT_SYSTEM)
        self._instance_template = cfg.get("instance_template", _DEFAULT_INSTANCE)
        self._store = ClassificationStore()

    def build_from_annotation(
        self,
        traj: NormalizedTrajectory,
        annotation: dict,
        run_dir: Path | None = None,
    ) -> str:
        """Build a tree using per-step annotation labels from ``step_N.jsonl``.

        Each step file contains ``llm_analysis.command_impact`` with:
        - ``assessment``: useful / correct / unuseful / incorrect
        - ``impact_type``: "just explore" / "state changes"

        Falls back to manifest-level ``incorrect_stages`` if step files
        are unavailable, and to ClassificationStore as a last resort.
        """
        step_annotations = _load_step_annotations(run_dir) if run_dir else {}

        incorrect_ids: set[int] = set()
        unuseful_ids: set[int] = set()
        for stage in annotation.get("incorrect_stages", []):
            incorrect_ids.update(stage.get("incorrect_step_ids", []))
            unuseful_ids.update(stage.get("unuseful_step_ids", []))

        metas: list[_StepMeta] = []
        for s in traj.steps:
            ann = step_annotations.get(s.step_id)
            if ann is not None:
                node_type = _annotation_to_node_type(ann)
                label = ann.get("stage_label") or _derive_label(s)
            elif s.step_id in incorrect_ids:
                node_type = "change"
                label = _derive_label(s)
            elif s.step_id in unuseful_ids:
                node_type = "explore"
                label = _derive_label(s)
            else:
                node_type = _classify_step(s, self._store)
                label = _derive_label(s)
            metas.append(_StepMeta(step_id=s.step_id, label=label, node_type=node_type))

        if not metas:
            return "root\n"
        parents = _build_parent_pointers(metas)
        root_children, _ = _build_tree_nodes(metas, parents)
        return TraceTree(root_children=root_children).render()

    def build(self, traj: NormalizedTrajectory) -> str:
        metas = [
            _StepMeta(
                step_id=s.step_id,
                label=_derive_label(s),
                node_type=_classify_step(s, self._store),
            )
            for s in traj.steps
        ]
        if not metas:
            return "root\n"
        parents = _build_parent_pointers(metas)
        root_children, _ = _build_tree_nodes(metas, parents)
        tree = TraceTree(root_children=root_children)
        return tree.render()

    def build_with_llm(self, traj: NormalizedTrajectory) -> str:
        if self._llm is None:
            return self.build(traj)

        step_summaries = [{"step_id": s.step_id, "action": (s.action or "")[:200]} for s in traj.steps]
        steps_json = json.dumps(step_summaries, ensure_ascii=False)
        instance = render_template(self._instance_template, steps_json=steps_json)
        messages = [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": instance},
        ]
        try:
            resp = self._llm.query(messages)
            content = resp.get("content", "")
            m = re.search(r"\[.*\]", content, re.DOTALL)
            if m:
                classifications = json.loads(m.group(0))
                id_to_cls = {c["step_id"]: c for c in classifications if isinstance(c, dict)}
                metas = []
                for s in traj.steps:
                    cls_entry = id_to_cls.get(s.step_id, {})
                    node_type = cls_entry.get("node_type") or _classify_step(s, self._store)
                    metas.append(
                        _StepMeta(
                            step_id=s.step_id,
                            label=cls_entry.get("label") or _derive_label(s),
                            node_type=node_type,
                        )
                    )
                    if cls_entry.get("node_type"):
                        self._store.store(s.action or "", cls_entry["node_type"])
                parents = _build_parent_pointers(metas)
                root_children, _ = _build_tree_nodes(metas, parents)
                return TraceTree(root_children=root_children).render()
        except Exception:
            pass
        return self.build(traj)
