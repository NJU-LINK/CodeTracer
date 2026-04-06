"""Tree index models for trajectory navigation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """One node in the trace tree index."""

    step_id: int
    label: str
    node_type: str  # "change" | "explore"
    children: list[TreeNode] = field(default_factory=list)


@dataclass
class TraceTree:
    """Hierarchical index of the trajectory for navigation."""

    root_children: list[TreeNode]

    def render(self) -> str:
        lines = ["root"]

        def rec(node: TreeNode, prefix: str, is_last: bool) -> None:
            lines.append(f"{prefix}{'└── ' if is_last else '├── '}[{node.step_id}] {node.label} ({node.node_type})")
            for i, child in enumerate(node.children):
                rec(child, prefix + ("    " if is_last else "│   "), i == len(node.children) - 1)

        for i, n in enumerate(self.root_children):
            rec(n, "", i == len(self.root_children) - 1)
        return "\n".join(lines) + "\n"
