"""YAML configuration loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_BUILTIN_CONFIG = Path(__file__).parent.parent / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive to base)."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(user_path: Path | None = None) -> dict[str, Any]:
    """Load the built-in default config and optionally merge a user override on top."""
    base: dict[str, Any] = yaml.safe_load(_BUILTIN_CONFIG.read_text(encoding="utf-8")) or {}
    if user_path is not None:
        override: dict[str, Any] = yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}
        base = _deep_merge(base, override)
    return base
