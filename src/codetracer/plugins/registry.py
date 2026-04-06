"""PluginRegistry: discovers and manages PluginAdapter instances."""

from __future__ import annotations

import importlib
import logging

from codetracer.plugins.adapter import PluginAdapter

logger = logging.getLogger(__name__)

# Built-in adapter class paths (resolved lazily)
_BUILTIN_ADAPTERS: dict[str, str] = {
    "openhands": "codetracer.plugins.adapter:OpenHandsAdapter",
    "swe_agent": "codetracer.plugins.adapter:SweAgentAdapter",
    "miniswe": "codetracer.plugins.adapter:MinisweAdapter",
}


class PluginRegistry:
    """Central registry of available PluginAdapter classes."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[PluginAdapter]] = {}

    def register(self, name: str, adapter_cls: type[PluginAdapter]) -> None:
        self._adapters[name] = adapter_cls

    def get(self, name: str) -> PluginAdapter:
        """Return an instantiated adapter by name.

        Tries registered adapters first, then falls back to built-in stubs.
        """
        cls = self._adapters.get(name)
        if cls is None:
            cls = self._resolve_builtin(name)
        if cls is None:
            raise KeyError(f"No adapter registered for '{name}'")
        return cls()

    def list_adapters(self) -> list[str]:
        all_names = set(self._adapters) | set(_BUILTIN_ADAPTERS)
        return sorted(all_names)

    def register_from_import_path(self, name: str, import_path: str) -> None:
        """Register an adapter by dotted import path 'module:ClassName'."""
        cls = self._import_class(import_path)
        self.register(name, cls)

    @staticmethod
    def _resolve_builtin(name: str) -> type[PluginAdapter] | None:
        path = _BUILTIN_ADAPTERS.get(name)
        if path is None:
            return None
        try:
            return PluginRegistry._import_class(path)
        except Exception:
            logger.warning("Failed to load built-in adapter %s from %s", name, path)
            return None

    @staticmethod
    def _import_class(path: str) -> type[PluginAdapter]:
        module_path, class_name = path.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        if not (isinstance(cls, type) and issubclass(cls, PluginAdapter)):
            raise TypeError(f"{path} is not a PluginAdapter subclass")
        return cls


# Module-level default instance
default_registry = PluginRegistry()
