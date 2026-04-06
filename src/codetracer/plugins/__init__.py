"""CodeTracer plugin system -- adapters, hooks, and registry."""

from codetracer.plugins.adapter import PluginAdapter
from codetracer.plugins.hooks import HookManager
from codetracer.plugins.registry import PluginRegistry

__all__ = ["PluginAdapter", "PluginRegistry", "HookManager"]
