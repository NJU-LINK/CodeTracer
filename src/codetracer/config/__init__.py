from pathlib import Path

builtin_config_dir = Path(__file__).parent
default_config_path = builtin_config_dir / "default.yaml"

__all__ = ["builtin_config_dir", "default_config_path"]
