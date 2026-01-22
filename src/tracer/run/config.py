from __future__ import annotations

from tracer import global_config_file, global_config_dir


def configure_if_first_time() -> None:
    """
    Ensure the global config file exists.

    The file is loaded on startup via python-dotenv. Users can put API keys and
    other settings in it (e.g., OPENAI_API_KEY).
    """

    global_config_dir.mkdir(parents=True, exist_ok=True)
    if global_config_file.exists():
        return
    global_config_file.write_text(
        "# tracer global config\n"
        "# Add environment variables here, for example:\n"
        "# OPENAI_API_KEY=...\n",
        encoding="utf-8",
    )


