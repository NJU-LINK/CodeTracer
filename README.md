# CodeTracer (mini-tracer)

CodeTracer is a minimal **tracer** framework built on a bash-only agent loop.

- **Input**: an agent run output directory (a folder containing trajectories/logs).
- **Output**: step labels (**only** `incorrect` / `unuseful`) plus aggregate counts.
- **Important**: the LLM is responsible for discovering the folder format, deciding step segmentation, and mapping each labeled step back to a file path + line range.

## Install

```bash
cd /data/terminalbench/mini-tracer
pip install -e .
```

## Configure model (OpenAI-compatible)

Edit `src/tracer/config/extra/tracer.yaml`:

- `model.api_base`: OpenAI-compatible base URL (must include scheme, e.g. `https://.../v1`)
- `model.api_key`: API key
- `model.model_name`: optional (if empty, tracer may auto-detect from `/v1/models`)

## Run

```bash
tracer <run_dir> -m <model_name>
```

Notes:
- Commands run locally via `subprocess.run` with `cwd=<run_dir>`.
- Use `--config` to point to a different YAML config.

## Output contract

The tracer run instructs the LLM to write these files **into `<run_dir>`**:

- `mini_tracer_labels.jsonl`: one JSON object per labeled step, including:
  - `command_index`, `command`, `trajectory_path`, `line_start`, `line_end`, `label`, `error_label`
- `mini_tracer_summary.json`: counts for `incorrect`/`unuseful` and per `error_label`
- `mini_tracer.traj.json`: the tracer’s own trajectory (for debugging)


