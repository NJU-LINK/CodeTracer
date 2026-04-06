# CodeTracer

Self-evolving agent trajectory diagnosis system. CodeTracer analyzes agent trajectories step-by-step, identifies incorrect and unuseful actions, and produces structured error labels.

## Installation

```bash
cd tracer
pip install -e .
```

Set environment variables for your LLM endpoint:

```bash
export CODETRACER_API_BASE="http://your-api-endpoint/v1"
export CODETRACER_API_KEY="your-api-key"
```

## Evaluating on Tracebench

[Tracebench](https://huggingface.co/datasets/Contextbench/Tracebench) is a benchmark of 4316 agent trajectories with human-verified step-level annotations for trajectory diagnosis evaluation.

| Split | Rows | Description |
|-------|------|-------------|
| `verified` | 1000 | Curated subset (489 SWE-bench + 511 terminal) |
| `full` | 3316 | All trajectories |

### Dataset Fields

Each manifest entry contains these fields. The ones used by CodeTracer are marked with `*`:

| Field | Description |
|-------|-------------|
| `traj_id` * | Unique trajectory identifier, used as output directory name |
| `annotation_relpath` * | Relative path to step annotation files (`step_N.jsonl`, `stage_ranges.json`) |
| `artifact_path` * | Path to the `.tar.zst` artifact containing the raw traj folder |
| `agent` | Agent name: `mini-SWE-agent`, `OpenHands`, `Terminus2`, `SWE-agent` |
| `model` | Model identifier (e.g. `OpenAI/GPT-5`, `Anthropic/Claude-Sonnet-4-20250514-Thinking`) |
| `stages` | Ground-truth stage ranges (`[{stage_id, start_step_id, end_step_id}]`) |
| `incorrect_stages` * | Per-stage incorrect/unuseful step annotations (evaluation ground truth) |
| `solved` | Whether the agent solved the task |
| `step_count` | Total number of steps |
| `category` | Task category (e.g. `software-engineering`) |
| `difficulty` | `easy` / `medium` / `hard` |

### Step 1: Load the Manifest

**From HuggingFace:**

```python
from datasets import load_dataset

ds = load_dataset("Contextbench/Tracebench", split="verified")
entry = ds[0]
print(entry["traj_id"], entry["agent"], entry["model"])
```

**From local JSONL:**

```python
import json

with open("bench_manifest.verified.jsonl") as f:
    entries = [json.loads(line) for line in f]

entry = entries[0]
```

### Step 2: Download and Extract the Artifact

Each entry's `artifact_path` points to a `.tar.zst` archive containing the raw trajectory folder (sessions, agent-logs, results.json, etc.).

```bash
# Download from HuggingFace (if not local)
huggingface-cli download Contextbench/Tracebench \
  --repo-type dataset \
  --include "bench_artifacts/full/<filename>.tar.zst" \
  --local-dir ./tracebench_data

# Extract
mkdir -p workspace
tar --zstd -xf tracebench_data/<artifact_path> -C workspace/
```

After extraction, the directory structure looks like:

```
workspace/<traj_id>/
  results.json            # task metadata and test results
  commands.txt            # (miniswe) keystroke log
  agent-logs/
    mini.traj.json        # (miniswe) structured trajectory
  sessions/
    agent.log             # agent session log
    agent.cast            # asciinema recording
    tests.log             # test output
```

The exact layout varies by agent type. CodeTracer auto-detects the format via its skill system.

### Step 3: Prepare the Annotation File

CodeTracer uses the manifest entry's stage/step annotations to build its analysis tree. Save the entry as a JSON file:

```python
import json

item = {
    "traj_id": entry["traj_id"],
    "annotation": {
        "stages": entry["stages"],
        "incorrect_stages": entry["incorrect_stages"],
    }
}

with open(f"workspace/{entry['traj_id']}.annotation.json", "w") as f:
    json.dump(item, f)
```

### Step 4: Run CodeTracer on a Single Trajectory

```bash
codetracer analyze workspace/<traj_id>/ \
  --model <your-model> \
  --api-base "$CODETRACER_API_BASE" \
  --api-key "$CODETRACER_API_KEY" \
  --output-dir outputs \
  --traj-id "<traj_id>" \
  --annotation-tree \
  --traj-annotation-path workspace/<traj_id>.annotation.json \
  --skip-discovery \
  --skip-sandbox
```

**Flag reference:**

| Flag | Purpose |
|------|---------|
| `--output-dir` | Root directory for all outputs |
| `--traj-id` | Subdirectory name under output-dir |
| `--annotation-tree` | Build the analysis tree from ground-truth annotations instead of heuristics |
| `--traj-annotation-path` | Path to the annotation JSON file |
| `--skip-discovery` | Fail fast if trajectory format is unrecognized (don't invoke LLM generator) |
| `--skip-sandbox` | Skip sandbox environment creation |
| `--cost-limit` | Max LLM spend in USD per trajectory (default: 3.0) |
| `--dry-run` | Normalize + tree only, skip the LLM trace agent |

For terminal-bench tasks, also pass `--tasks-root` pointing to the tasks directory:

```bash
codetracer analyze workspace/<traj_id>/ \
  ... \
  --tasks-root /path/to/terminal-bench/tasks
```

### Step 5: Batch Run (All 1000 Verified)

Use the provided batch script to run all verified entries in parallel:

```bash
bash scripts/run_verified_batch.sh \
  --parallel 4 \
  --model claude-sonnet-4-20250514 \
  --manifest /path/to/bench_manifest.verified.jsonl \
  --output /path/to/outputs
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--parallel N` | 4 | Number of parallel workers |
| `--model` | `claude-sonnet-4-20250514` | LLM model name |
| `--resume` | on | Skip trajectories that already have output |
| `--dry-run` | off | Normalize + tree only, skip LLM trace |
| `--cost-limit` | 3.0 | USD per trajectory |
| `--api-key` | env `CODETRACER_API_KEY` | API key |
| `--api-base` | env `CODETRACER_API_BASE` | API endpoint |

### Step 6: Output Structure

Each completed trajectory produces:

```
outputs/<traj_id>/
  steps.json                    # normalized trajectory (action/observation per step)
  tree.md                       # step classification tree (change/explore labels)
  stage_ranges.json             # stage segmentation
  codetracer_labels.json        # predicted labels (the main output)
  codetracer_labels.traj.json   # full agent reasoning trace
```

`codetracer_labels.json` format:

```json
[
  {
    "stage_id": 3,
    "incorrect_step_ids": [21, 22],
    "unuseful_step_ids": [],
    "reasoning": "Steps 21-22 edit the wrong file..."
  }
]
```

### Step 7: Evaluation

Compare CodeTracer's predicted labels against the ground-truth `incorrect_stages` from the manifest:

```python
import json

# Load prediction
with open("outputs/<traj_id>/codetracer_labels.json") as f:
    predicted = json.load(f)

# Load ground truth from manifest entry
ground_truth = entry["incorrect_stages"]

# Extract predicted incorrect step IDs
pred_steps = set()
for stage in predicted:
    pred_steps.update(stage.get("incorrect_step_ids", []))

# Extract ground-truth incorrect step IDs
gt_steps = set()
for stage in ground_truth:
    gt_steps.update(stage.get("incorrect_step_ids", []))

# Compute metrics
tp = len(pred_steps & gt_steps)
precision = tp / len(pred_steps) if pred_steps else 0
recall = tp / len(gt_steps) if gt_steps else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
```

## Configuration

Full configuration reference is in `src/codetracer/config/default.yaml`. Key sections:

- `llm` -- API endpoint, model name, model kwargs
- `trace` -- cost limit, step limit, prompt templates
- `discovery` -- skill generator settings (max attempts)
- `tree` -- tree builder templates
- `replay` -- replay engine settings

Override any value via a custom YAML file:

```bash
codetracer analyze <run_dir> --config my_config.yaml ...
```

## Supported Agent Formats

CodeTracer ships with built-in parsers (skills) for:

- **miniswe** -- mini-SWE-agent trajectories (`agent-logs/mini.traj.json`, `sessions/agent.log`, `commands.txt`)
- **openhands** -- OpenHands agent trajectories
- **terminus2** -- Terminus2 agent trajectories

For unknown formats, CodeTracer can auto-generate a parser via LLM (unless `--skip-discovery` is set).
