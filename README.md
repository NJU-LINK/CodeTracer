<p align="center">
<pre align="center">
<b>
   ██████╗ ██████╗ ██████╗ ███████╗████████╗██████╗  █████╗  ██████╗███████╗██████╗
  ██╔════╝██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗
  ██║     ██║   ██║██║  ██║█████╗     ██║   ██████╔╝███████║██║     █████╗  ██████╔╝
  ██║     ██║   ██║██║  ██║██╔══╝     ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗
  ╚██████╗╚██████╔╝██████╔╝███████╗   ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║
   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝
</b>
</pre>
</p>

<h3 align="center">Self-Evolving Agent Trajectory Diagnosis System</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2604.11641"><img src="https://img.shields.io/badge/arXiv-2604.11641-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
  <a href="https://NJU-LINK.github.io/CodeTracer/"><img src="https://img.shields.io/badge/docs-API%20Reference-blue.svg?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs"></a>
  <a href="https://huggingface.co/datasets/NJU-LINK/CodeTraceBench"><img src="https://img.shields.io/badge/🤗-CodeTraceBench-yellow.svg?style=for-the-badge" alt="Dataset"></a>
</p>
<p align="center">
  <a href="https://pypi.org/project/codetracer/"><img src="https://img.shields.io/pypi/v/codetracer.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/NJU-LINK/CodeTracer/actions"><img src="https://img.shields.io/github/actions/workflow/status/NJU-LINK/CodeTracer/ci.yml?branch=main&style=flat-square&logo=github" alt="CI"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#codetrace-bench">CodeTraceBench</a> &bull;
  <a href="#supported-agents">Supported Agents</a> &bull;
  <a href="#configuration">Configuration</a> &bull;
  <a href="#citation">Citation</a>
</p>

---

**CodeTracer** analyzes agent execution trajectories step-by-step, identifies *incorrect* and *unuseful* actions, and produces structured diagnostic labels. It operates as an autonomous diagnosis agent — navigating trajectory environments, inspecting evidence, and building root-cause chains — with cross-trajectory memory that accumulates experience over time.

> [!TIP]
> **New to CodeTracer?** Start with a single trajectory: `codetracer analyze /path/to/trajectory/ --model gpt-4o --profile detailed`

## Highlights

<table>
<tr>
<td width="50%">

**Autonomous Diagnosis Agent**
Iteratively explores trajectory data, inspects steps, gathers evidence, and produces structured error labels with full reasoning chains.

**Deep Recursive Discovery**
Three-phase trajectory discovery (marker scan → child preference → LLM-guided analysis) handles arbitrarily nested and messy data archives.

**Auto-Skill Generation**
Unknown trajectory formats are automatically parsed via LLM-generated skills; no manual parser authoring required.

</td>
<td width="50%">

**Cross-Trajectory Memory**
Online memory extraction during analysis accumulates agent-specific failure patterns and investigation strategies across runs.

**Resilient Context Management**
Two-tier compaction (LLM summarization → sliding window fallback) ensures analysis never stalls from context overflow.

**Replay Engine**
Resume failed trajectories from diagnosed breakpoints with corrective strategies injected.

</td>
</tr>
</table>

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CodeTracer Pipeline                              │
│                                                                         │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────┐  │
│   │ Discovery  │───▶│ Normalize │───▶│   Tree    │───▶│   Diagnosis   │  │
│   │           │    │           │    │  Builder  │    │    Agent      │  │
│   │ Deep scan │    │ Unify all │    │ Classify  │    │ Inspect steps │  │
│   │ + auto-   │    │ formats → │    │ steps →   │    │ + label       │  │
│   │ detect    │    │ steps.json│    │ tree.md   │    │ errors        │  │
│   └───────────┘    └───────────┘    └───────────┘    └──────┬────────┘  │
│                                                             │           │
│                              ┌───────────────┐              │           │
│                        ┌─────│    Memory      │◀─────────────┘           │
│                        │     │  Cross-traj    │                         │
│                        │     │  experience    │                         │
│                        │     └───────────────┘                         │
│                        ▼                                                │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐   │
│   │  Structured   │    │    Replay     │    │    Output Profiles    │   │
│   │  Error Labels │    │    Engine     │    │                       │   │
│   │               │    │              │    │ • tracebench (eval)   │   │
│   │ incorrect /   │    │ Resume from  │    │ • detailed (root     │   │
│   │ unuseful      │    │ breakpoint   │    │   cause analysis)    │   │
│   │ per step      │    │ + fix        │    │ • rl_feedback (RL)   │   │
│   └───────────────┘    └───────────────┘    └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/NJU-LINK/CodeTracer.git
cd CodeTracer
pip install -e .
```

### Configure LLM

```bash
export CODETRACER_API_BASE="https://api.openai.com/v1"
export CODETRACER_API_KEY="your-api-key"
```

### Analyze a Trajectory

```bash
# Single trajectory
codetracer analyze /path/to/trajectory/ \
  --model gpt-4o \
  --profile detailed

# Batch analysis
codetracer-batch \
  --manifest bench_manifest.verified.jsonl \
  --parallel 4 \
  --model gpt-4o
```

### Interactive REPL

```bash
codetracer repl /path/to/trajectory/
```

---

## Core Modules

| Module | Description |
|--------|-------------|
| `discovery.explorer` | Three-phase recursive trajectory discovery with LLM-guided fallback |
| `agents.trace_agent` | Autonomous diagnosis loop with bash execution in trajectory environments |
| `agents.compact` | Two-tier context management (LLM summarization + sliding window) |
| `services.memory` | Cross-trajectory memory with online mid-analysis extraction |
| `skills.*` | Pluggable format parsers with auto-generation for unknown formats |
| `agents.replay` | Replay engine for resuming trajectories from diagnosed breakpoints |
| `query.normalizer` | Unified trajectory normalization across all supported formats |
| `query.tree_builder` | Step classification tree construction (change/explore labeling) |

---

## CodeTrace Bench

[**CodeTraceBench**](https://huggingface.co/datasets/NJU-LINK/CodeTraceBench) is a benchmark of **4,316 agent trajectories** with human-verified step-level annotations for trajectory diagnosis evaluation.

| Split | Trajectories | Description |
|-------|:-------------|-------------|
| `verified` | 1,000 | Curated subset (489 SWE-bench + 511 TerminalBench) |
| `full` | 3,316 | All trajectories across agents and models |

> [!NOTE]
> CodeTraceBench covers **4 agents** (mini-SWE-agent, OpenHands, Terminus2, SWE-agent) × **5 models** (Claude Sonnet 4, DeepSeek-V3.2, Kimi-K2, GPT-5, Qwen3-Coder) across **26 task categories**.

### Quick Evaluation

```bash
# Download and extract
huggingface-cli download NJU-LINK/CodeTraceBench \
  --repo-type dataset \
  --local-dir ./tracebench_data

# Run on verified split
codetracer-batch \
  --manifest tracebench_data/bench_manifest.verified.jsonl \
  --model gpt-4o \
  --parallel 4 \
  --output outputs/
```

<details>
<summary><b>Dataset Fields</b></summary>

| Field | Description |
|-------|-------------|
| `traj_id` | Unique trajectory identifier |
| `agent` | Agent name (`mini-SWE-agent`, `OpenHands`, `Terminus2`, `SWE-agent`) |
| `model` | Model identifier |
| `stages` | Ground-truth stage ranges |
| `incorrect_stages` | Per-stage incorrect/unuseful step annotations |
| `solved` | Whether the agent solved the task |
| `step_count` | Total number of steps |
| `difficulty` | `easy` / `medium` / `hard` |

</details>

### Output Format

Each analysis produces structured diagnostic labels:

```json
[
  {
    "stage_id": 3,
    "incorrect_step_ids": [21, 22],
    "unuseful_step_ids": [],
    "reasoning": "Steps 21-22 edit the wrong file based on an incorrect localization hypothesis..."
  }
]
```

<details>
<summary><b>Detailed profile output (root-cause analysis)</b></summary>

```json
{
  "root_cause_chain": ["Final test failure", "Wrong file edited", "Incorrect grep interpretation"],
  "critical_decision_points": [
    {"step_id": 15, "decision": "Searched for pattern X", "should_have": "Searched for pattern Y"}
  ],
  "correct_strategy": "Should have verified file structure before editing",
  "stage_labels": [...],
  "summary": "Agent mislocalized the bug due to ambiguous grep results..."
}
```

</details>

### Evaluation Metrics

```python
from datasets import load_dataset
import json

ds = load_dataset("NJU-LINK/CodeTraceBench", split="verified")

for entry in ds:
    pred = json.load(open(f"outputs/{entry['traj_id']}/codetracer_labels.json"))
    gt = entry["incorrect_stages"]

    pred_steps = {s for stage in pred for s in stage.get("incorrect_step_ids", [])}
    gt_steps = {s for stage in gt for s in stage.get("incorrect_step_ids", [])}

    tp = len(pred_steps & gt_steps)
    precision = tp / len(pred_steps) if pred_steps else 0
    recall = tp / len(gt_steps) if gt_steps else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
```

---

## Supported Agents

CodeTracer ships with built-in parsers for major agent frameworks:

| Agent | Format | Auto-Detected |
|-------|--------|:------------:|
| [mini-SWE-agent](https://github.com/princeton-nlp/SWE-agent) | `agent-logs/mini.traj.json` + `sessions/` | Yes |
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | OpenHands event stream | Yes |
| [Terminus2](https://github.com/TerminalBench/Terminus2) | Session recordings + results | Yes |
| Custom | Any format | Via auto-generated skill |

### Auto-Skill Generation

For unknown trajectory formats, CodeTracer automatically generates a parser:

```bash
# Auto-detect and generate parser for unknown format
codetracer analyze /path/to/unknown/trajectory/ --model gpt-4o

# Generated skill is cached for future use
ls ~/.config/codetracer/skills/
```

---

## Configuration

Full configuration reference in [`src/codetracer/config/default.yaml`](src/codetracer/config/default.yaml).

| Section | Key Settings |
|---------|-------------|
| `llm` | `api_base`, `model_name`, `model_kwargs` |
| `trace` | `cost_limit`, `step_limit`, `timeout`, prompt templates |
| `discovery` | `max_depth`, `skip_dirs`, auto-generation settings |
| `memory` | `enabled`, `online_step_interval`, `online_token_threshold` |
| `output.profiles` | `tracebench`, `detailed`, `rl_feedback` |
| `replay` | `max_replay_steps`, `timeout` |

Override via custom YAML:

```bash
codetracer analyze <run_dir> --config my_config.yaml --model gpt-4o
```

### Output Profiles

| Profile | Output File | Description |
|---------|-----------|-------------|
| `tracebench` | `codetracer_labels.json` | Stage-level labels (benchmark evaluation) |
| `detailed` | `codetracer_analysis.json` | Root cause chains + critical decision points |
| `rl_feedback` | `codetracer_rl_feedback.json` | Per-step deviation analysis for RL training |

```bash
# Use specific profile
codetracer analyze <run_dir> --profile rl_feedback --model gpt-4o
```

---

## Project Structure

```
CodeTracer/
├── src/codetracer/
│   ├── agents/           # Core agent loops (trace, replay, compact)
│   ├── cli/              # CLI commands and interactive REPL
│   ├── config/           # Default configuration
│   ├── discovery/        # Deep recursive trajectory discovery
│   ├── llm/              # LLM client with Azure AD support
│   ├── models/           # Data models (trajectory, analysis, replay)
│   ├── plugins/          # Hook system and adapter interface
│   ├── query/            # Normalizer, tree builder, config loader
│   ├── scripts/          # Batch runner, dev tools, analysis scripts
│   ├── services/         # Memory, cost tracking, validation, complexity
│   ├── skills/           # Format parsers (built-in + generated)
│   ├── state/            # Output profiles and session persistence
│   └── utils/            # Template rendering, report generation
├── data/                 # Trajectory datasets
├── scripts/              # Shell scripts for batch operations
└── tests/                # Test suite
```

---

<details>
<summary><b>CLI Reference</b></summary>

```
Usage: codetracer [COMMAND] [OPTIONS]

Commands:
  analyze         Run trajectory diagnosis on a single run directory
  run             Full pipeline: normalize → tree → analyze → replay
  replay          Resume a trajectory from a diagnosed breakpoint
  repl            Interactive trajectory exploration shell
  normalize       Normalize a trajectory to steps.json format
  tree            Build step classification tree
  batch           Run batch analysis from manifest

Global Options:
  --model TEXT     LLM model name
  --api-base URL   API endpoint
  --api-key TEXT   API key
  --config PATH   Custom configuration file
  --profile TEXT   Output profile (tracebench/detailed/rl_feedback)
  --cost-limit $   Max LLM spend per trajectory (default: 3.0)
  --dry-run       Normalize + tree only, skip LLM analysis
```

</details>

## Citation

If you use CodeTracer or CodeTraceBench in your research, please cite:

```bibtex
@article{li2026codetracer,
  title={CodeTracer: Towards Traceable Agent States},
  author={Li, Han and Yao, Yifan and Zhu, Letian and Feng, Rili and Ye, Hongyi and Wang, Jiaming and He, Yancheng and Zou, Pengyu and Zhang, Lehan and Lei, Xinping and Huang, Haoyang and Deng, Ken and Sun, Ming and Zhang, Zhaoxiang and Ye, He and Liu, Jiaheng},
  journal={arXiv preprint arXiv:2604.11641},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with care at <a href="https://cs.nju.edu.cn">Nanjing University</a></sub>
</p>
