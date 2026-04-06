---
name: miniswe
description: >
  Parse MiniSWE agent trajectories. MiniSWE stores data in one of three
  formats: agent-logs/mini.traj.json (JSON with messages array),
  sessions/agent.log (structured text log), or commands.txt (keystroke lists).
fingerprints:
  - "agent-logs/mini.traj.json"
  - "sessions/agent.log"
  - "commands.txt"
priority: 10
metadata:
  version: "1.0"
  source: seed
---

# MiniSWE Trajectory Parser

## Directory Layout

### Primary: traj.json
```
run_dir/
  results.json
  agent-logs/
    mini.traj.json     # {messages: [{role, content, timestamp}, ...]}
```

### Alternative: agent.log
```
run_dir/
  sessions/
    agent.log          # Structured text with ```bash blocks and <returncode>
```

### Fallback: commands.txt
```
run_dir/
  commands.txt         # Each line is a Python list literal: ["cmd", "Enter"]
```

## Step Extraction Logic

- **traj.json**: Assistant messages with ```bash blocks are actions; next user message with `<returncode>` is observation
- **agent.log**: Scan for ```bash fences, pair with following `<returncode>` lines
- **commands.txt**: Each line is `ast.literal_eval` to get keystroke list, last element must be "Enter"

## Format Fingerprint

Presence of `agent-logs/mini.traj.json`, `sessions/agent.log`, or `commands.txt`.
