---
name: terminus2
description: >
  Parse Terminus2 agent trajectories. Terminus2 stores each episode as
  agent-logs/episode-N/response.txt (JSON with commands/analysis/plan)
  and the next episode's prompt.txt contains the observation
  (New Terminal Output section).
fingerprints:
  - "agent-logs/episode-*/response.txt"
priority: 50
metadata:
  version: "1.0"
  source: seed
---

# Terminus2 Trajectory Parser

## Directory Layout

```
run_dir/
  results.json                  # Task metadata
  agent-logs/
    episode-0/response.txt      # JSON: {commands, analysis, plan}
    episode-1/prompt.txt         # Contains "New Terminal Output:" section
    episode-1/response.txt
    ...
```

## Step Extraction Logic

- Each episode-N/response.txt is one action (step_id = N+1)
- Observation comes from episode-(N+1)/prompt.txt
- Commands field contains keystrokes list
- If response.txt is valid JSON, extract commands; otherwise use raw text

## Format Fingerprint

Presence of `agent-logs/episode-*/response.txt` files.
