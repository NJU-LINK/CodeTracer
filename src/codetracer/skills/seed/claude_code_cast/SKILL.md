---
name: claude_code_cast
description: >
  Parse Claude Code trajectories from the terminal-bench recording format.
  Extracts tool-use steps from sessions/claude_code.log (LiteLLM proxy log
  containing full Claude API request JSON) and metadata from results.json.
fingerprints:
  - "sessions/claude_code.log"
  - "commands.txt"
  - "results.json"
priority: 50
metadata:
  version: "1.0"
  source: seed
---

# Claude Code Cast Trajectory Parser

## Directory Layout

```
run_dir/
  # Often contains a nested trial dir:
  #   task_name/task_name.1-of-1.YYYY-MM-DD__HH-MM-SS/
  results.json            # trial metadata: id, instruction, is_resolved, ...
  commands.txt            # tmux send-keys log (not directly parsed)
  sessions/
    claude_code.log       # LiteLLM request/response log (primary source)
    agent.cast            # asciinema recording of agent session
    agent.log             # (optional) agent output log
    tests.cast            # asciinema recording of test run
    tests.log             # (optional) test output log
  panes/
    pre-agent.txt         # terminal state before agent
    post-agent.txt        # terminal state after agent
    post-test.txt         # terminal state after test
  agent-logs/             # (usually empty)
```

## Step Extraction Logic

1. Read `sessions/claude_code.log` line by line, parsing JSON objects
2. Collect all requests containing both `messages` and `tools` keys
   (these are the main Claude Code agent conversation turns)
3. Take the **last** such request (it contains the full accumulated conversation)
4. Walk through `messages`, extracting:
   - `assistant` messages with `tool_use` content blocks -> action steps
   - `tool` messages matched by `tool_call_id` -> observations
5. Map tool names (Bash, Write, Edit, Read, etc.) to human-readable actions
6. Task description from `results.json -> instruction`

## Format Fingerprint

Presence of `sessions/claude_code.log` file alongside `results.json`.
