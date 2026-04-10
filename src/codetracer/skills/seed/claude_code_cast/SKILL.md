---
name: claude_code_cast
description: >
  Parse Claude Code trajectories from the terminal-bench recording format.
  Extracts tool-use steps from sessions/claude_code.log (LiteLLM proxy log
  containing full Claude API request JSON) and metadata from results.json.
  Captures thinking blocks, parallel tool calls, and the full Claude Code
  tool taxonomy.
fingerprints:
  - "sessions/claude_code.log"
  - "commands.txt"
  - "results.json"
priority: 50
metadata:
  version: "2.0"
  source: seed
---

# Claude Code Cast Trajectory Parser

## Directory Layout

```
run_dir/
  # Often contains a nested trial dir:
  #   task_name/task_name.1-of-1.YYYY-MM-DD__HH-MM-SS/
  results.json            # trial metadata: id, instruction, is_resolved, ...
  commands.txt            # tmux send-keys log (fallback parser)
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
   - `thinking` content blocks -> attached to the first step in the message
5. Map tool names to human-readable actions (see Tool Taxonomy below)
6. Task description from `results.json -> instruction`

## Parallel Tool Call Detection

When an assistant message contains multiple `tool_use` blocks, all resulting
steps are assigned the same `parallel_group` ID. This allows downstream
analysis to evaluate them as a batch rather than sequentially.

## Thinking Block Capture

Claude Code may include `type: "thinking"` content blocks in assistant
messages. The thinking text is captured in `StepRecord.thinking` on the
first step of the message. This preserves the agent's reasoning chain
for analysis.

## Tool Taxonomy

The parser formats the following Claude Code tools:

### Core Tools
- **Bash** - Shell command execution
- **Read** - File reading
- **Write** - File creation/overwrite
- **Edit** - Targeted file edits (old_string -> new_string)
- **Glob** - File pattern matching
- **Grep** - Content search (ripgrep-based)

### Agent & Planning Tools
- **Agent** - Subagent spawning (with subagent_type)
- **EnterPlanMode** / **ExitPlanMode** - Plan mode transitions
- **TodoWrite** / **TodoRead** - Task tracking
- **SendMessage** - Inter-agent messaging
- **AskUserQuestion** - User interaction

### Notebook & Web Tools
- **NotebookEdit** - Jupyter notebook cell editing
- **WebSearch** - Web search
- **WebFetch** - URL content fetching

### Workspace & Environment Tools
- **EnterWorktree** / **ExitWorktree** - Git worktree isolation
- **Skill** - Skill invocation
- **CronCreate** / **CronDelete** / **CronList** - Scheduled tasks

### Task Management Tools
- **Task** - Task subagent management
- **TaskOutput** / **TaskStop** - Background task control
- **BashOutput** - Background bash output retrieval

### Other Tools
- **PowerShell** - Windows shell execution
- **FileRead** / **FileWrite** / **FileEdit** - Alternative file tool names

## Format Fingerprint

Presence of `sessions/claude_code.log` file alongside `results.json`.
