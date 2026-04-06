---
name: openhands
description: >
  Parse OpenHands agent trajectories. OpenHands stores events as either
  sharded JSON files under sessions/sessions/*/events/ (or event_cache/)
  or as flat session JSON files under sessions/.
fingerprints:
  - "sessions/sessions/*/events"
  - "sessions/*.json"
priority: 40
metadata:
  version: "1.0"
  source: seed
---

# OpenHands Trajectory Parser

## Directory Layout

### Sharded layout
```
run_dir/
  results.json
  sessions/
    sessions/
      <session-id>/
        events/
          0.json, 1.json, ...      # Individual event files
        event_cache/
          0-99.json, 100-199.json  # Batched event shards
```

### Flat layout
```
run_dir/
  results.json
  sessions/
    <session-id>.json              # All events in one file
```

## Step Extraction Logic

- Events with action "run" or "run_ipython" are actions
- Command extracted from tool_call_metadata.args.command or args.code
- Observations linked via cause field matching action event id
- Steps are ordered by event id

## Format Fingerprint

Presence of `sessions/sessions/*/events` directory or `sessions/*.json` files.
