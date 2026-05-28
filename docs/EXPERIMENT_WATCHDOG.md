# Mac Experiment Watchdog

Long-running SJJI experiments should run through the Mac watchdog instead of a
bare terminal command. The watchdog gives us:

- macOS notifications on start, stall, crash, and completion
- a live experiment log
- a structured status JSON file
- a JSONL event history for progress/error events
- optional launchd integration so runs survive terminal/session churn
- a local web console for status, logs, events, and safe controls

## Local Console

Start the console directly while developing:

```bash
cd /Users/edward/Projects/sjji-eeg
./venv/bin/python scripts/mac_watchdog_console.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Keep it running as a Mac LaunchAgent:

```bash
cd /Users/edward/Projects/sjji-eeg
./venv/bin/python scripts/mac_launch_console.py --load --port 8765
```

Stop and remove the console LaunchAgent:

```bash
./venv/bin/python scripts/mac_launch_console.py --unload
```

The console reads the same local files as the watchdog. It does not require a
database or external service.

The console currently includes:

- a global OpenClaw/Mac scope shell with domain/project breadcrumbs
- a live Mac system strip for host, load, LaunchAgents, local listeners, and tracked processes
- an attention queue for stalled, failed, quiet, or active watched runs
- a domain registry for Algorithms, System Runtime, Local Services, and Agent Surface
- a global activity feed built from watchdog event streams
- auto-refreshing run inventory and health counters
- selectable run detail panes for overview, logs, event history, and raw status
- a system detail pane for local process/service inventory
- progress-age visibility separate from status-file heartbeat age
- safe stop control for active LaunchAgent-backed runs

## Direct Run

Use this when you are already in a stable terminal/tmux session:

```bash
cd /Users/edward/Projects/sjji-eeg
./venv/bin/python scripts/mac_experiment_watchdog.py --name baseline_native --stall-seconds 900 -- ./venv/bin/python -u baseline.py
```

Outputs:

```text
runs/watchdog/baseline_native.log
runs/watchdog/baseline_native.status.json
runs/watchdog/baseline_native.events.jsonl
```

## LaunchAgent Run

Use this for overnight or high-value runs. It writes and starts a user
LaunchAgent under `~/Library/LaunchAgents`.

```bash
cd /Users/edward/Projects/sjji-eeg
./venv/bin/python scripts/mac_launch_experiment.py --name baseline_native --load --stall-seconds 900 -- ./venv/bin/python -u baseline.py
```

Stop and remove it:

```bash
./venv/bin/python scripts/mac_launch_experiment.py --name baseline_native --unload
```

## Check Status

```bash
cat runs/watchdog/baseline_native.status.json
tail -80 runs/watchdog/baseline_native.log
tail -20 runs/watchdog/baseline_native.events.jsonl
```

`state` values:

- `running`
- `completed`
- `failed`

Important fields:

- `pid`: active child process PID
- `last_progress_line`: most recent fold/epoch/results line seen
- `last_error_line`: most recent traceback/error line seen
- `stalled`: true when no log/progress has appeared beyond the configured threshold
- `exit_code`: process exit code after completion/failure

## Control Plane Contract

Every watched system should emit the same minimum shape:

- `<name>.status.json`: current state, PID, command, timestamps, progress, error, stall flag
- `<name>.events.jsonl`: append-only timeline of starts, progress, errors, stalls, exits
- `<name>.log`: raw stdout/stderr tail for detailed inspection
- LaunchAgent label: `com.sjji.experiment.<name>` for Mac-native lifecycle control

That contract is the bridge from experiment watchdog to broader local control
plane. Other microapps, crawlers, ontology jobs, and OpenClaw tasks can join the
console once they emit compatible status/events/log files.

## Current Policy

Run every experiment expected to last more than 10 minutes through the watchdog.
If a run crashes or stalls, fix the underlying issue and restart through the
watchdog again; do not silently continue with stale logs.
