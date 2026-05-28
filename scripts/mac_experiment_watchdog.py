#!/usr/bin/env python3
"""Mac-native experiment watchdog.

Runs an experiment command, writes structured status/event files, watches log
progress, and posts macOS notifications on start, stall, crash, and completion.

Example:
    python scripts/mac_experiment_watchdog.py \
      --name baseline_native \
      --stall-seconds 900 \
      -- ./venv/bin/python -u baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROGRESS_PATTERNS = [
    re.compile(r"Fold\s+\d+", re.IGNORECASE),
    re.compile(r"Mean:\s+bal_acc=", re.IGNORECASE),
    re.compile(r"epoch\s+\d+/\d+", re.IGNORECASE),
    re.compile(r"Results saved", re.IGNORECASE),
    re.compile(r"All done", re.IGNORECASE),
]

ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"RuntimeError:"),
    re.compile(r"ValueError:"),
    re.compile(r"Error:", re.IGNORECASE),
    re.compile(r"Exception:", re.IGNORECASE),
]


@dataclass
class WatchdogStatus:
    name: str
    pid: int | None
    command: list[str]
    cwd: str
    state: str
    started_at: str
    updated_at: str
    exit_code: int | None = None
    log_path: str | None = None
    last_log_update_at: str | None = None
    last_progress_at: str | None = None
    last_progress_line: str | None = None
    last_error_line: str | None = None
    stall_seconds: int = 0
    stalled: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": utc_now(), "event": event, **fields}
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def notify(title: str, body: str, enabled: bool = True) -> None:
    if not enabled or sys.platform != "darwin":
        return
    # osascript is built into macOS. Avoid shell=True so experiment output cannot
    # alter the notification command.
    script = (
        'display notification '
        + json.dumps(body)
        + ' with title '
        + json.dumps(title)
        + ' sound name "Glass"'
    )
    subprocess.run(["/usr/bin/osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_new_lines(log_path: Path, offset: int) -> tuple[list[str], int]:
    if not log_path.exists():
        return [], offset
    size = log_path.stat().st_size
    if size < offset:
        offset = 0
    with log_path.open("rb") as f:
        f.seek(offset)
        chunk = f.read()
        offset = f.tell()
    text = chunk.decode("utf-8", errors="replace")
    return text.splitlines(), offset


def classify_line(line: str) -> str | None:
    if any(p.search(line) for p in ERROR_PATTERNS):
        return "error"
    if any(p.search(line) for p in PROGRESS_PATTERNS):
        return "progress"
    return None


def terminate_child(proc: subprocess.Popen[Any], event_path: Path, signum: int, frame: Any) -> None:
    append_event(event_path, "watchdog_signal", signal=signum)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    raise SystemExit(128 + signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and monitor a local experiment on macOS.")
    parser.add_argument("--name", required=True, help="Stable run name, e.g. baseline_native.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for the command.")
    parser.add_argument("--run-dir", default="runs/watchdog", help="Directory for status, events, and logs.")
    parser.add_argument("--log", default=None, help="Log file path. Defaults to <run-dir>/<name>.log.")
    parser.add_argument("--check-interval", type=int, default=15, help="Seconds between checks.")
    parser.add_argument("--stall-seconds", type=int, default=900, help="Warn if log/progress is quiet this long.")
    parser.add_argument("--no-notify", action="store_true", help="Disable macOS notifications.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, after --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required; pass it after --")
    return args


def main() -> int:
    args = parse_args()
    cwd = Path(args.cwd).resolve()
    run_dir = (cwd / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    log_path = Path(args.log).resolve() if args.log else run_dir / f"{args.name}.log"
    status_path = run_dir / f"{args.name}.status.json"
    event_path = run_dir / f"{args.name}.events.jsonl"
    notify_enabled = not args.no_notify

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    append_event(event_path, "starting", command=args.command, cwd=str(cwd), log_path=str(log_path))
    notify("SJJI experiment started", f"{args.name}: {shlex.join(args.command)}", notify_enabled)

    with log_path.open("ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            args.command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        signal.signal(signal.SIGINT, lambda signum, frame: terminate_child(proc, event_path, signum, frame))
        signal.signal(signal.SIGTERM, lambda signum, frame: terminate_child(proc, event_path, signum, frame))

        started_at = utc_now()
        last_log_update = time.time()
        last_progress = time.time()
        last_progress_line: str | None = None
        last_error_line: str | None = None
        warned_stall = False
        offset = 0

        while True:
            now = time.time()
            lines, offset = read_new_lines(log_path, offset)
            if lines:
                last_log_update = now
            for line in lines:
                kind = classify_line(line)
                if kind == "progress":
                    last_progress = now
                    last_progress_line = line[-500:]
                    warned_stall = False
                    append_event(event_path, "progress", line=last_progress_line)
                elif kind == "error":
                    last_error_line = line[-500:]
                    append_event(event_path, "error_line", line=last_error_line)
                    notify("SJJI experiment error line", f"{args.name}: {last_error_line}", notify_enabled)

            quiet_for = int(now - max(last_log_update, last_progress))
            stalled = quiet_for >= args.stall_seconds
            if stalled and not warned_stall and proc.poll() is None:
                warned_stall = True
                append_event(event_path, "stall_warning", quiet_for=quiet_for)
                notify("SJJI experiment stalled", f"{args.name}: no log/progress for {quiet_for}s", notify_enabled)

            exit_code = proc.poll()
            state = "running" if exit_code is None else ("completed" if exit_code == 0 else "failed")
            status = WatchdogStatus(
                name=args.name,
                pid=proc.pid if exit_code is None else None,
                command=args.command,
                cwd=str(cwd),
                state=state,
                started_at=started_at,
                updated_at=utc_now(),
                exit_code=exit_code,
                log_path=str(log_path),
                last_log_update_at=datetime.fromtimestamp(last_log_update, timezone.utc).isoformat(),
                last_progress_at=datetime.fromtimestamp(last_progress, timezone.utc).isoformat(),
                last_progress_line=last_progress_line,
                last_error_line=last_error_line,
                stall_seconds=args.stall_seconds,
                stalled=stalled and exit_code is None,
            )
            atomic_write_json(status_path, asdict(status))

            if exit_code is not None:
                append_event(event_path, state, exit_code=exit_code)
                title = "SJJI experiment completed" if exit_code == 0 else "SJJI experiment failed"
                body = f"{args.name}: exit {exit_code}"
                if last_error_line:
                    body += f"; {last_error_line}"
                notify(title, body, notify_enabled)
                return int(exit_code)

            time.sleep(max(args.check_interval, 1))


if __name__ == "__main__":
    raise SystemExit(main())
