#!/usr/bin/env python3
"""Install and start an experiment watchdog as a macOS LaunchAgent.

This lets long SJJI runs survive terminal/session churn while still using the
Mac-native notification system.

Example:
    python scripts/mac_launch_experiment.py --name baseline_native --load -- \
      ./venv/bin/python -u baseline.py
"""

from __future__ import annotations

import argparse
import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path


def safe_label(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip("-.")
    return value or "experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a macOS LaunchAgent for a watched experiment.")
    parser.add_argument("--name", required=True, help="Stable run name, e.g. baseline_native.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for the experiment.")
    parser.add_argument("--run-dir", default="runs/watchdog", help="Status/event/log directory relative to cwd.")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--check-interval", type=int, default=15)
    parser.add_argument("--load", action="store_true", help="Load and start the LaunchAgent immediately.")
    parser.add_argument("--unload", action="store_true", help="Unload an existing LaunchAgent for this name.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.unload and not args.command:
        parser.error("command is required unless --unload is used")
    return args


def launchctl(*parts: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/bin/launchctl", *parts], text=True, capture_output=True)


def main() -> int:
    args = parse_args()
    cwd = Path(args.cwd).resolve()
    run_dir = (cwd / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    label = f"com.sjji.experiment.{safe_label(args.name)}"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    watchdog = Path(__file__).with_name("mac_experiment_watchdog.py").resolve()
    domain = f"gui/{os.getuid()}"

    if args.unload:
        launchctl("bootout", domain, str(plist_path))
        if plist_path.exists():
            plist_path.unlink()
        print(f"Unloaded {label}")
        return 0

    program_args = [
        sys.executable,
        str(watchdog),
        "--name",
        args.name,
        "--cwd",
        str(cwd),
        "--run-dir",
        str(run_dir),
        "--stall-seconds",
        str(args.stall_seconds),
        "--check-interval",
        str(args.check_interval),
        "--",
        *args.command,
    ]

    payload = {
        "Label": label,
        "ProgramArguments": program_args,
        "RunAtLoad": True,
        "KeepAlive": False,
        "WorkingDirectory": str(cwd),
        "StandardOutPath": str(run_dir / f"{args.name}.launchd.out.log"),
        "StandardErrorPath": str(run_dir / f"{args.name}.launchd.err.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "PATH": os.environ.get("PATH", "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"),
        },
    }

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    with plist_path.open("wb") as f:
        plistlib.dump(payload, f)

    print(f"Wrote {plist_path}")
    print(f"Label: {label}")
    print(f"Status: {run_dir / (args.name + '.status.json')}")
    print(f"Log:    {run_dir / (args.name + '.log')}")

    if args.load:
        # Replace an older version if it exists.
        launchctl("bootout", domain, str(plist_path))
        result = launchctl("bootstrap", domain, str(plist_path))
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            return result.returncode
        kick = launchctl("kickstart", "-k", f"{domain}/{label}")
        if kick.returncode != 0:
            print(kick.stderr or kick.stdout, file=sys.stderr)
            return kick.returncode
        print("Loaded and started.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
