#!/usr/bin/env python3
"""Install/start the local watchdog console as a macOS LaunchAgent."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path


LABEL = "com.sjji.watchdog.console"


def launchctl(*parts: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/bin/launchctl", *parts], text=True, capture_output=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LaunchAgent for the watchdog web console.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Project directory containing watchdog runs.")
    parser.add_argument("--run-dir", default="runs/watchdog")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--load", action="store_true", help="Load and start immediately.")
    parser.add_argument("--unload", action="store_true", help="Unload and remove the LaunchAgent.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path(args.cwd).resolve()
    run_dir = (cwd / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"

    if args.unload:
        launchctl("bootout", domain, str(plist_path))
        if plist_path.exists():
            plist_path.unlink()
        print(f"Unloaded {LABEL}")
        return 0

    console = Path(__file__).with_name("mac_watchdog_console.py").resolve()
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            str(console),
            "--cwd",
            str(cwd),
            "--run-dir",
            str(run_dir),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "WorkingDirectory": str(cwd),
        "StandardOutPath": str(run_dir / "console.launchd.out.log"),
        "StandardErrorPath": str(run_dir / "console.launchd.err.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "PATH": os.environ.get("PATH", "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"),
        },
    }

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    with plist_path.open("wb") as f:
        plistlib.dump(payload, f)

    print(f"Wrote {plist_path}")
    print(f"URL: http://{args.host}:{args.port}")

    if args.load:
        launchctl("bootout", domain, str(plist_path))
        result = launchctl("bootstrap", domain, str(plist_path))
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            return result.returncode
        kick = launchctl("kickstart", "-k", f"{domain}/{LABEL}")
        if kick.returncode != 0:
            print(kick.stderr or kick.stdout, file=sys.stderr)
            return kick.returncode
        print("Loaded and started.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
