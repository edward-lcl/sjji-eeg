#!/usr/bin/env python3
"""SJJI Experiment Scheduler — resource-aware job orchestrator.

Polls system state every N seconds, decides whether conditions allow launching
the next queued experiment, and fires it via mac_launch_experiment.py so it
runs as a supervised LaunchAgent (survives terminal closes, gets watchdog
stall-detection and macOS notifications).

Usage:
    # One-shot status check (what would run next, and why/why not):
    python scripts/mac_scheduler.py --status

    # Run the scheduler daemon (loops forever, launches jobs when resources clear):
    python scripts/mac_scheduler.py --daemon

    # Edit the queue:
    python scripts/mac_scheduler.py --queue-show
    python scripts/mac_scheduler.py --queue-add '{"name":"baseline_v2","cmd":["./venv/bin/python","-u","baseline.py"],"priority":10}'
    python scripts/mac_scheduler.py --queue-skip baseline_v2

Queue file: runs/scheduler/queue.json
Status file: runs/scheduler/status.json  (written every poll — feed this to a dashboard)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Thresholds ─────────────────────────────────────────────────────────────── #

RAM_FREE_GB_MIN      = 10.0   # don't launch if less than this is free
DISK_FREE_GB_MIN     = 20.0   # don't launch if disk is tight
POLL_INTERVAL_SEC    = 30     # how often the daemon checks
STALL_SECONDS        = 1800   # passed through to watchdog

# Process name fragments that indicate a heavy ML job is running
ML_PROCESS_PATTERNS  = [
    "baseline.py", "ssl_pilot.py", "train.py",
    "fingerprint", "subject_level", "tuh_ingest",
]
# Process name fragments that indicate data pipeline is running (allow ML alongside)
DATA_PIPELINE_PATTERNS = [
    "tuh_pipeline.sh", "aws s3 sync", "rsync.*nedc",
]


# ── Paths ──────────────────────────────────────────────────────────────────── #

REPO_ROOT   = Path(__file__).resolve().parent.parent
SCHED_DIR   = REPO_ROOT / "runs" / "scheduler"
QUEUE_FILE  = SCHED_DIR / "queue.json"
STATUS_FILE = SCHED_DIR / "status.json"
LAUNCH_SCRIPT = Path(__file__).with_name("mac_launch_experiment.py").resolve()


# ── Data types ─────────────────────────────────────────────────────────────── #

@dataclass
class Job:
    name: str
    cmd: list[str]
    priority: int = 50        # lower = higher priority
    status: str = "pending"   # pending | running | done | skipped | failed
    launched_at: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SystemState:
    ram_free_gb: float
    ram_total_gb: float
    ram_pressure: str          # normal | warn | critical (from memory_pressure)
    disk_free_gb: float
    disk_total_gb: float
    ml_jobs_running: list[str]
    data_pipelines_running: list[str]
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── System probes ───────────────────────────────────────────────────────────── #

def probe_system() -> SystemState:
    import subprocess, shutil

    # RAM — use vm_stat for free pages
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        pages_free = pages_inactive = pages_speculative = 0
        page_size = 4096
        for line in vm.stdout.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif "Pages free:" in line:
                pages_free = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split()[-1].rstrip("."))
            elif "Pages speculative:" in line:
                pages_speculative = int(line.split()[-1].rstrip("."))
        free_bytes = (pages_free + pages_speculative) * page_size
        # conservative: only count truly free pages
        ram_free_gb = free_bytes / 1e9
    except Exception:
        ram_free_gb = 0.0

    try:
        import subprocess
        sysctl = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram_total_gb = int(sysctl.stdout.strip()) / 1e9
    except Exception:
        ram_total_gb = 48.0

    # Memory pressure indicator
    try:
        mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
        pressure_line = [l for l in mp.stdout.splitlines() if "System-wide memory free percentage" in l or "pressure" in l.lower()]
        raw = mp.stdout.lower()
        if "critical" in raw:
            pressure = "critical"
        elif "warn" in raw:
            pressure = "warn"
        else:
            pressure = "normal"
    except Exception:
        pressure = "unknown"

    # Disk
    try:
        stat = shutil.disk_usage(str(REPO_ROOT))
        disk_free_gb  = stat.free  / 1e9
        disk_total_gb = stat.total / 1e9
    except Exception:
        disk_free_gb = disk_total_gb = 0.0

    # Running processes
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        procs = ps.stdout
    except Exception:
        procs = ""

    ml_running   = [p for p in ML_PROCESS_PATTERNS      if p in procs]
    data_running = [p for p in DATA_PIPELINE_PATTERNS    if any(pat in procs for pat in [p])]

    return SystemState(
        ram_free_gb=round(ram_free_gb, 2),
        ram_total_gb=round(ram_total_gb, 1),
        ram_pressure=pressure,
        disk_free_gb=round(disk_free_gb, 1),
        disk_total_gb=round(disk_total_gb, 1),
        ml_jobs_running=ml_running,
        data_pipelines_running=data_running,
    )


def can_launch(state: SystemState) -> tuple[bool, str]:
    """Return (ok, reason_string)."""
    if state.ml_jobs_running:
        return False, f"ML job already running: {state.ml_jobs_running}"
    if state.ram_pressure == "critical":
        return False, "Memory pressure is CRITICAL"
    if state.ram_free_gb < RAM_FREE_GB_MIN:
        return False, f"RAM free {state.ram_free_gb:.1f}GB < {RAM_FREE_GB_MIN}GB threshold"
    if state.disk_free_gb < DISK_FREE_GB_MIN:
        return False, f"Disk free {state.disk_free_gb:.1f}GB < {DISK_FREE_GB_MIN}GB threshold"
    return True, "ok"


# ── Queue I/O ───────────────────────────────────────────────────────────────── #

def load_queue() -> list[Job]:
    if not QUEUE_FILE.exists():
        return []
    data = json.loads(QUEUE_FILE.read_text())
    return [Job.from_dict(d) for d in data]


def save_queue(jobs: list[Job]) -> None:
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps([asdict(j) for j in jobs], indent=2) + "\n")


def next_pending(jobs: list[Job]) -> Job | None:
    pending = [j for j in jobs if j.status == "pending"]
    if not pending:
        return None
    return sorted(pending, key=lambda j: j.priority)[0]


# ── Launcher ────────────────────────────────────────────────────────────────── #

def launch_job(job: Job) -> bool:
    cmd = [
        sys.executable, str(LAUNCH_SCRIPT),
        "--name", job.name,
        "--cwd", str(REPO_ROOT),
        "--stall-seconds", str(STALL_SECONDS),
        "--load", "--",
        *job.cmd,
    ]
    print(f"[scheduler] Launching: {job.name}")
    print(f"  cmd: {' '.join(shlex.quote(c) for c in job.cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


# ── Status writer ────────────────────────────────────────────────────────────── #

def write_status(state: SystemState, jobs: list[Job], gate: tuple[bool, str]) -> None:
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "system": asdict(state),
        "can_launch": gate[0],
        "gate_reason": gate[1],
        "queue": [asdict(j) for j in jobs],
        "next_job": asdict(next_pending(jobs)) if next_pending(jobs) else None,
    }
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(STATUS_FILE)


# ── CLI ─────────────────────────────────────────────────────────────────────── #

def cmd_status() -> None:
    state = probe_system()
    jobs  = load_queue()
    gate  = can_launch(state)
    write_status(state, jobs, gate)

    print(f"\n{'='*55}")
    print(f"  SJJI Scheduler — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")
    print(f"  RAM free:      {state.ram_free_gb:.1f} GB / {state.ram_total_gb:.0f} GB  (pressure: {state.ram_pressure})")
    print(f"  Disk free:     {state.disk_free_gb:.1f} GB / {state.disk_total_gb:.0f} GB")
    print(f"  ML running:    {state.ml_jobs_running or 'none'}")
    print(f"  Data pipelines:{state.data_pipelines_running or 'none'}")
    print(f"  Can launch:    {'✅ YES' if gate[0] else '🚫 NO'} — {gate[1]}")
    nxt = next_pending(jobs)
    print(f"  Next job:      {nxt.name if nxt else '(queue empty)'}")
    print(f"{'='*55}")
    if jobs:
        print(f"\n  Queue ({len(jobs)} jobs):")
        for j in sorted(jobs, key=lambda x: x.priority):
            icon = {"pending":"⏳","running":"🔄","done":"✅","skipped":"⏭️","failed":"❌"}.get(j.status,"?")
            print(f"    {icon} [{j.priority:3d}] {j.name:<30} {j.status:<10} {j.note}")
    print()


def cmd_daemon() -> None:
    print(f"[scheduler] Daemon started. Poll interval: {POLL_INTERVAL_SEC}s")
    print(f"  Status file: {STATUS_FILE}")
    while True:
        state = probe_system()
        jobs  = load_queue()
        gate  = can_launch(state)
        write_status(state, jobs, gate)

        if gate[0]:
            nxt = next_pending(jobs)
            if nxt:
                print(f"[scheduler] Gate open — launching {nxt.name}")
                ok = launch_job(nxt)
                nxt.status  = "running" if ok else "failed"
                nxt.launched_at = datetime.now(timezone.utc).isoformat()
                nxt.note = "launched by scheduler" if ok else "launch failed"
                save_queue(jobs)
            else:
                print(f"[scheduler] Gate open but queue empty — nothing to do")
        else:
            nxt = next_pending(jobs)
            if nxt:
                print(f"[scheduler] Waiting: {gate[1]}")

        time.sleep(POLL_INTERVAL_SEC)


def cmd_queue_show() -> None:
    jobs = load_queue()
    if not jobs:
        print("Queue is empty. Add jobs with --queue-add.")
        return
    for j in sorted(jobs, key=lambda x: x.priority):
        print(json.dumps(asdict(j), indent=2))


def cmd_queue_add(raw: str) -> None:
    data = json.loads(raw)
    jobs = load_queue()
    job  = Job.from_dict(data)
    jobs.append(job)
    save_queue(jobs)
    print(f"Added: {job.name} (priority {job.priority})")


def cmd_queue_skip(name: str) -> None:
    jobs = load_queue()
    for j in jobs:
        if j.name == name:
            j.status = "skipped"
            save_queue(jobs)
            print(f"Skipped: {name}")
            return
    print(f"Not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--status",      action="store_true", help="One-shot system + queue status")
    g.add_argument("--daemon",      action="store_true", help="Run scheduler loop forever")
    g.add_argument("--queue-show",  action="store_true", help="Print current queue")
    g.add_argument("--queue-add",   metavar="JSON",      help="Add a job (JSON string)")
    g.add_argument("--queue-skip",  metavar="NAME",      help="Skip a pending job by name")
    args = parser.parse_args()

    if args.status:      cmd_status()
    elif args.daemon:    cmd_daemon()
    elif args.queue_show: cmd_queue_show()
    elif args.queue_add:  cmd_queue_add(args.queue_add)
    elif args.queue_skip: cmd_queue_skip(args.queue_skip)


if __name__ == "__main__":
    main()
