#!/usr/bin/env python3
"""Local web console for Mac watchdog runs.

The console is intentionally stdlib-only so it can run on a clean macOS Python
without becoming another service dependency. It reads the watchdog's structured
status/events/log files and exposes a small localhost UI plus JSON endpoints.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_RUN_DIR = "runs/watchdog"


def safe_label(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip("-.")
    return value or "experiment"


def atomic_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()[-limit:]
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "unparseable", "line": line})
    return events


def read_text_tail(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - max_bytes))
        data = f.read()
    return data.decode("utf-8", errors="replace")


def launchctl(*parts: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/bin/launchctl", *parts], text=True, capture_output=True)


def run_command(command: list[str], timeout: float = 2.0) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0 and not result.stdout:
        return result.stderr.strip()
    return result.stdout.strip()


def system_snapshot() -> dict[str, Any]:
    terms = ("openclaw", "sjji", "atlas", "uvicorn", "python", "node", "codex", "chrome")
    ps_output = run_command(["/bin/ps", "-axo", "pid=,pcpu=,pmem=,command="], timeout=3.0)
    processes: list[dict[str, Any]] = []
    for line in ps_output.splitlines()[1:]:
        if not any(term in line.lower() for term in terms):
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        command_path = parts[3].split(None, 1)[0]
        processes.append(
            {
                "pid": parts[0],
                "cpu": parts[1],
                "mem": parts[2],
                "command": Path(command_path).name,
                "args": parts[3][-220:],
            }
        )

    launch_output = run_command(["/bin/launchctl", "print", f"gui/{os.getuid()}"], timeout=3.0)
    labels: list[str] = []
    for line in launch_output.splitlines():
        match = re.search(r"\b(com\.(?:sjji|openclaw|atlas)[^\s=\"]+)", line)
        if match:
            labels.append(match.group(1))

    lsof_output = run_command(["/usr/sbin/lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], timeout=3.0)
    ports: list[dict[str, str]] = []
    for line in lsof_output.splitlines()[1:]:
        columns = line.split()
        if len(columns) < 9:
            continue
        name = columns[0]
        address = columns[-2] if columns[-1] == "(LISTEN)" else columns[-1]
        if any(term in name.lower() or term in line.lower() for term in terms):
            ports.append({"process": name, "pid": columns[1], "address": address})

    return {
        "host": platform.node() or "mac",
        "system": platform.platform(),
        "load": [round(value, 2) for value in os.getloadavg()],
        "uptime": run_command(["/usr/bin/uptime"], timeout=1.0),
        "processes": processes[:18],
        "launch_agents": sorted(set(labels))[:18],
        "ports": ports[:18],
        "ts": time.time(),
    }


def artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".status.json"):
        return "status"
    if name.endswith(".events.jsonl"):
        return "events"
    if name.endswith(".log") or name.endswith(".out") or name.endswith(".err"):
        return "log"
    if path.suffix.lower() in {".json", ".csv", ".npy", ".npz", ".pt", ".pth"}:
        return "data"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return "media"
    return path.suffix.lower().lstrip(".") or "file"


def collect_artifacts(cwd: Path, run_dir: Path, limit: int = 36) -> list[dict[str, Any]]:
    roots = [cwd / "results", run_dir]
    seen: set[Path] = set()
    artifacts: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                stat = path.stat()
            except OSError:
                continue
            artifacts.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "rel_path": str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path),
                    "kind": artifact_kind(path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    return sorted(artifacts, key=lambda item: float(item["mtime"]), reverse=True)[:limit]


DATASET_RE = re.compile(r"^---\s+(?P<name>ds\d+)\s+\((?P<meta>.*?)\)\s+---$")
FOLD_RE = re.compile(
    r"^\s*Fold\s+(?P<fold>\d+):\s+"
    r"bal_acc=(?P<bal_acc>-?\d+(?:\.\d+)?)\s+"
    r"sens=(?P<sens>-?\d+(?:\.\d+)?)\s+"
    r"spec=(?P<spec>-?\d+(?:\.\d+)?)"
)
MEAN_RE = re.compile(
    r"^\s*Mean:\s+"
    r"bal_acc=(?P<bal_acc>-?\d+(?:\.\d+)?)\s+"
    r"sens=(?P<sens>-?\d+(?:\.\d+)?)\s+"
    r"spec=(?P<spec>-?\d+(?:\.\d+)?)"
)


def metric_values(match: re.Match[str]) -> dict[str, float]:
    return {
        "bal_acc": float(match.group("bal_acc")),
        "sens": float(match.group("sens")),
        "spec": float(match.group("spec")),
    }


def parse_run_metrics(log_text: str) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in log_text.splitlines():
        dataset_match = DATASET_RE.match(raw_line)
        if dataset_match:
            current = {
                "name": dataset_match.group("name"),
                "meta": dataset_match.group("meta"),
                "folds": [],
                "mean": None,
                "state": "running",
            }
            datasets.append(current)
            continue

        if not current:
            continue

        fold_match = FOLD_RE.match(raw_line)
        if fold_match:
            current["folds"].append(
                {
                    "fold": int(fold_match.group("fold")),
                    **metric_values(fold_match),
                }
            )
            continue

        mean_match = MEAN_RE.match(raw_line)
        if mean_match:
            current["mean"] = metric_values(mean_match)
            current["state"] = "completed"

    completed = [dataset for dataset in datasets if dataset.get("mean")]
    active = next((dataset for dataset in reversed(datasets) if dataset.get("state") == "running"), None)
    fold_count = sum(len(dataset.get("folds", [])) for dataset in datasets)
    best_completed = None
    if completed:
        best_completed = max(completed, key=lambda item: float(item["mean"]["bal_acc"]))

    insights: list[dict[str, str]] = []
    if not datasets:
        insights.append(
            {
                "level": "info",
                "title": "Waiting for metrics",
                "body": "The run has not emitted dataset fold results yet.",
            }
        )
    if completed:
        avg_bal_acc = sum(float(dataset["mean"]["bal_acc"]) for dataset in completed) / len(completed)
        if avg_bal_acc < 0.60:
            insights.append(
                {
                    "level": "warn",
                    "title": "Baseline remains weak",
                    "body": f"Completed datasets average {avg_bal_acc * 100:.1f}% bal_acc; this still points toward reproducing TransformEEG more faithfully before treating SSL gains as conclusive.",
                }
            )
        else:
            insights.append(
                {
                    "level": "info",
                    "title": "Baseline has signal",
                    "body": f"Completed datasets average {avg_bal_acc * 100:.1f}% bal_acc; compare against SSL and cross-dataset transfer before drawing the paper story.",
                }
            )

        collapsed = []
        for dataset in completed:
            for fold in dataset.get("folds", []):
                if float(fold["sens"]) < 0.10 or float(fold["spec"]) < 0.10:
                    collapsed.append(f"{dataset['name']} F{fold['fold']}")
        if collapsed:
            insights.append(
                {
                    "level": "warn",
                    "title": "Fold-level class collapse",
                    "body": "Some folds heavily predict one class: " + ", ".join(collapsed[:6]) + ("..." if len(collapsed) > 6 else ""),
                }
            )

    if active:
        folds = active.get("folds", [])
        latest = folds[-1] if folds else None
        if latest:
            gap = abs(float(latest["sens"]) - float(latest["spec"]))
            level = "warn" if gap >= 0.45 else "info"
            insights.append(
                {
                    "level": level,
                    "title": f"{active['name']} still in progress",
                    "body": f"Latest fold is {float(latest['bal_acc']) * 100:.1f}% bal_acc with sens/spec gap {gap * 100:.1f} points.",
                }
            )

    return {
        "datasets": datasets,
        "active_dataset": active["name"] if active else None,
        "completed_datasets": len(completed),
        "folds_logged": fold_count,
        "best_completed": best_completed["name"] if best_completed else None,
        "best_completed_bal_acc": best_completed["mean"]["bal_acc"] if best_completed else None,
        "insights": insights,
    }


class WatchdogStore:
    def __init__(self, cwd: Path, run_dir: Path):
        self.cwd = cwd
        self.run_dir = run_dir

    def status_paths(self) -> list[Path]:
        return sorted(self.run_dir.glob("*.status.json"))

    def runs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self.status_paths():
            status = atomic_read_json(path)
            if not status:
                continue
            name = str(status.get("name") or path.name.removesuffix(".status.json"))
            status["_status_path"] = str(path)
            status["_event_path"] = str(self.event_path(name))
            status["_launch_label"] = f"com.sjji.experiment.{safe_label(name)}"
            rows.append(status)
        return sorted(rows, key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def run(self, name: str) -> dict[str, Any] | None:
        path = self.run_dir / f"{name}.status.json"
        status = atomic_read_json(path)
        if not status:
            return None
        status["_status_path"] = str(path)
        status["_event_path"] = str(self.event_path(name))
        status["_launch_label"] = f"com.sjji.experiment.{safe_label(name)}"
        return status

    def event_path(self, name: str) -> Path:
        return self.run_dir / f"{name}.events.jsonl"

    def log_path(self, status: dict[str, Any]) -> Path:
        raw = status.get("log_path")
        return Path(raw) if raw else self.run_dir / f"{status['name']}.log"

    def events(self, name: str, limit: int = 80) -> list[dict[str, Any]]:
        return read_jsonl_tail(self.event_path(name), limit)

    def log_tail(self, status: dict[str, Any], max_bytes: int = 80_000) -> str:
        return read_text_tail(self.log_path(status), max_bytes)

    def metrics(self, name: str) -> dict[str, Any] | None:
        status = self.run(name)
        if not status:
            return None
        return parse_run_metrics(self.log_tail(status, max_bytes=1_200_000))

    def stop(self, name: str) -> dict[str, Any]:
        status = self.run(name)
        if not status:
            return {"ok": False, "error": "run not found"}

        label = f"com.sjji.experiment.{safe_label(name)}"
        plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        domain = f"gui/{os.getuid()}"
        stopped: list[str] = []
        errors: list[str] = []

        if plist.exists():
            result = launchctl("bootout", domain, str(plist))
            if result.returncode == 0:
                stopped.append("launchagent")
            elif "No such process" not in (result.stderr + result.stdout):
                errors.append((result.stderr or result.stdout).strip())

        pid = status.get("pid")
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 15)
                stopped.append(f"pid:{pid}")
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                errors.append(str(exc))

        return {"ok": not errors, "stopped": stopped, "errors": errors}


class ConsoleHandler(BaseHTTPRequestHandler):
    store: WatchdogStore

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(render_index(self.store.runs(), self.store.run_dir))
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path == "/api/runs":
            self.send_json({"runs": self.store.runs(), "run_dir": str(self.store.run_dir), "ts": time.time()})
            return
        if parsed.path == "/api/system":
            self.send_json(system_snapshot())
            return
        if parsed.path == "/api/artifacts":
            self.send_json({"artifacts": collect_artifacts(self.store.cwd, self.store.run_dir), "ts": time.time()})
            return
        if parsed.path.startswith("/api/runs/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 3:
                self.send_json({"error": "missing run name"}, HTTPStatus.BAD_REQUEST)
                return
            name = parts[2]
            status = self.store.run(name)
            if not status:
                self.send_json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
                return
            qs = parse_qs(parsed.query)
            if len(parts) == 3:
                self.send_json({"run": status})
                return
            if parts[3] == "events":
                limit = int(qs.get("limit", ["80"])[0])
                self.send_json({"events": self.store.events(name, limit=limit)})
                return
            if parts[3] == "log":
                max_bytes = int(qs.get("bytes", ["80000"])[0])
                body = self.store.log_tail(status, max_bytes=max_bytes).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    return
                return
            if parts[3] == "metrics":
                self.send_json({"metrics": self.store.metrics(name), "ts": time.time()})
                return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4:
                self.send_json(self.store.stop(parts[2]))
                return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def state_class(state: str, stalled: bool) -> str:
    if stalled:
        return "stalled"
    return state if state in {"running", "completed", "failed"} else "unknown"


def render_index(runs: list[dict[str, Any]], run_dir: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenClaw Control Console</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111315;
      --panel: #191d21;
      --panel-2: #20262b;
      --text: #edf2f4;
      --muted: #9ea8b3;
      --line: #313942;
      --ok: #4dbd7a;
      --warn: #d9a441;
      --bad: #e05d5d;
      --info: #5ca8d8;
      --focus: #8fb5ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 20px 24px;
      border-bottom: 1px solid var(--line);
      background: #15181b;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 20px; font-weight: 700; letter-spacing: 0; }}
    h2 {{ font-size: 16px; font-weight: 700; letter-spacing: 0; }}
    .muted {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .crumbs {{ display: flex; gap: 7px; align-items: center; color: var(--muted); font-size: 12px; flex-wrap: wrap; }}
    .crumbs span:not(:last-child)::after {{ content: "/"; margin-left: 7px; color: #5f6b76; }}
    .header-actions {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .system {{
      display: grid;
      grid-template-columns: minmax(260px, 360px) 1fr;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #101315;
    }}
    .system-card, .scope-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-width: 0;
    }}
    .scope-card h2, .system-card h2 {{ margin-bottom: 8px; }}
    .scope-path {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
    .scope-path article {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px;
      min-height: 64px;
    }}
    .scope-path span, .service span {{ display: block; color: var(--muted); font-size: 12px; }}
    .scope-path strong, .service strong {{ display: block; margin-top: 4px; font-size: 13px; overflow-wrap: anywhere; }}
    .service-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
    .service {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 9px 10px;
      min-height: 58px;
    }}
    .dashboard {{
      display: grid;
      grid-template-columns: 1.05fr 1fr 1.15fr 1.1fr;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #111417;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      min-width: 0;
      padding: 12px;
    }}
    .panel h2 {{ margin-bottom: 10px; }}
    .attention-list, .activity-list, .domain-list, .artifact-list {{ display: grid; gap: 8px; }}
    .attention-item, .activity-item, .domain-item, .artifact-item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px;
      min-width: 0;
    }}
    .attention-item strong, .activity-item strong, .domain-item strong, .artifact-item strong {{
      display: block;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .attention-item span, .activity-item span, .domain-item span, .artifact-item span {{
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .domain-item {{
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
    }}
    .domain-item em {{
      color: var(--muted);
      font-style: normal;
      font-size: 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
    }}
    .palette {{
      position: fixed;
      inset: 0;
      display: none;
      place-items: start center;
      padding-top: 92px;
      background: rgba(9, 11, 13, .62);
      z-index: 5;
    }}
    .palette.open {{ display: grid; }}
    .palette-card {{
      width: min(680px, calc(100vw - 28px));
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #15191d;
      box-shadow: 0 24px 70px rgba(0,0,0,.36);
      padding: 12px;
    }}
    .palette-card input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-2);
      color: var(--text);
      padding: 10px 11px;
      font: inherit;
      outline: none;
    }}
    .palette-actions {{ display: grid; gap: 8px; margin-top: 10px; }}
    .palette-action {{
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      padding: 10px;
    }}
    .palette-action strong {{ display: block; font-size: 13px; }}
    .palette-action span {{ display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #121518;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px 12px;
      min-height: 62px;
    }}
    .metric strong {{ display: block; font-size: 20px; line-height: 1.1; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    main {{ display: grid; grid-template-columns: minmax(320px, 430px) 1fr; min-height: calc(100vh - 148px); }}
    .runs {{ border-right: 1px solid var(--line); padding: 16px; overflow: auto; }}
    .run {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--info);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 12px;
      cursor: pointer;
    }}
    .run:hover, .run.selected {{ border-color: var(--focus); }}
    .run.running {{ border-left-color: var(--ok); }}
    .run.completed {{ border-left-color: var(--info); }}
    .run.failed {{ border-left-color: var(--bad); }}
    .run.stalled {{ border-left-color: var(--warn); }}
    .run-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
    .badge {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; color: var(--muted); white-space: nowrap; }}
    .run.running .badge {{ color: var(--ok); }}
    .run.failed .badge {{ color: var(--bad); }}
    .run.stalled .badge {{ color: var(--warn); }}
    dl {{ display: grid; gap: 8px; margin: 0; }}
    dl div {{ display: grid; grid-template-columns: 74px 1fr; gap: 10px; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 0; font-size: 13px; overflow-wrap: anywhere; }}
    .actions {{ display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }}
    button {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 7px;
      padding: 7px 10px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    button:hover {{ border-color: #5b6672; }}
    button.active {{ border-color: var(--focus); color: #dce8ff; }}
    button.danger {{ color: #ffb8b8; }}
    button:disabled {{ color: #64707b; cursor: not-allowed; }}
    .detail {{ min-width: 0; display: flex; flex-direction: column; }}
    .detail-bar {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .detail-body {{ min-width: 0; flex: 1; overflow: auto; }}
    .overview {{ display: grid; gap: 14px; padding: 16px; }}
    .kv {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .kv article {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 11px 12px;
      min-height: 68px;
    }}
    .kv span {{ display: block; color: var(--muted); font-size: 12px; }}
    .kv strong {{ display: block; margin-top: 4px; font-size: 13px; overflow-wrap: anywhere; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
    }}
    .dataset-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-width: 0;
    }}
    .dataset-card h3 {{
      margin: 0;
      font-size: 14px;
    }}
    .dataset-card .meta {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }}
    .score-row {{
      display: grid;
      grid-template-columns: 78px 58px 1fr;
      align-items: center;
      gap: 8px;
      margin-top: 9px;
      font-size: 12px;
    }}
    .score-row span {{ color: var(--muted); }}
    .bar {{
      height: 7px;
      border-radius: 999px;
      background: #2d343b;
      overflow: hidden;
    }}
    .bar i {{
      display: block;
      height: 100%;
      width: 0%;
      background: var(--info);
    }}
    .dataset-card.running .bar i {{ background: var(--ok); }}
    .fold-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(58px, 1fr));
      gap: 6px;
      margin-top: 12px;
    }}
    .fold-chip {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-2);
      padding: 6px;
      font-size: 11px;
      color: var(--muted);
    }}
    .fold-chip strong {{ display: block; color: var(--text); font-size: 12px; }}
    .strip {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }}
    .strip h3 {{ margin: 0 0 8px; font-size: 13px; }}
    .audit-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    .audit-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-width: 0;
    }}
    .audit-card h3 {{ margin: 0; font-size: 13px; }}
    .audit-card p {{ margin: 6px 0 0; color: var(--muted); font-size: 12px; }}
    .audit-card ul {{ margin: 10px 0 0; padding-left: 18px; color: #dce5ea; font-size: 12px; }}
    .audit-card li {{ margin: 5px 0; }}
    .audit-card.keep {{ border-left: 4px solid var(--ok); }}
    .audit-card.enhance {{ border-left: 4px solid var(--info); }}
    .audit-card.pull {{ border-left: 4px solid var(--warn); }}
    .audit-card.surface {{ border-left: 4px solid var(--focus); }}
    .timeline {{ display: grid; gap: 8px; }}
    .event {{ display: grid; grid-template-columns: 150px 130px 1fr; gap: 10px; color: #dce5ea; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .event span {{ color: var(--muted); overflow-wrap: anywhere; }}
    pre {{
      margin: 0;
      padding: 16px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #dce5ea;
      flex: 1;
    }}
    .empty {{ color: var(--muted); padding: 16px; }}
    .pulse {{ display: inline-block; width: 8px; height: 8px; border-radius: 999px; background: var(--muted); margin-right: 7px; }}
    .pulse.running {{ background: var(--ok); box-shadow: 0 0 0 4px rgba(77, 189, 122, .12); }}
    .pulse.failed {{ background: var(--bad); }}
    .pulse.stalled {{ background: var(--warn); }}
    @media (max-width: 820px) {{
      .system {{ grid-template-columns: 1fr; }}
      .dashboard {{ grid-template-columns: 1fr; }}
      .scope-path, .service-grid {{ grid-template-columns: 1fr; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      main {{ grid-template-columns: 1fr; }}
      .runs {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .kv {{ grid-template-columns: 1fr; }}
      .event {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>OpenClaw Control Console</h1>
      <p class="crumbs"><span>Mac</span><span>Algorithms</span><span>SJJI</span><span>Experiments</span></p>
      <p class="muted">{html.escape(str(run_dir))}</p>
    </div>
    <div class="header-actions">
      <p class="muted" id="last-refresh">loading</p>
      <button id="command-open">Command</button>
      <button id="refresh">Refresh</button>
      <button id="auto-refresh" class="active">Auto</button>
    </div>
  </header>
  <section class="system">
    <section class="scope-card">
      <h2>Scope</h2>
      <div class="scope-path">
        <article><span>Global</span><strong>OpenClaw + Mac</strong></article>
        <article><span>Domain</span><strong>Algorithms</strong></article>
        <article><span>Project</span><strong>SJJI EEG</strong></article>
      </div>
    </section>
    <section class="system-card">
      <h2>Mac System</h2>
      <div class="service-grid" id="system-grid">
        <article class="service"><span>Host</span><strong>Loading</strong></article>
        <article class="service"><span>Load</span><strong>-</strong></article>
        <article class="service"><span>Services</span><strong>-</strong></article>
      </div>
    </section>
  </section>
  <section class="dashboard">
    <section class="panel">
      <h2>Attention</h2>
      <div class="attention-list" id="attention-list"><p class="empty">Loading attention queue.</p></div>
    </section>
    <section class="panel">
      <h2>Domains</h2>
      <div class="domain-list" id="domain-list"><p class="empty">Loading domains.</p></div>
    </section>
    <section class="panel">
      <h2>Activity</h2>
      <div class="activity-list" id="activity-list"><p class="empty">Loading activity.</p></div>
    </section>
    <section class="panel">
      <h2>Artifacts</h2>
      <div class="artifact-list" id="artifact-list"><p class="empty">Loading artifacts.</p></div>
    </section>
  </section>
  <section class="summary" id="summary"></section>
  <main>
    <section class="runs" id="runs"><p class="empty">Loading watchdog runs.</p></section>
    <section class="detail">
      <div class="detail-bar">
        <div>
          <p id="detail-title">Select a run</p>
          <p class="muted" id="detail-meta">status, events, and logs stay local</p>
        </div>
        <div class="tabs">
          <button data-view="overview" class="active">Overview</button>
          <button data-view="audit">Audit</button>
          <button data-view="metrics">Metrics</button>
          <button data-view="log">Log</button>
          <button data-view="events">Events</button>
          <button data-view="status">Status</button>
          <button data-view="artifacts">Artifacts</button>
          <button data-view="system">System</button>
          <button class="danger" id="stop-run" disabled>Stop</button>
        </div>
      </div>
      <div class="detail-body" id="detail-output"><p class="empty">No run selected.</p></div>
    </section>
  </main>
  <section class="palette" id="palette" aria-hidden="true">
    <div class="palette-card">
      <input id="palette-input" placeholder="Run a console action" autocomplete="off">
      <div class="palette-actions" id="palette-actions"></div>
    </div>
  </section>
  <script>
    const runsEl = document.getElementById('runs');
    const summaryEl = document.getElementById('summary');
    const systemGrid = document.getElementById('system-grid');
    const attentionList = document.getElementById('attention-list');
    const domainList = document.getElementById('domain-list');
    const activityList = document.getElementById('activity-list');
    const artifactList = document.getElementById('artifact-list');
    const palette = document.getElementById('palette');
    const paletteInput = document.getElementById('palette-input');
    const paletteActions = document.getElementById('palette-actions');
    const out = document.getElementById('detail-output');
    const title = document.getElementById('detail-title');
    const meta = document.getElementById('detail-meta');
    const refreshLabel = document.getElementById('last-refresh');
    const stopButton = document.getElementById('stop-run');
    const autoButton = document.getElementById('auto-refresh');
    let runs = {json.dumps(runs)};
    let systemState = null;
    let artifacts = [];
    let selected = runs[0]?.name || null;
    let view = 'overview';
    let autoRefresh = true;

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}}[char]));
    }}

    function stateClass(run) {{
      if (run?.stalled) return 'stalled';
      return ['running', 'completed', 'failed'].includes(run?.state) ? run.state : 'unknown';
    }}

    function timeAgo(value) {{
      if (!value) return 'unknown';
      const ms = Date.now() - new Date(value).getTime();
      if (!Number.isFinite(ms)) return value;
      const sec = Math.max(0, Math.round(ms / 1000));
      if (sec < 90) return sec + 's ago';
      const min = Math.round(sec / 60);
      if (min < 90) return min + 'm ago';
      const hr = Math.round(min / 60);
      return hr + 'h ago';
    }}

    function durationLabel(ms) {{
      if (!Number.isFinite(ms) || ms <= 0) return '-';
      const min = Math.max(1, Math.round(ms / 60000));
      if (min < 90) return min + 'm';
      const hr = Math.floor(min / 60);
      const rest = min % 60;
      return rest ? hr + 'h ' + rest + 'm' : hr + 'h';
    }}

    function metric(label, value) {{
      return '<article class="metric"><strong>' + esc(value) + '</strong><span>' + esc(label) + '</span></article>';
    }}

    function service(label, value) {{
      return '<article class="service"><span>' + esc(label) + '</span><strong>' + esc(value || '-') + '</strong></article>';
    }}

    function compactPath(value) {{
      const text = String(value || '');
      if (text.length <= 72) return text;
      return '...' + text.slice(-69);
    }}

    function bytesLabel(value) {{
      const size = Number(value || 0);
      if (size < 1024) return size + ' B';
      if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KB';
      return (size / 1024 / 1024).toFixed(1) + ' MB';
    }}

    function percent(value) {{
      const number = Number(value);
      if (!Number.isFinite(number)) return '-';
      return (number * 100).toFixed(1) + '%';
    }}

    function bar(label, value) {{
      const number = Number(value);
      const width = Number.isFinite(number) ? Math.max(0, Math.min(100, number * 100)) : 0;
      return '<div class="score-row"><span>' + esc(label) + '</span><strong>' + esc(percent(value)) + '</strong><div class="bar"><i style="width:' + width.toFixed(1) + '%"></i></div></div>';
    }}

    function estimateActiveRun(metrics, events) {{
      const activeName = metrics.active_dataset;
      const active = (metrics.datasets || []).find(dataset => dataset.name === activeName);
      const foldCount = (active?.folds || []).length;
      if (!active || foldCount < 2) return {{cadence: '-', remaining: '-'}};
      const foldEvents = (events || [])
        .filter(event => /^\\s*Fold\\s+\\d+:/.test(event.line || '') && event.ts)
        .slice(-foldCount);
      if (foldEvents.length < 2) return {{cadence: '-', remaining: '-'}};
      const first = new Date(foldEvents[0].ts).getTime();
      const last = new Date(foldEvents[foldEvents.length - 1].ts).getTime();
      const cadenceMs = (last - first) / Math.max(1, foldEvents.length - 1);
      const remainingFolds = Math.max(0, 10 - foldCount);
      return {{
        cadence: durationLabel(cadenceMs),
        remaining: remainingFolds ? durationLabel(cadenceMs * remainingFolds) : 'finalizing',
      }};
    }}

    function renderAttention() {{
      const items = [];
      for (const run of runs) {{
        const progressMs = run.last_progress_at ? Date.now() - new Date(run.last_progress_at).getTime() : 0;
        const quietFor = Math.round(progressMs / 60000);
        const progress = run.last_progress_line || '';
        const balAcc = /bal_acc=(\\d+(?:\\.\\d+)?)/.exec(progress);
        const sens = /sens=(\\d+(?:\\.\\d+)?)/.exec(progress);
        const spec = /spec=(\\d+(?:\\.\\d+)?)/.exec(progress);
        const latestBalAcc = balAcc ? Number(balAcc[1]) : null;
        const latestGap = sens && spec ? Math.abs(Number(sens[1]) - Number(spec[1])) : null;
        if (run.stalled || run.state === 'failed') {{
          items.push(['Needs action', run.name + ' is ' + (run.stalled ? 'stalled' : 'failed')]);
        }} else if (run.state === 'running' && latestBalAcc !== null && latestBalAcc < 0.50) {{
          items.push(['Quality watch', run.name + ' latest fold is ' + percent(latestBalAcc) + '; inspect fold balance before trusting the baseline.']);
        }} else if (run.state === 'running' && latestGap !== null && latestGap >= 0.45) {{
          items.push(['Balance warning', run.name + ' latest sensitivity/specificity gap is ' + percent(latestGap) + '.']);
        }} else if (run.state === 'running' && quietFor >= 10) {{
          items.push(['Watch closely', run.name + ' has no progress event for ' + quietFor + 'm']);
        }} else if (run.state === 'running') {{
          items.push(['Active run', run.name + ' is reporting progress']);
        }}
      }}
      if (!items.length) items.push(['System quiet', 'No watched runs need attention right now']);
      attentionList.innerHTML = items.slice(0, 4).map(item =>
        '<article class="attention-item"><strong>' + esc(item[0]) + '</strong><span>' + esc(item[1]) + '</span></article>'
      ).join('');
    }}

    function renderDomains() {{
      const agents = systemState?.launch_agents || [];
      const ports = systemState?.ports || [];
      const processes = systemState?.processes || [];
      const domains = [
        ['Algorithms', runs.length + ' watched run' + (runs.length === 1 ? '' : 's'), 'SJJI'],
        ['System Runtime', agents.length + ' launch agents', 'Mac'],
        ['Local Services', ports.length + ' listeners', 'Atlas'],
        ['Agent Surface', processes.filter(proc => /openclaw|codex|peekaboo|atlas/i.test(proc.args || '')).length + ' tracked processes', 'OpenClaw'],
      ];
      domainList.innerHTML = domains.map(domain =>
        '<article class="domain-item"><div><strong>' + esc(domain[0]) + '</strong><span>' + esc(domain[1]) + '</span></div><em>' + esc(domain[2]) + '</em></article>'
      ).join('');
    }}

    async function renderActivity() {{
      const events = [];
      for (const run of runs.slice(0, 6)) {{
        try {{
          const response = await fetch('/api/runs/' + encodeURIComponent(run.name) + '/events?limit=5');
          const payload = await response.json();
          for (const event of payload.events || []) {{
            events.push({{run: run.name, ...event}});
          }}
        }} catch (error) {{
          events.push({{run: run.name, event: 'activity_error', line: String(error), ts: new Date().toISOString()}});
        }}
      }}
      events.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());
      if (!events.length) {{
        activityList.innerHTML = '<p class="empty">No activity events yet.</p>';
        return;
      }}
      activityList.innerHTML = events.slice(0, 6).map(event => {{
        const detail = event.line ?? event.exit_code ?? event.quiet_for ?? '';
        return '<article class="activity-item"><strong>' + esc(event.run + ' / ' + event.event) + '</strong><span>' + esc(timeAgo(event.ts) + ' - ' + compactPath(detail)) + '</span></article>';
      }}).join('');
    }}

    function renderArtifacts() {{
      if (!artifacts.length) {{
        artifactList.innerHTML = '<p class="empty">No artifacts found yet.</p>';
        return;
      }}
      artifactList.innerHTML = artifacts.slice(0, 5).map(artifact =>
        '<article class="artifact-item"><strong>' + esc(artifact.name) + '</strong><span>' + esc(artifact.kind + ' - ' + bytesLabel(artifact.size) + ' - ' + timeAgo(artifact.mtime * 1000)) + '</span></article>'
      ).join('');
    }}

    function renderSystem() {{
      if (!systemState) {{
        systemGrid.innerHTML = [
          service('Host', 'Loading'),
          service('Load', '-'),
          service('Services', '-'),
        ].join('');
        return;
      }}
      const ports = systemState.ports || [];
      const agents = systemState.launch_agents || [];
      const processes = systemState.processes || [];
      const primaryPort = ports[0] ? ports[0].process + ' ' + ports[0].address : 'none found';
      const primaryAgent = agents[0] || 'none found';
      systemGrid.innerHTML = [
        service('Host', systemState.host),
        service('Load avg', (systemState.load || []).join(' / ')),
        service('Tracked processes', processes.length),
        service('LaunchAgents', agents.length),
        service('Primary service', primaryAgent),
        service('Local listener', primaryPort),
      ].join('');
    }}

    function renderSummary() {{
      const counts = {{running: 0, failed: 0, stalled: 0, completed: 0}};
      for (const run of runs) {{
        if (run.stalled) counts.stalled += 1;
        if (counts[run.state] !== undefined) counts[run.state] += 1;
      }}
      summaryEl.innerHTML = [
        metric('running jobs', counts.running),
        metric('stalled jobs', counts.stalled),
        metric('failed jobs', counts.failed),
        metric('completed jobs', counts.completed),
      ].join('');
    }}

    function renderControlPlane() {{
      renderAttention();
      renderDomains();
      renderSystem();
      renderArtifacts();
      renderSummary();
    }}

    function renderRuns() {{
      if (!runs.length) {{
        runsEl.innerHTML = '<p class="empty">No watchdog runs found yet.</p>';
        return;
      }}
      runsEl.innerHTML = runs.map(run => {{
        const klass = stateClass(run);
        const progress = run.last_progress_line || 'No progress line yet';
        const updated = timeAgo(run.updated_at);
        const progressAge = timeAgo(run.last_progress_at);
        const pid = run.pid || '-';
        const selectedClass = run.name === selected ? ' selected' : '';
        return `
          <article class="run ${{klass}}${{selectedClass}}" data-run="${{esc(run.name)}}">
            <div class="run-top">
              <div>
                <h2><span class="pulse ${{klass}}"></span>${{esc(run.name)}}</h2>
                <p class="muted">${{esc(run._launch_label || '')}}</p>
              </div>
              <span class="badge">${{esc(run.stalled ? 'stalled' : run.state || 'unknown')}}</span>
            </div>
            <dl>
              <div><dt>PID</dt><dd>${{esc(pid)}}</dd></div>
              <div><dt>Updated</dt><dd>${{esc(updated)}}</dd></div>
              <div><dt>Progress age</dt><dd>${{esc(progressAge)}}</dd></div>
              <div><dt>Progress</dt><dd>${{esc(progress)}}</dd></div>
            </dl>
          </article>
        `;
      }}).join('');
    }}

    function kv(label, value) {{
      return '<article><span>' + esc(label) + '</span><strong>' + esc(value || '-') + '</strong></article>';
    }}

    async function renderOverview(run) {{
      title.textContent = run.name;
      meta.textContent = (run.stalled ? 'stalled' : run.state) + ' · updated ' + timeAgo(run.updated_at);
      stopButton.disabled = run.state !== 'running';
      const body = `
        <section class="overview">
          <div class="kv">
            ${{kv('State', run.stalled ? 'stalled' : run.state)}}
            ${{kv('PID', run.pid)}}
            ${{kv('Exit code', run.exit_code ?? '-')}}
            ${{kv('Started', run.started_at)}}
            ${{kv('Updated', run.updated_at)}}
            ${{kv('Last progress', run.last_progress_at)}}
            ${{kv('Stall threshold', (run.stall_seconds || 0) + 's')}}
          </div>
          <section class="strip">
            <h3>Latest progress</h3>
            <pre>${{esc(run.last_progress_line || 'No progress line yet.')}}</pre>
          </section>
          <section class="strip">
            <h3>Latest error</h3>
            <pre>${{esc(run.last_error_line || 'No error line recorded.')}}</pre>
          </section>
          <section class="strip">
            <h3>Command</h3>
            <pre>${{esc((run.command || []).join(' '))}}</pre>
          </section>
          <section class="strip">
            <h3>Recent events</h3>
            <div class="timeline" id="recent-events"><p class="muted">Loading events.</p></div>
          </section>
        </section>
      `;
      out.innerHTML = body;
      const response = await fetch('/api/runs/' + encodeURIComponent(run.name) + '/events?limit=8');
      const payload = await response.json();
      const events = payload.events || [];
      const timeline = document.getElementById('recent-events');
      timeline.innerHTML = events.length ? events.reverse().map(event => (
        '<div class="event"><span>' + esc(timeAgo(event.ts)) + '</span><strong>' + esc(event.event) + '</strong><span>' + esc(event.line ?? event.exit_code ?? event.quiet_for ?? '') + '</span></div>'
      )).join('') : '<p class="muted">No events yet.</p>';
    }}

    async function renderLog(run) {{
      title.textContent = run.name + ' / log';
      meta.textContent = 'tailing last 100 KB';
      stopButton.disabled = run.state !== 'running';
      const response = await fetch('/api/runs/' + encodeURIComponent(run.name) + '/log?bytes=100000');
      const text = await response.text();
      out.innerHTML = '<pre>' + esc(text || '(empty)') + '</pre>';
      out.scrollTop = out.scrollHeight;
    }}

    async function renderEvents(run) {{
      title.textContent = run.name + ' / events';
      meta.textContent = 'latest 160 structured events';
      stopButton.disabled = run.state !== 'running';
      const response = await fetch('/api/runs/' + encodeURIComponent(run.name) + '/events?limit=160');
      out.innerHTML = '<pre>' + esc(JSON.stringify(await response.json(), null, 2)) + '</pre>';
    }}

    async function renderMetrics(run) {{
      title.textContent = run.name + ' / metrics';
      meta.textContent = 'parsed from the live experiment log';
      stopButton.disabled = run.state !== 'running';
      const [response, eventResponse] = await Promise.all([
        fetch('/api/runs/' + encodeURIComponent(run.name) + '/metrics'),
        fetch('/api/runs/' + encodeURIComponent(run.name) + '/events?limit=120'),
      ]);
      const payload = await response.json();
      const eventPayload = await eventResponse.json();
      const metrics = payload.metrics || {{}};
      const eta = estimateActiveRun(metrics, eventPayload.events || []);
      const datasets = metrics.datasets || [];
      const insights = metrics.insights || [];
      if (!datasets.length) {{
        out.innerHTML = '<p class="empty">No dataset metrics found in the log yet.</p>';
        return;
      }}
      const insightCards = insights.map(insight =>
        '<article class="attention-item"><strong>' + esc(insight.title) + '</strong><span>' + esc((insight.level || 'info') + ' - ' + insight.body) + '</span></article>'
      ).join('');
      const cards = datasets.map(dataset => {{
        const latestFold = (dataset.folds || [])[dataset.folds.length - 1];
        const values = dataset.mean || latestFold || {{}};
        const state = dataset.mean ? 'completed' : 'running';
        const folds = (dataset.folds || []).map(fold =>
          '<div class="fold-chip"><span>F' + esc(fold.fold) + '</span><strong>' + esc(percent(fold.bal_acc)) + '</strong></div>'
        ).join('');
        const label = dataset.mean ? 'mean' : 'latest fold';
        return '<article class="dataset-card ' + esc(state) + '">' +
          '<h3>' + esc(dataset.name) + ' <span class="badge">' + esc(state) + '</span></h3>' +
          '<p class="meta">' + esc(dataset.meta || '') + '</p>' +
          '<p class="meta">' + esc(label + ' · ' + (dataset.folds || []).length + ' folds logged') + '</p>' +
          bar('bal acc', values.bal_acc) +
          bar('sensitivity', values.sens) +
          bar('specificity', values.spec) +
          '<div class="fold-list">' + (folds || '<p class="muted">No folds yet.</p>') + '</div>' +
        '</article>';
      }}).join('');
      out.innerHTML =
        '<section class="overview">' +
          '<div class="kv">' +
            kv('Completed datasets', metrics.completed_datasets) +
            kv('Active dataset', metrics.active_dataset || '-') +
            kv('Folds logged', metrics.folds_logged) +
            kv('Best completed', metrics.best_completed || '-') +
            kv('Best completed bal acc', percent(metrics.best_completed_bal_acc)) +
            kv('Fold cadence', eta.cadence) +
            kv('ETA remaining', eta.remaining) +
          '</div>' +
          '<section class="strip"><h3>Run insights</h3><div class="attention-list">' + (insightCards || '<p class="muted">No derived insights yet.</p>') + '</div></section>' +
          '<section class="metrics-grid">' + cards + '</section>' +
        '</section>';
    }}

    function renderStatus(run) {{
      title.textContent = run.name + ' / status';
      meta.textContent = 'raw watchdog status';
      stopButton.disabled = run.state !== 'running';
      out.innerHTML = '<pre>' + esc(JSON.stringify(run, null, 2)) + '</pre>';
    }}

    function renderArtifactsDetail() {{
      title.textContent = 'Artifacts';
      meta.textContent = artifacts.length + ' recent files from results and watchdog runs';
      stopButton.disabled = true;
      if (!artifacts.length) {{
        out.innerHTML = '<p class="empty">No artifacts found yet.</p>';
        return;
      }}
      out.innerHTML =
        '<section class="overview"><section class="strip"><h3>Recent artifacts</h3><div class="timeline">' +
        artifacts.map(artifact => (
          '<div class="event"><span>' + esc(artifact.kind) + '</span><strong>' + esc(artifact.name) + '</strong><span>' + esc(bytesLabel(artifact.size) + ' - ' + timeAgo(artifact.mtime * 1000) + ' - ' + compactPath(artifact.rel_path)) + '</span></div>'
        )).join('') +
        '</div></section></section>';
    }}

    function renderSystemDetail() {{
      title.textContent = 'Mac / OpenClaw system';
      meta.textContent = systemState ? 'live local system snapshot' : 'loading system snapshot';
      stopButton.disabled = true;
      if (!systemState) {{
        out.innerHTML = '<p class="empty">System state is loading.</p>';
        return;
      }}
      const processes = (systemState.processes || []).map(proc => (
        '<div class="event"><span>' + esc(proc.pid) + '</span><strong>' + esc(proc.command) + '</strong><span>CPU ' + esc(proc.cpu) + ' / MEM ' + esc(proc.mem) + ' / ' + esc(proc.args) + '</span></div>'
      )).join('');
      const ports = (systemState.ports || []).map(port => (
        '<div class="event"><span>' + esc(port.pid) + '</span><strong>' + esc(port.process) + '</strong><span>' + esc(port.address) + '</span></div>'
      )).join('');
      const agents = (systemState.launch_agents || []).map(agent => (
        '<div class="event"><span>launchd</span><strong>' + esc(agent) + '</strong><span>loaded in user domain</span></div>'
      )).join('');
      out.innerHTML =
        '<section class="overview">' +
          '<div class="kv">' +
            kv('Host', systemState.host) +
            kv('Platform', systemState.system) +
            kv('Load average', (systemState.load || []).join(' / ')) +
          '</div>' +
          '<section class="strip"><h3>Uptime</h3><pre>' + esc(systemState.uptime || '-') + '</pre></section>' +
          '<section class="strip"><h3>Tracked processes</h3><div class="timeline">' + (processes || '<p class="muted">No matching processes found.</p>') + '</div></section>' +
          '<section class="strip"><h3>LaunchAgents</h3><div class="timeline">' + (agents || '<p class="muted">No matching LaunchAgents found.</p>') + '</div></section>' +
          '<section class="strip"><h3>Local listeners</h3><div class="timeline">' + (ports || '<p class="muted">No matching listeners found.</p>') + '</div></section>' +
        '</section>';
    }}

    function auditCard(kind, titleText, body, items) {{
      const rows = items.map(item => '<li>' + esc(item) + '</li>').join('');
      return '<article class="audit-card ' + esc(kind) + '"><h3>' + esc(titleText) + '</h3><p>' + esc(body) + '</p><ul>' + rows + '</ul></article>';
    }}

    function renderAuditDetail() {{
      title.textContent = 'Console visual audit';
      meta.textContent = 'product surface, useful signals, and next build threads';
      stopButton.disabled = true;
      const active = runs.find(item => item.state === 'running');
      const agents = systemState?.launch_agents || [];
      const ports = systemState?.ports || [];
      const mediaArtifacts = artifacts.filter(item => item.kind === 'media').length;
      out.innerHTML =
        '<section class="overview">' +
          '<div class="kv">' +
            kv('Current useful core', 'watchdog + launchd + live metrics') +
            kv('Active thread', active ? active.name : 'none') +
            kv('Mapped services', agents.length + ' agents / ' + ports.length + ' listeners') +
            kv('Visual artifacts', mediaArtifacts) +
          '</div>' +
          '<section class="strip"><h3>Audit read</h3><pre>' + esc('The console has the right backbone now: local-first state, live run status, metrics parsing, Mac service inventory, artifacts, and safe controls. The weak spot is product hierarchy: it should make the next decision obvious without requiring log reading.') + '</pre></section>' +
          '<section class="audit-grid">' +
            auditCard('keep', 'Keep', 'These are already carrying the control-plane idea.', [
              'Local-only localhost console with no external dependency.',
              'LaunchAgent-backed watchdog process that survives chat and terminal churn.',
              'Structured status/events/log files as a reusable contract.',
              'Metrics and insight extraction from raw experiment output.',
              'Mac service/process/listener inventory as the seed of global system mapping.',
            ]) +
            auditCard('enhance', 'Enhance', 'The page should feel more like an operating console than a report.', [
              'Create a real domain/project registry instead of hard-coded SJJI breadcrumbs.',
              'Add run ETA, fold cadence, elapsed time, and expected next milestone.',
              'Turn artifacts into previewable/openable objects with kind-specific affordances.',
              'Promote attention items into a decision queue with recommended next actions.',
              'Make command palette actions state-aware instead of just view switches.',
            ]) +
            auditCard('surface', 'Surface next', 'These signals reduce ambiguity during long-running work.', [
              'Resource pressure: CPU, memory, MPS/GPU, disk, battery/power source.',
              'Experiment phase: preprocessing, per-dataset baseline, cross-dataset, SSL pretrain, eval.',
              'Quality flags: class collapse, sensitivity/specificity imbalance, chance-level folds.',
              'Owner and intent: why this job exists, what output decides, who needs the result.',
              'Artifact lineage: which command produced which file and which paper/table it feeds.',
            ]) +
            auditCard('pull', 'Threads to keep pulling', 'These become the broader OpenClaw console.', [
              'Global Mac/OpenClaw system map: agents, IDEs, browsers, local APIs, schedulers.',
              'Algorithms domain: SJJI now, JKJ/Algoverse next, then shared experiment templates.',
              'Microapp registry: every local tool gets health, URL, owner, logs, and controls.',
              'Incident history: failed/stalled jobs become searchable learning records.',
              'Discord/OpenClaw bridge: notify the right channel when a run needs a human decision.',
            ]) +
          '</section>' +
        '</section>';
    }}

    async function renderDetail() {{
      if (view === 'system') return renderSystemDetail();
      if (view === 'audit') return renderAuditDetail();
      const run = runs.find(item => item.name === selected);
      if (!run) {{
        title.textContent = 'Select a run';
        meta.textContent = 'status, events, and logs stay local';
        stopButton.disabled = true;
        out.innerHTML = '<p class="empty">No run selected.</p>';
        return;
      }}
      if (view === 'log') return renderLog(run);
      if (view === 'events') return renderEvents(run);
      if (view === 'metrics') return renderMetrics(run);
      if (view === 'status') return renderStatus(run);
      if (view === 'artifacts') return renderArtifactsDetail();
      return renderOverview(run);
    }}

    async function refreshRuns() {{
      const [runsResponse, systemResponse, artifactsResponse] = await Promise.all([
        fetch('/api/runs'),
        fetch('/api/system'),
        fetch('/api/artifacts'),
      ]);
      const payload = await runsResponse.json();
      systemState = await systemResponse.json();
      artifacts = (await artifactsResponse.json()).artifacts || [];
      runs = payload.runs || [];
      if (!selected && runs.length) selected = runs[0].name;
      if (selected && !runs.some(run => run.name === selected)) selected = runs[0]?.name || null;
      renderControlPlane();
      renderRuns();
      await renderDetail();
      await renderActivity();
      refreshLabel.textContent = 'updated ' + new Date().toLocaleTimeString();
    }}

    async function stopRun() {{
      const run = runs.find(item => item.name === selected);
      if (!run || run.state !== 'running') return;
      if (!confirm('Stop ' + run.name + '?')) return;
      const response = await fetch('/api/runs/' + encodeURIComponent(run.name) + '/stop', {{method: 'POST'}});
      out.innerHTML = '<pre>' + esc(JSON.stringify(await response.json(), null, 2)) + '</pre>';
      await refreshRuns();
    }}

    const commands = [
      ['Refresh console', 'Pull latest runs, system state, artifacts, and activity', () => refreshRuns()],
      ['Open active run overview', 'Return to the selected watchdog run summary', () => setView('overview')],
      ['Open visual audit', 'Show what to keep, improve, surface, and keep pulling', () => setView('audit')],
      ['Show run metrics', 'Parse dataset means, fold scores, sensitivity, and specificity', () => setView('metrics')],
      ['Tail selected run log', 'Inspect stdout and stderr from the selected run', () => setView('log')],
      ['Show artifacts', 'List result files, logs, events, and status artifacts', () => setView('artifacts')],
      ['Show Mac system', 'Inspect tracked processes, LaunchAgents, and local listeners', () => setView('system')],
      ['Show event stream', 'Inspect structured watchdog events', () => setView('events')],
    ];

    function setView(nextView) {{
      view = nextView;
      document.querySelectorAll('[data-view]').forEach(item => item.classList.toggle('active', item.dataset.view === nextView));
      renderDetail();
    }}

    function closePalette() {{
      palette.classList.remove('open');
      palette.setAttribute('aria-hidden', 'true');
    }}

    function renderPalette() {{
      const query = paletteInput.value.toLowerCase();
      const visible = commands.filter(command => (command[0] + ' ' + command[1]).toLowerCase().includes(query));
      paletteActions.innerHTML = visible.map((command, index) =>
        '<button class="palette-action" data-command="' + index + '"><strong>' + esc(command[0]) + '</strong><span>' + esc(command[1]) + '</span></button>'
      ).join('');
      paletteActions.querySelectorAll('[data-command]').forEach((button, index) => {{
        button.addEventListener('click', () => {{
          visible[index][2]();
          closePalette();
        }});
      }});
    }}

    function openPalette() {{
      palette.classList.add('open');
      palette.setAttribute('aria-hidden', 'false');
      paletteInput.value = '';
      renderPalette();
      paletteInput.focus();
    }}

    runsEl.addEventListener('click', event => {{
      const card = event.target.closest('[data-run]');
      if (!card) return;
      selected = card.dataset.run;
      renderRuns();
      renderDetail();
    }});

    document.querySelectorAll('[data-view]').forEach(button => {{
      button.addEventListener('click', () => {{
        setView(button.dataset.view);
      }});
    }});

    document.getElementById('refresh').addEventListener('click', refreshRuns);
    document.getElementById('command-open').addEventListener('click', openPalette);
    stopButton.addEventListener('click', stopRun);
    paletteInput.addEventListener('input', renderPalette);
    palette.addEventListener('click', event => {{
      if (event.target === palette) closePalette();
    }});
    document.addEventListener('keydown', event => {{
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {{
        event.preventDefault();
        openPalette();
      }} else if (event.key === 'Escape') {{
        closePalette();
      }}
    }});
    autoButton.addEventListener('click', () => {{
      autoRefresh = !autoRefresh;
      autoButton.classList.toggle('active', autoRefresh);
    }});

    renderControlPlane();
    renderRuns();
    renderDetail();
    refreshRuns();
    setInterval(() => {{
      if (autoRefresh) refreshRuns();
    }}, 5000);
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local web console for watchdog runs.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Project directory containing the run directory.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Watchdog run directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path(args.cwd).resolve()
    run_dir = (cwd / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ConsoleHandler.store = WatchdogStore(cwd=cwd, run_dir=run_dir)
    server = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    print(f"SJJI Control Console: http://{args.host}:{args.port}")
    print(f"Run directory: {run_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
