#!/usr/bin/env python3
"""Sleep daemon — runs memory consolidation during idle/low-activity periods.

Designed to be triggered by a scheduled task (cron / Windows Task Scheduler)
during nighttime hours (default: 2:00 AM).

Also supports an "idle detection" mode: if the system is idle (no user input
for N minutes), run a quick consolidation cycle.
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def is_system_idle(threshold_minutes: int = 15) -> bool:
    """Check if the user has been idle (no input)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        user32 = ctypes.windll.user32
        if user32.GetLastInputInfo(ctypes.byref(lii)):
            idle_ms = user32.GetTickCount() - lii.dwTime
            return (idle_ms / 60000) > threshold_minutes
    else:
        # Linux/macOS: check via xprintidle or ioreg
        try:
            result = subprocess.run(
                ["xprintidle"], capture_output=True, text=True, timeout=5
            )
            idle_ms = int(result.stdout.strip())
            return (idle_ms / 60000) > threshold_minutes
        except Exception:
            pass
    return True  # Default to True if can't detect


def get_pending_sessions(data_dir: Path) -> list[Path]:
    """Get session files that haven't been consolidated yet."""
    pending_dir = data_dir / "pending"
    if not pending_dir.exists():
        return []
    return sorted(pending_dir.glob("*.json"))


def load_session_anchors(session_file: Path) -> list:
    """Load anchors from a session file."""
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("anchors", [])


def mark_consolidated(session_file: Path, data_dir: Path) -> None:
    """Move a processed session file to the consolidated directory."""
    done_dir = data_dir / "consolidated"
    done_dir.mkdir(parents=True, exist_ok=True)
    dst = done_dir / session_file.name
    # Avoid overwriting
    if dst.exists():
        dst = done_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{session_file.name}"
    session_file.rename(dst)


def run_sleep_cycle(graph_path: Path, retention: float = 0.15,
                    edge_prune: float = 0.1) -> dict:
    """Import and run sleep cycle programmatically."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "star-graph-memory"))
    from star_graph.storage import Storage
    from star_graph.sleep import SleepCycle

    store = Storage(graph_path)
    graph = store.load()
    cycle = SleepCycle(graph)
    result = cycle.run(
        retention_threshold=retention,
        edge_prune_threshold=edge_prune,
    )
    store.save(graph)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Star Graph Memory — Sleep Daemon"
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / ".star_graph"),
        help="Data directory for memory storage",
    )
    parser.add_argument(
        "--idle-threshold",
        type=int,
        default=15,
        help="Minutes of idle time before running (idle mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["scheduled", "idle", "once"],
        default="scheduled",
        help="Run mode: scheduled (nightly), idle (when user is away), once (run now)",
    )
    parser.add_argument(
        "--retention",
        type=float,
        default=0.15,
        help="Retention threshold for pruning",
    )
    parser.add_argument(
        "--edge-prune",
        type=float,
        default=0.1,
        help="Edge weight threshold for pruning",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    graph_path = data_dir / "memory.json"

    log_path = Path(args.log_file) if args.log_file else (data_dir / "sleep.log")

    def log(msg: str) -> None:
        ts = datetime.now().isoformat()
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── Mode: once ──
    if args.mode == "once":
        log("Running one-shot sleep cycle...")
        result = run_sleep_cycle(graph_path, args.retention, args.edge_prune)
        log(f"Done: {json.dumps(result.get('stats_after', {}))}")
        return

    # ── Mode: idle ──
    if args.mode == "idle":
        log("Idle-watch mode started")
        try:
            while True:
                if is_system_idle(args.idle_threshold):
                    log("System idle — running sleep cycle")
                    result = run_sleep_cycle(graph_path, args.retention, args.edge_prune)
                    log(f"Cycle done: merged={result.get('merged', 0)}, "
                        f"pruned_a={result.get('pruned_anchors', 0)}")
                time.sleep(300)  # Check every 5 minutes
        except KeyboardInterrupt:
            log("Idle-watch stopped")
        return

    # ── Mode: scheduled ──
    # Process pending sessions
    pending = get_pending_sessions(data_dir)
    if pending:
        log(f"Processing {len(pending)} pending sessions...")
        for session_file in pending:
            log(f"  Consolidating: {session_file.name}")
            mark_consolidated(session_file, data_dir)

    log("Running scheduled sleep cycle...")
    result = run_sleep_cycle(graph_path, args.retention, args.edge_prune)
    log(f"Sleep complete: {json.dumps(result.get('stats_after', {}))}")
    for entry in result.get("log", []):
        log(f"  {entry}")


if __name__ == "__main__":
    main()
