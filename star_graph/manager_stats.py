"""Manager stats dataclass — extracted from runtime.py (P1 module split)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ManagerStats:
    """Snapshot of the memory system state."""
    anchors: int = 0
    edges: int = 0
    ghosts: int = 0
    schemas: int = 0
    abstracts: int = 0
    working_memory: int = 0
    cortices: int = 0
    hubs: int = 0
    clusters: int = 0
    cold_anchors: int = 0
    sleep_cycles: int = 0
    total_evolutions: int = 0
    auto_micro_sleeps: int = 0
    auto_full_sleeps: int = 0
    anchors_since_micro: int = 0
    uptime_seconds: float = 0.0
    cognitive_health: dict | None = None
