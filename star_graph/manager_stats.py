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

    def to_dict(self) -> dict:
        """JSON-serializable representation for REST / API responses."""
        return {
            "anchors": self.anchors,
            "edges": self.edges,
            "ghosts": self.ghosts,
            "schemas": self.schemas,
            "abstracts": self.abstracts,
            "working_memory": self.working_memory,
            "cortices": self.cortices,
            "hubs": self.hubs,
            "clusters": self.clusters,
            "cold_anchors": self.cold_anchors,
            "sleep_cycles": self.sleep_cycles,
            "total_evolutions": self.total_evolutions,
            "auto_micro_sleeps": self.auto_micro_sleeps,
            "auto_full_sleeps": self.auto_full_sleeps,
            "anchors_since_micro": self.anchors_since_micro,
            "uptime_seconds": self.uptime_seconds,
            "cognitive_health": self.cognitive_health,
        }
