"""Sleep report data classes — extracted from sleep.py (P1 module split)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseMetrics:
    """Metrics for one sleep phase."""
    phase: str = ""
    duration_ms: float = 0.0
    items_processed: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class SleepReport:
    """Rich, human-readable report from one full sleep cycle."""
    cycle: int = 0
    total_duration_ms: float = 0.0
    phases: list[PhaseMetrics] = field(default_factory=list)

    # Aggregate counts
    memories_replayed: int = 0
    memories_merged: int = 0
    memories_pruned: int = 0
    ghosts_created: int = 0
    schemas_formed: int = 0
    abstractions_formed: int = 0
    bridges_created: int = 0
    edges_pruned: int = 0
    emotional_decoupled: int = 0
    cortical_transferred: int = 0

    # Before/after
    anchors_before: int = 0
    anchors_after: int = 0
    edges_before: int = 0
    edges_after: int = 0
    avg_retention_before: float = 0.0
    avg_retention_after: float = 0.0
    compression_ratio: float = 1.0

    def summary(self) -> str:
        """One-line summary of the sleep cycle."""
        parts = []
        if self.memories_replayed:
            parts.append(f"Replayed {self.memories_replayed}")
        if self.memories_merged:
            parts.append(f"Merged {self.memories_merged}")
        if self.schemas_formed:
            parts.append(f"Created {self.schemas_formed} schemas")
        if self.memories_pruned:
            parts.append(f"Pruned {self.memories_pruned} ({self.ghosts_created} ghosts)")
        if self.bridges_created:
            parts.append(f"Bridged {self.bridges_created}")
        if not parts:
            return "Sleep cycle complete — no significant changes"
        return " | ".join(parts)

    def detailed(self) -> str:
        """Multi-line detailed report."""
        lines = [
            f"╔{'═' * 50}╗",
            f"║  Sleep Cycle #{self.cycle} Report ({self.total_duration_ms:.0f}ms)",
            f"╟{'═' * 50}╣",
        ]
        for p in self.phases:
            name = p.phase.ljust(26)
            lines.append(f"║  {name} {p.items_processed:>4} items ({p.duration_ms:>6.0f}ms)")
        lines.append(f"╟{'═' * 50}╣")
        lines.append(f"║  Anchors:  {self.anchors_before:>4} → {self.anchors_after:<4} "
                     f"(compression: {self.compression_ratio:.2f}x)")
        lines.append(f"║  Edges:    {self.edges_before:>4} → {self.edges_after:<4}")
        lines.append(f"║  Retention:{self.avg_retention_before:>5.3f} → "
                     f"{self.avg_retention_after:<5.3f}")
        lines.append(f"╟{'═' * 50}╣")
        if self.memories_merged:
            lines.append(f"║  Merged:   {self.memories_merged} near-duplicate anchors")
        if self.schemas_formed:
            lines.append(f"║  Schemas:  {self.schemas_formed} new abstractions")
        if self.memories_pruned:
            lines.append(f"║  Pruned:   {self.memories_pruned} low-retention anchors "
                         f"→ {self.ghosts_created} ghosts")
        if self.emotional_decoupled:
            lines.append(f"║  Emotion:  decoupled from {self.emotional_decoupled} memories")
        if self.cortical_transferred:
            lines.append(f"║  Cortical: {self.cortical_transferred} memories transferred")
        if self.bridges_created:
            lines.append(f"║  Bridges:  {self.bridges_created} cross-constellation links")
        lines.append(f"╚{'═' * 50}╝")
        return "\n".join(lines)
