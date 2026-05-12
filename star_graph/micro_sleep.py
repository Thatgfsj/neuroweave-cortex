"""Micro-Sleep Scheduler — incremental non-blocking sleep consolidation.

Splits the full 8-phase sleep cycle into bite-sized steps that can run
during agent idle time. Each call to micro_sleep() runs 1-2 phases,
records progress, and returns immediately. The next call resumes from
the checkpoint.

Usage:
    scheduler = MicroSleepScheduler(sleep_cycle)
    while not scheduler.is_complete():
        result = scheduler.run_next(steps=2)
        # agent can do other work between calls
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ── Phase definitions ─────────────────────────────────────────

@dataclass
class MicroPhase:
    """One atomic step in the micro-sleep sequence."""
    name: str                       # human-readable name
    method: str                     # method name on SleepCycle
    description: str = ""
    items_processed: int = 0
    duration_ms: float = 0.0
    error: str | None = None


# All 8 phases in order
_PHASES: list[MicroPhase] = [
    MicroPhase("N1 Replay Indexing", "_swr_replay_phase",
               "SWR replay with priority-weighted sampling"),
    MicroPhase("N2 Weak Merge", "_merge_bridge_phase",
               "Merge similar anchors + bridge distant constellations"),
    MicroPhase("N3 Compression", "_compression_phase",
               "Systems consolidation + schema extraction + Hebbian update"),
    MicroPhase("REM Emotional Decoupling", "_rem_phase",
               "Emotional stripping + synaptic homeostasis"),
    MicroPhase("Wake-prep Schema Synthesis", "_prune_phase",
               "Prune weak anchors + prune weak edges"),
    MicroPhase("5b Compression", "_compression_clusters_phase",
               "Multi-level compression: RAW→EPISODIC→STRATEGIC→META"),
    MicroPhase("5c Atom Facts", "_atom_facts_phase",
               "LLM-assisted entity-centric fact extraction"),
    MicroPhase("6 Hub Connection", "_hub_phase",
               "Cross-cortex hub pattern detection + linking"),
    MicroPhase("7 Forgetting/Degradation", "_thermal_phase",
               "Thermal lifecycle downgrade (HOT→WARM→COLD→DEAD)"),
    MicroPhase("8 Index Rebuild", "_index_phase",
               "ANN refresh + BrainSphere cache rebuild"),
]


@dataclass
class MicroSleepProgress:
    """Checkpoint state for resumable micro-sleep."""
    cycle: int = 0
    phase_index: int = 0           # next phase to run (0-9)
    phases_completed: list[str] = field(default_factory=list)
    phases_remaining: list[str] = field(default_factory=list)
    total_phases: int = 10
    started_at: float = 0.0
    last_phase_at: float = 0.0
    total_duration_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.phase_index >= self.total_phases

    @property
    def progress_pct(self) -> float:
        return self.phase_index / self.total_phases if self.total_phases else 1.0


@dataclass
class MicroSleepResult:
    """Result from one micro_sleep() call."""
    phases_run: list[str] = field(default_factory=list)
    phases_processed: int = 0
    is_complete: bool = False
    progress: MicroSleepProgress | None = None
    errors: list[str] = field(default_factory=list)
    items_processed: int = 0
    duration_ms: float = 0.0


class MicroSleepScheduler:
    """Incremental non-blocking sleep scheduler.

    Wraps a SleepCycle instance and runs its phases incrementally.
    Each call to run_next() executes 1-2 phases and returns.
    The progress is checkpointed so the next call resumes cleanly.

    Usage:
        sleep = SleepCycle(graph, config)
        micro = MicroSleepScheduler(sleep, brain=..., hublayer=..., cortices=[...])
        while not micro.is_complete():
            result = micro.run_next(steps=2)
            print(f"Completed phases: {result.phases_run}")
    """

    def __init__(self, sleep_cycle=None, graph=None, config=None,
                 brain=None, hublayer=None, cortices=None):
        self._sleep = sleep_cycle
        self._graph = graph
        self._config = config
        self.brain = brain
        self.hublayer = hublayer
        self.cortices = cortices or []
        self._progress = MicroSleepProgress(total_phases=len(_PHASES))
        self._phase_results: list[MicroSleepResult] = []

    @property
    def is_complete(self) -> bool:
        return self._progress.is_complete

    @property
    def progress(self) -> MicroSleepProgress:
        return self._progress

    @property
    def sleep_cycle(self):
        """Lazy-init the SleepCycle if not provided."""
        if self._sleep is None:
            from .sleep import SleepCycle
            from .config import Config
            from .graph import StarGraph
            g = self._graph or StarGraph()
            c = self._config or Config.get()
            self._sleep = SleepCycle(g, c)
        return self._sleep

    def resume_from(self, phase_index: int) -> MicroSleepProgress:
        """Resume from a specific phase checkpoint.

        Args:
            phase_index: 0-based index of the phase to resume from.
                         0 = restart from beginning.
        """
        self._progress.phase_index = max(0, min(phase_index, len(_PHASES)))
        self._progress.phases_completed = [p.name for p in _PHASES[:self._progress.phase_index]]
        self._progress.phases_remaining = [p.name for p in _PHASES[self._progress.phase_index:]]
        return self._progress

    def run_next(self, steps: int = 2) -> MicroSleepResult:
        """Run the next 'steps' phases (1-2 typically).

        Args:
            steps: Number of phases to run in this micro-sleep call (default 2).

        Returns:
            MicroSleepResult with phases completed and progress state.
        """
        t0 = time.time()
        result = MicroSleepResult(progress=self._progress)

        if self._progress.is_complete:
            result.is_complete = True
            return result

        self._progress.started_at = self._progress.started_at or t0
        sleep = self.sleep_cycle

        for _ in range(steps):
            if self._progress.is_complete:
                break

            idx = self._progress.phase_index
            phase = _PHASES[idx]
            phase_start = time.time()

            try:
                items = self._run_phase(sleep, phase)
                phase.items_processed = items
                phase.duration_ms = (time.time() - phase_start) * 1000
                phase.error = None
                result.items_processed += items
            except Exception as exc:
                phase.error = str(exc)
                result.errors.append(f"{phase.name}: {exc}")

            phase.duration_ms = (time.time() - phase_start) * 1000
            result.phases_run.append(phase.name)
            result.phases_processed += 1
            self._progress.phases_completed.append(phase.name)
            self._progress.last_phase_at = time.time()
            self._progress.phase_index += 1

            if self._progress.phase_index < len(_PHASES):
                self._progress.phases_remaining = [
                    p.name for p in _PHASES[self._progress.phase_index:]
                ]
            else:
                self._progress.phases_remaining = []

        result.is_complete = self._progress.is_complete
        result.duration_ms = (time.time() - t0) * 1000
        self._progress.total_duration_ms += result.duration_ms
        self._phase_results.append(result)

        return result

    def run_all(self) -> MicroSleepResult:
        """Run all remaining phases (convenience wrapper)."""
        remaining = len(_PHASES) - self._progress.phase_index
        return self.run_next(steps=remaining)

    # ── Phase implementations ──────────────────────────────────

    def _run_phase(self, sleep, phase: MicroPhase) -> int:
        """Execute a single micro-sleep phase. Returns items processed count."""
        sleep_cfg = sleep.cfg.sleep if hasattr(sleep, 'cfg') else None
        merge_threshold = getattr(getattr(sleep_cfg, 'merge', None), 'default_threshold', 0.85)
        prune_threshold = getattr(getattr(sleep_cfg, 'prune', None), 'default_retention_threshold', 0.15)
        edge_threshold = getattr(getattr(sleep_cfg, 'prune', None), 'default_edge_threshold', 0.1)

        if phase.method == "_swr_replay_phase":
            # Collect recent DORMANT/CONSOLIDATING anchors for SWR replay
            from .anchor import MemoryState
            candidates = [a for a in sleep.graph.anchors.values()
                         if a.state in (MemoryState.DORMANT, MemoryState.CONSOLIDATING)]
            if candidates:
                sleep._swr_replay(candidates, merge_threshold)
            return len(candidates)

        elif phase.method == "_merge_bridge_phase":
            merged = sleep._merge_similar(merge_threshold)
            bridges = sleep._bridge_distant()
            return merged + bridges

        elif phase.method == "_compression_phase":
            sleep._systems_consolidation()
            schemas = sleep._schema_extraction()
            sleep._hebbian_update()
            return schemas

        elif phase.method == "_rem_phase":
            sleep._emotional_stripping()
            sleep._synaptic_homeostasis()
            return sum(1 for a in sleep.graph.anchors.values()
                      if abs(a.vector.emotional_valence) < 0.1)

        elif phase.method == "_prune_phase":
            pruned_a = sleep._prune_anchors(prune_threshold)
            pruned_e = sleep._prune_edges(edge_threshold)
            return len(pruned_a) + len(pruned_e)

        elif phase.method == "_compression_clusters_phase":
            stats = sleep._compress_clusters()
            return stats.get("compressed_anchors", 0)

        elif phase.method == "_atom_facts_phase":
            stats = sleep._extract_atom_facts()
            return stats.get("facts_extracted", 0)

        elif phase.method == "_hub_phase":
            if self.hublayer and self.cortices:
                return sleep._connect_cross_cortex_hubs(self.hublayer, self.cortices)
            return 0

        elif phase.method == "_thermal_phase":
            stats = sleep._apply_thermal_forgetting()
            return sum(stats.values())

        elif phase.method == "_index_phase":
            sleep._refresh_cortical_index()
            sleep._rebuild_ann_index()
            if self.brain and self.cortices:
                self.brain.refresh_cache(self.cortices)
            return 0

        return 0

    def get_summary(self) -> str:
        """One-line summary of current micro-sleep progress."""
        p = self._progress
        done = len(p.phases_completed)
        total = len(_PHASES)
        pct = p.progress_pct * 100
        remaining = p.phases_remaining[:3] if p.phases_remaining else []
        tail = f" → {', '.join(remaining)}..." if remaining else "✓ done"
        return f"Micro-sleep: {done}/{total} ({pct:.0f}%){tail}"

    def reset(self):
        """Reset to beginning — start a fresh micro-sleep cycle."""
        self._progress = MicroSleepProgress(total_phases=len(_PHASES))
        self._phase_results.clear()
        if self._sleep:
            self._sleep._cycle_count = getattr(self._sleep, '_cycle_count', 0) + 1
