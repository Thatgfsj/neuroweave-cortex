"""MemoryManager — thin facade for the cognitive memory runtime.

Single entry point for AI agents. Delegates to MemoryRuntime (lifecycle/subsystems)
and RetrievalPipeline (all retrieval paths).

Usage:
    manager = MemoryManager()
    manager.remember("User prefers concise answers", tags=["preference", "style"])
    manager.remember("Debugged Redis timeout — pool size was 10, fixed to 20",
                     tags=["debug", "redis"])

    # Context-aware recall
    ctx = AgentContext(task_type="debugging", active_goals=["fix Redis"])
    memories = manager.recall("Redis connection issues", context=ctx)

    # Cognitive maintenance
    manager.micro_consolidate()   # light online update
    report = manager.sleep()      # full consolidation

    # Persistence
    manager.save("agent_memory.db")
    manager.load("agent_memory.db")

    # Health
    manager.print_health()
"""

from __future__ import annotations

from .graph import StarGraph
from .config import Config
from .runtime import MemoryRuntime, ManagerStats
from .retrieval_pipeline import RetrievalPipeline


class MemoryManager:
    """Thin facade composing MemoryRuntime + RetrievalPipeline.

    All methods not defined here are auto-delegated to the runtime
    or retrieval pipeline via __getattr__.

    Inheritable by wrappers (e.g. AsyncMemoryManager) that need to
    intercept or wrap individual methods.
    """

    def __init__(self, graph: StarGraph | None = None,
                 config: Config | None = None,
                 storage_path: str = ""):
        self._rt = MemoryRuntime(graph=graph, config=config, storage_path=storage_path)
        self._rp = RetrievalPipeline(self._rt)

    # ── Explicit delegations for IDE discoverability ──────────
    # Properties that callers access directly

    @property
    def graph(self):
        return self._rt.graph

    @graph.setter
    def graph(self, value):
        self._rt.graph = value

    @property
    def cfg(self):
        return self._rt.cfg

    @property
    def storage_path(self):
        return self._rt.storage_path

    @storage_path.setter
    def storage_path(self, value):
        self._rt.storage_path = value

    @property
    def stats(self) -> ManagerStats:
        return self._rt.stats

    # ── Auto-delegation ──────────────────────────────────────

    def __getattr__(self, name: str):
        """Delegate missing attributes to runtime, then retrieval pipeline."""
        # Avoid infinite recursion: __getattr__ is only called when normal lookup fails
        rt = self.__dict__.get('_rt')
        if rt is not None and hasattr(rt, name):
            return getattr(rt, name)
        rp = self.__dict__.get('_rp')
        if rp is not None and hasattr(rp, name):
            return getattr(rp, name)
        raise AttributeError(
            f"MemoryManager has no attribute '{name}' "
            f"(not found on MemoryRuntime or RetrievalPipeline)"
        )

    # ── Methods requiring inter-component coordination ──────
    # (most methods are auto-delegated; only add explicit ones
    #  when the method needs to coordinate between runtime and pipeline)
