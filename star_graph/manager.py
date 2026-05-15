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

    # ── Sync/Async unification ──────────────────────────────

    def to_async(self):
        """Return an AsyncMemoryManager wrapping this manager.

        Provides the same API but with async/await support for all operations.
        """
        from .async_manager import AsyncMemoryManager
        return AsyncMemoryManager(self)

    async def async_remember(self, text: str, **kwargs):
        """Async wrapper for remember()."""
        import asyncio
        return await asyncio.to_thread(self.remember, text, **kwargs)

    async def async_recall(self, query: str = "", **kwargs):
        """Async wrapper for recall()."""
        import asyncio
        return await asyncio.to_thread(self.recall, query, **kwargs)

    async def async_sleep(self, **kwargs):
        """Async wrapper for sleep()."""
        import asyncio
        return await asyncio.to_thread(self.sleep, **kwargs)

    async def async_micro_consolidate(self, **kwargs):
        """Async wrapper for micro_consolidate()."""
        import asyncio
        return await asyncio.to_thread(self.micro_consolidate, **kwargs)

    async def async_stats(self, **kwargs):
        """Async wrapper for stats()."""
        import asyncio
        return await asyncio.to_thread(lambda: self.stats, **kwargs)
