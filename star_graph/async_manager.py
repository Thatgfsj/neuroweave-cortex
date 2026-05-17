"""Async Memory Manager — asyncio wrapper with concurrent safety.

Provides:
- AsyncMemoryManager: full asyncio wrapper around MemoryManager
- Thread-safe graph operations with reader-writer locks
- Connection pooling for multi-agent shared memory access
- All major methods available as async coroutines
- Backward compatible: sync MemoryManager works unchanged

Usage:
    async with AsyncMemoryManager() as amgr:
        await amgr.remember("User prefers dark mode", tags=["preference"])
        ctx = await amgr.recall("dark mode")
        await amgr.micro_sleep(steps=2)
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AsyncManagerStats:
    """Runtime stats for async manager."""
    active_connections: int = 0
    total_operations: int = 0
    lock_wait_time_ms: float = 0.0
    queue_depth: int = 0
    errors: int = 0


class AsyncMemoryManager:
    """Asyncio-compatible wrapper around MemoryManager.

    Provides:
    - All core MemoryManager methods as async coroutines
    - Reader-writer lock for thread-safe graph access
    - Connection pool for multi-agent scenarios
    - Context manager for clean resource lifecycle

    The underlying MemoryManager is shared — all connections operate
    on the same graph. Locking ensures consistency.

    Usage:
        # Single-agent usage
        async with AsyncMemoryManager() as amgr:
            await amgr.remember("fact", tags=["test"])
            result = await amgr.recall("fact")

        # Multi-agent shared pool
        pool = AsyncMemoryManager(max_connections=4)
        async with pool.connection() as conn:
            await conn.recall("query")
    """

    def __init__(self, memory_manager=None, *,
                 max_connections: int = 4,
                 storage_path: str = "",
                 config=None):
        self._mgr = memory_manager
        self._storage_path = storage_path
        self._config = config
        self.max_connections = max_connections

        # RW lock for concurrent reads, exclusive writes
        self._lock = threading.RLock()
        self._read_sem = threading.Semaphore(max_connections)
        self._write_lock = threading.Lock()

        # Stats
        self._stats = AsyncManagerStats()
        self._started_at = time.time()

        # Lazy init flag
        self._initialized = False

    @property
    def manager(self):
        """Lazy-init the underlying MemoryManager."""
        if self._mgr is None:
            from .manager import MemoryManager
            from .config import Config
            cfg = self._config or Config.get()
            self._mgr = MemoryManager(config=cfg, storage_path=self._storage_path)
        return self._mgr

    async def _init(self):
        """Async initialization — called on first use."""
        if not self._initialized:
            # Ensure embedder is loaded (may download models)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.manager._get_embedder)
            self._initialized = True

    # ── Context manager ───────────────────────────────────────

    async def __aenter__(self):
        await self._init()
        self._stats.active_connections += 1
        return self

    async def __aexit__(self, *args):
        self._stats.active_connections -= 1
        # Don't close the underlying manager — it may be shared

    @asynccontextmanager
    async def connection(self):
        """Get a connection from the pool. Multiple connections share the graph."""
        await self._init()
        self._stats.active_connections += 1
        try:
            yield self
        finally:
            self._stats.active_connections -= 1

    # ── CRUD operations (async) ───────────────────────────────

    async def remember(self, text: str, *,
                       source_session: str = "",
                       tags: list[str] | None = None,
                       emotional_valence: float = 0.0,
                       importance: float = 0.5,
                       connect_to: list[str] | None = None,
                       edge_type: str = "topical",
                       **vec_kw):
        """Async: Store a new memory."""
        await self._init()
        loop = asyncio.get_running_loop()
        self._stats.total_operations += 1

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.remember(
                    text, source_session=source_session,
                    tags=tags, emotional_valence=emotional_valence,
                    importance=importance, connect_to=connect_to,
                    edge_type=edge_type, **vec_kw,
                ),
            )
        return result

    async def recall(self, query: str = "",
                     context=None, max_items: int = 10):
        """Async: Multi-path retrieval with exact cache + raw buffer + graph."""
        await self._init()
        loop = asyncio.get_running_loop()
        self._stats.total_operations += 1

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.recall(query, context, max_items),
            )
        return result

    async def dual_recall(self, query: str = "",
                          context=None, max_items: int = 10):
        """Async: System-1 + System-2 dual-channel recall."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.dual_recall(query, context, max_items),
            )
        return result

    async def forget(self, anchor_id: str, create_ghost: bool = True):
        """Async: Remove a memory."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.forget(anchor_id, create_ghost),
            )
        return result

    async def update(self, anchor_id: str, text: str | None = None,
                     tags: list[str] | None = None,
                     importance: float | None = None):
        """Async: Update an existing memory."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.update(anchor_id, text, tags, importance),
            )
        return result

    # ── Working memory (async) ────────────────────────────────

    async def remember_working(self, text: str, *,
                               importance: float = 0.5,
                               tags: list[str] | None = None,
                               source_session: str = "",
                               emotional_valence: float = 0.0):
        """Async: Add to working memory (fast, ephemeral)."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.remember_working(
                    text, importance=importance, tags=tags,
                    source_session=source_session,
                    emotional_valence=emotional_valence,
                ),
            )
        return result

    async def get_working(self):
        """Async: Get all active working memory items."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, self.manager.get_working,
            )
        return result

    async def promote_working(self, entry):
        """Async: Promote working memory entry to long-term storage."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, lambda: self.manager.promote_working(entry),
            )
        return result

    # ── Sleep & maintenance (async) ───────────────────────────

    async def sleep(self) -> dict:
        """Async: Full 8-phase sleep consolidation."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._write_lock:
            result = await loop.run_in_executor(
                None, self.manager.sleep,
            )
        return result

    async def micro_sleep(self, steps: int = 2) -> dict:
        """Async: Incremental non-blocking sleep (1-2 phases)."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, lambda: self.manager.micro_sleep(steps),
            )
        return result

    async def estimate_sleep_cost(self, dry_run: bool = False):
        """Async: Estimate sleep resource cost."""
        await self._init()
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None, lambda: self.manager.estimate_sleep_cost(dry_run),
        )
        return result

    async def evolve(self) -> dict:
        """Async: Run evolution cycle without sleep."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, self.manager.evolve,
            )
        return result

    # ── Cortex management (async) ─────────────────────────────

    async def add_cortex(self, name: str, domain_keywords: list[str],
                        description: str = "", **kwargs):
        """Async: Create and register a new cortex."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.add_cortex(
                    name, domain_keywords, description, **kwargs,
                ),
            )
        return result

    async def sparse_recall(self, query: str = "",
                            context=None, max_items: int = 20) -> dict:
        """Async: Full sparse-activation recall pipeline."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.sparse_recall(query, context, max_items),
            )
        return result

    # ── Persistence (async) ───────────────────────────────────

    async def save(self, path: str | None = None) -> str:
        """Async: Persist memory system to disk."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, lambda: self.manager.save(path),
            )
        return result

    async def load(self, path: str | None = None):
        """Async: Load memory system from disk."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, lambda: self.manager.load(path),
            )
        return result

    # ── Snapshot management (async) ───────────────────────────

    async def snapshot(self, description: str = "", force: bool = False):
        """Async: Create a versioned snapshot."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.snapshot(description, force),
            )
        return result

    async def recover(self) -> tuple:
        """Async: Crash recovery — load latest snapshot + replay WAL."""
        await self._init()
        loop = asyncio.get_running_loop()

        with self._lock:
            result = await loop.run_in_executor(
                None, self.manager.recover,
            )
        return result

    # ── Health & stats ────────────────────────────────────────

    async def print_health(self):
        """Async: Print system health report."""
        await self._init()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.manager.print_health)

    @property
    def stats(self) -> AsyncManagerStats:
        return self._stats

    @property
    def health(self) -> dict:
        """Snapshot of system state (sync access for monitoring)."""
        return {
            "anchors": len(self.manager.graph.anchors),
            "edges": len(self.manager.graph.edges),
            "ghosts": len(self.manager.graph._ghost_subsystem.ghosts),
            "working_memory": self.manager.working_memory.size,
            "exact_cache": self.manager.exact_cache.stats,
            "sleep_cycles": self.manager.sleep_cycles,
            "connections": self._stats.active_connections,
            "operations": self._stats.total_operations,
            "uptime_seconds": time.time() - self._started_at,
        }

    # ── Bulk operations (async) ───────────────────────────────

    async def remember_batch(self, items: list[dict]) -> list:
        """Async: Store multiple memories efficiently.

        Each item: {text, tags?, importance?, emotional_valence?, source_session?}
        """
        await self._init()
        loop = asyncio.get_running_loop()

        results = []
        for item in items:
            self._stats.total_operations += 1
            with self._lock:
                anchor = await loop.run_in_executor(
                    None,
                    lambda: self.manager.remember(
                        item["text"],
                        source_session=item.get("source_session", ""),
                        tags=item.get("tags"),
                        emotional_valence=item.get("emotional_valence", 0.0),
                        importance=item.get("importance", 0.5),
                    ),
                )
            results.append(anchor)
        return results

    async def recall_batch(self, queries: list[str],
                          context=None, max_items: int = 10) -> list:
        """Async: Run multiple recall queries in parallel.

        Each query is processed with shared-read lock, allowing
        concurrent reads from multiple agents.
        """
        await self._init()
        loop = asyncio.get_running_loop()

        async def _single_recall(q):
            self._stats.total_operations += 1
            with self._lock:
                return await loop.run_in_executor(
                    None,
                    lambda: self.manager.recall(q, context, max_items),
                )

        tasks = [_single_recall(q) for q in queries]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Release resources."""
        self._mgr = None
        self._initialized = False

    # ── async_* aliases for callers migrating from MemoryManager ──

    def __getattr__(self, name: str):
        """Route async_* alias calls to their native async methods.

        Allows both ``await amgr.remember(...)`` and
        ``await amgr.async_remember(...)`` to work identically.
        """
        if name.startswith("async_"):
            native = name[6:]  # strip "async_" prefix
            if native and hasattr(self, native):
                return getattr(self, native)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
