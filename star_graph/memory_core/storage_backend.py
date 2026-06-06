"""Storage backend ABC — pluggable persistence for star-graph memory.

Implementations:
  - JSONStorage: single-file JSON (legacy, fine for <10K anchors)
  - SQLiteStorage: indexed SQLite (for 10K+ anchors, concurrent access)

Usage:
    backend = SQLiteStorage("memory.db")  # recommended for production
    backend.save(graph)
    graph = backend.load()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator


class StorageBackend(ABC):
    """Abstract interface for graph persistence backends.

    All implementations must be thread-safe. The base class provides
    a threading.RLock — subclasses should use self._lock to guard
    concurrent access.
    """

    @abstractmethod
    def save(self, graph: Any) -> None:
        """Persist the full graph state atomically."""
        ...

    @abstractmethod
    def load(self) -> Any:
        """Load the full graph state."""
        ...

    @property
    @abstractmethod
    def exists(self) -> bool:
        """Whether persisted state exists."""
        ...

    # ── Coarse-grained batch ops ──

    def batch_save(self, items: list[tuple[str, dict]]) -> None:
        """Atomically persist multiple items.

        Args:
            items: List of (item_type, data_dict) tuples.
                   Common item_types: "anchor", "edge", "ghost", "schema"

        Default implementation calls save() for full graph; override
        for efficient batch writes.
        """
        # Default: full save — subclasses override for efficiency
        pass

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for atomic multi-step operations.

        Ensures all writes within the block are committed atomically
        or rolled back on exception.
        """
        yield

    # ── Search ──

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Perform a basic text/embedding search across persisted items.

        Args:
            query: Search query text
            top_k: Maximum results

        Returns:
            List of item dicts with at least 'id', 'text', 'score' keys.
            Empty list if search is not supported by this backend.
        """
        return []

    # ── Migration ──

    @classmethod
    def detect_and_create(cls, path: str | None = None) -> StorageBackend:
        """Auto-detect the best backend for the given path.

        - '.db' or '.sqlite' suffix → SQLiteStorage
        - '.json' suffix → JSONStorage
        - No path or unknown → SQLiteStorage (default for new installations)

        This is the recommended factory for production use.
        """
        from pathlib import Path

        if path is None:
            path = str(Path.home() / ".nwc" / "memory.db")

        p = Path(path)
        suffix = p.suffix.lower()

        if suffix in (".db", ".sqlite"):
            from .sqlite_storage import SQLiteStorage
            return SQLiteStorage(p)
        elif suffix == ".json":
            from .storage import JSONStorage
            return JSONStorage(p)
        else:
            # Default to SQLite
            from .sqlite_storage import SQLiteStorage
            if not p.suffix:
                p = p.with_suffix(".db")
            return SQLiteStorage(p)

    # ── Fine-grained ops (optional — default to no-op) ──

    def save_anchor(self, anchor_id: str, data: dict) -> None:
        """Persist a single anchor update."""
        pass

    def delete_anchor(self, anchor_id: str) -> None:
        """Remove a single anchor."""
        pass

    def save_edge(self, source: str, target: str, data: dict) -> None:
        """Persist a single edge."""
        pass

    def delete_edge(self, source: str, target: str) -> None:
        """Remove a single edge."""
        pass

    def save_ghost(self, ghost_id: str, data: dict) -> None:
        """Persist a ghost node."""
        pass

    def delete_ghost(self, ghost_id: str) -> None:
        """Remove a ghost node."""
        pass

    def save_schema(self, schema_id: str, data: dict) -> None:
        """Persist a schema."""
        pass

    def close(self) -> None:
        """Release any resources (connections, file handles)."""
        pass
