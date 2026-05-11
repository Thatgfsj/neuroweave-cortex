"""Storage backend ABC — pluggable persistence for star-graph memory.

Implementations:
  - JSONStorage: single-file JSON (legacy, fine for <10K anchors)
  - SQLiteStorage: indexed SQLite (for 10K+ anchors, concurrent access)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .graph import StarGraph


class StorageBackend(ABC):
    """Abstract interface for graph persistence backends."""

    @abstractmethod
    def save(self, graph: StarGraph) -> None:
        """Persist the full graph state atomically."""
        ...

    @abstractmethod
    def load(self) -> StarGraph:
        """Load the full graph state."""
        ...

    @property
    @abstractmethod
    def exists(self) -> bool:
        """Whether persisted state exists."""
        ...

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
