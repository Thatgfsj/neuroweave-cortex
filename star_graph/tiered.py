"""Tiered storage — HOT (RAM) / WARM (RAM+flush) / COLD (disk-only).

Hooks into ThermalState transitions to actually move data between storage
media, not just label the tier. COLD anchors are serialized to disk and
removed from the in-memory graph; they are transparently thawed on access.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


class TieredStorage:
    """Three-tier storage backed by the file system.

    HOT:  Normal in-memory dict (graph.anchors) — full text + embedding.
    WARM: Same as HOT, but candidate for periodic flush to disk.
    COLD: Disk-only — serialized to JSON, only metadata in memory.
    DEAD: Purged entirely (no storage).

    Usage:
        cold = TieredStorage(path="memory_cold.json")
        cold.offload(anchor_id, anchor_data_dict)
        ...
        data = cold.load(anchor_id)  # returns dict or None
    """

    def __init__(self, path: str = ""):
        self._path = path
        self._store: dict[str, dict] = {}  # anchor_id → serialized anchor data
        self._loaded = False
        self._dirty = False

    # ── Public API ─────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._store)

    def offload(self, anchor_id: str, data: dict) -> None:
        """Move an anchor to cold storage (serialize to disk)."""
        entry = {
            "id": anchor_id,
            "text": data.get("text", ""),
            "embedding": data.get("embedding"),
            "tags": data.get("tags", []),
            "importance": data.get("importance", 0.5),
            "emotional_valence": data.get("emotional_valence", 0.0),
            "source_session": data.get("source_session", ""),
            "created_at": data.get("created_at", time.time()),
            "last_activated_at": data.get("last_activated_at", time.time()),
            "community_id": data.get("community_id", ""),
            "offloaded_at": time.time(),
        }
        self._store[anchor_id] = entry
        self._dirty = True

    def load(self, anchor_id: str) -> dict | None:
        """Load an anchor's data from cold storage (thaw)."""
        self._ensure_loaded()
        return self._store.get(anchor_id)

    def remove(self, anchor_id: str) -> None:
        """Permanently delete from cold storage."""
        self._ensure_loaded()
        if anchor_id in self._store:
            del self._store[anchor_id]
            self._dirty = True

    def contains(self, anchor_id: str) -> bool:
        self._ensure_loaded()
        return anchor_id in self._store

    def ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._store.keys())

    def flush(self) -> None:
        """Write cold store to disk."""
        if not self._path or not self._dirty:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)
        self._dirty = False

    def compact(self) -> int:
        """Rewrite the cold store file, removing any deleted entries.

        Returns the number of entries compacted (difference in size).
        """
        if not self._path:
            return 0
        self._ensure_loaded()
        before = len(self._store)
        # The store only contains live entries — removed entries are deleted from dict.
        # compact() just ensures the file on disk matches the in-memory dict.
        self._dirty = True
        self.flush()
        return before  # all entries currently in dict are live

    def reload(self) -> None:
        """Reload from disk, discarding in-memory changes."""
        self._loaded = False
        self._store.clear()
        self._ensure_loaded()

    @property
    def stats(self) -> dict:
        self._ensure_loaded()
        sizes = [len(e.get("text", "")) for e in self._store.values()]
        return {
            "cold_anchors": len(self._store),
            "total_text_bytes": sum(sizes),
            "avg_text_bytes": sum(sizes) / max(1, len(sizes)),
            "dirty": self._dirty,
            "path": self._path or "(memory only)",
        }

    # ── Internal ───────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._path and os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._store = {}
        self._loaded = True


def offload_anchor_to_cold(anchor, cold_store: TieredStorage) -> dict:
    """Serialize anchor data and move to cold store. Returns the serialized dict."""
    data = {
        "text": getattr(anchor, "text", ""),
        "embedding": getattr(anchor, "embedding", None),
        "tags": list(getattr(anchor, "tags", [])),
        "importance": getattr(getattr(anchor, "vector", None), "importance", 0.5),
        "emotional_valence": getattr(getattr(anchor, "vector", None), "emotional_valence", 0.0),
        "source_session": getattr(anchor, "source_session", ""),
        "created_at": getattr(anchor, "created_at", time.time()),
        "last_activated_at": getattr(anchor, "last_activated_at", time.time()),
        "community_id": getattr(anchor, "community_id", ""),
    }
    cold_store.offload(anchor.id, data)
    return data
