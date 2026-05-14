"""Thermal Store — 3-tier auto hot/cold/archive with promotion & demotion.

Three tiers:
  HOT (RAM)   — frequently accessed, kept in graph.anchors
  COLD (disk) — infrequently accessed, offloaded to TieredStorage
  ARCHIVE     — almost never accessed, compressed + minimal metadata

Auto-promotion: archive→cold on first access, cold→hot on >=2 accesses within 24h
Auto-demotion:  hot→cold after idle>72h, cold→archive after idle>720h

Wired into runtime.sleep() (demotion scan) and runtime.remember() (promotion on touch).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Optional

from .anchor import Anchor, ThermalState
from .tiered import TieredStorage


# ── Tier thresholds ──
HOT_TO_COLD_IDLE_HOURS = 72.0       # hot → cold after 3 days idle
COLD_TO_ARCHIVE_IDLE_HOURS = 720.0  # cold → archive after 30 days idle
PROMOTE_COLD_TO_HOT_ACCESSES = 2    # access count in window to promote cold→hot
PROMOTE_WINDOW_HOURS = 24.0         # time window for access count tracking
MAX_COLD_ITEMS = 5000               # max cold items before archive compaction


class ThermalStore:
    """3-tier auto storage manager with promotion/demotion.

    Usage:
        store = ThermalStore(storage_dir="memory_data")
        store.touch(anchor_id)           # record access (may trigger promotion)
        store.demote_scan(graph)         # scan for idle anchors to demote (call during sleep)
        store.stats                      # tier occupancy stats
    """

    def __init__(self, storage_dir: str = "memory",
                 hot_to_cold_hours: float = HOT_TO_COLD_IDLE_HOURS,
                 cold_to_archive_hours: float = COLD_TO_ARCHIVE_IDLE_HOURS,
                 promote_accesses: int = PROMOTE_COLD_TO_HOT_ACCESSES,
                 promote_window_hours: float = PROMOTE_WINDOW_HOURS):
        self.hot_to_cold_hours = hot_to_cold_hours
        self.cold_to_archive_hours = cold_to_archive_hours
        self.promote_accesses = promote_accesses
        self.promote_window_hours = promote_window_hours

        # Tier tracking
        self._cold_ids: set[str] = set()        # anchor IDs in cold (disk) tier
        self._archive_ids: set[str] = set()     # anchor IDs in archive tier

        # Access tracking for promotion
        self._access_log: dict[str, list[float]] = defaultdict(list)

        # Disk storage
        cold_path = os.path.join(storage_dir, "thermal_cold.json")
        archive_path = os.path.join(storage_dir, "thermal_archive.json")
        self._cold_store = TieredStorage(path=cold_path)
        self._archive_store = TieredStorage(path=archive_path)

        # Stats
        self._total_promotions = 0
        self._total_demotions = 0
        self._total_archived = 0

    # ── Access tracking (promotion) ──────────────────────────

    def touch(self, anchor_id: str):
        """Record an access to an anchor. May trigger promotion."""
        now = time.time()
        self._access_log[anchor_id].append(now)

        # Clean old entries outside the window
        cutoff = now - self.promote_window_hours * 3600
        self._access_log[anchor_id] = [
            t for t in self._access_log[anchor_id] if t > cutoff
        ]

        # Check promotion eligibility
        count = len(self._access_log[anchor_id])
        if count >= self.promote_accesses:
            if anchor_id in self._archive_ids:
                self._promote_archive_to_cold(anchor_id)
            elif anchor_id in self._cold_ids:
                self._promote_cold_to_hot(anchor_id)

    def _promote_archive_to_cold(self, anchor_id: str):
        """Load archive data into cold tier."""
        data = self._archive_store.load(anchor_id)
        if data is None:
            return
        self._archive_store.remove(anchor_id)
        self._archive_ids.discard(anchor_id)
        self._cold_store.offload(anchor_id, data)
        self._cold_ids.add(anchor_id)
        self._total_promotions += 1

    def _promote_cold_to_hot(self, anchor_id: str):
        """Reconstruct anchor from cold storage into graph."""
        data = self._cold_store.load(anchor_id)
        if data is None:
            self._cold_ids.discard(anchor_id)
            return
        # Promotion is handled by the caller (runtime) — we just mark it
        self._cold_store.remove(anchor_id)
        self._cold_ids.discard(anchor_id)
        self._total_promotions += 1

    # ── Demotion scan (called during sleep) ──────────────────

    def demote_scan(self, graph, now: float | None = None) -> dict:
        """Scan graph for idle anchors and demote tiers.

        Returns stats dict: {hot_to_cold, cold_to_archive, archived_removed}.
        """
        if now is None:
            now = time.time()
        stats = {"hot_to_cold": 0, "cold_to_archive": 0, "archived_removed": 0}

        # Scan hot anchors for demotion to cold
        hot_to_demote = []
        for aid, anchor in list(graph.anchors.items()):
            idle_hours = (now - anchor.last_activated_at) / 3600
            if idle_hours >= self.hot_to_cold_hours:
                hot_to_demote.append(aid)

        for aid in hot_to_demote:
            anchor = graph.anchors.pop(aid, None)
            if anchor is None:
                continue
            data = self._serialize_anchor(anchor)
            self._cold_store.offload(aid, data)
            self._cold_ids.add(aid)
            stats["hot_to_cold"] += 1
            self._total_demotions += 1

        # Scan cold anchors for demotion to archive
        cold_to_demote = []
        for aid in list(self._cold_store.ids()):
            data = self._cold_store.load(aid)
            if data is None:
                continue
            last_accessed = data.get("last_activated_at", 0)
            idle_hours = (now - last_accessed) / 3600
            if idle_hours >= self.cold_to_archive_hours:
                cold_to_demote.append(aid)

        for aid in cold_to_demote:
            data = self._cold_store.load(aid)
            if data is None:
                continue
            # Compress text for archive
            data["text"] = data.get("text", "")[:200]
            data["archived_at"] = now
            self._archive_store.offload(aid, data)
            self._cold_store.remove(aid)
            self._cold_ids.discard(aid)
            self._archive_ids.add(aid)
            stats["cold_to_archive"] += 1
            self._total_archived += 1

        # Compact cold store if over max
        if len(self._cold_ids) > MAX_COLD_ITEMS:
            self._cold_store.compact()

        # Purge ancient archives (> 2 years)
        ancient_cutoff = now - 730 * 24 * 3600  # 2 years
        archive_to_purge = []
        for aid in list(self._archive_store.ids()):
            data = self._archive_store.load(aid)
            if data and data.get("archived_at", 0) < ancient_cutoff:
                archive_to_purge.append(aid)
        for aid in archive_to_purge:
            self._archive_store.remove(aid)
            self._archive_ids.discard(aid)
            stats["archived_removed"] += 1

        return stats

    # ── Cold data access (on retrieval miss) ─────────────────

    def load_cold(self, anchor_id: str) -> dict | None:
        """Load anchor data from cold tier. Returns data dict or None."""
        data = self._cold_store.load(anchor_id)
        if data is not None:
            self.touch(anchor_id)
        return data

    def load_archive(self, anchor_id: str) -> dict | None:
        """Load anchor data from archive tier. Returns data dict or None."""
        data = self._archive_store.load(anchor_id)
        if data is not None:
            self.touch(anchor_id)
        return data

    def thaw_anchor(self, anchor_id: str, graph) -> Anchor | None:
        """Full thaw: cold/archive → hot. Returns reconstructed Anchor or None."""
        data = self.load_cold(anchor_id) or self.load_archive(anchor_id)
        if data is None:
            return None

        anchor = Anchor(
            id=anchor_id,
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            tags=data.get("tags", []),
            source_session=data.get("source_session", ""),
            created_at=data.get("created_at", time.time()),
            last_activated_at=data.get("last_activated_at", time.time()),
            community_id=data.get("community_id", ""),
            importance=data.get("importance", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
        )
        # Remove from cold/archive
        self._cold_store.remove(anchor_id)
        self._cold_ids.discard(anchor_id)
        self._archive_store.remove(anchor_id)
        self._archive_ids.discard(anchor_id)
        # Add to graph (hot)
        graph.add_anchor(anchor)
        self._total_promotions += 1
        return anchor

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _serialize_anchor(anchor: Anchor) -> dict:
        return {
            "id": anchor.id,
            "text": anchor.text,
            "embedding": anchor.embedding,
            "tags": anchor.tags,
            "source_session": anchor.source_session,
            "created_at": anchor.created_at,
            "last_activated_at": anchor.last_activated_at,
            "importance": anchor.vector.importance,
            "emotional_valence": anchor.vector.emotional_valence,
            "community_id": getattr(anchor, 'community_id', ''),
        }

    def flush(self):
        """Flush cold and archive stores to disk."""
        self._cold_store.flush()
        self._archive_store.flush()

    @property
    def stats(self) -> dict:
        return {
            "hot_count": 0,  # filled by runtime
            "cold_count": len(self._cold_ids),
            "archive_count": len(self._archive_ids),
            "total_promotions": self._total_promotions,
            "total_demotions": self._total_demotions,
            "total_archived": self._total_archived,
        }
