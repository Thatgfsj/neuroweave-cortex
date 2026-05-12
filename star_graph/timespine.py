"""Time Spine — pseudo-2D temporal index with priority scanning.

Biological analogy: episodic memory timeline. The Time Spine organizes memories
along a primary temporal axis, with each time point holding a bounded set of
memory clusters. This enables:

1. "Upper-right to lower-left" priority scanning (recent+important first)
2. Dimensional reduction fallback: when semantic search fails, scan the timeline
3. Single-day explosion prevention: max clusters per time unit
4. Time-window queries ("last week", "Tuesday's debugging session")

Structure:
    Day Spine → Memory Clusters (per day, max N)
    Each Cluster → anchor IDs (grouped by topic)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryCluster:
    """A group of related memories at a specific time point."""
    id: str
    anchor_ids: list[str] = field(default_factory=list)
    topic: str = ""               # auto-extracted topic label
    importance: float = 0.0       # max importance of contained anchors
    created_at: float = field(default_factory=time.time)
    summary: str = ""             # compressed summary of cluster contents

    @property
    def size(self) -> int:
        return len(self.anchor_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.anchor_ids) == 0


@dataclass
class TimeBucket:
    """A single time unit (e.g., one day) holding memory clusters."""
    timestamp: float               # epoch seconds for this bucket's start
    clusters: list[MemoryCluster] = field(default_factory=list)
    max_clusters: int = 10         # prevent single-day explosion

    @property
    def is_full(self) -> bool:
        return len(self.clusters) >= self.max_clusters

    def add_cluster(self, cluster: MemoryCluster) -> bool:
        """Add a cluster. Returns False if bucket is full and cluster is low priority."""
        if not self.is_full:
            self.clusters.append(cluster)
            return True

        # If full, only replace a lower-importance cluster
        min_cluster = min(self.clusters, key=lambda c: c.importance)
        if cluster.importance > min_cluster.importance:
            self.clusters.remove(min_cluster)
            self.clusters.append(cluster)
            return True
        return False

    def top_clusters(self, n: int = 5) -> list[MemoryCluster]:
        """Return top-n clusters by importance."""
        sorted_clusters = sorted(self.clusters, key=lambda c: -c.importance)
        return sorted_clusters[:n]


class TimeSpine:
    """Pseudo-2D temporal memory index.

    Organizes memory clusters along a primary temporal axis (day spine).
    Each day holds at most `max_clusters_per_day` clusters.
    The spine supports "upper-right to lower-left" priority scanning:
    scan from most recent + most important → older + less important.

    Usage:
        spine = TimeSpine(max_clusters_per_day=10)
        spine.index_anchor(anchor, day_timestamp)

        # Priority scan: recent+important first
        clusters = spine.scan_priority(max_days=30, max_clusters=20)

        # Time window query
        clusters = spine.query_window(start_time, end_time)
    """

    def __init__(self, max_clusters_per_day: int = 10,
                 cluster_similarity_threshold: float = 0.5):
        self.max_clusters_per_day = max_clusters_per_day
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.buckets: dict[int, TimeBucket] = {}  # day_key → TimeBucket
        self._cluster_id_counter: int = 0

    # ── Indexing ─────────────────────────────────────────

    def index_anchor(self, anchor_id: str,
                     timestamp: float | None = None,
                     importance: float = 0.5,
                     embedding: list[float] | None = None,
                     topic: str = "") -> str | None:
        """Index an anchor into the time spine.

        Finds or creates the appropriate day bucket, then finds or creates
        a cluster for this anchor's topic. Returns the cluster ID.
        """
        if timestamp is None:
            timestamp = time.time()

        day_key = self._day_key(timestamp)
        bucket = self._get_or_create_bucket(day_key, timestamp)

        # Find existing cluster for this topic, or create new
        cluster = self._find_or_create_cluster(
            bucket, anchor_id, importance, embedding, topic, timestamp)

        return cluster.id if cluster else None

    def remove_anchor(self, anchor_id: str):
        """Remove an anchor from its cluster. Cleans up empty clusters."""
        for bucket in self.buckets.values():
            for cluster in bucket.clusters:
                if anchor_id in cluster.anchor_ids:
                    cluster.anchor_ids.remove(anchor_id)
            # Remove empty clusters
            bucket.clusters = [c for c in bucket.clusters if not c.is_empty]

    def update_importance(self, anchor_id: str, importance: float):
        """Update an anchor's importance, propagating to its cluster."""
        for bucket in self.buckets.values():
            for cluster in bucket.clusters:
                if anchor_id in cluster.anchor_ids:
                    cluster.importance = max(cluster.importance, importance)

    # ── Scanning (dimensional reduction Level 3) ─────────

    def scan_priority(self, max_days: int = 30,
                      max_clusters: int = 20,
                      min_importance: float = 0.0) -> list[MemoryCluster]:
        """"Upper-right to lower-left" priority scan.

        Scans from most recent day → oldest day.
        Within each day, clusters are ordered by importance (high → low).

        upper-right = recent + high importance  (scanned first)
        lower-left  = old + low importance      (scanned last)

        This is Level 3 of dimensional reduction retrieval.
        """
        if not self.buckets:
            return []

        # Sort day keys descending (most recent first)
        sorted_days = sorted(self.buckets.keys(), reverse=True)
        if max_days:
            cutoff = self._day_key(time.time()) - max_days
            sorted_days = [d for d in sorted_days if d >= cutoff]

        results: list[MemoryCluster] = []
        for day_key in sorted_days:
            bucket = self.buckets[day_key]
            # Within each day: top clusters by importance (descending)
            day_clusters = sorted(bucket.clusters, key=lambda c: -c.importance)
            for cluster in day_clusters:
                if cluster.importance >= min_importance and not cluster.is_empty:
                    results.append(cluster)
                    if len(results) >= max_clusters:
                        return results

        return results

    def query_window(self, start_time: float, end_time: float,
                     max_clusters: int = 20) -> list[MemoryCluster]:
        """Query memory clusters within a specific time window."""
        start_day = self._day_key(start_time)
        end_day = self._day_key(end_time)

        results: list[MemoryCluster] = []
        for day_key in range(start_day, end_day + 1):
            bucket = self.buckets.get(day_key)
            if bucket:
                results.extend(bucket.top_clusters(5))
                if len(results) >= max_clusters:
                    break

        # Sort by importance within the window
        results.sort(key=lambda c: -c.importance)
        return results[:max_clusters]

    def scan_timeline(self, start_time: float,
                      direction: str = "backward",
                      max_clusters: int = 20) -> list[MemoryCluster]:
        """Linear timeline scan forward or backward from a point."""
        sorted_days = sorted(self.buckets.keys(), reverse=(direction == "backward"))
        start_day = self._day_key(start_time)

        if direction == "backward":
            sorted_days = [d for d in sorted_days if d <= start_day]
        else:
            sorted_days = [d for d in sorted_days if d >= start_day]

        results: list[MemoryCluster] = []
        for day_key in sorted_days:
            bucket = self.buckets[day_key]
            for cluster in sorted(bucket.clusters, key=lambda c: -c.importance):
                if not cluster.is_empty:
                    results.append(cluster)
                    if len(results) >= max_clusters:
                        return results
        return results

    # ── Internal ─────────────────────────────────────────

    def _day_key(self, timestamp: float) -> int:
        """Convert epoch seconds to day index (days since epoch)."""
        return int(timestamp // 86400)

    def _get_or_create_bucket(self, day_key: int, timestamp: float) -> TimeBucket:
        if day_key not in self.buckets:
            self.buckets[day_key] = TimeBucket(
                timestamp=day_key * 86400,
                max_clusters=self.max_clusters_per_day,
            )
        return self.buckets[day_key]

    def _find_or_create_cluster(self, bucket: TimeBucket,
                                 anchor_id: str,
                                 importance: float,
                                 embedding: list[float] | None,
                                 topic: str,
                                 timestamp: float) -> MemoryCluster | None:
        """Find an existing cluster for this topic, or create a new one."""
        # Try to match an existing cluster
        best_cluster: MemoryCluster | None = None
        best_sim = self.cluster_similarity_threshold

        for cluster in bucket.clusters:
            # Topic match
            if topic and cluster.topic == topic:
                best_cluster = cluster
                break
            # Tag-based match: if no topic, just use the most similar cluster
            # (For now, simple topic-based matching)

        if best_cluster:
            best_cluster.anchor_ids.append(anchor_id)
            best_cluster.importance = max(best_cluster.importance, importance)
            return best_cluster

        # Create new cluster
        self._cluster_id_counter += 1
        cluster = MemoryCluster(
            id=f"mc_{self._cluster_id_counter}",
            anchor_ids=[anchor_id],
            topic=topic,
            importance=importance,
            created_at=timestamp,
        )

        if not bucket.add_cluster(cluster):
            return None  # bucket full and cluster too low priority

        return cluster

    # ── Health ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total_clusters = sum(len(b.clusters) for b in self.buckets.values())
        total_anchors = sum(
            len(c.anchor_ids) for b in self.buckets.values() for c in b.clusters)
        return {
            "days_indexed": len(self.buckets),
            "total_clusters": total_clusters,
            "total_anchors_indexed": total_anchors,
            "max_clusters_per_day": self.max_clusters_per_day,
        }

    @property
    def most_recent_day(self) -> int | None:
        if not self.buckets:
            return None
        return max(self.buckets.keys())

    @property
    def oldest_day(self) -> int | None:
        if not self.buckets:
            return None
        return min(self.buckets.keys())
