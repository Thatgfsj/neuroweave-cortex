"""Cluster Memory — retrieval-integrated community pre-filtering.

Maps queries to nearest community/cluster centroids, then searches within
the matched cluster rather than the full graph. Reduces search scope from
O(all_anchors) to O(cluster_size).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .math_utils import cosine_sim as _cosine_sim


@dataclass
class ClusterCentroid:
    """A memory cluster with a centroid embedding for routing."""
    cluster_id: str
    centroid: list[float]
    anchor_ids: set[str] = field(default_factory=set)
    keywords: list[str] = field(default_factory=list)
    label: str = ""
    size: int = 0
    coherence: float = 0.0
    created_at: float = field(default_factory=time.time)


class ClusterRouter:
    """Routes queries to the nearest memory cluster for scoped retrieval.

    Usage:
        cr = ClusterRouter()
        cr.build_index(graph)                          # build clusters from communities
        cluster_ids, centroid = cr.route(query_emb)    # find best cluster
        scope = cr.get_cluster_scope(cluster_id)       # anchor IDs in cluster
    """

    def __init__(self,
                 min_cluster_size: int = 5,
                 max_clusters: int = 20,
                 similarity_threshold: float = 0.4):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold
        self._clusters: dict[str, ClusterCentroid] = {}
        self._anchor_to_cluster: dict[str, str] = {}  # anchor_id → cluster_id

    # ── Index building ─────────────────────────────────────

    def build_index(self, graph) -> int:
        """Build cluster centroids from graph communities and embeddings.

        Returns number of clusters built.
        """
        self._clusters.clear()
        self._anchor_to_cluster.clear()

        # Group anchors by community_id
        communities: dict[str, list] = defaultdict(list)
        unassigned: list = []
        for aid, anchor in graph.anchors.items():
            if not anchor.is_retrievable or not anchor.embedding:
                continue
            cid = getattr(anchor, 'community_id', '')
            if cid:
                communities[cid].append(anchor)
            else:
                unassigned.append(anchor)

        cluster_idx = 0
        # Build clusters from existing communities
        for cid, anchors in communities.items():
            if len(anchors) < self.min_cluster_size:
                unassigned.extend(anchors)
                continue
            cluster = self._build_cluster(f"cluster_{cid}", anchors)
            if cluster:
                self._clusters[cluster.cluster_id] = cluster
                for aid in cluster.anchor_ids:
                    self._anchor_to_cluster[aid] = cluster.cluster_id
                cluster_idx += 1

        # Build clusters from unassigned anchors via greedy clustering
        if unassigned and cluster_idx < self.max_clusters:
            clustered_ids: set[str] = set()
            for i, anchor in enumerate(unassigned):
                if anchor.id in clustered_ids:
                    continue
                cluster_anchors = [anchor]
                clustered_ids.add(anchor.id)
                for j in range(i + 1, len(unassigned)):
                    other = unassigned[j]
                    if other.id in clustered_ids:
                        continue
                    sim = _cosine_sim(anchor.embedding, other.embedding)
                    if sim >= self.similarity_threshold:
                        cluster_anchors.append(other)
                        clustered_ids.add(other.id)
                if len(cluster_anchors) >= self.min_cluster_size:
                    cluster = self._build_cluster(f"auto_cluster_{cluster_idx}", cluster_anchors)
                    if cluster:
                        self._clusters[cluster.cluster_id] = cluster
                        for aid in cluster.anchor_ids:
                            self._anchor_to_cluster[aid] = cluster.cluster_id
                        cluster_idx += 1
                if cluster_idx >= self.max_clusters:
                    break

        return len(self._clusters)

    def _build_cluster(self, cluster_id: str, anchors: list) -> ClusterCentroid | None:
        """Build a ClusterCentroid from a list of anchors."""
        if not anchors:
            return None
        dim = len(anchors[0].embedding)
        centroid = [0.0] * dim
        for a in anchors:
            for d in range(dim):
                centroid[d] += a.embedding[d]
        for d in range(dim):
            centroid[d] /= len(anchors)

        # Collect keywords from tags
        keyword_counter: dict[str, int] = defaultdict(int)
        all_text = ""
        for a in anchors:
            for tag in a.tags:
                keyword_counter[tag] += 1
            all_text += a.text[:80] + " "
        top_kw = [kw for kw, _ in sorted(keyword_counter.items(), key=lambda x: -x[1])[:5]]

        # Simple label from top keywords
        label = "_".join(top_kw[:3]) if top_kw else cluster_id

        return ClusterCentroid(
            cluster_id=cluster_id,
            centroid=centroid,
            anchor_ids={a.id for a in anchors},
            keywords=top_kw,
            label=label,
            size=len(anchors),
        )

    # ── Routing ────────────────────────────────────────────

    def route(self, query_embedding: list[float] | None = None,
              top_k: int = 2) -> list[tuple[str, float]]:
        """Find nearest clusters to the query embedding.

        Returns list of (cluster_id, similarity_score) sorted by similarity desc.
        """
        if not query_embedding or not self._clusters:
            return []
        scored = []
        for cid, cluster in self._clusters.items():
            sim = _cosine_sim(query_embedding, cluster.centroid)
            if sim >= self.similarity_threshold:
                scored.append((cid, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def get_cluster_scope(self, cluster_id: str) -> set[str]:
        """Get all anchor IDs in a cluster."""
        cluster = self._clusters.get(cluster_id)
        return cluster.anchor_ids if cluster else set()

    def get_anchor_cluster(self, anchor_id: str) -> str:
        """Get the cluster ID for a specific anchor."""
        return self._anchor_to_cluster.get(anchor_id, "")

    def get_cluster_info(self, cluster_id: str) -> dict:
        """Get metadata about a cluster."""
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return {}
        return {
            "cluster_id": cluster.cluster_id,
            "label": cluster.label,
            "size": cluster.size,
            "keywords": cluster.keywords,
            "coherence": cluster.coherence,
        }

    # ── Stats ───────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total_anchors = sum(c.size for c in self._clusters.values())
        return {
            "total_clusters": len(self._clusters),
            "total_anchors_indexed": total_anchors,
            "avg_cluster_size": round(total_anchors / max(1, len(self._clusters)), 1),
            "clusters": [
                {"id": c.cluster_id, "label": c.label, "size": c.size}
                for c in sorted(self._clusters.values(), key=lambda x: -x.size)[:10]
            ],
        }
