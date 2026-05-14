"""Self-Organization — auto-cluster, merge similar, detect emergent topics.

Three mechanisms run during sleep:
  1. Community detection: label propagation on the graph → auto-assign community_id
  2. Near-duplicate merge: cosine-similar anchors with overlapping tags → merge into one
  3. Emergent topic labeling: key-term extraction from cluster text → topic labels

Prevents graph fragmentation and keeps the knowledge structure tidy without
manual curation.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .math_utils import cosine_sim as _cosine_sim


@dataclass
class EmergentTopic:
    """A cluster of related anchors with an auto-generated label."""
    name: str
    keywords: list[str]
    anchor_ids: set[str] = field(default_factory=set)
    centroid_embedding: list[float] | None = None
    coherence: float = 0.0         # how well the cluster holds together (0-1)
    size: int = 0
    created_at: float = field(default_factory=time.time)


class SelfOrganization:
    """Auto-organization engine for the memory graph.

    Usage:
        so = SelfOrganization()
        report = so.organize(graph)  # call during sleep
    """

    def __init__(self,
                 merge_threshold: float = 0.88,
                 cluster_similarity: float = 0.55,
                 min_cluster_size: int = 3,
                 max_topics: int = 30):
        self.merge_threshold = merge_threshold
        self.cluster_similarity = cluster_similarity
        self.min_cluster_size = min_cluster_size
        self.max_topics = max_topics
        self._topics: dict[str, EmergentTopic] = {}
        self._total_merged = 0
        self._total_clustered = 0

    # ── Main entry point ─────────────────────────────────────

    def organize(self, graph, current_time: float | None = None) -> dict:
        """Run full self-organization cycle. Returns report dict."""
        report = {
            "topics_detected": 0,
            "anchors_clustered": 0,
            "merges": 0,
            "communities_assigned": 0,
        }

        # Step 1: Auto-assign communities (label propagation)
        report["communities_assigned"] = self._auto_assign_communities(graph)

        # Step 2: Detect emergent topics from clusters
        topic_result = self._detect_topics(graph)
        report["topics_detected"] = topic_result["topics"]
        report["anchors_clustered"] = topic_result["anchors"]

        # Step 3: Merge near-duplicate anchors
        report["merges"] = self._merge_near_duplicates(graph)

        return report

    # ── Community assignment ─────────────────────────────────

    def _auto_assign_communities(self, graph) -> int:
        """Assign community_id to anchors via label propagation."""
        assigned = 0
        # Build adjacency-based communities
        visited: set[str] = set()
        community_idx = 0

        for anchor_id in graph.anchors:
            if anchor_id in visited:
                continue
            # BFS from this anchor
            community_id = f"auto_comm_{community_idx}"
            queue = [anchor_id]
            visited.add(anchor_id)
            members = [anchor_id]

            while queue:
                node = queue.pop(0)
                for neighbor in graph._adjacency.get(node, set()):
                    if neighbor not in visited and neighbor in graph.anchors:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        members.append(neighbor)

            if len(members) >= 2:  # only assign to non-trivial communities
                for mid in members:
                    anchor = graph.anchors.get(mid)
                    if anchor:
                        anchor.community_id = community_id
                        assigned += 1
                community_idx += 1

        return assigned

    # ── Emergent topic detection ─────────────────────────────

    def _detect_topics(self, graph) -> dict:
        """Detect emergent topics from anchor clusters."""
        anchors_with_emb = {
            aid: a for aid, a in graph.anchors.items()
            if a.embedding and a.is_retrievable
        }
        if len(anchors_with_emb) < self.min_cluster_size:
            return {"topics": 0, "anchors": 0}

        # Simple greedy clustering
        ids = list(anchors_with_emb.keys())
        clustered: set[str] = set()
        topics: list[EmergentTopic] = []

        for i, aid in enumerate(ids):
            if aid in clustered:
                continue
            anchor = anchors_with_emb[aid]
            cluster_ids = {aid}
            cluster_texts = [anchor.text]

            for j in range(i + 1, len(ids)):
                other_id = ids[j]
                if other_id in clustered:
                    continue
                other = anchors_with_emb[other_id]
                sim = _cosine_sim(anchor.embedding, other.embedding)
                if sim >= self.cluster_similarity:
                    cluster_ids.add(other_id)
                    cluster_texts.append(other.text)

            if len(cluster_ids) >= self.min_cluster_size:
                clustered.update(cluster_ids)
                topic = self._label_topic(cluster_texts, cluster_ids, anchor.embedding)
                topics.append(topic)

            if len(topics) >= self.max_topics:
                break

        # Store topics
        self._topics.clear()
        for topic in topics:
            self._topics[topic.name] = topic

        total_clustered = sum(t.size for t in topics)
        self._total_clustered += total_clustered
        return {"topics": len(topics), "anchors": total_clustered}

    def _label_topic(self, texts: list[str], anchor_ids: set[str],
                     centroid: list[float] | None) -> EmergentTopic:
        """Generate a topic label from key terms in the cluster texts."""
        combined = " ".join(texts).lower()

        # Extract meaningful tokens (3+ char words)
        tokens = re.findall(r'[a-z]{3,}|[一-鿿]{1,4}', combined)
        # Count frequency
        counter = Counter(tokens)
        # Remove stop words
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from',
                     'have', 'has', 'was', 'are', 'not', 'but', 'all', 'can'}
        for sw in stop_words:
            counter.pop(sw, None)

        # Top keywords
        top_kw = [kw for kw, _ in counter.most_common(5)]
        topic_name = "_".join(top_kw[:3]) if top_kw else f"topic_{len(self._topics)}"

        # Coherence: ratio of top keywords to total
        total_weight = sum(counter.values())
        top_weight = sum(c for _, c in counter.most_common(5))
        coherence = top_weight / max(1, total_weight)

        return EmergentTopic(
            name=topic_name,
            keywords=top_kw,
            anchor_ids=anchor_ids,
            centroid_embedding=centroid,
            coherence=coherence,
            size=len(anchor_ids),
        )

    # ── Near-duplicate merge ─────────────────────────────────

    def _merge_near_duplicates(self, graph) -> int:
        """Merge anchors that are near-duplicates (high sim + tag overlap)."""
        anchors_with_emb = {
            aid: a for aid, a in graph.anchors.items()
            if a.embedding and a.is_retrievable
        }
        if len(anchors_with_emb) < 2:
            return 0

        merges = 0
        ids = list(anchors_with_emb.keys())
        merged: set[str] = set()

        for i, aid1 in enumerate(ids):
            if aid1 in merged:
                continue
            anchor1 = anchors_with_emb[aid1]

            for j in range(i + 1, len(ids)):
                aid2 = ids[j]
                if aid2 in merged:
                    continue
                anchor2 = anchors_with_emb[aid2]

                sim = _cosine_sim(anchor1.embedding, anchor2.embedding)
                if sim < self.merge_threshold:
                    continue

                # Check tag overlap
                tags1 = set(anchor1.tags) if anchor1.tags else set()
                tags2 = set(anchor2.tags) if anchor2.tags else set()
                tag_overlap = len(tags1 & tags2)
                if tag_overlap == 0 and len(tags1) > 0 and len(tags2) > 0:
                    continue  # no tag overlap, keep separate

                # Merge anchor2 into anchor1
                anchor1.text = self._merge_text(anchor1.text, anchor2.text)
                if anchor2.tags:
                    anchor1.tags = list(set(anchor1.tags + anchor2.tags))
                anchor1.vector.importance = max(
                    anchor1.vector.importance, anchor2.vector.importance)
                anchor1.vector.stability = max(
                    anchor1.vector.stability, anchor2.vector.stability)
                anchor1.last_activated_at = max(
                    anchor1.last_activated_at, anchor2.last_activated_at)

                # Rewire edges: all edges from anchor2 → anchor1
                for neighbor_id in list(graph._adjacency.get(aid2, set())):
                    if neighbor_id != aid1:
                        edge_key = graph._key(aid2, neighbor_id)
                        edge = graph.edges.get(edge_key)
                        if edge:
                            graph.add_edge(aid1, neighbor_id,
                                          weight=edge.weight,
                                          edge_type=edge.edge_type)
                    graph._adjacency[neighbor_id].discard(aid2)

                # Remove anchor2
                graph.anchors.pop(aid2, None)
                graph._adjacency.pop(aid2, None)
                # Remove edges from anchor2
                for key in list(graph.edges.keys()):
                    if aid2 in key:
                        del graph.edges[key]
                merged.add(aid2)
                merges += 1
                self._total_merged += 1

        return merges

    @staticmethod
    def _merge_text(text1: str, text2: str) -> str:
        """Merge two similar texts into one."""
        if len(text1) >= len(text2):
            return text1
        return text2

    # ── Query API ────────────────────────────────────────────

    def get_topics(self, min_coherence: float = 0.0) -> list[EmergentTopic]:
        """Get detected emergent topics."""
        return [t for t in self._topics.values()
                if t.coherence >= min_coherence]

    def get_topic_anchors(self, topic_name: str) -> set[str]:
        """Get anchor IDs for a given topic."""
        topic = self._topics.get(topic_name)
        return topic.anchor_ids if topic else set()

    @property
    def stats(self) -> dict:
        return {
            "topics_detected": len(self._topics),
            "total_merged": self._total_merged,
            "total_clustered": self._total_clustered,
            "topics": {
                name: {"size": t.size, "coherence": round(t.coherence, 3)}
                for name, t in sorted(self._topics.items(),
                                     key=lambda x: -x[1].size)[:10]
            },
        }
