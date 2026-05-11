"""Memory Competition — interference-based forgetting and competitive retrieval.

Real cognitive systems are competitive, not just additive. This module implements:

1. Competitive Retrieval: strongly activated anchor A suppresses similar anchor B
2. Interference Forgetting: new knowledge inhibits old contradictory knowledge
3. Emotional Priority: emotional memories outcompete neutral ones
4. Retrieval-Induced Forgetting: retrieving A weakens competing B

This moves beyond "decay over time" to active, adaptive memory management.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph
from .config import Config


class MemoryCompetition:
    """Manages competitive dynamics between anchors.

    Not "delete old → add new" but "new inhibits old when contradictory,
    emotional outcompetes neutral, similar memories interfere."
    """

    def __init__(self, graph: StarGraph, config: Config | None = None):
        self.graph = graph
        self.cfg = config if config is not None else Config.get()

    def apply_competition(self, activated_anchor_id: str,
                          suppression_radius: float | None = None,
                          base_suppression: float | None = None) -> dict:
        """When anchor A is strongly activated, suppress competitors.

        Competitors are anchors that are:
        - Semantically similar but distinct (nearby in embedding space)
        - Share tags with the activated anchor
        - Connected by edges to the activated anchor

        Returns dict of {suppressed_id: suppression_amount}.
        """
        c = self.cfg.competition
        if suppression_radius is None:
            suppression_radius = c.suppression_radius
        if base_suppression is None:
            base_suppression = c.base_suppression

        if activated_anchor_id not in self.graph.anchors:
            return {}

        activated = self.graph.anchors[activated_anchor_id]
        suppressed = {}

        competitors = self._find_competitors(activated, suppression_radius)

        for comp_id, similarity in competitors:
            if comp_id == activated_anchor_id:
                continue
            comp = self.graph.anchors.get(comp_id)
            if not comp:
                continue

            activation_strength = activated.retention_score
            suppression = base_suppression * similarity * activation_strength

            if comp.retention_score < activation_strength:
                suppression *= 1.5  # winner-take-more

            if abs(activated.vector.emotional_valence) > c.emotional_boost_threshold:
                suppression *= c.emotional_suppression_boost

            comp.vector.importance *= (1.0 - suppression)
            comp.vector.stability *= (1.0 - suppression * c.winner_stability_factor)
            comp.vector.recency *= (1.0 - suppression * c.winner_recency_factor)

            suppressed[comp_id] = suppression

            key = self.graph._key(activated_anchor_id, comp_id)
            edge = self.graph.edges.get(key)
            if edge:
                edge.weaken(suppression * c.edge_weaken_factor)

        return suppressed

    def _find_competitors(self, anchor: Anchor,
                          radius: float) -> list[tuple[str, float]]:
        """Find anchors that compete with the given anchor."""
        competitors = []

        for other_id, other in self.graph.anchors.items():
            if other_id == anchor.id:
                continue

            # Competition factor 1: share tags
            tag_overlap = len(set(anchor.tags) & set(other.tags))
            tag_sim = tag_overlap / max(1, len(anchor.tags))

            # Competition factor 2: embedding similarity
            if anchor.embedding and other.embedding:
                emb_sim = self._cosine_sim(anchor.embedding, other.embedding)
            else:
                emb_sim = 0.0

            # Competition factor 3: graph distance (1-hop or 2-hop)
            graph_dist = self._graph_distance(anchor.id, other_id)
            if graph_dist is None or graph_dist > radius:
                graph_sim = 0.0
            else:
                graph_sim = 1.0 / max(1, graph_dist)

            # Combined competition score
            c = self.cfg.competition
            comp_score = c.tag_weight * tag_sim + c.embedding_weight * emb_sim + c.graph_weight * graph_sim

            if comp_score > c.competitor_threshold:
                competitors.append((other_id, comp_score))

        return sorted(competitors, key=lambda x: -x[1])

    def _graph_distance(self, aid_a: str, aid_b: str) -> float | None:
        """BFS distance between two anchors in the graph."""
        if aid_a == aid_b:
            return 0.0
        if aid_a not in self.graph.anchors or aid_b not in self.graph.anchors:
            return None

        from collections import deque
        visited = {aid_a}
        queue = deque([(aid_a, 0)])

        while queue:
            node, dist = queue.popleft()
            if dist > 5:  # limit search depth
                continue
            for neighbor in self.graph._adjacency.get(node, set()):
                if neighbor == aid_b:
                    return dist + 1.0
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return None

    def interference_forget(self, new_anchor: Anchor,
                           interference_threshold: float | None = None) -> list[str]:
        """New knowledge inhibits old contradictory/similar knowledge.

        When a new anchor is created, check if it interferes with existing
        anchors. If similarity is high but details differ (e.g., old API
        knowledge vs new API knowledge), the old one is suppressed.

        Returns list of suppressed anchor IDs.
        """
        c = self.cfg.competition
        if interference_threshold is None:
            interference_threshold = c.interference_threshold

        suppressed = []

        for other_id, other in self.graph.anchors.items():
            if other_id == new_anchor.id:
                continue

            if new_anchor.embedding and other.embedding:
                sim = self._cosine_sim(new_anchor.embedding, other.embedding)
            else:
                continue

            if sim > interference_threshold:
                if new_anchor.created_at > other.created_at:
                    other.vector.importance *= c.interference_importance_factor
                    other.vector.stability *= c.interference_stability_factor
                    if "contradicted" not in other.tags:
                        other.tags.append("contradicted")
                    suppressed.append(other_id)

                    key = self.graph._key(new_anchor.id, other_id)
                    if key in self.graph.edges:
                        self.graph.edges[key].edge_type = "revision"

        return suppressed

    def retrieval_induced_forgetting(self, retrieved_id: str,
                                     competing_ids: list[str]) -> None:
        """Retrieving anchor A weakens competing anchors B, C, D.

        This is the "retrieval practice effect" — practicing one memory
        strengthens it while weakening related unpracticed memories.
        """
        if retrieved_id not in self.graph.anchors:
            return

        retrieved = self.graph.anchors[retrieved_id]

        for comp_id in competing_ids:
            if comp_id not in self.graph.anchors:
                continue
            comp = self.graph.anchors[comp_id]

            # The more similar, the more retrieval-induced forgetting
            if retrieved.embedding and comp.embedding:
                sim = self._cosine_sim(retrieved.embedding, comp.embedding)
            else:
                continue

            if sim > self.cfg.competition.rif_similarity_threshold:
                rif = self.cfg.competition.rif_factor * sim
                comp.vector.importance *= (1.0 - rif)
                comp.vector.stability *= (1.0 - rif * self.cfg.competition.rif_stability_factor)

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(min_len))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-8)

    @property
    def competition_stats(self) -> dict:
        """Stats on competitive dynamics."""
        contradicted = sum(
            1 for a in self.graph.anchors.values()
            if "contradicted" in a.tags
        )
        revisions = sum(
            1 for e in self.graph.edges.values()
            if e.edge_type == "revision"
        )
        return {
            "contradicted_anchors": contradicted,
            "revision_edges": revisions,
            "competition_active": contradicted > 0 or revisions > 0,
        }
