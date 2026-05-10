"""Resonance-based memory retrieval.

Instead of keyword search or cosine similarity lookup, we find memories by
"resonance" — the current context activates seed anchors, spreading activation
traverses the graph, and the constellation that lights up is the retrieved memory.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph, Constellation


class Resonator:
    """Finds which star strings resonate with a given context."""

    def __init__(self, graph: StarGraph):
        self.graph = graph

    def find_seeds(self, context_text: str, top_k: int = 5) -> list[Anchor]:
        """Find seed anchors that resonate with the context.

        If embeddings are available, uses semantic similarity.
        Otherwise falls back to text overlap as a simple heuristic.
        """
        # Try embedding-based first
        embedded_anchors = [a for a in self.graph.anchors.values() if a.embedding]
        if embedded_anchors and hasattr(self, '_embedder'):
            return self._semantic_seeds(context_text, embedded_anchors, top_k)
        return self._text_overlap_seeds(context_text, top_k)

    def _text_overlap_seeds(self, context: str, top_k: int) -> list[Anchor]:
        """Simple overlap heuristic when embeddings aren't available."""
        scored = []
        ctx_chars = set(context)
        for anchor in self.graph.anchors.values():
            anchor_chars = set(anchor.text)
            overlap = len(ctx_chars & anchor_chars) / max(1, len(ctx_chars | anchor_chars))
            score = overlap * anchor.retention_score
            scored.append((score, anchor))
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored[:top_k]]

    def _semantic_seeds(self, context: str, anchors: list[Anchor],
                        top_k: int) -> list[Anchor]:
        """Use stored embeddings for semantic resonance."""
        ctx_emb = np.array(self._embedder.encode(context))
        anchor_ids = []
        anchor_embs = []
        for a in anchors:
            if a.embedding:
                anchor_ids.append(a.id)
                anchor_embs.append(a.embedding)
        if not anchor_embs:
            return self._text_overlap_seeds(context, top_k)

        anchor_matrix = np.array(anchor_embs)
        similarities = np.dot(anchor_matrix, ctx_emb) / (
            np.linalg.norm(anchor_matrix, axis=1) * np.linalg.norm(ctx_emb) + 1e-8
        )
        # Weight by retention score
        retention_scores = np.array([
            self.graph.anchors[aid].retention_score for aid in anchor_ids
        ])
        combined = similarities * (0.7 + 0.3 * retention_scores)
        top_indices = np.argsort(combined)[-top_k:][::-1]

        return [self.graph.anchors[anchor_ids[i]] for i in top_indices]

    def resonate(self, context: str, spread_steps: int = 3,
                 min_activation: float = 0.1) -> list[Constellation]:
        """Full resonance retrieval:
        1. Find seed anchors from context
        2. Spread activation
        3. Extract constellations that lit up
        4. Return them ranked by activation strength
        """
        seeds = self.find_seeds(context)
        if not seeds:
            return []

        seed_ids = [s.id for s in seeds]
        activation = self.graph.spread_activation(seed_ids, steps=spread_steps)

        # Group activated anchors into constellations
        activated = {aid for aid, level in activation.items()
                     if level >= min_activation}

        constellations: dict[str, Constellation] = {}
        for aid in activated:
            constellation = self.graph.find_constellation(aid)
            if constellation.anchors:
                # Use the most-activated anchor as constellation key
                rep = constellation.anchors[0]
                key = rep.id
                if key not in constellations or constellation.total_weight > constellations[key].total_weight:
                    constellations[key] = constellation

        # Rank by activation sum
        def activation_sum(c: Constellation) -> float:
            return sum(activation.get(a.id, 0) for a in c.anchors)

        ranked = sorted(constellations.values(), key=activation_sum, reverse=True)
        return ranked

    def bridge_score(self, constellation_a: Constellation,
                     constellation_b: Constellation) -> float:
        """Score the potential bridge value between two constellations.

        High score = surprising but useful connection (structural hole bridging).
        """
        # If they're already connected, score is low
        a_ids = {a.id for a in constellation_a.anchors}
        b_ids = {a.id for a in constellation_b.anchors}
        existing = 0
        for aid_a in a_ids:
            for aid_b in b_ids:
                key = self.graph._key(aid_a, aid_b)
                if key in self.graph.edges:
                    existing += self.graph.edges[key].weight

        if existing > 1.0:  # Already well-connected
            return 0.0

        # Semantic similarity of centroids
        va = constellation_a.centroid_vector
        vb = constellation_b.centroid_vector
        dot = sum(a * b for a, b in zip(va.to_list(), vb.to_list()))
        norm_a = np.sqrt(sum(x ** 2 for x in va.to_list()))
        norm_b = np.sqrt(sum(x ** 2 for x in vb.to_list()))
        similarity = dot / (norm_a * norm_b + 1e-8)

        # Bridge score: similar but not connected = high value
        return similarity * (1.0 - min(1.0, existing))
