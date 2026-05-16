"""Resonance-based memory retrieval — v0.2 oscillatory phase-locking.

Two retrieval pathways (dual-process model):
1. Hippocampal: oscillatory phase-locking + spreading activation (graph traversal)
2. Cortical: direct embedding lookup (for well-consolidated memories)

Plus: prediction-error-driven retrieval — the best constellation is the one
that minimizes prediction error against current context.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

from ..anchor import Anchor, AnchorPrediction
from ..graph import StarGraph, Constellation


def _derive_oscillation(embedding: list[float] | None) -> tuple[float, float]:
    """Derive driving frequency and phase from a context embedding.

    Maps the embedding's statistical properties to oscillatory parameters.
    Frequency = how "fast"/"intense" the context is (variance)
    Phase = where in the cycle we are (mean direction)
    """
    if not embedding or len(embedding) < 4:
        return (0.5, 0.0)
    arr = np.array(embedding)
    # Frequency from normalized variance (0.1..1.0)
    var = float(np.var(arr))
    freq = 0.2 + 0.8 * min(1.0, var / (np.mean(np.abs(arr)) + 1e-8))
    # Phase from angular components
    angles = np.arctan2(arr[1::2], arr[::2])  # pairwise as angle vectors
    phase = float(np.mean(angles)) % (2 * math.pi)
    return (freq, phase)


class Resonator:
    """Finds which star strings resonate with a given context.

    Uses dual-pathway retrieval:
    1. Oscillatory phase-locking for hippocampal (recent/episodic) memories
    2. Cortical direct lookup for consolidated (semantic) memories
    """

    def __init__(self, graph: StarGraph):
        self.graph = graph
        self._embedder = None

    def set_embedder(self, embedder) -> None:
        """Attach a sentence-transformers model for embeddings."""
        self._embedder = embedder

    def find_seeds(self, context_text: str, embedding: list[float] | None = None,
                   top_k: int = 5) -> list[Anchor]:
        """Find seed anchors using hybrid retrieval."""
        seeds: list[Anchor] = []

        # Pathway 1: Oscillatory resonance (hippocampal)
        driving_freq, driving_phase = _derive_oscillation(embedding)
        resonance = self.graph.oscillatory_resonance(driving_freq, driving_phase)
        for aid, strength in sorted(resonance.items(), key=lambda x: -x[1])[:top_k * 2]:
            if aid in self.graph.anchors:
                seeds.append(self.graph.anchors[aid])

        # Pathway 2: Cortical direct lookup (for consolidated memories)
        if embedding:
            cortical = self.graph.cortical_lookup(embedding, top_k=top_k)
            for aid, score in cortical:
                if aid not in {s.id for s in seeds} and aid in self.graph.anchors:
                    seeds.append(self.graph.anchors[aid])

        # Deduplicate and rank by combined score
        seen: set[str] = set()
        unique: list[Anchor] = []
        for a in seeds:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)
        return unique[:top_k]

    def resonate(self, context: str, embedding: list[float] | None = None,
                 spread_steps: int = 3, min_activation: float = 0.1,
                 use_prediction: bool = True) -> list[Constellation]:
        """Full resonance retrieval with prediction error minimization."""
        seeds = self.find_seeds(context, embedding)
        if not seeds:
            return []

        seed_ids = [s.id for s in seeds]
        activation = self.graph.spread_activation(seed_ids, steps=spread_steps)
        activated = {aid for aid, level in activation.items()
                     if level >= min_activation}

        constellations: dict[str, Constellation] = {}
        for aid in activated:
            c = self.graph.find_constellation(aid)
            if c.anchors:
                rep = c.anchors[0]
                key = rep.id
                if key not in constellations or c.total_weight > constellations[key].total_weight:
                    constellations[key] = c

        ranked = list(constellations.values())

        if use_prediction and len(ranked) > 1:
            ranked = self._rerank_by_prediction_error(ranked, context)

        def activation_sum(c: Constellation) -> float:
            return sum(activation.get(a.id, 0) for a in c.anchors)

        ranked.sort(key=activation_sum, reverse=True)
        return ranked

    def _rerank_by_prediction_error(self, constellations: list[Constellation],
                                    context: str) -> list[Constellation]:
        """Re-rank constellations: minimize prediction error against context.

        The constellation that best predicts the current context is the
        most relevant one — even if it has lower raw activation.
        """
        # Create a simple prediction from context as "actual"
        ctx_prediction = AnchorPrediction(emotional_tone=0.0, confidence=0.5)

        scored = []
        for c in constellations:
            # Aggregate predictions from anchors in constellation
            predictions = [a.prediction for a in c.anchors if a.prediction]
            if not predictions:
                scored.append((0.5, c))  # neutral if no predictions
                continue

            total_error = 0.0
            for p in predictions:
                total_error += p.error(ctx_prediction)
            avg_error = total_error / len(predictions)

            # Lower error = higher score
            error_score = 1.0 - min(1.0, avg_error)
            scored.append((error_score, c))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    def bridge_score(self, constellation_a: Constellation,
                     constellation_b: Constellation) -> float:
        """Score the potential bridge value between two constellations.

        Incorporates phase compatibility: two constellations whose dominant
        oscillations are harmonics of each other have higher bridge potential.
        """
        a_ids = {a.id for a in constellation_a.anchors}
        b_ids = {a.id for a in constellation_b.anchors}
        existing = 0.0
        for aid_a in a_ids:
            for aid_b in b_ids:
                key = self.graph._key(aid_a, aid_b)
                if key in self.graph.edges:
                    existing += self.graph.edges[key].weight

        if existing > 1.0:
            return 0.0

        # Semantic similarity of centroids
        va = constellation_a.centroid_vector
        vb = constellation_b.centroid_vector
        la, lb = va.to_list(), vb.to_list()
        dot = sum(a * b for a, b in zip(la, lb))
        na = math.sqrt(sum(x ** 2 for x in la))
        nb = math.sqrt(sum(x ** 2 for x in lb))
        similarity = dot / (na * nb + 1e-8)

        # Phase compatibility bonus
        freq_a, phase_a = constellation_a.dominant_oscillation
        freq_b, phase_b = constellation_b.dominant_oscillation
        freq_ratio = max(freq_a, freq_b) / (min(freq_a, freq_b) + 1e-8)
        is_harmonic = abs(freq_ratio - round(freq_ratio)) < 0.15
        harmonic_bonus = 0.2 if is_harmonic else 0.0

        return similarity * (1.0 - min(1.0, existing)) + harmonic_bonus * (1.0 - similarity)

    # ── Predictive retrieval ─────────────────────────────

    def predictive_retrieve(self, context: str,
                            embedding: list[float] | None = None
                            ) -> tuple[Constellation | None, str]:
        """Retrieve with prediction error guiding the decision.

        Returns:
            (constellation, action) where action is 'confirm', 'update', or 'novel'
        """
        constellations = self.resonate(context, embedding, use_prediction=True)

        if not constellations:
            return None, "novel"

        best = constellations[0]
        predictions = [a.prediction for a in best.anchors if a.prediction]

        if not predictions:
            return best, "confirm"

        # Compute aggregate prediction error
        ctx_pred = AnchorPrediction()
        total_error = sum(p.error(ctx_pred) for p in predictions) / len(predictions)

        if total_error < 0.15:
            return best, "confirm"
        elif total_error < 0.50:
            return best, "update"
        else:
            return best, "novel"
