"""NREM sleep phase mixin — SWR replay, prioritised sampling, systems consolidation,
Hebbian learning, and synaptic homeostasis.

Provides the core slow-wave-sleep mechanisms that transfer memories from
hippocampal to cortical representations.
"""

from __future__ import annotations

import math
import time


class SleepNREM:
    """Mixin: NREM (slow-wave) sleep consolidation routines."""

    def _hebbian_and_homeostasis(self) -> None:
        self._hebbian_update()
        self._synaptic_homeostasis()
        self._refresh_cortical_index()

    # ── Phase 1: Prioritized SWR Replay ──────────────────

    def _constrained_candidates(self, anchor: Anchor, existing: Anchor,
                                 window_hours: float = 168.0) -> dict[str, Anchor]:
        """Build a constrained candidate set for edge creation.

        Instead of full-graph O(n²) similarity search, only consider anchors
        that share: same session, same topic (tag overlap), same time window,
        or same entities. This keeps edge creation O(k) where k << n.
        """
        candidates: dict[str, Anchor] = {}
        now = time.time()
        anchor_time = existing.created_at if existing else now

        for other_id, other in self.graph.anchors.items():
            if other_id == (existing.id if existing else anchor.id):
                continue

            # Constraint 1: same session
            if existing.source_session and other.source_session == existing.source_session:
                candidates[other_id] = other
                continue

            # Constraint 2: same topic (tag overlap)
            if set(existing.tags) & set(other.tags):
                candidates[other_id] = other
                continue

            # Constraint 3: same time window
            time_diff = abs(anchor_time - other.created_at) / 3600
            if time_diff < window_hours:
                candidates[other_id] = other
                continue

        return candidates

    def _swr_replay(self, recent: list[Anchor], threshold: float) -> None:
        """Prioritized experience replay — like RL's PER but biologically motivated.

        Priority = weighted combination of:
          |emotional_valence| × 0.25  — emotional salience
          surprise × 0.25              — novelty (unexpected = needs consolidation)
          retrieval frequency × 0.20   — how often accessed
          graph centrality × 0.15      — hub position in the graph
          |1 - stability| × 0.15       — unresolved / still-labile memories
        """
        c = self.cfg.sleep.swr
        for anchor in recent:
            centrality = len(self.graph._adjacency.get(anchor.id, set()))
            centrality_norm = min(1.0, centrality / max(1, len(self.graph.anchors)))

            anchor._replay_priority = (
                abs(anchor.vector.emotional_valence) * c.valence_weight
                + anchor.vector.surprise * c.surprise_weight
                + anchor.vector.frequency * c.frequency_weight
                + centrality_norm * c.centrality_weight
                + abs(1.0 - anchor.vector.stability) * c.instability_weight
            )

        prioritized = sorted(recent, key=lambda a: a._replay_priority, reverse=True)

        # Stochastic sampling: top fraction always replayed, rest sampled by priority
        top_half = max(1, int(len(prioritized) * c.top_fraction))
        guaranteed = prioritized[:top_half]
        remaining = prioritized[top_half:]
        if remaining:
            import random
            weights = [a._replay_priority for a in remaining]
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                sample_count = max(1, int(len(remaining) * c.sample_fraction))
                sampled = random.choices(remaining, weights=probs, k=sample_count)
                prioritized = guaranteed + sampled
            else:
                prioritized = guaranteed

        embedder = self._get_embedder()

        for anchor in prioritized:
            existing = self.graph.anchors.get(anchor.id)
            if existing:
                existing.transition('replay')  # ACTIVE/DORMANT → REHEARSING
                existing.activate()
                blend = self.cfg.sleep.merge.importance_blend
                existing.vector.importance = (
                    blend * existing.vector.importance
                    + (1 - blend) * anchor.vector.importance
                )
                existing.vector.surprise = max(existing.vector.surprise, anchor.vector.surprise)
                existing.tags = list(set(existing.tags + anchor.tags))
            else:
                if not anchor.embedding:
                    anchor.embedding = embedder.encode(anchor.text)
                self.graph.add_anchor(anchor)

            # Connect using topology-constrained similarity (NOT full-graph O(n²))
            anchor_emb = existing.embedding if existing else anchor.embedding
            if not anchor_emb:
                anchor_emb = embedder.encode(anchor.text)

            anchor_id = existing.id if existing else anchor.id
            candidates = self._constrained_candidates(anchor, existing or anchor, window_hours=168)
            for other_id, other in candidates.items():
                if other_id == anchor_id:
                    continue
                if not other.embedding:
                    continue
                sim = self._embedding_similarity(anchor_emb, other.embedding)
                if sim > threshold:
                    edge_type = self._infer_edge_type(anchor, other)
                    weight = sim * (1.0 + self.cfg.sleep.edge_formation.emotion_weight_boost * abs(anchor.vector.emotional_valence))
                    self.graph.add_edge(anchor_id, other_id,
                                        weight=min(1.0, weight),
                                        edge_type=edge_type)

        self._log_event(f"SWR Replay: replayed {len(prioritized)} anchors "
                        f"(topology-constrained linking, compression ~{max(1, len(recent)//3)}:1)")

    # ── Phase 2: Systems Consolidation ──────────────────

    def _systems_consolidation(self) -> None:
        c = self.cfg.sleep.systems

        for anchor in self.graph.anchors.values():
            replay_factor = anchor.replay_count / max(1, self._cycle_count)
            anchor.vector.hippocampal_dependency = math.exp(
                -replay_factor * self._cycle_count / c.tau
            )
            anchor.vector.hippocampal_dependency = max(c.min_hippocampal_dep,
                anchor.vector.hippocampal_dependency)

            if anchor.is_cortical:
                if anchor.vector.stability > c.schema_stability_threshold and len(anchor.text) > c.schema_text_threshold:
                    sentences = anchor.text.replace('！', '。').replace('？', '。').split('。')
                    anchor.text = sentences[0].strip()[:200]

                for neighbor in self.graph._adjacency.get(anchor.id, set()):
                    key = self.graph._key(anchor.id, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        if edge.edge_type == "temporal":
                            edge.weaken(c.temporal_edge_weaken)
                        elif edge.edge_type in ("topical", "causal"):
                            edge.strengthen(c.topical_edge_strengthen)

            anchor.vector.stability = min(1.0,
                1.0 - c.cortical_stability_factor * anchor.vector.hippocampal_dependency)

        cortical_count = sum(1 for a in self.graph.anchors.values() if a.is_cortical)
        self._log_event(f"Systems Consolidation: {cortical_count}/{len(self.graph.anchors)} "
                        f"memories cortical")

    # ── Phase 8: Hebbian Update ─────────────────────────

    def _hebbian_update(self) -> None:
        c = self.cfg.sleep.hebbian
        now = time.time()
        for edge in self.graph.edges.values():
            hours = (now - edge.last_activated_at) / 3600
            if edge.co_activation_count > 0 and hours < c.active_window_hours:
                edge.strengthen(c.strengthen_delta)
            else:
                decay = c.decay_log_factor * math.log(1 + hours / c.active_window_hours)
                edge.weaken(min(c.max_decay, decay))

    # ── Phase 9: Synaptic Homeostasis ───────────────────

    def _synaptic_homeostasis(self) -> None:
        if not self.graph.edges:
            return

        c = self.cfg.sleep.homeostasis
        weights = [e.weight for e in self.graph.edges.values()]
        mean_w = sum(weights) / len(weights)

        if mean_w > c.target_mean:
            scale = c.target_mean / mean_w
            blend = c.scale_blend
            for edge in self.graph.edges.values():
                edge.weight *= (blend + (1 - blend) * scale)

        for anchor in self.graph.anchors.values():
            hours = (time.time() - anchor.last_activated_at) / 3600
            anchor.decay(hours)
