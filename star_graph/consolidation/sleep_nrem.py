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

    # ── Phase N2b: Conflict Detection & Resolution (#63) ────

    def _detect_and_resolve_conflicts(self) -> dict:
        """Detect semantic contradictions between similar memories.

        Finds anchor pairs with high embedding similarity (>0.85) but
        opposing emotional valences (one positive, one negative) — these
        represent potential contradictions. Resolves using one of:
          overwrite, coexist, deprecate.

        Returns a dict with conflict counts and resolution breakdown.
        """
        from .conflict_detection import ConflictDetector

        cfg = self.cfg
        cc_cfg = cfg.get_path('conflict', {}) if hasattr(cfg, 'get_path') else {}

        detector = ConflictDetector(
            similarity_threshold=cc_cfg.get('similarity_threshold', 0.85),
            sentiment_threshold=cc_cfg.get('sentiment_threshold', 0.3),
            overwrite_confidence=cc_cfg.get('overwrite_confidence', 0.7),
            deprecate_retention=cc_cfg.get('deprecate_retention', 0.2),
        )
        all_anchors = list(self.graph.anchors.values())
        result = detector.detect_and_resolve(all_anchors)

        if result["conflicts_detected"]:
            self._log_event(
                f"Conflict Detection: {result['conflicts_detected']} conflicts "
                f"(overwrite={result['resolutions']['overwrite']}, "
                f"coexist={result['resolutions']['coexist']}, "
                f"deprecate={result['resolutions']['deprecate']})"
            )
        return result

    # ── Phase N2c: Memory Revision (#65) ─────────────────────

    def _revise_memories(self) -> dict:
        """Revise low-confidence / high-surprise memories during sleep.

        Identifies anchors with confidence below threshold or surprise
        above threshold, then re-summarizes or merges them to improve
        memory quality. Uses template-based revision by default, with
        optional LLM-assisted re-summarization.

        Returns a dict with revision stats.
        """
        from .memory_revision import MemoryRevisionEngine

        cfg = self.cfg
        rev_cfg = cfg.get_path('revision', {}) if hasattr(cfg, 'get_path') else {}

        # Build LLM function if provider is configured
        llm_fn = None
        provider = rev_cfg.get('llm_provider', 'template')
        if provider in ('openai', 'anthropic'):
            try:
                llm_fn = self._build_revision_llm(provider, rev_cfg)
            except Exception:
                pass

        engine = MemoryRevisionEngine(
            confidence_threshold=rev_cfg.get('confidence_threshold', 0.35),
            surprise_threshold=rev_cfg.get('surprise_threshold', 0.7),
            max_candidates=rev_cfg.get('max_candidates', 50),
            similarity_threshold=rev_cfg.get('similarity_threshold', 0.75),
            strengthen_boost=rev_cfg.get('strengthen_boost', 0.15),
            llm_fn=llm_fn,
        )
        all_anchors = list(self.graph.anchors.values())
        result = engine.revise(all_anchors)

        if result.revised or result.merged_into_existing:
            self._log_event(
                f"Memory Revision: {result.candidates_scanned} candidates scanned, "
                f"{result.revised} revised, "
                f"{result.merged_into_existing} merged into existing"
            )
        return {
            "candidates_scanned": result.candidates_scanned,
            "revised": result.revised,
            "merged": result.merged_into_existing,
            "skipped": result.skipped,
        }

    def _build_revision_llm(self, provider: str, rev_cfg: dict):
        """Build an LLM callable for memory revision."""
        import os
        if provider == "openai":
            import openai
            client = openai.OpenAI(
                api_key=rev_cfg.get('api_key', os.getenv('OPENAI_API_KEY', '')),
                base_url=rev_cfg.get('base_url', os.getenv('OPENAI_BASE_URL', '')),
            )
            model = rev_cfg.get('model', 'gpt-4o-mini')

            def _openai_revise(anchor, similar):
                similar_texts = "\n".join(
                    f"- [{a.id[:8]}] {a.text[:100]}" for a in similar[:3]
                )
                prompt = (
                    f"Revise this low-quality memory into a clear, factual statement "
                    f"(max 280 chars).\n\n"
                    f"Memory: {anchor.text}\n\n"
                    f"Related context:\n{similar_texts}\n\n"
                    f"Revised statement:"
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()[:280]

            return _openai_revise

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(
                api_key=rev_cfg.get('api_key', os.getenv('ANTHROPIC_API_KEY', '')),
            )
            model = rev_cfg.get('model', 'claude-haiku-4-5')

            def _anthropic_revise(anchor, similar):
                similar_texts = "\n".join(
                    f"- [{a.id[:8]}] {a.text[:100]}" for a in similar[:3]
                )
                prompt = (
                    f"Revise this low-quality memory into a clear, factual statement "
                    f"(max 280 chars).\n\n"
                    f"Memory: {anchor.text}\n\n"
                    f"Related context:\n{similar_texts}\n\n"
                    f"Revised statement:"
                )
                resp = client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()[:280]

            return _anthropic_revise

        return None

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
