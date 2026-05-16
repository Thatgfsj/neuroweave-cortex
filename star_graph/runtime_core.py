"""RuntimeCore — mixin providing CRUD, working memory, multimodal, cortex, and ghost operations.

Extracted from runtime.py MemoryRuntime to reduce module size.
Inherited by MemoryRuntime alongside RuntimeLifecycle.
"""

from __future__ import annotations

import os

from .anchor import Anchor
from .math_utils import cosine_sim as _cosine_sim
from .working_memory import WorkingMemoryEntry
from .cortex import MemoryCortex
from .write_gate import GateDecision
from .multimodal import MultimodalAnchor


class RuntimeCore:
    """Mixin that provides CRUD, working memory, multimodal, cortex, and ghost operations."""

    # ── CRUD: Basic memory operations ─────────────────────────

    def remember(self, text: str, *, source_session: str = "",
                 tags: list[str] | None = None,
                 emotional_valence: float = 0.0,
                 importance: float = 0.5,
                 connect_to: list[str] | None = None,
                 edge_type: str = "topical",
                 anchor_id: str | None = None,
                 skip_gate: bool = False,
                 **vec_kw) -> Anchor | None:
        """Store a new memory. Returns the anchor, or None if rejected by write gate."""
        embedder = self._get_embedder()
        embedding = embedder.encode(text)

        # ── Stage 0: Write Gate pre-filter (#50) ──
        wg_enabled = False
        wg_cfg = getattr(self.cfg, 'write_gate', None)
        if wg_cfg:
            wg_enabled = getattr(wg_cfg, 'enabled', True)
        if wg_enabled and not skip_gate:
            gate_result = self.write_gate.evaluate(
                text=text,
                embedding=embedding,
                tags=tags,
                importance=importance,
                emotional_valence=emotional_valence,
                graph=self.graph,
            )
            if gate_result.decision == GateDecision.REJECT:
                return None
            if gate_result.decision == GateDecision.MERGE and gate_result.merge_target_id:
                self.update(gate_result.merge_target_id, text=text, tags=tags)
                return self.graph.anchors.get(gate_result.merge_target_id)
            if gate_result.decision == GateDecision.DEFER:
                # Only store in hippocampus; skip graph insertion
                self.hippocampus.ingest(
                    text=text, tags=tags or [], importance=importance,
                    emotional_valence=emotional_valence,
                    source_session=source_session, embedding=embedding,
                )
                return None

        if anchor_id:
            anchor = Anchor(
                id=anchor_id, text=text, embedding=embedding,
                source_session=source_session,
                tags=tags or [],
                vector=Anchor.__dataclass_fields__["vector"].default_factory(),
                **vec_kw,
            )
        else:
            anchor = Anchor.create(
                text=text, source_session=source_session,
                embedding=embedding,
                emotional_valence=emotional_valence,
                importance=importance,
                tags=tags,
                **vec_kw,
            )

        # Hard cap: evict before insert if at capacity
        max_total = int(getattr(self.cfg.graph, 'max_total_anchors', 0) if self.cfg else 0)
        if max_total > 0 and len(self.graph.anchors) >= max_total:
            policy = getattr(self.cfg.graph, 'eviction_policy', 'lowest_retention') if self.cfg else 'lowest_retention'
            evicted = self.graph._evict_anchors(count=1, policy=policy)
            if evicted and self._bm25 is not None:
                for eid in evicted:
                    self._bm25.remove(eid)

        # Cortex capacity: auto-consolidate overfull cortices before adding
        for cortex in self.router.cortices:
            if cortex.is_overfull():
                cortex.ensure_capacity()

        # Hippocampus buffer: ingest into L1 (transient cache)
        hc_enabled = True
        hc_cfg = getattr(self.cfg, 'hippocampus', None)
        if hc_cfg:
            hc_enabled = getattr(hc_cfg, 'enabled', True)
        if hc_enabled:
            self.hippocampus.ingest(
                text=text,
                tags=tags or [],
                importance=importance,
                emotional_valence=emotional_valence,
                source_session=source_session,
                embedding=embedding,
            )

        self.graph.add_anchor(anchor)
        self._anchors_since_micro += 1

        # Index on the temporal spine for time-windowed retrieval (Layer 3)
        self.timespine.index_anchor(
            anchor.id,
            timestamp=anchor.created_at,
            importance=importance,
            embedding=embedding,
            topic=(tags[0] if tags else ""),
        )

        # Index in BM25 keyword index for sparse retrieval channel
        if self._bm25 is not None:
            self._bm25.add(anchor.id, text)

        # Index in domain router for hierarchical retrieval pre-filtering (#48)
        dr_enabled = True
        dr_cfg = getattr(self.cfg, 'domain_router', None)
        if dr_cfg:
            dr_enabled = getattr(dr_cfg, 'enabled', True)
        if dr_enabled:
            self.domain_router.index_anchor(anchor.id, text=text, tags=tags)

        # Ingest into four-layer compression pipeline (#51)
        fl_enabled = True
        fl_cfg = getattr(self.cfg, 'four_layer', None)
        if fl_cfg:
            fl_enabled = getattr(fl_cfg, 'enabled', True)
        if fl_enabled:
            self.four_layer.ingest_message(
                text=text,
                embedding=embedding,
                importance=importance,
                tags=tags or [],
            )

        # Incremental personality model update (#56)
        self.personality.ingest_anchor(anchor)

        # Episodic memory recording — time-ordered event stream (B-13)
        self.episodic_memory.record_episode(
            session_id=source_session,
            summary=text[:200],
            anchor_ids=[anchor.id],
            tags=tags or [],
            emotional_valence=emotional_valence,
            importance=importance,
        )

        # Auto-sleep check: trigger micro/full consolidation on thresholds
        self._check_auto_sleep()

        # Dual-write: also store raw uncompressed chunk in L0 buffer
        self.raw_buffer.add(
            text=text, session_id=source_session,
            embedding=embedding, tags=tags,
            importance=importance, anchor_id=anchor.id,
        )

        # Harvest exact match keys into deterministic KV cache
        auto_harvest = True
        ec_cfg = getattr(self.cfg, 'exact_cache', None)
        if ec_cfg:
            auto_harvest = getattr(ec_cfg, 'auto_harvest', True)
        if auto_harvest:
            self.exact_cache.harvest_from_anchor(anchor)

        # Auto-connect to specified anchors
        if connect_to:
            for target_id in connect_to:
                if target_id in self.graph.anchors:
                    target_emb = self.graph.anchors[target_id].embedding
                    sim = 0.5
                    if target_emb and embedding:
                        sim = _cosine_sim(embedding, target_emb)
                    edge = self.graph.add_edge(
                        anchor.id, target_id,
                        weight=sim, edge_type=edge_type,
                    )
                    # Enforce edge budget after each edge addition (#49)
                    if edge:
                        self.edge_budget.enforce(self.graph, anchor.id)
                        self.edge_budget.enforce(self.graph, target_id)

        return anchor

    def forget(self, anchor_id: str, create_ghost: bool = True) -> Anchor | None:
        """Remove a memory. If create_ghost=True, preserves a latent trace."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None

        if create_ghost:
            residual = {}
            for neighbor_id in self.graph._adjacency.get(anchor_id, set()):
                edge_key = self.graph._key(anchor_id, neighbor_id)
                edge = self.graph.edges.get(edge_key)
                if edge:
                    residual[neighbor_id] = edge.weight * 0.3
            self.ghosts.create(anchor, residual)

        self.graph.remove_anchor(anchor_id)
        self.timespine.remove_anchor(anchor_id)
        if self._bm25 is not None:
            self._bm25.remove(anchor_id)
        return anchor

    # ── Working Memory (short-term buffer) ───────────────────

    def remember_working(self, text: str, *,
                         importance: float = 0.5,
                         tags: list[str] | None = None,
                         source_session: str = "",
                         emotional_valence: float = 0.0) -> WorkingMemoryEntry:
        """Add an item to working memory — fast, ephemeral, high-priority."""
        embedder = self._get_embedder()
        embedding = embedder.encode(text)
        return self.working_memory.add(
            text=text,
            embedding=embedding,
            importance=importance,
            tags=tags,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )

    def get_working(self) -> list[WorkingMemoryEntry]:
        return self.working_memory.get_all()

    def promote_working(self, entry: WorkingMemoryEntry) -> str | None:
        if entry not in self.working_memory._entries:
            return None
        return self.working_memory.promote(entry, self)

    def clear_working_memory(self, session_id: str = ""):
        if session_id:
            self.working_memory.clear_session(session_id)
        else:
            self.working_memory.clear()

    def update(self, anchor_id: str, text: str | None = None,
               tags: list[str] | None = None,
               importance: float | None = None) -> Anchor | None:
        """Update an existing memory with new information."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None

        if text is not None:
            anchor.text = text
            embedder = self._get_embedder()
            anchor.embedding = embedder.encode(text)
        if tags is not None:
            anchor.tags = list(set(anchor.tags + tags))
        if importance is not None:
            anchor.vector.importance = max(0.0, min(1.0, importance))

        anchor.activate()
        return anchor

    # ── Multimodal operations ─────────────────────────────────

    def remember_image(self, image_path: str, caption: str = "",
                      tags: list[str] | None = None,
                      importance: float = 0.5,
                      source_session: str = "",
                      emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Store a memory with an image attachment."""
        anchor = MultimodalAnchor.from_image(
            image_path=image_path,
            provider=self.multimodal,
            caption=caption,
            tags=tags,
            importance=importance,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )
        self.graph.add_anchor(anchor)
        self.raw_buffer.add(
            text=caption or f"[image] {os.path.basename(image_path)}",
            session_id=source_session,
            embedding=anchor.embedding,
            tags=tags,
            importance=importance,
            anchor_id=anchor.id,
        )
        return anchor

    def remember_text_and_image(self, text: str, image_path: str,
                               tags: list[str] | None = None,
                               importance: float = 0.5,
                               source_session: str = "",
                               emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Store a memory with both text content and an associated image."""
        anchor = MultimodalAnchor.from_text_and_image(
            text=text,
            image_path=image_path,
            provider=self.multimodal,
            tags=tags,
            importance=importance,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )
        self.graph.add_anchor(anchor)
        self.raw_buffer.add(
            text=text,
            session_id=source_session,
            embedding=anchor.embedding,
            tags=tags,
            importance=importance,
            anchor_id=anchor.id,
        )
        return anchor

    # ── Cortex management ────────────────────────────────────

    def add_cortex(self, name: str, domain_keywords: list[str],
                   description: str = "", **kwargs) -> MemoryCortex:
        """Create and register a new domain-specific memory cortex."""
        cortex = self.router.find_or_create_cortex(
            name=name,
            domain_keywords=domain_keywords,
            description=description,
            **kwargs,
        )
        self.brain.register_cortex(
            cortex_name=name,
            entry_embedding=cortex.centroid or [],
            summary=description,
            node_count=0,
        )
        return cortex

    # ── Ghost operations ──────────────────────────────────────

    def fuzzy_recall(self, embedding: list[float] | None = None,
                     query: str = "", threshold: float | None = None):
        """Fuzzy recall from ghost traces — 'I seem to remember...' """
        if embedding is None:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        return self.ghosts.fuzzy_recall(embedding, threshold)

    def ghost_intensity_recall(self, query: str = "",
                               embedding: list[float] | None = None,
                               top_k: int = 10) -> list[dict]:
        """Rank ghost traces by intensity for retrieval boosting."""
        if embedding is None:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        ranked = self.ghosts.ranked_resonance(embedding, top_k=top_k)
        return [
            {
                "ghost_id": ghost.id,
                "intensity": round(intensity, 4),
                "semantic_shadow": ghost.semantic_shadow,
                "original_tags": ghost.original_tags,
                "original_importance": ghost.original_importance,
                "emotion_trace": ghost.emotion_trace,
                "revival_count": ghost.revival_count,
            }
            for ghost, intensity in ranked
        ]

    def create_negative_ghost(self, original_text: str,
                             contradiction_text: str,
                             target_anchor_id: str = "",
                             original_importance: float = 0.5,
                             contradiction_type: str = "direct") -> str:
        """Create a negative ghost to track a contradiction."""
        embedder = self._get_embedder()
        embedding = embedder.encode(original_text)
        ghost = self.ghosts.create_negative(
            original_text=original_text,
            contradiction_text=contradiction_text,
            target_anchor_id=target_anchor_id,
            original_importance=original_importance,
            contradiction_type=contradiction_type,
            embedding=embedding,
        )
        return ghost.id

    def check_ghost_suppression(self, embedding: list[float] | None = None,
                               query: str = "") -> dict:
        """Check if a query or embedding is suppressed by negative ghosts."""
        if embedding is None and query:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        if embedding is None:
            return {"suppression_factor": 1.0, "active_negatives": []}

        factor = self.ghosts.check_suppression(embedding)
        active = []
        for ghost in self.ghosts.negative_ghosts:
            resonance = ghost.resonance(embedding)
            if resonance > 0.1:
                active.append({
                    "ghost_id": ghost.id,
                    "resonance": round(resonance, 4),
                    "intensity": round(ghost.intensity, 4),
                    "contradiction_type": ghost.contradiction_type,
                    "contradiction_text": ghost.contradiction_text[:120],
                })

        return {
            "suppression_factor": round(factor, 4),
            "active_negatives": active,
        }

    def _infer_domain(self, anchor) -> str:
        """Infer the memory domain from anchor tags."""
        tags_lower = [t.lower() for t in anchor.tags]
        tag_text = " ".join(tags_lower) + " " + anchor.text.lower()

        procedural_kw = {"how", "fix", "solution", "workflow", "deploy", "install",
                         "config", "setup", "debug", "error", "bug", "patch"}
        semantic_kw = {"preference", "knowledge", "concept", "fact", "opinion",
                       "user", "style", "likes", "dislikes", "background"}
        reflection_kw = {"strategy", "lesson", "pattern", "summary", "insight",
                         "reflection", "meta", "review", "analysis"}

        if any(kw in tag_text for kw in reflection_kw):
            return "reflection"
        if any(kw in tag_text for kw in procedural_kw):
            return "procedural"
        if any(kw in tag_text for kw in semantic_kw):
            return "semantic"
        return "episodic"  # default: conversations and events
