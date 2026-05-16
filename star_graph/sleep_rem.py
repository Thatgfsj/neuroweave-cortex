"""REM (rapid eye movement) sleep phase mixin — emotional decoupling, schema
extraction, and emergent abstraction discovery.

Provides the dream-state mechanisms that strip emotional valence from
consolidated memories and discover higher-order conceptual categories.
"""

from __future__ import annotations

from collections import defaultdict

from .anchor import Anchor, MemoryState
from .graph import Schema


class SleepREM:
    """Mixin: REM (dream) sleep consolidation routines."""

    # ── Phase 3: Emotional Stripping ────────────────────

    def _emotional_stripping(self) -> None:
        c = self.cfg.sleep.emotional

        for anchor in self.graph.anchors.values():
            if anchor.vector.stability > c.strip_stability_threshold:
                old_valence = anchor.vector.emotional_valence
                anchor.vector.emotional_valence *= c.decay
                anchor.vector.importance = max(
                    anchor.vector.importance * c.importance_min_factor,
                    abs(old_valence) * c.importance_emotional_residual + c.importance_baseline
                )

        self._log_event("Emotional Stripping: decoupled emotion from consolidated memories")

    # ── Phase 4: Schema Extraction + Abstraction Emergence ─

    def _schema_extraction(self) -> int:
        """Extract schemas AND discover emergent abstract categories.

        Two-stage process:
        1. Tag-based schema extraction (legacy, for tagged anchors)
        2. Embedding-cluster-based abstraction (new — emergent categories)
        """
        c = self.cfg.sleep.schema
        formed = 0

        tag_groups: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            for tag in anchor.tags:
                tag_groups[tag].append(anchor)

        for tag, group in tag_groups.items():
            if len(group) < c.min_instances:
                continue

            existing_schema = any(
                s for s in self.graph.schemas.values()
                if tag in s.tags
            )
            if existing_schema:
                continue

            sorted_group = sorted(group, key=lambda a: -a.vector.stability)

            # Use embedding similarity for schema validation
            similarities = []
            for i in range(min(c.min_instances, len(sorted_group))):
                for j in range(i + 1, min(c.min_instances, len(sorted_group))):
                    if sorted_group[i].embedding and sorted_group[j].embedding:
                        sim = self._embedding_similarity(
                            sorted_group[i].embedding, sorted_group[j].embedding)
                    else:
                        sim = 0.0
                    similarities.append(sim)

            avg_sim = sum(similarities) / max(1, len(similarities))
            if avg_sim < c.min_similarity:
                continue

            template_anchor = sorted_group[0]
            schema_id = f"schema_{tag}_{self._cycle_count}"
            schema = Schema(
                id=schema_id,
                template=template_anchor.text,
                slots={"topic": "specific topic instance", "context": "conversation context"},
                instance_ids=[a.id for a in sorted_group[:c.min_instances]],
                confidence=avg_sim,
                tags=[tag],
            )
            self.graph.schemas[schema_id] = schema
            formed += 1

            for a in sorted_group[:c.min_instances]:
                a.schema_ref = schema_id

        if formed:
            self._log_event(f"Schema Extraction: formed {formed} new schemas (embedding-based)")

        # Phase 4b: Abstraction Emergence — discover emergent categories
        abstract_formed = self._abstraction_emergence()

        return formed + abstract_formed

    def _abstraction_emergence(self) -> int:
        """Discover emergent higher-order categories from anchor clusters.

        Uses embedding-cluster-based detection. When multiple anchors share
        a semantic subspace, generates AbstractNode capturing the invariant.

        Example: "likes Python" + "writes Flask" + "deploys FastAPI"
                 -> AbstractNode "Backend Python Developer"
        """
        try:
            from .abstraction import AbstractionEngine
        except ImportError:
            return 0

        if not hasattr(self, '_abstraction_engine'):
            self._abstraction_engine = AbstractionEngine(
                min_cluster_size=self.cfg.abstraction.min_cluster_size,
                similarity_threshold=self.cfg.abstraction.similarity_threshold,
            )

        # Collect anchor embeddings
        anchors = {}
        embeddings = {}
        for aid, a in self.graph.anchors.items():
            if a.embedding and a.state.name in ('ACTIVE', 'DORMANT', 'CONSOLIDATING'):
                anchors[aid] = a
                embeddings[aid] = a.embedding

        new_abstracts = self._abstraction_engine.discover(anchors, embeddings)

        for abstract in new_abstracts:
            # Store in graph
            self.graph.abstracts[abstract.id] = abstract
            # Tag source anchors with abstraction reference
            for aid in abstract.source_anchor_ids:
                if aid in self.graph.anchors:
                    self.graph.anchors[aid].tags.append(f"abstract:{abstract.label}")

        if new_abstracts:
            self._log_event(
                f"Abstraction Emergence: discovered {len(new_abstracts)} "
                f"new concepts: {[a.label for a in new_abstracts]}"
            )

        return len(new_abstracts)
