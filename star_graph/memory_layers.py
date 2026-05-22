"""4-Layer Memory Pyramid — working, episodic, semantic, core-identity layers.

Each layer has distinct recall strategies, compression policies, TTLs,
decay rates, protection levels, and item caps. LayerManager integrates
with StarGraph to classify, manage, and enforce per-layer constraints.

Layer boundary: this module lives at Layer 1 (Storage) and imports
only from memory_core (L1). It provides the layer taxonomy that
cognitive modules (L2) and behavior modules (L3) consume.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor


class MemoryLayer(str, Enum):
    """Four-layer cognitive memory taxonomy."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CORE_IDENTITY = "core_identity"


@dataclass
class LayerPolicy:
    """Configuration for one memory layer's behavior."""

    layer: MemoryLayer
    default_ttl_seconds: float
    recall_strategy: str
    compression_strategy: str
    max_items: int
    decay_rate: float
    protection_level: int
    auto_archive_to: MemoryLayer | None = None


_WORKING_POLICY = LayerPolicy(
    layer=MemoryLayer.WORKING,
    default_ttl_seconds=3600.0,
    recall_strategy="full_inject",
    compression_strategy="none",
    max_items=200,
    decay_rate=1.0,
    protection_level=0,
    auto_archive_to=None,
)

_EPISODIC_POLICY = LayerPolicy(
    layer=MemoryLayer.EPISODIC,
    default_ttl_seconds=2592000.0,
    recall_strategy="semantic_search",
    compression_strategy="summarize",
    max_items=30000,
    decay_rate=0.1,
    protection_level=0,
    auto_archive_to=None,
)

_SEMANTIC_POLICY = LayerPolicy(
    layer=MemoryLayer.SEMANTIC,
    default_ttl_seconds=31536000.0,
    recall_strategy="graph_traversal",
    compression_strategy="recursive_summarize",
    max_items=15000,
    decay_rate=0.01,
    protection_level=1,
    auto_archive_to=None,
)

_CORE_IDENTITY_POLICY = LayerPolicy(
    layer=MemoryLayer.CORE_IDENTITY,
    default_ttl_seconds=float("inf"),
    recall_strategy="force_inject",
    compression_strategy="manual_only",
    max_items=5000,
    decay_rate=0.0,
    protection_level=2,
    auto_archive_to=None,
)

_LAYER_POLICIES: dict[MemoryLayer, LayerPolicy] = {
    MemoryLayer.WORKING: _WORKING_POLICY,
    MemoryLayer.EPISODIC: _EPISODIC_POLICY,
    MemoryLayer.SEMANTIC: _SEMANTIC_POLICY,
    MemoryLayer.CORE_IDENTITY: _CORE_IDENTITY_POLICY,
}

_WORKING_TAGS: set[str] = {
    "current_task", "debug", "working", "active", "now", "in_progress",
}

_IDENTITY_TAGS: set[str] = {
    "identity", "personality", "core_value", "belief", "value", "self",
}

_SEMANTIC_TAGS: set[str] = {
    "knowledge", "concept", "pattern", "fact", "skill", "expertise",
    "preference",
}

_TASK_TAGS: set[str] = {"task", "todo", "backlog"}
_COMPLETED_TAGS: set[str] = {"completed", "done", "finished", "resolved"}


class LayerManager:
    """Manages four-layer memory assignment and enforcement on a StarGraph."""

    def __init__(self, graph: StarGraph) -> None:
        self._graph = graph
        self._layer_assignments: dict[str, MemoryLayer] = {}

    # ── Layer access ────────────────────────────────────────

    def get_layer(self, anchor_id: str) -> MemoryLayer:
        """Get anchor's current memory layer.

        Checks explicit assignment first, then dynamic attr on anchor,
        then falls back to inference via classify_anchor.
        """
        layer = self._layer_assignments.get(anchor_id)
        if layer is not None:
            return layer
        anchor = self._graph.anchors.get(anchor_id)
        if anchor is not None:
            dynamic = getattr(anchor, "memory_layer", None)
            if dynamic is not None:
                try:
                    return MemoryLayer(dynamic)
                except ValueError:
                    pass
            return self.classify_anchor(anchor)
        return MemoryLayer.EPISODIC

    def set_layer(self, anchor_id: str, layer: MemoryLayer) -> None:
        """Manually assign an anchor to a memory layer."""
        self._layer_assignments[anchor_id] = layer
        anchor = self._graph.anchors.get(anchor_id)
        if anchor is not None:
            anchor.memory_layer = layer.value

    def classify_anchor(self, anchor: Anchor) -> MemoryLayer:
        """Infer the appropriate memory layer from anchor properties.

        Priority: Core Identity > Working > Semantic > Episodic (default).
        """
        tags_lower = {t.lower() for t in anchor.tags}
        vec = anchor.vector

        if self._is_core_identity(anchor, tags_lower, vec):
            return MemoryLayer.CORE_IDENTITY
        if self._is_working(anchor, tags_lower, vec):
            return MemoryLayer.WORKING
        if self._is_semantic(anchor, tags_lower, vec):
            return MemoryLayer.SEMANTIC
        return MemoryLayer.EPISODIC

    # ── Classification helpers ──────────────────────────────

    def _is_working(self, anchor: Anchor, tags: set[str], vec) -> bool:
        if tags & _WORKING_TAGS:
            return True
        if tags & _SEMANTIC_TAGS or tags & _IDENTITY_TAGS:
            return False
        has_task_tag = bool(tags & _TASK_TAGS) and not bool(tags & _COMPLETED_TAGS)
        age_seconds = time.time() - anchor.created_at
        if getattr(anchor, "memory_tier", None) == "hot":
            if has_task_tag:
                return True
            if vec.importance < 0.4 and not bool(tags & _COMPLETED_TAGS):
                return True
            return False
        if has_task_tag:
            return True
        if age_seconds < 3600 and vec.importance < 0.4:
            if not bool(tags & _COMPLETED_TAGS):
                return True
        return False

    def _is_core_identity(self, anchor: Anchor, tags: set[str], vec) -> bool:
        if tags & _IDENTITY_TAGS:
            return True
        if vec.importance > 0.9 and vec.emotional_valence != 0.0:
            return True
        if "preference" in tags and vec.importance > 0.8:
            return True
        return False

    def _is_semantic(self, anchor: Anchor, tags: set[str], vec) -> bool:
        if tags & _SEMANTIC_TAGS:
            return True
        if "knowledge" in tags:
            return True
        if anchor.replay_count > 10:
            return True
        from .memory_core.anchor import MemoryState
        consolidated_states = {
            MemoryState.CONSOLIDATING,
            MemoryState.DORMANT,
            MemoryState.GHOST,
            MemoryState.REACTIVATED,
        }
        if anchor.state in consolidated_states:
            return True
        return False

    # ── Layer config ────────────────────────────────────────

    def get_layer_config(self, layer: MemoryLayer) -> LayerPolicy:
        """Get the LayerPolicy for a given memory layer."""
        return _LAYER_POLICIES[layer]

    def get_layer_counts(self) -> dict[str, int]:
        """Count anchors assigned to each memory layer."""
        counts: dict[str, int] = {lyr.value: 0 for lyr in MemoryLayer}
        for aid in self._graph.anchors:
            layer = self.get_layer(aid)
            counts[layer.value] += 1
        return counts

    # ── Limit enforcement ───────────────────────────────────

    def enforce_limits(self) -> dict[str, list[str]]:
        """Evict excess anchors to stay within per-layer max_items.

        Returns {layer_name: [evicted_anchor_ids]}.
        """
        evictions: dict[str, list[str]] = {}
        anchors_by_layer = self._partition_by_layer()

        for layer in MemoryLayer:
            policy = _LAYER_POLICIES[layer]
            items = anchors_by_layer[layer]
            excess = len(items) - policy.max_items
            if excess <= 0:
                continue

            items.sort(key=lambda aid: self._graph.anchors[aid].retention_score)
            to_evict = items[:excess]
            evicted = []
            for aid in to_evict:
                anchor = self._graph.anchors.get(aid)
                if anchor is not None:
                    pl = policy.protection_level
                    if pl >= 1:
                        continue
                self._graph.remove_anchor(aid)
                self._layer_assignments.pop(aid, None)
                evicted.append(aid)
            if evicted:
                evictions[layer.value] = evicted

        return evictions

    # ── Decay ───────────────────────────────────────────────

    def should_decay(self, anchor_id: str) -> bool:
        """Check if an anchor is eligible for decay based on layer protection."""
        layer = self.get_layer(anchor_id)
        policy = _LAYER_POLICIES[layer]
        if policy.protection_level >= 2:
            return False
        return True

    # ── Injection ───────────────────────────────────────────

    def get_injectable_items(self, max_chars: int) -> list:
        """Collect anchors for context injection.

        All working + all core_identity (forced) + top episodic by relevance,
        capped by max_chars total text length.
        """
        working: list[tuple[float, str, str]] = []
        core_ids: list[tuple[float, str, str]] = []
        episodic: list[tuple[float, str, str]] = []
        semantic: list[tuple[float, str, str]] = []

        for aid, anchor in self._graph.anchors.items():
            if not anchor.is_retrievable:
                continue
            layer = self.get_layer(aid)
            score = anchor.retention_score
            entry = (score, aid, anchor.text)
            if layer == MemoryLayer.WORKING:
                working.append(entry)
            elif layer == MemoryLayer.CORE_IDENTITY:
                core_ids.append(entry)
            elif layer == MemoryLayer.EPISODIC:
                episodic.append(entry)
            elif layer == MemoryLayer.SEMANTIC:
                semantic.append(entry)

        core_ids.sort(key=lambda x: -x[0])

        result: list = []
        used_chars = 0

        for score, aid, text in working:
            cost = len(text)
            used_chars += cost
            result.append((aid, text, MemoryLayer.WORKING))
        for score, aid, text in core_ids:
            cost = len(text)
            used_chars += cost
            result.append((aid, text, MemoryLayer.CORE_IDENTITY))

        remaining = max(0, max_chars - used_chars)
        episodic.sort(key=lambda x: -x[0])
        for score, aid, text in episodic:
            cost = len(text)
            if cost > remaining:
                continue
            remaining -= cost
            result.append((aid, text, MemoryLayer.EPISODIC))

        remaining_s = max(0, remaining)
        semantic.sort(key=lambda x: -x[0])
        for score, aid, text in semantic:
            cost = len(text)
            if cost > remaining_s:
                continue
            remaining_s -= cost
            result.append((aid, text, MemoryLayer.SEMANTIC))

        return result

    # ── Layer movement ──────────────────────────────────────

    def promote(self, anchor_id: str, to_layer: MemoryLayer) -> bool:
        """Move anchor to a higher memory layer.

        Layer ordering: WORKING < EPISODIC < SEMANTIC < CORE_IDENTITY.
        Returns True if the promotion was valid.
        """
        current = self.get_layer(anchor_id)
        layer_order = {
            MemoryLayer.WORKING: 0,
            MemoryLayer.EPISODIC: 1,
            MemoryLayer.SEMANTIC: 2,
            MemoryLayer.CORE_IDENTITY: 3,
        }
        if layer_order.get(to_layer, 0) <= layer_order.get(current, 0):
            return False
        self.set_layer(anchor_id, to_layer)
        return True

    def demote(self, anchor_id: str, to_layer: MemoryLayer) -> bool:
        """Move anchor to a lower memory layer for re-evaluation.

        Layer ordering: WORKING < EPISODIC < SEMANTIC < CORE_IDENTITY.
        Returns True if the demotion was valid.
        """
        current = self.get_layer(anchor_id)
        layer_order = {
            MemoryLayer.WORKING: 0,
            MemoryLayer.EPISODIC: 1,
            MemoryLayer.SEMANTIC: 2,
            MemoryLayer.CORE_IDENTITY: 3,
        }
        if layer_order.get(to_layer, 0) >= layer_order.get(current, 0):
            return False
        self.set_layer(anchor_id, to_layer)
        return True

    # ── Internal helpers ────────────────────────────────────

    def _partition_by_layer(self) -> dict[MemoryLayer, list[str]]:
        """Group anchor IDs by their current memory layer."""
        partition: dict[MemoryLayer, list[str]] = defaultdict(list)
        for aid in self._graph.anchors:
            partition[self.get_layer(aid)].append(aid)
        return partition
