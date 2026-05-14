"""Memory Tiering — explicit STM/MTM/LTM/Core four-layer architecture.

v1.2.0: The graph was degenerating into a "high-dimensional linked list" where
everything connects to everything via cosine similarity. The fix is explicit
tiering — different storage strategies for different memory lifetimes.

Architecture:
  Input → STM (deque, no graph, high churn, 2h TTL)
       ↘ MTM (StarGraph, topic clusters, medium stability, days-weeks)
       ↘ LTM (summary nodes only, high compression, low write, months)
       ↘ Core (user profile, capability model, worldview — near-immutable)

Promotion pipeline (during sleep):
  Micro-sleep: STM → MTM (topic clustering)
  Full sleep:  MTM → LTM (compression into summary nodes)
  Deep sleep:  LTM → Core (worldview extraction into profile)
"""

from __future__ import annotations

import enum
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from .config import Config


class MemoryTier(enum.Enum):
    STM = "stm"    # Short-Term: seconds-hours
    MTM = "mtm"    # Middle-Term: days-weeks
    LTM = "ltm"    # Long-Term: months-years
    CORE = "core"  # Core identity: near-immutable


# Tier decay parameters
TIER_DECAY_HALF_LIFE: dict[MemoryTier, float] = {
    MemoryTier.STM: 2.0 / 24,       # 2 hours → days
    MemoryTier.MTM: 7.0,            # 7 days
    MemoryTier.LTM: 180.0,          # 180 days
    MemoryTier.CORE: 365.0 * 2,     # 2 years
}

TIER_MAX_ITEMS: dict[MemoryTier, int] = {
    MemoryTier.STM: 100,
    MemoryTier.MTM: 2000,
    MemoryTier.LTM: 500,
    MemoryTier.CORE: 50,
}

TIER_PROMOTION_THRESHOLD: dict[tuple[MemoryTier, MemoryTier], float] = {
    (MemoryTier.STM, MemoryTier.MTM): 0.3,
    (MemoryTier.MTM, MemoryTier.LTM): 0.5,
    (MemoryTier.LTM, MemoryTier.CORE): 0.7,
}


@dataclass
class TierEntry:
    """A single entry in any memory tier."""
    id: str
    text: str
    tier: MemoryTier
    embedding: list[float] | None = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    emotional_valence: float = 0.0
    source_session: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    reinforcement: float = 0.0       # 0..1, grows with successful recall
    stability: float = 0.3            # 0..1, resistance to decay
    decay_rate: float = 0.0           # computed from tier + reinforcement
    metadata: dict = field(default_factory=dict)
    promoted_from: str = ""           # source entry ID if promoted
    promoted_to: str = ""             # target entry ID after promotion

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    @property
    def idle_hours(self) -> float:
        return (time.time() - self.last_accessed_at) / 3600

    @property
    def retention_score(self) -> float:
        """Composite retention score adjusted by tier and reinforcement."""
        hours = self.age_hours
        half_life_days = TIER_DECAY_HALF_LIFE.get(self.tier, 7.0)
        half_life_hours = half_life_days * 24
        # Exponential decay: e^(-ln(2) * t / half_life)
        time_decay = math.exp(-0.693 * hours / max(1, half_life_hours))
        # Reinforcement slows decay
        reinforcement_damping = 1.0 + self.reinforcement * 0.5
        return min(1.0, time_decay * reinforcement_damping * self.importance * self.stability)

    def access(self) -> None:
        """Record an access — boosts reinforcement."""
        self.last_accessed_at = time.time()
        self.access_count += 1
        self.reinforcement = min(1.0, self.reinforcement + 0.03)


class ShortTermMemory:
    """STM — deque-based transient buffer. No graph, no complex indexing.

    Fast append/pop, automatic TTL eviction, embedding-only search.
    This is the entry point for ALL new input.
    """

    def __init__(self, max_items: int = 100, ttl_hours: float = 2.0):
        self.max_items = max_items
        self.ttl_hours = ttl_hours
        self._entries: deque[TierEntry] = deque()
        self._by_id: dict[str, TierEntry] = {}
        self._counter = 0

    def add(self, text: str, *,
            embedding: list[float] | None = None,
            tags: list[str] | None = None,
            importance: float = 0.5,
            emotional_valence: float = 0.0,
            source_session: str = "") -> TierEntry:
        """Add an entry to STM. Auto-evicts oldest if at capacity."""
        self._counter += 1
        entry = TierEntry(
            id=f"stm_{self._counter}_{time.time():.0f}",
            text=text,
            tier=MemoryTier.STM,
            embedding=embedding,
            tags=tags or [],
            importance=importance,
            emotional_valence=emotional_valence,
            source_session=source_session,
            decay_rate=TIER_DECAY_HALF_LIFE[MemoryTier.STM],
        )

        # Evict if at capacity
        while len(self._entries) >= self.max_items:
            old = self._entries.popleft()
            self._by_id.pop(old.id, None)

        self._entries.append(entry)
        self._by_id[entry.id] = entry
        return entry

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[TierEntry, float]]:
        """Search STM by embedding similarity (brute-force, bounded by max_items=100)."""
        results: list[tuple[TierEntry, float]] = []
        for entry in self._entries:
            if entry.embedding:
                sim = _cosine_sim(query_embedding, entry.embedding)
                results.append((entry, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def get(self, entry_id: str) -> TierEntry | None:
        return self._by_id.get(entry_id)

    def evict_expired(self) -> list[TierEntry]:
        """Remove entries past TTL. Returns evicted entries for promotion check."""
        now = time.time()
        expired = []
        keep = deque()
        for entry in self._entries:
            if (now - entry.created_at) / 3600 >= self.ttl_hours:
                expired.append(entry)
                self._by_id.pop(entry.id, None)
            else:
                keep.append(entry)
        self._entries = keep
        return expired

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def all_entries(self) -> list[TierEntry]:
        return list(self._entries)


class MiddleTermMemory:
    """MTM — StarGraph-backed topic-cluster memory.

    Stores topic-level clusters, not individual messages. This is where the
    star graph structure provides maximum value.
    """

    def __init__(self, graph=None, max_topics: int = 2000):
        from .graph import StarGraph
        self.graph = graph or StarGraph()
        self.max_topics = max_topics
        self._topic_centroids: dict[str, list[float]] = {}  # tag → centroid
        self._counter = 0

    def add_topic(self, text: str, *,
                  embedding: list[float] | None = None,
                  tags: list[str] | None = None,
                  importance: float = 0.5,
                  source_session: str = "",
                  source_stm_ids: list[str] | None = None,
                  **kwargs) -> str:
        """Add a topic node (summary of multiple STM entries) to MTM."""
        from .anchor import Anchor
        self._counter += 1
        anchor = Anchor.create(
            text=text,
            tags=(tags or []) + ["mtm", "topic_cluster"],
            importance=importance,
            source_session=source_session,
            emotional_valence=0.3,
            **kwargs,
        )
        anchor.id = f"mtm_{self._counter}_{time.time():.0f}"
        if embedding:
            anchor.embedding = embedding
        # MTM-specific: lower decay rate
        anchor.vector.decay_rate = TIER_DECAY_HALF_LIFE[MemoryTier.MTM] / 365.0
        anchor.vector.stability = 0.5
        # Track source STM IDs
        if source_stm_ids:
            anchor.exact_match_keys = source_stm_ids

        self.graph.add_anchor(anchor)

        # Update topic centroids
        if tags and embedding:
            for tag in tags:
                if tag in ('mtm', 'topic_cluster'):
                    continue
                if tag not in self._topic_centroids:
                    self._topic_centroids[tag] = list(embedding)
                else:
                    n = sum(1 for a in self.graph.anchors.values()
                           if tag in a.tags)
                    old = self._topic_centroids[tag]
                    for i in range(len(embedding)):
                        old[i] = (old[i] * (n - 1) + embedding[i]) / n

        # Evict if over capacity
        if len(self.graph.anchors) > self.max_topics:
            self._evict_lowest_retention()

        return anchor.id

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple]:
        """Search MTM via ANN or brute-force on topic centroids."""
        from .anchor import Anchor
        results: list[tuple[Anchor, float]] = []

        # Try ANN first
        ann = self.graph._get_ann_index() if self.graph._ann_index is not None else None
        if ann is not None and ann.size > 0:
            neighbors = ann.query(query_embedding, k=top_k)
            for nid, sim in neighbors:
                anchor = self.graph.anchors.get(nid)
                if anchor and anchor.is_retrievable:
                    results.append((anchor, sim))
        else:
            # Fallback: brute-force on all anchors
            for aid, anchor in self.graph.anchors.items():
                if anchor.embedding and anchor.is_retrievable:
                    sim = _cosine_sim(query_embedding, anchor.embedding)
                    results.append((anchor, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _evict_lowest_retention(self) -> int:
        """Remove the lowest-retention anchors to stay under capacity."""
        candidates = sorted(
            self.graph.anchors.items(),
            key=lambda kv: kv[1].retention_score,
        )
        to_remove = candidates[:max(1, len(candidates) - self.max_topics)]
        for aid, _ in to_remove:
            self.graph.remove_anchor(aid)
        return len(to_remove)

    def __len__(self) -> int:
        return len(self.graph.anchors)


class LongTermMemory:
    """LTM — summary-only nodes with high stability and low write frequency.

    Each LTM node is a compressed concept. No raw messages here.
    """

    def __init__(self, graph=None, max_summaries: int = 500):
        from .graph import StarGraph
        self.graph = graph or StarGraph()
        self.max_summaries = max_summaries
        self._counter = 0

    def add_summary(self, text: str, *,
                    embedding: list[float] | None = None,
                    tags: list[str] | None = None,
                    importance: float = 0.7,
                    confidence: float = 0.6,
                    source_mtm_ids: list[str] | None = None,
                    **kwargs) -> str:
        """Add a summary node to LTM. High stability, very slow decay."""
        from .anchor import Anchor
        self._counter += 1
        anchor = Anchor.create(
            text=text,
            tags=(tags or []) + ["ltm", "summary"],
            importance=importance,
            emotional_valence=0.2,
            **kwargs,
        )
        anchor.id = f"ltm_{self._counter}_{time.time():.0f}"
        if embedding:
            anchor.embedding = embedding
        anchor.vector.decay_rate = TIER_DECAY_HALF_LIFE[MemoryTier.LTM] / 365.0
        anchor.vector.stability = 0.8
        anchor.vector.confidence = confidence
        if source_mtm_ids:
            anchor.exact_match_keys = source_mtm_ids

        self.graph.add_anchor(anchor)

        if len(self.graph.anchors) > self.max_summaries:
            self._evict_lowest_retention()

        return anchor.id

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple]:
        """Search LTM — small, high-quality results."""
        from .anchor import Anchor
        results: list[tuple[Anchor, float]] = []
        ann = self.graph._get_ann_index() if self.graph._ann_index is not None else None
        if ann is not None and ann.size > 0:
            neighbors = ann.query(query_embedding, k=top_k * 2)
            for nid, sim in neighbors:
                anchor = self.graph.anchors.get(nid)
                if anchor and anchor.is_retrievable:
                    # Boost by confidence
                    boost = 1.0 + anchor.vector.confidence * 0.3
                    results.append((anchor, sim * boost))
        else:
            for aid, anchor in self.graph.anchors.items():
                if anchor.embedding and anchor.is_retrievable:
                    sim = _cosine_sim(query_embedding, anchor.embedding)
                    boost = 1.0 + anchor.vector.confidence * 0.3
                    results.append((anchor, sim * boost))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _evict_lowest_retention(self) -> int:
        candidates = sorted(
            self.graph.anchors.items(),
            key=lambda kv: kv[1].retention_score,
        )
        to_remove = candidates[:max(1, len(candidates) - self.max_summaries)]
        for aid, _ in to_remove:
            self.graph.remove_anchor(aid)
        return len(to_remove)

    def __len__(self) -> int:
        return len(self.graph.anchors)


class CoreMemory:
    """Core — near-immutable key-value profile store.

    Stores user identity, preferences, capability models, and worldview.
    Almost never changes. Changes only via deep sleep consensus.
    """

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self._store: dict[str, TierEntry] = {}
        self._counter = 0

    def set(self, key: str, value: str, *,
            confidence: float = 0.5,
            tags: list[str] | None = None,
            source_ltm_id: str = "") -> TierEntry:
        """Set a core profile entry. Auto-increment confidence on repeated sets."""
        self._counter += 1
        existing = self._store.get(key)
        if existing:
            # Reinforce existing belief
            existing.text = value  # update value
            existing.access_count += 1
            existing.reinforcement = min(1.0, existing.reinforcement + 0.1)
            existing.stability = min(1.0, existing.stability + 0.05)
            return existing

        entry = TierEntry(
            id=f"core_{self._counter}_{time.time():.0f}",
            text=value,
            tier=MemoryTier.CORE,
            tags=tags or [],
            importance=0.9,
            stability=0.9,
            decay_rate=TIER_DECAY_HALF_LIFE[MemoryTier.CORE] / 365.0,
            metadata={"key": key, "confidence": confidence,
                     "source_ltm_id": source_ltm_id},
        )
        self._store[key] = entry

        # Evict lowest confidence if over capacity
        if len(self._store) > self.max_entries:
            lowest = min(self._store.items(),
                        key=lambda kv: kv[1].metadata.get("confidence", 0.5))
            del self._store[lowest[0]]

        return entry

    def get(self, key: str) -> TierEntry | None:
        return self._store.get(key)

    def search(self, query_text: str, top_k: int = 5) -> list[tuple[TierEntry, float]]:
        """Simple keyword search over core entries."""
        q_words = set(query_text.lower().split())
        results: list[tuple[TierEntry, float]] = []
        for entry in self._store.values():
            e_words = set(entry.text.lower().split())
            overlap = len(q_words & e_words)
            if overlap > 0:
                score = overlap / max(1, len(q_words))
                score *= entry.metadata.get("confidence", 0.5)
                results.append((entry, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def to_dict(self) -> dict[str, str]:
        return {k: v.text for k, v in self._store.items()}

    def __len__(self) -> int:
        return len(self._store)

    @property
    def all_entries(self) -> list[TierEntry]:
        return list(self._store.values())


class MemoryTierManager:
    """Orchestrates memory across all four tiers.

    Usage:
        mgr = MemoryTierManager()
        mgr.remember("user is working on a memory system project")
        mgr.remember("user prefers Python for backend")

        # On micro-sleep:
        mgr.promote_stm_to_mtm()

        # On full sleep:
        mgr.promote_mtm_to_ltm()

        # Recall searches all tiers
        results = mgr.recall("what language does user prefer?")
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config.get()
        self.stm = ShortTermMemory(
            max_items=getattr(self.cfg, 'stm_max_items', 100),
            ttl_hours=getattr(self.cfg, 'stm_ttl_hours', 2.0),
        )
        self.mtm = MiddleTermMemory(
            max_topics=getattr(self.cfg, 'mtm_max_topics', 2000),
        )
        self.ltm = LongTermMemory(
            max_summaries=getattr(self.cfg, 'ltm_max_summaries', 500),
        )
        self.core = CoreMemory(
            max_entries=getattr(self.cfg, 'core_max_entries', 50),
        )
        self._embedder = None
        self._promotion_log: list[dict] = []

    @property
    def embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    # ── Public API ──────────────────────────────────────

    def remember(self, text: str, *,
                 tags: list[str] | None = None,
                 importance: float = 0.5,
                 emotional_valence: float = 0.0,
                 source_session: str = "") -> TierEntry:
        """Store new input. ALWAYS goes to STM first."""
        embedding = self.embedder.encode(text)
        entry = self.stm.add(
            text=text,
            embedding=embedding,
            tags=tags,
            importance=importance,
            emotional_valence=emotional_valence,
            source_session=source_session,
        )
        return entry

    def recall(self, query: str, *,
               max_items: int = 10,
               search_core: bool = True,
               search_ltm: bool = True,
               search_mtm: bool = True,
               search_stm: bool = True) -> list[dict]:
        """Search all tiers in priority order: Core → LTM → MTM → STM.

        Returns list of dicts with {tier, id, text, score, ...}.
        """
        query_emb = self.embedder.encode(query)
        results: list[dict] = []

        # Core first (identity/preferences)
        if search_core:
            core_results = self.core.search(query, top_k=3)
            for entry, score in core_results:
                entry.access()
                results.append({
                    "tier": "core", "id": entry.id, "text": entry.text,
                    "score": score, "tags": entry.tags,
                    "confidence": entry.metadata.get("confidence", 0.5),
                })

        # LTM (summary concepts)
        if search_ltm and len(results) < max_items:
            ltm_results = self.ltm.search(query_emb, top_k=max_items)
            for anchor, score in ltm_results:
                anchor.activate()
                results.append({
                    "tier": "ltm", "id": anchor.id, "text": anchor.text,
                    "score": score, "tags": anchor.tags,
                    "stability": anchor.vector.stability,
                })

        # MTM (topic clusters)
        if search_mtm and len(results) < max_items * 2:
            mtm_results = self.mtm.search(query_emb, top_k=max_items)
            for anchor, score in mtm_results:
                anchor.activate()
                results.append({
                    "tier": "mtm", "id": anchor.id, "text": anchor.text,
                    "score": score, "tags": anchor.tags,
                })

        # STM (recent raw input)
        if search_stm and len(results) < max_items * 2:
            stm_results = self.stm.search(query_emb, top_k=max_items)
            for entry, score in stm_results:
                entry.access()
                results.append({
                    "tier": "stm", "id": entry.id, "text": entry.text,
                    "score": score, "tags": entry.tags, "age_hours": entry.age_hours,
                })

        # Sort by tier priority, then score
        tier_order = {"core": 0, "ltm": 1, "mtm": 2, "stm": 3}
        results.sort(key=lambda r: (tier_order.get(r["tier"], 9), -r["score"]))
        return results[:max_items]

    # ── Promotion Pipeline ──────────────────────────────

    def promote_stm_to_mtm(self) -> dict:
        """Micro-sleep: cluster STM entries by topic, promote to MTM.

        Evicts expired STM entries first, then groups remaining entries
        by tag similarity and creates MTM topic nodes.

        Returns promotion stats.
        """
        # Evict expired STM entries
        expired = self.stm.evict_expired()

        entries = self.stm.all_entries
        if not entries:
            return {"stm_before": len(entries), "topics_created": 0,
                    "stm_evicted": len(expired), "stm_after": len(self.stm)}

        # Group STM entries by tag overlap
        clusters = self._cluster_by_tags(entries)

        topics_created = 0
        for tag, cluster in clusters.items():
            if len(cluster) < 2:
                continue

            # Compute cluster centroid
            embs = [e.embedding for e in cluster if e.embedding]
            if not embs:
                continue
            dim = len(embs[0])
            centroid = [0.0] * dim
            for emb in embs:
                for i in range(dim):
                    centroid[i] += emb[i]
            for i in range(dim):
                centroid[i] /= len(embs)

            # Generate topic summary from cluster texts
            texts = [e.text for e in cluster]
            summary = self._summarize_cluster(texts, tag)

            # Create MTM topic node
            source_ids = [e.id for e in cluster]
            avg_importance = sum(e.importance for e in cluster) / len(cluster)
            mtm_id = self.mtm.add_topic(
                text=summary,
                embedding=centroid,
                tags=[tag, "promoted_stm"],
                importance=avg_importance,
                source_session=cluster[0].source_session,
                source_stm_ids=source_ids,
            )

            # Mark STM entries as promoted
            for e in cluster:
                e.promoted_to = mtm_id

            # Remove promoted entries from STM
            for e in cluster:
                if e.id in self.stm._by_id:
                    del self.stm._by_id[e.id]
            self.stm._entries = deque(
                e for e in self.stm._entries if e.id not in {c.id for c in cluster}
            )

            topics_created += 1

        self._promotion_log.append({
            "phase": "stm_to_mtm",
            "ts": time.time(),
            "stm_before": len(entries),
            "stm_after": len(self.stm),
            "topics_created": topics_created,
        })

        return {
            "stm_before": len(entries),
            "stm_after": len(self.stm),
            "stm_evicted": len(expired),
            "topics_created": topics_created,
        }

    def promote_mtm_to_ltm(self) -> dict:
        """Full sleep: compress MTM topic clusters into LTM summary nodes.

        Finds MTM topics with high stability and compresses related clusters
        into LTM summary nodes. Returns promotion stats.
        """
        mtm_anchors = list(self.mtm.graph.anchors.values())
        if not mtm_anchors:
            return {"mtm_before": len(mtm_anchors), "summaries_created": 0}

        # Group MTM anchors by tag clusters
        tag_groups: dict[str, list] = defaultdict(list)
        for anchor in mtm_anchors:
            for tag in anchor.tags:
                if tag not in ('mtm', 'topic_cluster', 'promoted_stm'):
                    tag_groups[tag].append(anchor)

        summaries_created = 0
        for tag, group in tag_groups.items():
            if len(group) < 3:
                continue

            # Only promote stable topics
            avg_stability = sum(a.vector.stability for a in group) / len(group)
            if avg_stability < TIER_PROMOTION_THRESHOLD[(MemoryTier.MTM, MemoryTier.LTM)]:
                continue

            # Compute centroid
            embs = [a.embedding for a in group if a.embedding]
            if not embs:
                continue
            dim = len(embs[0])
            centroid = [0.0] * dim
            for emb in embs:
                for i in range(dim):
                    centroid[i] += emb[i]
            for i in range(dim):
                centroid[i] /= len(embs)

            # Generate LTM summary
            texts = [a.text for a in group]
            summary_text = self._summarize_for_ltm(texts, tag)

            mtm_ids = [a.id for a in group]
            avg_importance = sum(a.vector.importance for a in group) / len(group)
            avg_confidence = sum(a.vector.confidence for a in group) / len(group)

            ltm_id = self.ltm.add_summary(
                text=summary_text,
                embedding=centroid,
                tags=[tag, "ltm_summary", "promoted_mtm"],
                importance=avg_importance,
                confidence=avg_confidence,
                source_mtm_ids=mtm_ids,
            )
            summaries_created += 1

        self._promotion_log.append({
            "phase": "mtm_to_ltm",
            "ts": time.time(),
            "mtm_before": len(mtm_anchors),
            "summaries_created": summaries_created,
        })

        return {
            "mtm_before": len(mtm_anchors),
            "summaries_created": summaries_created,
        }

    def promote_ltm_to_core(self) -> dict:
        """Deep sleep: extract worldview patterns from LTM into Core profile.

        Only runs when LTM has high-confidence, stable summaries.
        Returns promotion stats.
        """
        ltm_anchors = list(self.ltm.graph.anchors.values())
        if not ltm_anchors:
            return {"ltm_before": len(ltm_anchors), "core_entries_added": 0}

        # Only consider very stable LTM summaries
        candidates = [
            a for a in ltm_anchors
            if a.vector.stability >= TIER_PROMOTION_THRESHOLD[(MemoryTier.LTM, MemoryTier.CORE)]
            and a.vector.confidence >= 0.7
        ]
        if not candidates:
            return {"ltm_before": len(ltm_anchors), "core_entries_added": 0}

        core_added = 0
        for anchor in candidates:
            # Extract core belief from LTM summary
            tags = [t for t in anchor.tags if t not in ('ltm', 'summary', 'ltm_summary', 'promoted_mtm')]
            if not tags:
                continue

            # Use the primary tag as the core key
            key = f"belief:{tags[0]}"
            existing = self.core.get(key)
            if existing and existing.metadata.get("confidence", 0) >= anchor.vector.confidence:
                continue  # already have a higher-confidence belief

            self.core.set(
                key=key,
                value=anchor.text[:300],
                confidence=anchor.vector.confidence,
                tags=tags,
                source_ltm_id=anchor.id,
            )
            core_added += 1

        self._promotion_log.append({
            "phase": "ltm_to_core",
            "ts": time.time(),
            "ltm_before": len(ltm_anchors),
            "core_entries_added": core_added,
        })

        return {
            "ltm_before": len(ltm_anchors),
            "core_entries_added": core_added,
        }

    # ── Internal Helpers ────────────────────────────────

    def _cluster_by_tags(self, entries: list[TierEntry]) -> dict[str, list[TierEntry]]:
        """Group entries by common tags."""
        clusters: dict[str, list[TierEntry]] = defaultdict(list)
        for entry in entries:
            if entry.tags:
                for tag in entry.tags:
                    clusters[tag].append(entry)
            else:
                clusters["general"].append(entry)

        # Merge small clusters into larger ones
        result = {}
        for tag, group in clusters.items():
            if len(group) >= 2:
                result[tag] = group
        return result

    def _summarize_cluster(self, texts: list[str], topic: str) -> str:
        """Generate a short topic summary from a cluster of STM texts."""
        if len(texts) == 1:
            return f"[{topic}] {texts[0][:200]}"
        # Take the shortest and a representative text
        shortest = min(texts, key=len)
        longest = max(texts, key=len)
        return f"[{topic}] {len(texts)} related items: {shortest[:150]} ... {longest[:150]}"

    def _summarize_for_ltm(self, texts: list[str], topic: str) -> str:
        """Generate a long-term summary from MTM topic texts."""
        if len(texts) <= 2:
            return f"[LTM] {topic}: {'; '.join(t[:200] for t in texts)}"
        shortest = min(texts, key=len)
        return f"[LTM] {topic} ({len(texts)} instances): pattern={shortest[:300]}"

    # ── Health ──────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "stm": {"count": len(self.stm), "max": self.stm.max_items},
            "mtm": {"count": len(self.mtm), "max": self.mtm.max_topics},
            "ltm": {"count": len(self.ltm), "max": self.ltm.max_summaries},
            "core": {"count": len(self.core), "max": self.core.max_entries},
            "total_items": len(self.stm) + len(self.mtm) + len(self.ltm) + len(self.core),
            "promotion_log": self._promotion_log[-5:],
        }


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)


# ═══════════════════════════════════════════════════════════════
# TieredStorage — HOT/WARM/COLD filesystem-backed storage
# (merged from tiered.py, P2 architecture slimdown)
# ═══════════════════════════════════════════════════════════════

import json as _json
import os as _os


class TieredStorage:
    """Three-tier storage backed by the file system.

    HOT:  Normal in-memory dict (graph.anchors) — full text + embedding.
    WARM: Same as HOT, but candidate for periodic flush to disk.
    COLD: Disk-only — serialized to JSON, only metadata in memory.
    DEAD: Purged entirely (no storage).
    """

    def __init__(self, path: str = ""):
        self._path = path
        self._store: dict[str, dict] = {}
        self._loaded = False
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self._store)

    def offload(self, anchor_id: str, data: dict) -> None:
        """Move an anchor to cold storage (serialize to disk)."""
        entry = {
            "id": anchor_id,
            "text": data.get("text", ""),
            "embedding": data.get("embedding"),
            "tags": data.get("tags", []),
            "importance": data.get("importance", 0.5),
            "emotional_valence": data.get("emotional_valence", 0.0),
            "source_session": data.get("source_session", ""),
            "created_at": data.get("created_at", time.time()),
            "last_activated_at": data.get("last_activated_at", time.time()),
            "community_id": data.get("community_id", ""),
            "offloaded_at": time.time(),
        }
        self._store[anchor_id] = entry
        self._dirty = True

    def load(self, anchor_id: str) -> dict | None:
        """Load an anchor's data from cold storage (thaw)."""
        self._ensure_loaded()
        return self._store.get(anchor_id)

    def remove(self, anchor_id: str) -> None:
        """Permanently delete from cold storage."""
        self._ensure_loaded()
        if anchor_id in self._store:
            del self._store[anchor_id]
            self._dirty = True

    def contains(self, anchor_id: str) -> bool:
        self._ensure_loaded()
        return anchor_id in self._store

    def ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._store.keys())

    def flush(self) -> None:
        """Write cold store to disk."""
        if not self._path or not self._dirty:
            return
        _os.makedirs(_os.path.dirname(self._path) or ".", exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(self._store, f, ensure_ascii=False, indent=2)
        _os.replace(tmp, self._path)
        self._dirty = False

    def compact(self) -> int:
        """Rewrite the cold store file, removing any deleted entries."""
        if not self._path:
            return 0
        self._ensure_loaded()
        before = len(self._store)
        self._dirty = True
        self.flush()
        return before

    def reload(self) -> None:
        """Reload from disk, discarding in-memory changes."""
        self._loaded = False
        self._store.clear()
        self._ensure_loaded()

    @property
    def stats(self) -> dict:
        self._ensure_loaded()
        sizes = [len(e.get("text", "")) for e in self._store.values()]
        return {
            "cold_anchors": len(self._store),
            "total_text_bytes": sum(sizes),
            "avg_text_bytes": sum(sizes) / max(1, len(sizes)),
            "dirty": self._dirty,
            "path": self._path or "(memory only)",
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._path and _os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._store = _json.load(f)
            except (_json.JSONDecodeError, OSError):
                self._store = {}
        self._loaded = True


def offload_anchor_to_cold(anchor, cold_store: TieredStorage) -> dict:
    """Serialize anchor data and move to cold store."""
    data = {
        "text": getattr(anchor, "text", ""),
        "embedding": getattr(anchor, "embedding", None),
        "tags": list(getattr(anchor, "tags", [])),
        "importance": getattr(getattr(anchor, "vector", None), "importance", 0.5),
        "emotional_valence": getattr(getattr(anchor, "vector", None), "emotional_valence", 0.0),
        "source_session": getattr(anchor, "source_session", ""),
        "created_at": getattr(anchor, "created_at", time.time()),
        "last_activated_at": getattr(anchor, "last_activated_at", time.time()),
        "community_id": getattr(anchor, "community_id", ""),
    }
    cold_store.offload(anchor.id, data)
    return data
