"""Four-Layer Memory Compression Architecture.

Fundamental architecture change: memory is not stored at message-level.
Instead, there are four abstraction layers:

  Layer 0: MESSAGE  — raw input, TTL 2h, no graph, deque buffer (STM)
  Layer 1: EVENT    — extracted events from messages, TTL 7d, light graph
  Layer 2: SEMANTIC — knowledge distilled from events, TTL 90d, medium graph
  Layer 3: PERSONALITY — user traits/preferences, TTL ∞, core graph

Promotion path:
  message → event → semantic → personality

Each layer has:
  - TTL (time to live before auto-decay)
  - Max items (hard cap, triggers compression)
  - Compression ratio (how many items compress to one at next layer)
  - Stability threshold (how stable before promotion)

This prevents the "message-level memory" problem where every chat line
becomes an anchor, flooding the graph with noise.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .config import Config
from .math_utils import cosine_sim as _cosine_sim


class CompressLayer(IntEnum):
    MESSAGE = 0
    EVENT = 1
    SEMANTIC = 2
    PERSONALITY = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class LayerConfig:
    """Configuration for one compression layer."""
    max_items: int = 500
    ttl_hours: float = 2.0
    compression_ratio: int = 5       # N items → 1 item at next layer
    promote_stability: float = 0.3   # stability threshold for promotion
    auto_promote: bool = True
    decay_half_life_hours: float = 48.0


@dataclass
class CompressedMemory:
    """A memory entry at any compression layer."""
    id: str
    text: str
    layer: CompressLayer
    embedding: list[float] | None = None
    source_ids: list[str] = field(default_factory=list)
    stability: float = 0.5
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    promoted: bool = False

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    @property
    def idle_hours(self) -> float:
        return (time.time() - self.last_accessed_at) / 3600

    def access(self):
        self.access_count += 1
        self.last_accessed_at = time.time()


class FourLayerCompressor:
    """Manages the four-layer memory compression architecture.

    Layer 0 (MESSAGE):    deque of raw inputs, auto-compress to events
    Layer 1 (EVENT):      extracted events, auto-promote to semantic
    Layer 2 (SEMANTIC):   distilled knowledge, auto-promote to personality
    Layer 3 (PERSONALITY): stable user traits, near-immutable

    Usage:
        flc = FourLayerCompressor()
        flc.ingest_message(text, embedding, importance)   # → Layer 0
        flc.compress_layer0()  # called periodically → Layer 1
        flc.compress_layer1()  # called during sleep → Layer 2
        flc.compress_layer2()  # called rarely → Layer 3
    """

    def __init__(self):
        c = Config.get()
        comp_cfg = getattr(c, 'four_layer', None) or {}

        self.layers: dict[CompressLayer, dict[str, CompressedMemory]] = {
            CompressLayer.MESSAGE: {},
            CompressLayer.EVENT: {},
            CompressLayer.SEMANTIC: {},
            CompressLayer.PERSONALITY: {},
        }

        self.layer_configs: dict[CompressLayer, LayerConfig] = {
            CompressLayer.MESSAGE: LayerConfig(
                max_items=getattr(comp_cfg, 'message_max', 200),
                ttl_hours=getattr(comp_cfg, 'message_ttl_hours', 2.0),
                compression_ratio=getattr(comp_cfg, 'message_ratio', 8),
                promote_stability=getattr(comp_cfg, 'message_promote_stability', 0.2),
            ),
            CompressLayer.EVENT: LayerConfig(
                max_items=getattr(comp_cfg, 'event_max', 500),
                ttl_hours=getattr(comp_cfg, 'event_ttl_hours', 168.0),  # 7 days
                compression_ratio=getattr(comp_cfg, 'event_ratio', 5),
                promote_stability=getattr(comp_cfg, 'event_promote_stability', 0.4),
            ),
            CompressLayer.SEMANTIC: LayerConfig(
                max_items=getattr(comp_cfg, 'semantic_max', 200),
                ttl_hours=getattr(comp_cfg, 'semantic_ttl_hours', 2160.0),  # 90 days
                compression_ratio=getattr(comp_cfg, 'semantic_ratio', 4),
                promote_stability=getattr(comp_cfg, 'semantic_promote_stability', 0.6),
            ),
            CompressLayer.PERSONALITY: LayerConfig(
                max_items=getattr(comp_cfg, 'personality_max', 50),
                ttl_hours=getattr(comp_cfg, 'personality_ttl_hours', float('inf')),
                compression_ratio=0,  # top layer, no further compression
                promote_stability=0.8,
                auto_promote=False,
            ),
        }

        self._counter = 0
        self._total_compressed = 0

    # ── Ingestion ───────────────────────────────────────────

    def ingest_message(self, text: str,
                       embedding: list[float] | None = None,
                       importance: float = 0.5,
                       tags: list[str] | None = None) -> CompressedMemory | None:
        """Ingest a raw message into Layer 0 (MESSAGE).

        Returns the entry if accepted, None if rejected (TTL expired / full).
        """
        layer = CompressLayer.MESSAGE
        cfg = self.layer_configs[layer]

        # Check capacity — if full, compress first
        if len(self.layers[layer]) >= cfg.max_items:
            self.compress_layer0()

        # If still full after compression, evict oldest
        if len(self.layers[layer]) >= cfg.max_items:
            self._evict_oldest(layer)

        self._counter += 1
        entry = CompressedMemory(
            id=f"msg_{self._counter}",
            text=text[:1000],
            layer=layer,
            embedding=embedding,
            importance=importance,
            tags=tags or [],
        )

        self.layers[layer][entry.id] = entry
        return entry

    # ── Layer compression ───────────────────────────────────

    def compress_layer0(self) -> int:
        """Compress Layer 0 (MESSAGE) → Layer 1 (EVENT).

        Groups related messages into events by semantic similarity.
        Returns number of events created.
        """
        src_layer = CompressLayer.MESSAGE
        dst_layer = CompressLayer.EVENT
        cfg = self.layer_configs[src_layer]
        ratio = cfg.compression_ratio

        entries = list(self.layers[src_layer].values())
        if len(entries) < ratio:
            return 0

        # Sort by time (oldest first)
        entries.sort(key=lambda e: e.created_at)

        # Group into batches of `ratio` messages
        events_created = 0
        for i in range(0, len(entries) - ratio + 1, ratio):
            batch = entries[i:i + ratio]

            # Skip if too spread out in time (> 1 hour apart)
            time_span = batch[-1].created_at - batch[0].created_at
            if time_span > 3600:
                continue

            event = self._synthesize_event(batch)
            self.layers[dst_layer][event.id] = event

            # Mark source messages as processed
            for e in batch:
                e.promoted = True
                # Don't delete yet — keep for dedup reference

            events_created += 1

        # Clean up promoted messages that are old enough
        self._clean_promoted(src_layer, min_age_hours=1.0)

        return events_created

    def compress_layer1(self) -> int:
        """Compress Layer 1 (EVENT) → Layer 2 (SEMANTIC).

        Distills recurring event patterns into semantic knowledge.
        Returns number of semantic entries created.
        """
        src_layer = CompressLayer.EVENT
        dst_layer = CompressLayer.SEMANTIC
        cfg = self.layer_configs[src_layer]
        ratio = cfg.compression_ratio

        entries = [e for e in self.layers[src_layer].values()
                  if not e.promoted and e.stability >= cfg.promote_stability]

        if len(entries) < ratio:
            return 0

        # Cluster by tag similarity and embedding proximity
        clusters = self._cluster_entries(entries, ratio)

        created = 0
        for cluster in clusters:
            semantic = self._synthesize_semantic(cluster)
            self.layers[dst_layer][semantic.id] = semantic

            for e in cluster:
                e.promoted = True

            created += 1

        self._clean_promoted(src_layer, min_age_hours=24.0)
        return created

    def compress_layer2(self) -> int:
        """Compress Layer 2 (SEMANTIC) → Layer 3 (PERSONALITY).

        Extracts stable personality traits from semantic knowledge.
        Returns number of personality entries created.
        """
        src_layer = CompressLayer.SEMANTIC
        dst_layer = CompressLayer.PERSONALITY
        cfg = self.layer_configs[src_layer]
        ratio = cfg.compression_ratio

        entries = [e for e in self.layers[src_layer].values()
                  if not e.promoted and e.stability >= cfg.promote_stability]

        if len(entries) < ratio:
            return 0

        clusters = self._cluster_entries(entries, ratio, threshold=0.4)

        created = 0
        for cluster in clusters:
            personality = self._synthesize_personality(cluster)
            self.layers[dst_layer][personality.id] = personality

            for e in cluster:
                e.promoted = True

            created += 1

        self._clean_promoted(src_layer, min_age_hours=720.0)  # 30 days
        return created

    # ── Synthesis helpers ───────────────────────────────────

    def _synthesize_event(self, messages: list[CompressedMemory]) -> CompressedMemory:
        """Synthesize an event from a batch of related messages.

        Extracts: what happened, key entities, actions, outcomes.
        """
        texts = [m.text for m in messages]
        combined = " ".join(texts)

        # Extract key action verbs and entities
        key_terms = self._extract_key_terms(texts, top_k=5)

        # Simple template-based synthesis
        if key_terms:
            event_text = f"[Event] Interaction about {key_terms[0]}: {', '.join(key_terms[1:4])}"
        else:
            event_text = f"[Event] {texts[0][:150]}"

        # Average embedding
        embs = [m.embedding for m in messages if m.embedding]
        avg_emb = None
        if embs:
            avg_emb = [sum(vals) / len(vals) for vals in zip(*embs)]

        # Collect all tags
        all_tags = list(set(t for m in messages for t in m.tags))

        self._counter += 1
        return CompressedMemory(
            id=f"event_{self._counter}",
            text=event_text[:500],
            layer=CompressLayer.EVENT,
            embedding=avg_emb,
            source_ids=[m.id for m in messages],
            stability=0.3,
            importance=sum(m.importance for m in messages) / len(messages),
            tags=all_tags,
        )

    def _synthesize_semantic(self, events: list[CompressedMemory]) -> CompressedMemory:
        """Distill recurring events into semantic knowledge.

        Looks for patterns: repeated topics, consistent user behaviors,
        frequently used tools/approaches.
        """
        texts = [e.text for e in events]
        all_tags = list(set(t for e in events for t in e.tags))
        key_terms = self._extract_key_terms(texts, top_k=4)

        # Detect recurring patterns
        pattern = self._detect_pattern(texts)

        if pattern:
            semantic_text = f"[Knowledge] {pattern}"
        elif key_terms:
            semantic_text = f"[Knowledge] User works with {', '.join(key_terms[:3])}"
        else:
            semantic_text = f"[Knowledge] Pattern from {len(events)} events"

        embs = [e.embedding for e in events if e.embedding]
        avg_emb = None
        if embs:
            avg_emb = [sum(vals) / len(vals) for vals in zip(*embs)]

        self._counter += 1
        return CompressedMemory(
            id=f"semantic_{self._counter}",
            text=semantic_text[:400],
            layer=CompressLayer.SEMANTIC,
            embedding=avg_emb,
            source_ids=[e.id for e in events],
            stability=0.5,
            importance=0.7,
            tags=all_tags,
        )

    def _synthesize_personality(self,
                                 semantics: list[CompressedMemory]) -> CompressedMemory:
        """Extract personality trait from semantic knowledge.

        Personality = stable, long-term patterns about the user's
        preferences, expertise, working style, and values.
        """
        texts = [s.text for s in semantics]
        all_tags = list(set(t for s in semantics for t in s.tags))

        # Determine trait type
        trait_type = self._classify_trait(texts, all_tags)

        # Generate trait description
        if trait_type == "expertise":
            trait_text = f"[Trait:Expertise] User is experienced in {self._summarize_domain(texts)}"
        elif trait_type == "preference":
            trait_text = f"[Trait:Preference] User prefers {self._summarize_domain(texts)}"
        elif trait_type == "workflow":
            trait_text = f"[Trait:Workflow] User works with {self._summarize_domain(texts)}"
        else:
            trait_text = f"[Trait] {self._summarize_domain(texts)}"

        embs = [s.embedding for s in semantics if s.embedding]
        avg_emb = None
        if embs:
            avg_emb = [sum(vals) / len(vals) for vals in zip(*embs)]

        self._counter += 1
        return CompressedMemory(
            id=f"personality_{self._counter}",
            text=trait_text[:300],
            layer=CompressLayer.PERSONALITY,
            embedding=avg_emb,
            source_ids=[s.id for s in semantics],
            stability=0.7,
            importance=0.9,
            tags=all_tags + [f"trait:{trait_type}"],
        )

    # ── Clustering ──────────────────────────────────────────

    def _cluster_entries(self, entries: list[CompressedMemory],
                         min_size: int,
                         threshold: float = 0.5) -> list[list[CompressedMemory]]:
        """Cluster entries by embedding similarity (BFS connected components)."""
        n = len(entries)
        if n < min_size:
            return []

        # Build similarity adjacency
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if entries[i].embedding and entries[j].embedding:
                    sim = _cosine_sim(entries[i].embedding, entries[j].embedding)
                    if sim > threshold:
                        adj[i].add(j)
                        adj[j].add(i)

        visited: set[int] = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                node = queue.pop(0)
                component.append(entries[node])
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= min_size:
                clusters.append(component)

        return clusters

    # ── Text analysis helpers ────────────────────────────────

    def _extract_key_terms(self, texts: list[str], top_k: int = 5) -> list[str]:
        """TF-IDF style key term extraction."""
        import re
        from collections import Counter

        STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "have", "has", "had", "do", "does", "did", "will", "would",
                 "could", "should", "to", "of", "in", "for", "on", "with",
                 "at", "by", "from", "and", "or", "but", "it", "this", "that",
                 "not", "so", "as", "if", "then", "than"}

        N = len(texts)
        doc_freq: Counter = Counter()
        term_freq: Counter = Counter()

        for text in texts:
            tokens = re.findall(r'[a-z0-9]+', text.lower())
            tokens = [t for t in tokens if len(t) > 3 and t not in STOP]
            for t in set(tokens):
                doc_freq[t] += 1
            for t in tokens:
                term_freq[t] += 1

        scored = []
        for term, tf in term_freq.items():
            df = doc_freq.get(term, 1)
            idf = __import__('math').log((N + 1) / (df + 1)) + 1.0
            scored.append((term, tf * idf))

        scored.sort(key=lambda x: -x[1])
        return [term for term, _ in scored[:top_k]]

    @staticmethod
    def _detect_pattern(texts: list[str]) -> str:
        """Detect recurring patterns across texts."""
        import re
        from collections import Counter

        # Look for repeated 3-grams
        trigram_counter: Counter = Counter()
        for text in texts:
            words = re.findall(r'[a-z一-鿿]+', text.lower())
            for i in range(len(words) - 2):
                trigram = ' '.join(words[i:i + 3])
                if len(trigram) > 10:
                    trigram_counter[trigram] += 1

        recurring = [(t, c) for t, c in trigram_counter.most_common(3) if c >= 2]
        if recurring:
            return f"Recurring pattern: {recurring[0][0]}"
        return ""

    @staticmethod
    def _classify_trait(texts: list[str], tags: list[str]) -> str:
        """Classify what type of personality trait this represents."""
        combined = " ".join(texts + tags).lower()
        if any(w in combined for w in ("prefer", "like", "dislike", "favorite", "偏好")):
            return "preference"
        if any(w in combined for w in ("expert", "skill", "experience", "熟练", "经验")):
            return "expertise"
        if any(w in combined for w in ("workflow", "process", "approach", "method", "流程")):
            return "workflow"
        if any(w in combined for w in ("value", "important", "priority", "重要", "优先")):
            return "value"
        return "general"

    @staticmethod
    def _summarize_domain(texts: list[str]) -> str:
        """Summarize the domain/topic from texts."""
        import re
        from collections import Counter

        STOP = {"the", "a", "an", "is", "are", "was", "with", "for", "and", "that",
                "事件", "知识", "人格", "event", "knowledge", "personality", "trait"}

        word_counter: Counter = Counter()
        for text in texts:
            words = re.findall(r'[a-z一-鿿]{2,}', text.lower())
            for w in words:
                if w not in STOP:
                    word_counter[w] += 1

        top = [w for w, c in word_counter.most_common(5) if c >= 2]
        if top:
            return " / ".join(top[:3])
        return "multiple domains"

    # ── Maintenance ─────────────────────────────────────────

    def _evict_oldest(self, layer: CompressLayer):
        """Remove the oldest entry from a layer."""
        entries = list(self.layers[layer].values())
        if not entries:
            return
        entries.sort(key=lambda e: e.created_at)
        del self.layers[layer][entries[0].id]

    def _clean_promoted(self, layer: CompressLayer, min_age_hours: float):
        """Remove promoted entries older than min_age_hours."""
        now = time.time()
        to_remove = []
        for eid, entry in self.layers[layer].items():
            if entry.promoted and (now - entry.created_at) / 3600 >= min_age_hours:
                to_remove.append(eid)
        for eid in to_remove:
            del self.layers[layer][eid]

    def decay_all(self) -> dict:
        """Apply time decay across all layers. Returns stats."""
        removed = {layer: 0 for layer in CompressLayer}
        now = time.time()

        for layer in CompressLayer:
            cfg = self.layer_configs[layer]
            if cfg.ttl_hours == float('inf'):
                continue
            to_remove = []
            for eid, entry in self.layers[layer].items():
                if (now - entry.last_accessed_at) / 3600 >= cfg.ttl_hours:
                    to_remove.append(eid)
            for eid in to_remove:
                del self.layers[layer][eid]
            removed[layer] = len(to_remove)

        return {"removed": removed, "total": sum(removed.values())}

    def get_for_retrieval(self, query: str = "",
                          max_per_layer: int = 10) -> dict[str, list[CompressedMemory]]:
        """Get entries from all layers for retrieval.

        Returns {layer_name: [entries]} prioritized by layer depth
        (personality first, then semantic, then event, then message).
        """
        result: dict[str, list[CompressedMemory]] = {}

        # Search in order: PERSONALITY → SEMANTIC → EVENT → MESSAGE
        for layer in reversed(list(CompressLayer)):
            entries = list(self.layers[layer].values())
            if query:
                # Simple keyword match
                query_lower = query.lower()
                scored = []
                for e in entries:
                    score = sum(1 for w in query_lower.split()
                              if w in e.text.lower())
                    if score > 0:
                        scored.append((e, score))
                scored.sort(key=lambda x: -x[1])
                entries = [e for e, _ in scored[:max_per_layer]]
            else:
                entries.sort(key=lambda e: -(e.stability * e.importance))
                entries = entries[:max_per_layer]

            if entries:
                result[layer.label] = entries

        return result

    @property
    def stats(self) -> dict:
        return {
            "message_count": len(self.layers[CompressLayer.MESSAGE]),
            "event_count": len(self.layers[CompressLayer.EVENT]),
            "semantic_count": len(self.layers[CompressLayer.SEMANTIC]),
            "personality_count": len(self.layers[CompressLayer.PERSONALITY]),
            "total_compressed": self._total_compressed,
            "layer_configs": {
                layer.label: {
                    "max_items": cfg.max_items,
                    "ttl_hours": cfg.ttl_hours,
                    "compression_ratio": cfg.compression_ratio,
                }
                for layer, cfg in self.layer_configs.items()
            },
        }
