"""Memory Lifecycle Engine — autonomous memory layer migration.

Implements cognitive memory migration between layers:
  L0 (Input)    → ephemeral, per-session, not persisted
  L1 (Working)  → recent active memories, LLM-maintained, fast read/write
  L2 (Long-term) → stable consolidated memories, strength-based retrieval
  L3 (Archive)  → compressed summaries, not in real-time retrieval

Migration rules:
  L1→L2: access_count > threshold OR age > 30 days OR user re-mentions
  L2→L3: 90 days no access OR importance < threshold → compress → archive
  L3→L2: query embedding matches archived summary → reactivate → restore
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum


class MemoryLayer(Enum):
    L0_INPUT = "input"
    L1_WORKING = "working"
    L2_LONG_TERM = "long_term"
    L3_ARCHIVE = "archive"


@dataclass
class MemoryMigration:
    """A single migration event between layers."""
    anchor_id: str
    from_layer: MemoryLayer
    to_layer: MemoryLayer
    reason: str
    timestamp: float
    compressed_text: str = ""


# Layer transition thresholds
L1_TO_L2_ACCESS_THRESHOLD = 3      # promoted after 3 accesses
L1_TO_L2_AGE_DAYS = 30              # promoted after 30 days
L2_TO_L3_AGE_DAYS = 90              # archived after 90 days
L2_TO_L3_IMPORTANCE_THRESHOLD = 0.3  # archived if importance below this
L2_REACTIVATE_SIMILARITY = 0.55     # reactivate from archive if cosine > this


class MemoryLifecycleEngine:
    """Manages autonomous memory migration across L0-L3 layers.
    
    Called during sleep consolidation. Does NOT use LLM for routing —
    all decisions are based on access patterns and temporal decay.
    LLM is only used for: compressing L2→L3 summaries, expanding L3→L2.
    """

    def __init__(self, graph, llm_fn=None):
        self.graph = graph
        self.llm_fn = llm_fn  # optional LLM for compression/expansion
        self.migrations: list[MemoryMigration] = []
        # Archive store: anchor_id → compressed_text
        self._archive: dict[str, str] = getattr(graph, '_archive', {})
        if not hasattr(graph, '_archive'):
            graph._archive = self._archive

    # ── L1 → L2 promotion ──────────────────────────────────

    def promote_l1_to_l2(self, now: float | None = None) -> list[MemoryMigration]:
        """Promote eligible L1 memories to L2 (long-term).
        
        Eligibility:
          - replay_count >= L1_TO_L2_ACCESS_THRESHOLD
          - OR age > L1_TO_L2_AGE_DAYS
        """
        if now is None:
            now = time.time()
        promoted: list[MemoryMigration] = []
        
        for aid, anchor in list(self.graph.anchors.items()):
            # Skip if already L2
            if getattr(anchor, '_layer', None) == MemoryLayer.L2_LONG_TERM:
                continue
            
            age_days = (now - anchor.created_at) / 86400.0
            replay = getattr(anchor, 'replay_count', 0)
            
            if replay >= L1_TO_L2_ACCESS_THRESHOLD or age_days >= L1_TO_L2_AGE_DAYS:
                anchor._layer = MemoryLayer.L2_LONG_TERM
                reason = f"replay_count={replay}" if replay >= L1_TO_L2_ACCESS_THRESHOLD else f"age={age_days:.0f}d"
                self.migrations.append(MemoryMigration(
                    anchor_id=aid, from_layer=MemoryLayer.L1_WORKING,
                    to_layer=MemoryLayer.L2_LONG_TERM, reason=reason, timestamp=now
                ))
                promoted.append(self.migrations[-1])
        
        return promoted

    # ── L2 → L3 archival ───────────────────────────────────

    def archive_l2_to_l3(self, now: float | None = None) -> list[MemoryMigration]:
        """Move stale/lowl-importance L2 memories to L3 archive.
        
        Eligibility:
          - 90 days since last_activated_at
          - OR vector.importance < L2_TO_L3_IMPORTANCE_THRESHOLD
        """
        if now is None:
            now = time.time()
        archived: list[MemoryMigration] = []
        
        for aid, anchor in list(self.graph.anchors.items()):
            if getattr(anchor, '_layer', None) not in (None, MemoryLayer.L2_LONG_TERM):
                continue
            
            age_days = (now - anchor.last_activated_at) / 86400.0
            importance = anchor.vector.importance if hasattr(anchor, 'vector') else 0.5
            
            if age_days < L2_TO_L3_AGE_DAYS and importance >= L2_TO_L3_IMPORTANCE_THRESHOLD:
                continue  # not eligible
            
            # Compress using LLM if available, else simple truncation
            if self.llm_fn and len(anchor.text) > 100:
                compressed = self._llm_compress(anchor.text)
            else:
                compressed = anchor.text[:120] + "..." if len(anchor.text) > 120 else anchor.text
            
            self._archive[aid] = compressed
            anchor._layer = MemoryLayer.L3_ARCHIVE
            
            reason = f"age={age_days:.0f}d" if age_days >= L2_TO_L3_AGE_DAYS else f"importance={importance:.2f}"
            self.migrations.append(MemoryMigration(
                anchor_id=aid, from_layer=MemoryLayer.L2_LONG_TERM,
                to_layer=MemoryLayer.L3_ARCHIVE, reason=reason,
                timestamp=now, compressed_text=compressed
            ))
            archived.append(self.migrations[-1])
        
        return archived

    # ── L3 → L2 reactivation ───────────────────────────────

    def reactivate_l3_to_l2(self, query_embedding: list[float],
                            top_k: int = 3) -> list[MemoryMigration]:
        """Check if any archived memory matches current query.
        
        If similarity between query and archived summary > threshold,
        restore the memory to L2.
        """
        from star_graph.math_utils import cosine_sim as _cos
        
        reactivated: list[MemoryMigration] = []
        scored: list[tuple[float, str]] = []
        
        for aid, compressed in self._archive.items():
            anchor = self.graph.anchors.get(aid)
            if anchor is None or not anchor.embedding:
                continue
            sim = _cos(query_embedding, anchor.embedding)
            if sim >= L2_REACTIVATE_SIMILARITY:
                scored.append((sim, aid))
        
        scored.sort(key=lambda x: -x[0])
        for sim, aid in scored[:top_k]:
            anchor = self.graph.anchors.get(aid)
            if anchor and hasattr(anchor, '_layer'):
                anchor._layer = MemoryLayer.L2_LONG_TERM
                anchor.last_activated_at = time.time()
                self.migrations.append(MemoryMigration(
                    anchor_id=aid, from_layer=MemoryLayer.L3_ARCHIVE,
                    to_layer=MemoryLayer.L2_LONG_TERM,
                    reason=f"reactivated(sim={sim:.2f})",
                    timestamp=time.time()
                ))
                reactivated.append(self.migrations[-1])
        
        return reactivated

    # ── LLM compression helper ─────────────────────────────

    def _llm_compress(self, text: str) -> str:
        """Use optional LLM to compress memory text.
        
        Falls back to truncation if no LLM available.
        """
        if not self.llm_fn:
            return text[:120] + "..." if len(text) > 120 else text
        try:
            return self.llm_fn(text)
        except Exception:
            return text[:120] + "..." if len(text) > 120 else text

    # ── Stats ──────────────────────────────────────────────

    def get_layer_counts(self) -> dict[str, int]:
        """Count anchors in each layer."""
        counts = {"L0_input": 0, "L1_working": 0, "L2_long_term": 0, "L3_archive": 0, "unlabeled": 0}
        for anchor in self.graph.anchors.values():
            layer = getattr(anchor, '_layer', None)
            if layer is None:
                counts["unlabeled"] += 1
            elif layer == MemoryLayer.L1_WORKING:
                counts["L1_working"] += 1
            elif layer == MemoryLayer.L2_LONG_TERM:
                counts["L2_long_term"] += 1
            elif layer == MemoryLayer.L3_ARCHIVE:
                counts["L3_archive"] += 1
            else:
                counts["L0_input"] += 1
        return counts
