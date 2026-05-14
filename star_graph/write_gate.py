"""Memory Write Gate — pre-write filtering to prevent memory pollution.

Before any memory enters the graph, the write gate checks:
  1. Importance: is this worth remembering?
  2. Duplicate: do we already have a very similar memory?
  3. Short-term noise: is this a transient chat message with no lasting value?
  4. Emotional noise: is this just emotional venting without substance?
  5. Similar node: can we merge into an existing anchor instead?

Returns: ACCEPT (write new), REJECT (skip), MERGE (update existing), DEFER (short buffer)

This is the single most important quality filter in the system.
Without it, the graph fills with noise and retrieval degrades to uselessness.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .config import Config
from .math_utils import cosine_sim as _cosine_sim


class GateDecision(Enum):
    ACCEPT = "accept"     # Write as new anchor
    REJECT = "reject"     # Skip entirely (noise / too low value)
    MERGE = "merge"       # Merge into existing similar anchor
    DEFER = "defer"       # Put in short-term buffer for later decision


@dataclass
class GateResult:
    """Result of write gate evaluation."""

    decision: GateDecision
    reason: str = ""
    score: float = 0.0
    merge_target_id: str = ""       # if MERGE: which anchor to merge into
    confidence: float = 0.0


class MemoryWriteGate:
    """Pre-write filter that decides whether to store, merge, or discard.

    Usage:
        gate = MemoryWriteGate()
        result = gate.evaluate(text, embedding, tags, importance, emotional_valence)
        if result.decision == GateDecision.ACCEPT:
            graph.add_anchor(anchor)
        elif result.decision == GateDecision.MERGE:
            update_existing(result.merge_target_id, text)
        elif result.decision == GateDecision.REJECT:
            pass  # skip
    """

    def __init__(self):
        c = Config.get()
        gate_cfg = getattr(c, 'write_gate', None) or {}

        # Thresholds
        self.min_importance = getattr(gate_cfg, 'min_importance', 0.15)
        self.min_text_length = getattr(gate_cfg, 'min_text_length', 8)
        self.max_text_length = getattr(gate_cfg, 'max_text_length', 4000)
        self.duplicate_threshold = getattr(gate_cfg, 'duplicate_threshold', 0.92)
        self.merge_threshold = getattr(gate_cfg, 'merge_threshold', 0.75)
        self.noise_pattern_score = getattr(gate_cfg, 'noise_pattern_score', -0.3)
        self.emotional_noise_threshold = getattr(gate_cfg, 'emotional_noise_threshold', 0.85)

        # Cache for recent evaluations to debounce rapid writes
        self._recent_texts: list[tuple[str, float]] = []  # (text_hash, timestamp)
        self._debounce_window = getattr(gate_cfg, 'debounce_window_seconds', 10.0)

    def evaluate(self, text: str,
                 embedding: list[float] | None = None,
                 tags: list[str] | None = None,
                 importance: float = 0.5,
                 emotional_valence: float = 0.0,
                 graph=None) -> GateResult:
        """Evaluate whether a memory should be written, merged, or rejected.

        Args:
            text: The memory text
            embedding: Optional embedding vector
            tags: Optional tags
            importance: User/source-assigned importance
            emotional_valence: Emotional tone (-1 to +1)
            graph: Optional StarGraph for duplicate/similarity checks

        Returns:
            GateResult with decision and reasoning
        """
        # ── Stage 0: Input sanity ──
        if not text or not text.strip():
            return GateResult(GateDecision.REJECT, "empty_text", 0.0)

        text = text.strip()
        if len(text) < self.min_text_length:
            return GateResult(GateDecision.REJECT,
                            f"too_short ({len(text)} < {self.min_text_length})", 0.0)

        # ── Stage 1: Noise pattern check ──
        noise_score = self._check_noise_patterns(text)
        if noise_score > 0.7:
            return GateResult(GateDecision.REJECT,
                            f"noise_pattern (score={noise_score:.2f})", noise_score)

        # ── Stage 2: Emotional noise check ──
        emotional_score = self._check_emotional_noise(text, emotional_valence)
        if emotional_score > self.emotional_noise_threshold:
            return GateResult(GateDecision.DEFER,
                            f"emotional_noise (score={emotional_score:.2f})",
                            emotional_score)

        # ── Stage 3: Importance threshold ──
        importance_score = self._check_importance(text, importance, tags)
        if importance_score < self.min_importance:
            if importance_score < 0.05:
                return GateResult(GateDecision.REJECT,
                                f"too_low_importance ({importance_score:.2f})",
                                importance_score)
            else:
                return GateResult(GateDecision.DEFER,
                                f"low_importance_defer ({importance_score:.2f})",
                                importance_score)

        # ── Stage 4: Duplicate & similarity check (requires graph) ──
        if graph and embedding:
            dup_result = self._check_duplicate(text, embedding, graph)
            if dup_result is not None:
                return dup_result

        # ── Stage 5: Debounce check ──
        text_hash = self._hash_text(text)
        now = time.time()
        # Clean old entries
        self._recent_texts = [
            (h, t) for h, t in self._recent_texts
            if now - t < self._debounce_window
        ]
        # Check for recent duplicates
        for h, _ in self._recent_texts:
            if h == text_hash:
                return GateResult(GateDecision.REJECT,
                                "debounce_duplicate", 0.5)
        self._recent_texts.append((text_hash, now))

        # ── All checks passed ──
        final_score = (importance_score * 0.5 +
                      (1.0 - noise_score) * 0.3 +
                      (1.0 - emotional_score * 0.5) * 0.2)

        return GateResult(
            decision=GateDecision.ACCEPT,
            reason=f"passed_all_checks",
            score=final_score,
            confidence=min(1.0, final_score),
        )

    # ── Stage implementations ───────────────────────────────

    _NOISE_PATTERNS = [
        # Pure reactions / acknowledgment
        (r'^(ok|okay|好的|嗯|哦|知道了|明白了|懂了|收到)[\s\.。!！]*$', 0.95),
        (r'^(thanks|谢谢|thank you|thx|3q)[\s\.。!！]*$', 0.9),
        (r'^(ha{2,}|呵{2,}|笑|w{2,}|草{2,})[\s\.。!！]*$', 0.95),
        # Very short generic responses
        (r'^(yes|no|对|不对|是|不是|可以|不可以)$', 0.8),
        (r'^[a-z]{1,3}$', 0.85),
        # Pure emoji/emoticon messages
        (r'^[😀-🙏👆-👇👈-👉🖕👍👎💪🙏🤲🐀-🐿️🦀-🦿🩰-🪿\U0001F300-\U0001F9FF☀-➿]+$', 0.98),
        # Greetings only
        (r'^(hi|hello|hey|嗨|你好|早|早上好|晚上好)[\s\.。!！]*$', 0.7),
        # Bot commands
        (r'^[/!][a-z]+(\s|$)', 0.6),
    ]

    def _check_noise_patterns(self, text: str) -> float:
        """Check text against known noise patterns. Returns 0-1 noise score."""
        text_clean = text.strip().lower()
        for pattern, score in self._NOISE_PATTERNS:
            if re.match(pattern, text_clean):
                return score

        # Check information density (ratio of meaningful tokens to total)
        tokens = re.findall(r'[一-鿿]+|[a-z]{3,}', text_clean)
        if len(tokens) == 0:
            return 0.9  # no meaningful tokens

        # Ratio of stop words / filler
        filler_words = {'um', 'uh', 'er', '嗯', '啊', '哦', '呃', '那个', '这个', '就是',
                       'maybe', 'perhaps', '大概', '可能', '好像'}
        filler_count = sum(1 for t in tokens if t in filler_words)
        total = len(tokens)
        if total > 0 and filler_count / total > 0.5:
            return 0.6

        return max(0.0, min(1.0, 1.0 - len(tokens) / max(5, len(text_clean.split()))))

    def _check_emotional_noise(self, text: str, emotional_valence: float) -> float:
        """Check if this is emotional venting without substance."""
        abs_emotion = abs(emotional_valence)

        # High emotion + short text → likely noise
        if abs_emotion > 0.8 and len(text) < 50:
            return 0.9

        # High emotion + few unique words → probably just emoting
        if abs_emotion > 0.6:
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(1, len(words))
            if unique_ratio < 0.3 and len(words) < 30:
                return 0.8

        return abs_emotion * 0.5  # moderate emotion is fine

    def _check_importance(self, text: str, importance: float,
                          tags: list[str] | None) -> float:
        """Assess importance score based on content signals."""
        score = importance  # base from caller

        # Length bonus: very short messages are less likely important
        text_len = len(text)
        if text_len < 20:
            score *= 0.5
        elif text_len > 100:
            score *= 1.2  # longer = more likely substantive

        # Tag bonus
        if tags:
            high_value_tags = {'preference', 'workflow', 'solution', 'error', 'fix',
                              'knowledge', 'fact', 'opinion', 'user', 'project',
                              '偏好', '工作流', '解决方案', '错误', '知识'}
            tag_match = sum(1 for t in tags if t.lower() in high_value_tags)
            score *= 1.0 + tag_match * 0.15

        # Content signals
        text_lower = text.lower()
        substantive_signals = {
            '因为', '所以', '原因', '方案', '步骤', '配置', '部署', '测试',
            'because', 'therefore', 'solution', 'step', 'config', 'deploy', 'test',
            'error', 'bug', 'fix', '解决', '修复',
            'prefer', 'like', 'want', 'need', '喜欢', '想要', '需要', '偏好',
        }
        signal_count = sum(1 for s in substantive_signals if s in text_lower)
        score *= 1.0 + min(0.5, signal_count * 0.1)

        return min(1.0, max(0.0, score))

    def _check_duplicate(self, text: str, embedding: list[float],
                         graph) -> GateResult | None:
        """Check if this memory duplicates or is similar to an existing one.

        Uses ANN for fast pre-filter, then cosine similarity on candidates.
        Returns None if no duplicate found, GateResult otherwise.
        """
        if not hasattr(graph, '_ann_index') or graph._ann_index is None:
            if len(graph.anchors) < 50:
                # Brute force for small graphs
                best_sim = 0.0
                best_id = ""
                for aid, anchor in graph.anchors.items():
                    if not anchor.embedding or not anchor.is_retrievable:
                        continue
                    sim = _cosine_sim(embedding, anchor.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = aid

                if best_sim >= self.duplicate_threshold:
                    return GateResult(
                        GateDecision.REJECT,
                        f"duplicate (sim={best_sim:.3f})",
                        best_sim,
                        merge_target_id=best_id,
                    )
                if best_sim >= self.merge_threshold:
                    return GateResult(
                        GateDecision.MERGE,
                        f"similar_exists (sim={best_sim:.3f})",
                        best_sim,
                        merge_target_id=best_id,
                    )
                return None

        # ANN-accelerated for larger graphs
        try:
            results = graph._ann_index.query(embedding, k=5)
            for candidate_id, sim in results:
                if sim >= self.duplicate_threshold:
                    return GateResult(
                        GateDecision.REJECT,
                        f"duplicate_ann (sim={sim:.3f})",
                        sim,
                        merge_target_id=candidate_id,
                    )
                if sim >= self.merge_threshold:
                    return GateResult(
                        GateDecision.MERGE,
                        f"similar_exists_ann (sim={sim:.3f})",
                        sim,
                        merge_target_id=candidate_id,
                    )
        except Exception:
            pass

        return None

    @staticmethod
    def _hash_text(text: str) -> str:
        """Fast content hash for debounce."""
        import hashlib
        return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]

    @property
    def stats(self) -> dict:
        return {
            "min_importance": self.min_importance,
            "duplicate_threshold": self.duplicate_threshold,
            "merge_threshold": self.merge_threshold,
            "debounce_window": self._debounce_window,
        }
