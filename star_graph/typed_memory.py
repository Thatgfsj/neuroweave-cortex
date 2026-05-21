"""Typed Memory — type-specific embedding, compression, and retrieval strategies."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor


class MemoryType(str, Enum):
    """Seven-domain memory classification for type-aware processing."""
    CODE = "code"
    TASK = "task"
    DIALOGUE = "dialogue"
    TOOL_CALL = "tool_call"
    KNOWLEDGE = "knowledge"
    EVENT = "event"
    PREFERENCE = "preference"


_DEFAULT_STRATEGIES: dict[MemoryType, dict] = {
    MemoryType.CODE: {
        "embedding_model": "code",
        "compression_strategy": "dedup",
        "retrieval_priority": 7,
        "index_method": "signature",
        "dedup_method": "signature",
        "max_retention_days": float("inf"),
        "mergeable": False,
    },
    MemoryType.TASK: {
        "embedding_model": "general",
        "compression_strategy": "state_machine",
        "retrieval_priority": 10,
        "index_method": "temporal",
        "dedup_method": "semantic",
        "max_retention_days": 90,
        "mergeable": True,
    },
    MemoryType.DIALOGUE: {
        "embedding_model": "general",
        "compression_strategy": "summarize",
        "retrieval_priority": 5,
        "index_method": "semantic",
        "dedup_method": "semantic",
        "max_retention_days": 365,
        "mergeable": True,
    },
    MemoryType.TOOL_CALL: {
        "embedding_model": "tool",
        "compression_strategy": "dedup",
        "retrieval_priority": 6,
        "index_method": "signature",
        "dedup_method": "signature",
        "max_retention_days": 30,
        "mergeable": True,
    },
    MemoryType.KNOWLEDGE: {
        "embedding_model": "knowledge",
        "compression_strategy": "summarize",
        "retrieval_priority": 8,
        "index_method": "hybrid",
        "dedup_method": "semantic",
        "max_retention_days": float("inf"),
        "mergeable": True,
    },
    MemoryType.EVENT: {
        "embedding_model": "general",
        "compression_strategy": "none",
        "retrieval_priority": 4,
        "index_method": "temporal",
        "dedup_method": "none",
        "max_retention_days": 365,
        "mergeable": False,
    },
    MemoryType.PREFERENCE: {
        "embedding_model": "general",
        "compression_strategy": "none",
        "retrieval_priority": 9,
        "index_method": "semantic",
        "dedup_method": "semantic",
        "max_retention_days": float("inf"),
        "mergeable": True,
    },
}

_ACTION_VERBS = frozenset({
    "do", "fix", "implement", "create", "build", "add",
    "remove", "update", "refactor", "deploy", "test", "run",
    "write", "design", "configure", "setup", "install",
    "migrate", "upgrade", "optimize", "develop", "release",
    "debug", "audit", "benchmark", "profile", "compile",
    "package", "publish", "dockerize", "containerize",
})

_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this",
    "have", "been", "was", "are", "has", "had", "will", "shall",
    "can", "may", "must", "need", "all", "some", "any", "not",
    "but", "or", "nor", "just", "only", "also", "very", "too",
    "then", "now", "here", "there", "when", "where", "which",
    "who", "whom", "what", "how", "why", "if", "else", "each",
    "every", "both", "few", "more", "most", "other", "such",
    "about", "into", "over", "after", "before", "between",
    "under", "again", "further", "once",
})

_CODE_TAGS = frozenset({"code", "function", "class", "script"})
_TASK_TAGS = frozenset({"task", "todo", "goal", "objective", "action"})
_TOOL_CALL_TAGS = frozenset({"tool_call", "api_call", "command"})
_PREFERENCE_TAGS = frozenset({"preference", "like", "dislike", "style", "prefer"})
_EVENT_TAGS = frozenset({"event", "meeting", "milestone", "happened", "occurred"})
_KNOWLEDGE_TAGS = frozenset({"knowledge", "fact", "concept", "learning", "info"})

_CODE_LINE_PATTERN = re.compile(r'\b(def |class |import |function |\{)')
_FUNC_CALL_PATTERN = re.compile(r'\b[A-Za-z_]\w*\s*\(')
_CODE_NAME_PATTERN = re.compile(r'def (\w+)|class (\w+)')
_WORD_PATTERN = re.compile(r'\b[a-zA-Z]{3,}\b')
_TOOL_NAME_PATTERN = re.compile(r'(\w+)\s*\(')
_TOOL_ACTION_PATTERN = re.compile(
    r'\b(call|invoke|execute|run|query|fetch|send|dispatch)\b', re.IGNORECASE
)

_PREFERENCE_PHRASES = (
    "prefer ", "i like ", "i don't like ", "favorite", "i love ",
    "i hate ", "i enjoy ", "i dislike ", "i'd rather ", "i would rather ",
)


@dataclass
class TypeStrategy:
    """Per-type configuration for embedding, compression, and retrieval."""
    memory_type: MemoryType
    embedding_model: str = "general"
    compression_strategy: str = "none"
    retrieval_priority: int = 5
    index_method: str = "semantic"
    dedup_method: str = "none"
    max_retention_days: float = float("inf")
    mergeable: bool = False

    @classmethod
    def default_for(cls, memory_type: MemoryType) -> "TypeStrategy":
        """Create a TypeStrategy with defaults for the given memory type."""
        defaults = _DEFAULT_STRATEGIES.get(memory_type, {})
        return cls(memory_type=memory_type, **defaults)


class TypeManager:
    """Manages typed-memory strategies, classification, and deduplication."""

    def __init__(self, graph: "StarGraph", embedder=None):
        self._graph = graph
        self._embedder = embedder
        self._strategies: dict[MemoryType, TypeStrategy] = {
            mt: TypeStrategy.default_for(mt) for mt in MemoryType
        }
        self._anchor_types: dict[str, MemoryType] = {}

    # ── Classification ────────────────────────────────────

    def classify(self, text: str, tags: list[str] | None = None,
                 existing_type: str = "") -> MemoryType:
        """Infer memory type from text patterns and tags."""
        tags_lower = [t.lower() for t in (tags or [])]
        text_lower = text.lower().strip()

        if existing_type:
            try:
                return MemoryType(existing_type)
            except ValueError:
                pass

        tag_set = frozenset(tags_lower)

        if tag_set & _CODE_TAGS or _CODE_LINE_PATTERN.search(text):
            return MemoryType.CODE

        if tag_set & _TASK_TAGS:
            return MemoryType.TASK

        first_word = text_lower.split()[0] if text_lower else ""
        if first_word in _ACTION_VERBS:
            return MemoryType.TASK

        if tag_set & _TOOL_CALL_TAGS:
            return MemoryType.TOOL_CALL
        func_match = _FUNC_CALL_PATTERN.search(text)
        if func_match and not _CODE_LINE_PATTERN.search(text):
            if any(kw in text_lower for kw in ("call ", "invoke", "execute", "run ")):
                return MemoryType.TOOL_CALL

        if tag_set & _PREFERENCE_TAGS:
            return MemoryType.PREFERENCE
        if any(phrase in text_lower for phrase in _PREFERENCE_PHRASES):
            return MemoryType.PREFERENCE

        if tag_set & _EVENT_TAGS:
            return MemoryType.EVENT

        if tag_set & _KNOWLEDGE_TAGS:
            return MemoryType.KNOWLEDGE

        words = set(text_lower.split())
        if not words & _ACTION_VERBS:
            return MemoryType.KNOWLEDGE

        return MemoryType.DIALOGUE

    # ── Strategy access ───────────────────────────────────

    def get_strategy(self, memory_type: MemoryType) -> TypeStrategy:
        """Get the strategy for a memory type."""
        return self._strategies.get(
            memory_type, TypeStrategy(memory_type=memory_type)
        )

    # ── Anchor type management ────────────────────────────

    def set_type(self, anchor_id: str, memory_type: MemoryType) -> None:
        """Explicitly set the type on an anchor."""
        self._anchor_types[anchor_id] = memory_type

    def get_type(self, anchor_id: str) -> MemoryType:
        """Get the anchor's memory type (defaults to DIALOGUE)."""
        return self._anchor_types.get(anchor_id, MemoryType.DIALOGUE)

    def filter_by_type(self, anchor_ids: list[str],
                       memory_type: MemoryType) -> list[str]:
        """Filter anchor IDs by memory type."""
        return [aid for aid in anchor_ids
                if self._anchor_types.get(aid) == memory_type]

    # ── Retrieval ordering ────────────────────────────────

    def get_retrieval_order(self,
                            query_types: list[MemoryType] | None = None
                            ) -> list[MemoryType]:
        """Return memory types ordered by retrieval_priority (highest first)."""
        types = query_types if query_types is not None else list(MemoryType)
        return sorted(
            types,
            key=lambda mt: self._strategies.get(
                mt, TypeStrategy(mt)
            ).retrieval_priority,
            reverse=True,
        )

    # ── Type signatures ───────────────────────────────────

    def compute_type_signature(self, text: str,
                               memory_type: MemoryType) -> str | None:
        """Compute a type-specific signature for deduplication."""
        if memory_type == MemoryType.CODE:
            names = _CODE_NAME_PATTERN.findall(text)
            flat = sorted({n[0] or n[1] for n in names})
            return "code:" + ":".join(flat) if flat else None

        if memory_type == MemoryType.TASK:
            words = _WORD_PATTERN.findall(text.lower())
            verbs = [w for w in words if w in _ACTION_VERBS]
            nouns = [w for w in words
                     if w not in _ACTION_VERBS and w not in _STOP_WORDS]
            verb = verbs[0] if verbs else "action"
            noun = nouns[0] if nouns else "item"
            return f"{verb}:{noun}:goal"

        if memory_type == MemoryType.TOOL_CALL:
            tool_match = _TOOL_NAME_PATTERN.search(text)
            tool_name = tool_match.group(1) if tool_match else "unknown"
            action_match = _TOOL_ACTION_PATTERN.search(text)
            action = action_match.group(1) if action_match else "call"
            params_hash = hashlib.blake2b(
                text.encode(), digest_size=4
            ).hexdigest()
            return f"{tool_name}:{action}:{params_hash}"

        return None

    # ── Deduplication ─────────────────────────────────────

    def dedup_candidates(self, text: str,
                         memory_type: MemoryType) -> list[str]:
        """Find potential duplicate anchor IDs of the same type."""
        strategy = self.get_strategy(memory_type)
        candidates: list[str] = []

        same_type_anchors = [
            aid for aid, atype in self._anchor_types.items()
            if atype == memory_type and aid in self._graph.anchors
        ]

        if strategy.dedup_method == "none":
            return candidates

        if strategy.dedup_method == "exact":
            for aid in same_type_anchors:
                anchor = self._graph.anchors[aid]
                if anchor.text.strip() == text.strip():
                    candidates.append(aid)

        elif strategy.dedup_method == "signature":
            sig = self.compute_type_signature(text, memory_type)
            if sig is None:
                return candidates
            for aid in same_type_anchors:
                anchor = self._graph.anchors[aid]
                existing_sig = self.compute_type_signature(
                    anchor.text, memory_type
                )
                if existing_sig == sig:
                    candidates.append(aid)

        elif strategy.dedup_method == "semantic":
            embedder = self._embedder
            if embedder is None or not same_type_anchors:
                return candidates
            try:
                embedding = embedder.encode(text)
            except Exception:
                return candidates
            for aid in same_type_anchors:
                anchor = self._graph.anchors[aid]
                if anchor.embedding is None:
                    continue
                dot = sum(a * b for a, b in zip(embedding, anchor.embedding))
                na = (sum(x * x for x in embedding)) ** 0.5
                nb = (sum(x * x for x in anchor.embedding)) ** 0.5
                if dot / (na * nb + 1e-8) > 0.92:
                    candidates.append(aid)

        return candidates

    # ── Snapshot ──────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return type distribution statistics."""
        type_counts: dict[str, int] = Counter()
        for atype in self._anchor_types.values():
            type_counts[atype.value] += 1

        total = sum(type_counts.values())
        all_types = [mt.value for mt in MemoryType]
        distribution = {t: type_counts.get(t, 0) for t in all_types}

        strategy_summary = {}
        for mt in MemoryType:
            s = self._strategies[mt]
            retention = s.max_retention_days
            strategy_summary[mt.value] = {
                "embedding_model": s.embedding_model,
                "compression_strategy": s.compression_strategy,
                "retrieval_priority": s.retrieval_priority,
                "index_method": s.index_method,
                "dedup_method": s.dedup_method,
                "max_retention_days": (
                    "inf" if retention == float("inf") else retention
                ),
                "mergeable": s.mergeable,
            }

        return {
            "total_typed_anchors": total,
            "distribution": distribution,
            "strategies": strategy_summary,
            "untyped_anchors": len(self._graph.anchors) - total,
            "timestamp": time.time(),
        }
