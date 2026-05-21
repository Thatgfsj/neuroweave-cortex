"""Memory Budget — capacity management and eviction across cognitive layers.

Enforces per-layer anchor quotas and token consumption limits for LLM calls.
Wired into memory gate, recall, and prompt injection codepaths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor


@dataclass
class MemoryBudgetConfig:
    """Per-layer quotas."""

    total_anchors: int = 50_000
    working_memory: int = 200
    episodic: int = 30_000
    semantic: int = 15_000
    core_identity: int = 5_000
    enforce_on_remember: bool = True
    eviction_policy: str = "lowest_quality"  # lowest_quality | oldest | coldest


@dataclass
class TokenBudgetConfig:
    """Token consumption limits."""

    per_recall_max_tokens: int = 4_000
    daily_llm_call_limit: int = 100
    prompt_injection_max_chars: int = 8_000
    truncation_strategy: str = "importance_desc"  # importance_desc | recency_desc | hybrid


def infer_memory_layer(anchor: Anchor) -> str:
    """Infer working/episodic/semantic/core_identity from anchor properties."""
    tags_lower = {t.lower() for t in (anchor.tags or [])}

    core_identity_indicators = {"identity", "personality", "core_value", "belief"}
    if tags_lower & core_identity_indicators:
        return "core_identity"
    imp = anchor.vector.importance if anchor.vector else 0.0
    val = anchor.vector.emotional_valence if anchor.vector else 0.0
    if imp > 0.9 and val != 0.0:
        return "core_identity"

    working_indicators = {"current_task", "debug", "working"}
    if tags_lower & working_indicators:
        return "working_memory"
    if getattr(anchor, "memory_tier", "") == "hot":
        return "working_memory"

    semantic_indicators = {"knowledge", "concept", "pattern", "preference", "fact", "skill"}
    if tags_lower & semantic_indicators:
        return "semantic"

    return "episodic"


def _importance_sort_key(item):
    """Extract importance for sorting, falling back to retention_score."""
    if hasattr(item, "vector") and item.vector is not None:
        imp = getattr(item.vector, "importance", 0.0)
        if isinstance(imp, (int, float)) and imp > 0.0:
            return imp
    if hasattr(item, "retention_score"):
        rs = item.retention_score
        if isinstance(rs, (int, float)):
            return rs
    return 0.0


class MemoryBudget:
    """Capacity controller managing anchor quotas across cognitive layers."""

    def __init__(self, graph: StarGraph, config: MemoryBudgetConfig | None = None):
        """Bind to a StarGraph with optional per-layer quota overrides."""
        self._graph = graph
        self._config = config or MemoryBudgetConfig()

    @property
    def config(self) -> MemoryBudgetConfig:
        """Active budget configuration."""
        return self._config

    def _quota_for_layer(self, layer: str) -> int:
        """Map a layer name to its configured quota."""
        mapping = {
            "working_memory": self._config.working_memory,
            "episodic": self._config.episodic,
            "semantic": self._config.semantic,
            "core_identity": self._config.core_identity,
        }
        return mapping.get(layer, self._config.total_anchors)

    def _is_valid_layer(self, layer: str) -> bool:
        """Check whether a layer name is recognised."""
        return layer in {"working_memory", "episodic", "semantic", "core_identity"}

    def check_capacity(self, layer: str = "episodic") -> bool:
        """Return True if the layer has room for more anchors."""
        return self._anchor_count(layer) < self._quota_for_layer(layer)

    def enforce(self, layer: str = "episodic") -> list[str]:
        """Evict weakest anchors to bring the layer within quota. Returns evicted ids."""
        quota = self._quota_for_layer(layer)
        count = self._anchor_count(layer)
        if count <= quota:
            return []
        excess = count - quota
        return self.evict(excess, layer, self._config.eviction_policy)

    def quota_used(self, layer: str) -> float:
        """Fraction of the layer quota currently consumed (0.0-1.0)."""
        quota = self._quota_for_layer(layer)
        if quota <= 0:
            return 1.0
        return min(1.0, self._anchor_count(layer) / quota)

    def quota_remaining(self, layer: str) -> int:
        """Absolute number of free slots remaining in the layer."""
        return max(0, self._quota_for_layer(layer) - self._anchor_count(layer))

    def evict(self, count: int, layer: str, policy: str = "lowest_quality") -> list[str]:
        """Evict *count* anchors from *layer* using the given eviction policy."""
        if count <= 0:
            return []

        candidates = [(aid, a) for aid, a in self._graph.anchors.items()
                       if self.layer_for_anchor(a) == layer]
        if not candidates:
            return []

        if policy == "oldest":
            candidates.sort(key=lambda x: getattr(x[1], "last_activated_at", 0.0))
        elif policy == "coldest":
            candidates.sort(key=lambda x: (
                getattr(x[1], "retention_score", 0.0),
                getattr(x[1], "last_activated_at", 0.0),
            ))
        else:
            candidates.sort(key=lambda x: getattr(x[1], "retention_score", 0.0))

        evicted: list[str] = []
        for anchor_id, _ in candidates[:count]:
            self._graph.remove_anchor(anchor_id)
            evicted.append(anchor_id)
        return evicted

    def layer_for_anchor(self, anchor: Anchor) -> str:
        """Infer the cognitive layer an anchor belongs to."""
        explicit = getattr(anchor, "memory_layer", None)
        if explicit and self._is_valid_layer(explicit):
            return explicit
        return infer_memory_layer(anchor)

    def snapshot(self) -> dict:
        """Full budget status across all layers."""
        layers = ["working_memory", "episodic", "semantic", "core_identity"]
        result: dict[str, object] = {
            "total_anchors_in_graph": len(self._graph.anchors),
        }
        for layer in layers:
            quota = self._quota_for_layer(layer)
            used = self._anchor_count(layer)
            result[layer] = {
                "quota": quota,
                "used": used,
                "remaining": max(0, quota - used),
                "fraction_used": round(used / quota, 4) if quota > 0 else 1.0,
            }
        return result

    def _anchor_count(self, layer: str) -> int:
        """Count anchors currently assigned to a layer."""
        return sum(1 for a in self._graph.anchors.values()
                   if self.layer_for_anchor(a) == layer)


class TokenBudget:
    """Token consumption tracker with daily LLM call limits."""

    def __init__(self, config: TokenBudgetConfig | None = None):
        """Initialise with optional config overrides."""
        self._config = config or TokenBudgetConfig()
        self._daily_llm_calls: int = 0
        self._daily_tokens: int = 0
        self._date: str = time.strftime("%Y-%m-%d")

    @property
    def config(self) -> TokenBudgetConfig:
        """Active token budget configuration."""
        return self._config

    def _check_date(self) -> None:
        """Auto-reset if the date has rolled over."""
        today = time.strftime("%Y-%m-%d")
        if today != self._date:
            self.reset_daily()

    def check_recall_budget(self, requested_tokens: int) -> bool:
        """Return True if the requested tokens fit within per_recall_max_tokens."""
        return requested_tokens <= self._config.per_recall_max_tokens

    def check_daily_llm_budget(self) -> bool:
        """Return True if the daily LLM call limit has not been reached."""
        self._check_date()
        return self._daily_llm_calls < self._config.daily_llm_call_limit

    def truncate_items(self, items: list, max_chars: int, strategy: str = "") -> list:
        """Truncate a list of memory items to fit within *max_chars* total characters."""
        if not items:
            return []

        strategy = strategy or self._config.truncation_strategy

        if strategy == "importance_desc":
            sorted_items = sorted(items, key=_importance_sort_key, reverse=True)
        elif strategy == "recency_desc":
            sorted_items = sorted(
                items,
                key=lambda x: getattr(x, "last_activated_at", 0.0),
                reverse=True,
            )
        elif strategy == "hybrid":
            now = time.time() or 1.0
            def _hybrid_key(item):
                imp = _importance_sort_key(item)
                rec = getattr(item, "last_activated_at", 0.0) / now
                return imp * 0.6 + rec * 0.4
            sorted_items = sorted(items, key=_hybrid_key, reverse=True)
        else:
            sorted_items = list(items)

        kept: list = []
        total_chars = 0
        for item in sorted_items:
            text = getattr(item, "text", str(item))
            total_chars += len(text)
            if total_chars > max_chars:
                break
            kept.append(item)
        return kept

    def record_llm_call(self, tokens_used: int = 0) -> None:
        """Record an LLM call against the daily budget."""
        self._check_date()
        self._daily_llm_calls += 1
        self._daily_tokens += tokens_used

    def daily_stats(self) -> dict:
        """Return today's consumption statistics."""
        self._check_date()
        return {
            "date": self._date,
            "llm_calls": self._daily_llm_calls,
            "llm_call_limit": self._config.daily_llm_call_limit,
            "tokens_used": self._daily_tokens,
        }

    def reset_daily(self) -> None:
        """Reset daily counters to zero."""
        self._daily_llm_calls = 0
        self._daily_tokens = 0
        self._date = time.strftime("%Y-%m-%d")
