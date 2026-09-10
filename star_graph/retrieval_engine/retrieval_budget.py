"""Retrieval Budget — prevent spreading activation and recall from exploding.

This module is the SINGLE source of truth for per-query retrieval budget
values (S-5). The authoritative numbers live in ``defaults.yaml`` under the
``retrieval_budget:`` section; the module constants below are only the
hard-coded fallback used when no config is available. Build instances via
:meth:`RetrievalBudget.from_config` rather than hand-rolling getattr chains.

Enforced caps (post-fusion, wired into RetrievalCore.recall):
  - MAX_HOPS = 3   — max BFS depth from seed nodes (advisory: callers run
                     their own BFS with hardcoded depths; see allow_hop)
  - MAX_NODES = 24 — max items kept after RRF fusion per query
  - MAX_TOKENS = 6000 — max token budget for retrieved content (char/4 est.)

Layering note: the activation-spread layer has its own ceiling,
``activation_graph.MAX_ACTIVATED_NODES = 50`` (this is the ``C`` in the
O(C·b) bound of docs/complexity_analysis.md, Theorem 2). That ceiling sits
*below* this budget: recall() currently spreads with ``max_nodes=max_items*2``
which is <= 50 for max_items <= 25, so the two layers never multiply into
runaway growth. Keep spread ceilings >= retrieval_budget.max_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Default fallbacks — mirrors defaults.yaml `retrieval_budget:` section.
MAX_HOPS = 3
MAX_NODES = 24
MAX_TOKENS = 6000


@dataclass
class BudgetState:
    """Runtime budget tracking during a single query."""
    hops_used: int = 0
    nodes_activated: int = 0
    tokens_used: int = 0
    truncated: bool = False
    truncation_reason: str = ""


class RetrievalBudget:
    """Hard budget limits for retrieval operations.

    Usage:
        budget = RetrievalBudget.from_config()   # reads defaults.yaml
        state = budget.begin()

        # During BFS:
        if not budget.allow_hop(state, depth=2):
            break

        # After activation:
        truncated = budget.enforce_nodes(activated, state)

        # After recall:
        truncated = budget.enforce_tokens(items, state)
    """

    def __init__(self,
                 max_hops: int = MAX_HOPS,
                 max_nodes: int = MAX_NODES,
                 max_tokens: int = MAX_TOKENS):
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_tokens = max_tokens

    @classmethod
    def from_config(cls, cfg=None) -> "RetrievalBudget":
        """Build a budget from the ``retrieval_budget`` config section.

        This is the single construction entry point — the authoritative
        values live in defaults.yaml under ``retrieval_budget:``; the
        module constants (MAX_HOPS/MAX_NODES/MAX_TOKENS) are only the
        fallback when the section or a key is missing.

        Args:
            cfg: a :class:`star_graph.config.Config` instance. ``None``
                falls back to the global singleton (``Config.get()``),
                which loads defaults.yaml on first use.
        """
        if cfg is None:
            from ..config import Config
            cfg = Config.get()
        section = getattr(cfg, "retrieval_budget", None)
        return cls(
            max_hops=getattr(section, "max_hops", MAX_HOPS) if section else MAX_HOPS,
            max_nodes=getattr(section, "max_nodes", MAX_NODES) if section else MAX_NODES,
            max_tokens=getattr(section, "max_tokens", MAX_TOKENS) if section else MAX_TOKENS,
        )

    def begin(self) -> BudgetState:
        """Start a new budget tracking session."""
        return BudgetState()

    def allow_hop(self, state: BudgetState, depth: int) -> bool:
        """Check if BFS can proceed to the next depth level."""
        if depth >= self.max_hops:
            state.truncated = True
            state.truncation_reason = f"max_hops({self.max_hops}) reached at depth {depth}"
            return False
        state.hops_used = max(state.hops_used, depth + 1)
        return True

    def allow_node(self, state: BudgetState) -> bool:
        """Check if we can activate one more node."""
        if state.nodes_activated >= self.max_nodes:
            state.truncated = True
            state.truncation_reason = f"max_nodes({self.max_nodes}) reached"
            return False
        state.nodes_activated += 1
        return True

    def count_tokens(self, text: str) -> int:
        """Estimate token count from text (char/4 heuristic)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def allow_tokens(self, state: BudgetState, text: str) -> bool:
        """Check if adding this text stays within token budget."""
        est = self.count_tokens(text)
        if state.tokens_used + est > self.max_tokens:
            state.truncated = True
            state.truncation_reason = f"max_tokens({self.max_tokens}) reached"
            return False
        state.tokens_used += est
        return True

    def enforce_nodes(self, items: list, state: BudgetState | None = None) -> list:
        """Truncate a list of items to max_nodes."""
        if state is None:
            state = self.begin()
        if len(items) > self.max_nodes:
            state.truncated = True
            state.truncation_reason = f"max_nodes({self.max_nodes}) truncated from {len(items)}"
            return items[:self.max_nodes]
        state.nodes_activated = len(items)
        return items

    def enforce_tokens(self, items: list, state: BudgetState | None = None,
                       text_attr: str = "text") -> list:
        """Truncate items to fit within max_tokens budget."""
        if state is None:
            state = self.begin()
        kept = []
        for item in items:
            if isinstance(item, str):
                text = item
            else:
                text = getattr(item, text_attr, "") or ""
            if self.allow_tokens(state, text):
                kept.append(item)
            else:
                break
        return kept

    @property
    def stats(self) -> dict:
        return {
            "max_hops": self.max_hops,
            "max_nodes": self.max_nodes,
            "max_tokens": self.max_tokens,
        }
