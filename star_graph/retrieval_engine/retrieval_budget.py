"""Retrieval Budget — prevent spreading activation and recall from exploding.

Enforces three hard limits during retrieval:
  - MAX_HOPS = 3   — max BFS depth from seed nodes
  - MAX_NODES = 24 — max activated nodes per query
  - MAX_TOKENS = 6000 — max token budget for retrieved content

Wired into spreading.activate() and retrieval_pipeline.recall().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
        budget = RetrievalBudget(max_hops=3, max_nodes=24, max_tokens=6000)
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
                 max_hops: int = 3,
                 max_nodes: int = 24,
                 max_tokens: int = 6000):
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_tokens = max_tokens

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
