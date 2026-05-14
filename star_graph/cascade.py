"""Cascade Recall — causal chain traversal along temporal-causal edges.

Not random graph walk. Not similarity search. Cascade recall follows explicit
causal and temporal edges to reconstruct event chains:

    "Tkinter → GUI development phase → campus network project → Selenium"

Each step follows edges marked as 'caused_by', 'derived_from', or temporal
'before'/'after' relations. This is directional, bounded-depth traversal
that reconstructs the narrative of how memories connect through time and
causation, not through semantic similarity.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph, RichEdge


@dataclass
class CausalChain:
    """A chain of causally-connected memories."""
    anchors: list[Anchor] = field(default_factory=list)
    edges: list[RichEdge] = field(default_factory=list)
    chain_type: str = ""         # "causal", "temporal", "derived"
    total_confidence: float = 0.0
    depth: int = 0

    @property
    def narrative(self) -> str:
        """Human-readable narrative of the causal chain."""
        if not self.anchors:
            return ""
        parts = [a.text[:80] for a in self.anchors]
        connector = " → " if self.chain_type in ("causal", "derived") else " then "
        return connector.join(parts)

    @property
    def is_valid(self) -> bool:
        return len(self.anchors) >= 2 and self.total_confidence > 0.1


class CascadeRecall:
    """Causal chain memory recall.

    Traces causal and temporal edges to reconstruct event chains.
    Directional: can trace forward ("what did this cause?") or
    backward ("what caused this?").

    Usage:
        cascade = CascadeRecall(graph)
        chains = cascade.trace_backward(anchor_id, max_depth=5)
        for chain in chains:
            print(chain.narrative)
    """

    def __init__(self, graph: StarGraph):
        self.graph = graph

    # ── Backward trace: "what caused this?" ──────────────

    def trace_backward(self, anchor_id: str,
                       max_depth: int = 5,
                       min_causal_strength: float = 0.2,
                       max_chains: int = 3) -> list[CausalChain]:
        """Trace backward along causal edges to find root causes.

        Follows edges: caused_by ←, derived_from ←, temporal before ←
        """
        return self._trace(
            start_id=anchor_id,
            max_depth=max_depth,
            min_strength=min_causal_strength,
            max_chains=max_chains,
            direction="backward",
        )

    # ── Forward trace: "what did this cause?" ────────────

    def trace_forward(self, anchor_id: str,
                      max_depth: int = 5,
                      min_causal_strength: float = 0.2,
                      max_chains: int = 3) -> list[CausalChain]:
        """Trace forward along causal edges to find consequences.

        Follows edges: caused_by →, derived_from →, temporal after →
        """
        return self._trace(
            start_id=anchor_id,
            max_depth=max_depth,
            min_strength=min_causal_strength,
            max_chains=max_chains,
            direction="forward",
        )

    # ── Core traversal ───────────────────────────────────

    def _trace(self, start_id: str, max_depth: int,
               min_strength: float, max_chains: int,
               direction: str) -> list[CausalChain]:
        """BFS/DFS hybrid: explore causal edges up to max_depth."""
        start_anchor = self.graph.anchors.get(start_id)
        if not start_anchor:
            return []

        chains: list[CausalChain] = []

        # Initialize frontier with immediate causal neighbors
        frontier: list[CausalChain] = []
        edges = self._get_causal_edges(start_id, direction, min_strength)

        for neighbor_id, edge in edges:
            neighbor = self.graph.anchors.get(neighbor_id)
            if not neighbor:
                continue
            chain = CausalChain(
                anchors=[start_anchor, neighbor],
                edges=[edge],
                chain_type=self._classify_edge(edge),
                total_confidence=edge.confidence if isinstance(edge, RichEdge) else 0.5,
                depth=1,
            )
            frontier.append(chain)

        # Expand frontier
        for _ in range(max_depth - 1):
            next_frontier: list[CausalChain] = []
            for chain in frontier:
                last_id = chain.anchors[-1].id
                next_edges = self._get_causal_edges(
                    last_id, direction, min_strength)

                for next_id, edge in next_edges:
                    # Avoid cycles
                    if next_id in {a.id for a in chain.anchors}:
                        continue
                    next_anchor = self.graph.anchors.get(next_id)
                    if not next_anchor:
                        continue

                    new_chain = CausalChain(
                        anchors=chain.anchors + [next_anchor],
                        edges=chain.edges + [edge],
                        chain_type=chain.chain_type,
                        total_confidence=chain.total_confidence * (
                            edge.confidence if isinstance(edge, RichEdge) else 0.5),
                        depth=chain.depth + 1,
                    )
                    next_frontier.append(new_chain)
                    chains.append(new_chain)

            frontier = next_frontier
            if not frontier:
                break

        # Also add single-hop chains
        chains.extend(frontier)

        # Sort by confidence * depth (longer, more confident chains first)
        chains.sort(key=lambda c: -c.total_confidence * (1 + 0.2 * c.depth))
        return chains[:max_chains]

    def _get_causal_edges(self, anchor_id: str, direction: str,
                          min_strength: float) -> list[tuple[str, RichEdge | object]]:
        """Get causal/temporal edges for an anchor, ranked by traversal weight."""
        results: list[tuple[str, RichEdge | object]] = []
        causal_types = {"caused_by", "derived_from", "causal"}
        temporal_types = {"before", "after"}

        for neighbor_id in self.graph._adjacency.get(anchor_id, set()):
            edge_key = self.graph._key(anchor_id, neighbor_id)
            edge = self.graph.edges.get(edge_key)
            if edge is None:
                continue

            if isinstance(edge, RichEdge):
                etype = edge.edge_type
                causal_str = getattr(edge, 'causal_strength', 0.0)
                # Causal edge types that indicate direction
                forward_causal = {"causes", "causal"}      # A causes B (A→B)
                backward_causal = {"caused_by", "derived_from"}  # A caused by B (B→A)
                bidirectional_causal = {"depends_on", "fixes", "resolves"}

                if direction == "backward":
                    if etype in backward_causal and causal_str >= min_strength:
                        results.append((neighbor_id, edge))
                    elif etype == "temporal" and getattr(edge, 'temporal_order', '') == "before":
                        if causal_str >= min_strength:
                            results.append((neighbor_id, edge))
                else:
                    if etype in forward_causal and causal_str >= min_strength:
                        results.append((neighbor_id, edge))
                    elif etype in bidirectional_causal and causal_str >= min_strength:
                        results.append((neighbor_id, edge))
                    elif etype == "temporal" and getattr(edge, 'temporal_order', '') == "after":
                        if causal_str >= min_strength:
                            results.append((neighbor_id, edge))
            else:
                etype = edge.edge_type
                if etype in causal_types:
                    results.append((neighbor_id, edge))
                elif etype in temporal_types:
                    if (direction == "backward" and etype == "before") or \
                       (direction == "forward" and etype == "after"):
                        results.append((neighbor_id, edge))

        # Rank by traversal_weight (higher = more important causal path)
        def _sort_key(item):
            e = item[1]
            tw = e.traversal_weight if hasattr(e, 'traversal_weight') else e.weight
            causal = getattr(e, 'causal_strength', 0.0) if isinstance(e, RichEdge) else 0.0
            return tw * (1.0 + causal)

        results.sort(key=_sort_key, reverse=True)
        return results

    def _classify_edge(self, edge) -> str:
        if isinstance(edge, RichEdge):
            if edge.edge_type in ("caused_by",):
                return "causal"
            if edge.edge_type in ("derived_from",):
                return "derived"
            if getattr(edge, 'temporal_order', ''):
                return "temporal"
        if hasattr(edge, 'edge_type'):
            if edge.edge_type in ("caused_by", "causal"):
                return "causal"
            if edge.edge_type in ("derived_from",):
                return "derived"
        return "temporal"

    # ── Full cascade from a query ────────────────────────

    def cascade_from_seeds(self, seed_ids: list[str],
                           max_depth: int = 5,
                           max_chains: int = 5) -> list[CausalChain]:
        """Run cascade recall from multiple seed anchors.

        Seeds are typically the top retrieval results. Cascade expands each
        seed into a causal chain, potentially revealing the full narrative
        context around the query.
        """
        all_chains: list[CausalChain] = []
        for seed_id in seed_ids[:3]:  # limit to top 3 seeds
            backward = self.trace_backward(seed_id, max_depth, max_chains=2)
            forward = self.trace_forward(seed_id, max_depth, max_chains=2)
            all_chains.extend(backward)
            all_chains.extend(forward)

        # Deduplicate and sort
        seen_narratives: set[str] = set()
        unique: list[CausalChain] = []
        for chain in sorted(all_chains, key=lambda c: -c.total_confidence):
            key = chain.narrative
            if key not in seen_narratives:
                seen_narratives.add(key)
                unique.append(chain)

        return unique[:max_chains]
