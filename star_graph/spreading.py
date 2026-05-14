"""Spreading Activation — local subgraph activation from seed nodes.

Not full-graph traversal. Not similarity-only search. Spreading activation
finds seed anchors by embedding, then walks the graph from seeds using
edge-type-weighted traversal:

  activation = seed_activation × traversal_weight × decay^depth

Edge types matter: causal edges (×1.5) propagate more activation than
contradiction edges (×0.5). This prevents the "everything is vaguely related"
problem by keeping activation local and structured.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph
from .config import Config


@dataclass
class ActivatedNode:
    """A node that received activation during spreading."""
    anchor_id: str
    anchor: Anchor | None = None
    accumulated_activation: float = 0.0
    activation_depth: int = 0       # how many hops from nearest seed
    source_seeds: list[str] = field(default_factory=list)  # which seeds activated this node
    path: list[str] = field(default_factory=list)  # trace of node IDs from seed to here

    @property
    def text(self) -> str:
        return self.anchor.text if self.anchor else ""

    @property
    def tags(self) -> list[str]:
        return self.anchor.tags if self.anchor else []


class SpreadingActivation:
    """Local subgraph activation with edge-type-weighted traversal.

    Usage:
        sa = SpreadingActivation(graph)
        activated = sa.activate(query_embedding, top_k=10)
        # activated is a list of ActivatedNode sorted by activation descending
    """

    def __init__(self, graph: StarGraph, config: Config | None = None):
        self.graph = graph
        self.cfg = config or Config.get()
        self._decay: float = 0.6
        self._min_traversal_weight: float = 0.05
        self._max_depth: int = 3
        self._top_k_seeds: int = 5
        self._top_k_results: int = 10
        self._seed_initial_activation: float = 1.0

    # ── Configuration ────────────────────────────────────

    def configure(self, *, decay: float | None = None,
                  min_traversal_weight: float | None = None,
                  max_depth: int | None = None,
                  top_k_seeds: int | None = None,
                  top_k_results: int | None = None):
        """Override default hyperparameters."""
        if decay is not None:
            self._decay = decay
        if min_traversal_weight is not None:
            self._min_traversal_weight = min_traversal_weight
        if max_depth is not None:
            self._max_depth = max_depth
        if top_k_seeds is not None:
            self._top_k_seeds = top_k_seeds
        if top_k_results is not None:
            self._top_k_results = top_k_results
        return self

    # ── Main API ─────────────────────────────────────────

    def activate(self, query_embedding: list[float] | None = None,
                 seed_ids: list[str] | None = None,
                 top_k: int | None = None) -> list[ActivatedNode]:
        """Run spreading activation from seeds.

        Args:
            query_embedding: embedding to find seeds via cortical_lookup
            seed_ids: explicit seed anchor IDs (overrides query_embedding)
            top_k: max results to return (defaults to self._top_k_results)

        Returns list of ActivatedNode sorted by accumulated activation desc.
        """
        if top_k is None:
            top_k = self._top_k_results

        # Find seeds
        seeds: list[str] = []
        if seed_ids:
            seeds = [sid for sid in seed_ids if sid in self.graph.anchors]
        elif query_embedding:
            seeds = self._find_seeds(query_embedding)
        else:
            return []

        if not seeds:
            return []

        # BFS spreading
        activated: dict[str, ActivatedNode] = {}
        current_wave: dict[str, float] = {}
        next_wave: dict[str, float] = {}

        for seed_id in seeds:
            current_wave[seed_id] = self._seed_initial_activation
            activated[seed_id] = ActivatedNode(
                anchor_id=seed_id,
                anchor=self.graph.anchors.get(seed_id),
                accumulated_activation=self._seed_initial_activation,
                activation_depth=0,
                source_seeds=[seed_id],
                path=[seed_id],
            )

        for depth in range(self._max_depth):
            if not current_wave:
                break
            next_wave.clear()

            for node_id, level in current_wave.items():
                neighbors = self.graph.neighbors(node_id, min_weight=self._min_traversal_weight)
                for neighbor_id, traversal_weight in neighbors:
                    propagated = level * traversal_weight * (self._decay ** (depth + 1))
                    if propagated < self._min_traversal_weight:
                        continue

                    if neighbor_id not in activated:
                        activated[neighbor_id] = ActivatedNode(
                            anchor_id=neighbor_id,
                            anchor=self.graph.anchors.get(neighbor_id),
                            accumulated_activation=0.0,
                            activation_depth=depth + 1,
                            source_seeds=list(activated[node_id].source_seeds),
                            path=activated[node_id].path + [neighbor_id],
                        )
                        next_wave[neighbor_id] = propagated
                    else:
                        # Already activated — sum activation from multiple paths
                        node = activated[neighbor_id]
                        merged_sources = list(set(node.source_seeds + activated[node_id].source_seeds))
                        node.source_seeds = merged_sources

                    activated[neighbor_id].accumulated_activation += propagated

            current_wave = dict(next_wave)

        # Collect and rank
        ranked = sorted(
            activated.values(),
            key=lambda n: -n.accumulated_activation,
        )
        return ranked[:top_k]

    def _find_seeds(self, query_embedding: list[float]) -> list[str]:
        """Find seed anchors via cortical (ANN) lookup."""
        results = self.graph.cortical_lookup(query_embedding, top_k=self._top_k_seeds * 2)
        if not results:
            return []
        # Filter to retrievable anchors
        seeds = []
        for aid, score in results:
            anchor = self.graph.anchors.get(aid)
            if anchor and anchor.is_retrievable:
                seeds.append(aid)
                if len(seeds) >= self._top_k_seeds:
                    break
        return seeds

    # ── Convenience ──────────────────────────────────────

    def activate_from_text(self, query: str,
                           top_k: int | None = None) -> list[ActivatedNode]:
        """Run spreading activation from a query string (encodes + activates)."""
        from .embedding import get_embedder
        embedder = get_embedder()
        query_emb = embedder.encode(query)
        return self.activate(query_embedding=query_emb, top_k=top_k)

    def get_subgraph(self, activated: list[ActivatedNode]) -> dict:
        """Extract the activated subgraph as {anchor_id: {neighbor_id, weight, ...}}.

        Useful for visualization and debugging of the activation spread.
        """
        activated_ids = {n.anchor_id for n in activated}
        subgraph: dict[str, list[dict]] = {}
        for aid in activated_ids:
            neighbors = self.graph.neighbors(aid, min_weight=0.0)
            subgraph[aid] = [
                {"neighbor": nid, "traversal_weight": tw}
                for nid, tw in neighbors
                if nid in activated_ids
            ]
        return subgraph
