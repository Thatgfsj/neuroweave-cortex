"""Observatory — activation-based retrieval engine.

This is NOT an LLM-driven brightness calculator.

Core principle:
  Brightness = f(historical_access, recency, edge_strength)
  
  NOT f(LLM_generated_vector)

The observatory provides a thin wrapper over ActivationGraph for 
backward compatibility. All real logic lives in activation_graph.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .activation_graph import (
    ActivationGraph, ActivatedNode, compute_edge_strength,
    get_activation_graph, ACTIVATION_THRESHOLD, MAX_SPREAD_DEPTH, MAX_ACTIVATED_NODES,
)
from star_graph.math_utils import cosine_sim as _cosine_sim


@dataclass
class ObservationResult:
    """Result of an observation (retrieval)."""
    node_id: str
    activation: float
    path: list[str]
    source: str
    depth: int


class Observatory:
    """The observatory — orchestrates activation-based retrieval.
    
    Retrieval flow:
      1. encode(query) → query_embedding
      2. find_seeds(query_embedding) → seed nodes
      3. spread(seeds) → activated subgraph
      4. reinforce_path(path_of_top_result) → strengthen edges
    
    No LLM calls at any stage. LLM only touches L1 maintenance.
    """

    def __init__(self, graph, embedder=None):
        self.graph = graph
        self._embedder = embedder
        self._act = get_activation_graph(graph, embedder)
        self._observation_history: list[dict] = []

    def observe(self, query_embedding: list[float],
                top_k: int = 10,
                max_depth: int = MAX_SPREAD_DEPTH,
                reinforce: bool = True) -> list[ObservationResult]:
        """Retrieve memories via activation spreading.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of results to return
            max_depth: Maximum BFS depth for activation spread
            reinforce: Whether to reinforce edges along the retrieval path
            
        Returns:
            Sorted list of ObservationResult, highest activation first
        """
        # 1. Find seed nodes
        seeds = self._act.find_seeds(query_embedding, top_k=max(top_k, 5))
        if not seeds:
            return []
        
        # 2. Activation spread
        activated = self._act.spread(seeds, max_depth=max_depth, max_nodes=top_k * 2)
        
        # 3. Convert to ObservationResult
        results = [
            ObservationResult(
                node_id=n.node_id,
                activation=n.activation,
                path=n.path,
                source=n.source,
                depth=n.depth,
            )
            for n in activated[:top_k]
        ]
        
        # 4. Reinforce edges along retrieval path
        if reinforce and results:
            top_path = results[0].path
            self._act.reinforce_path(top_path)
        
        # 5. Record observation
        self._observation_history.append({
            "timestamp": time.time(),
            "num_seeds": len(seeds),
            "num_results": len(results),
            "top_activation": results[0].activation if results else 0.0,
        })
        
        return results

    def decay(self) -> int:
        """Apply time decay to all edges. Call during sleep."""
        return self._act.decay_all()

    def get_stats(self) -> dict:
        """Get observation statistics."""
        if not self._observation_history:
            return {"total_observations": 0}
        return {
            "total_observations": len(self._observation_history),
            "last_observation": self._observation_history[-1],
        }
