"""Graph topology ranking — score nodes by structural centrality and edge richness.

For graph-first retrieval: instead of ranking nodes purely by embedding similarity,
rank them by their position in the graph structure — degree centrality, edge type
diversity, causal chain membership, and community role.

Central functions:
  topology_rank()         — combined graph-structure score for a set of nodes
  graph_first_recall()    — seed-by-embedding, then walk the graph for context
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional


# Edge type richness weight — edges of these types contribute more to topology score
EDGE_TYPE_RICHNESS_WEIGHTS = {
    "causes": 1.5,
    "causal": 1.5,
    "fixes": 1.4,
    "resolves": 1.4,
    "depends_on": 1.3,
    "before": 1.2,
    "after": 1.2,
    "preference": 1.2,
    "compresses": 1.1,
    "summarizes": 1.1,
    "topical": 1.0,
    "related": 0.8,
    "contradicts": 0.5,
    "invalidated_by": 0.3,
}


def topology_rank(graph,
                  candidate_ids: list[str] | None = None,
                  query_embedding: list[float] | None = None,
                  top_k: int = 20,
                  graph_weight: float = 0.6,
                  embedding_weight: float = 0.4) -> list[tuple[str, float]]:
    """Rank nodes by combined graph topology score and optional embedding similarity.

    The topology score combines:
      - Degree centrality: how many connections does this node have?
      - Edge type diversity: does it have causal, preference, workflow edges?
      - Betweenness proxy: is it a bridge between different communities?
      - Causal involvement: is it part of cause→effect chains?
      - Recency: has it been recently activated?

    Args:
        graph: StarGraph instance
        candidate_ids: Optional list of anchor IDs to rank. If None, ranks all.
        query_embedding: Optional query embedding for similarity component
        top_k: How many top-ranked nodes to return
        graph_weight: Weight of graph topology score (default 0.6)
        embedding_weight: Weight of embedding similarity (default 0.4)

    Returns:
        List of (anchor_id, combined_score) sorted descending
    """
    if candidate_ids is None:
        candidate_ids = list(graph.anchors.keys())

    scores: list[tuple[str, float]] = []

    for aid in candidate_ids:
        anchor = graph.anchors.get(aid)
        if anchor is None or not anchor.is_retrievable:
            continue

        # ── Graph topology component ──
        topo = _node_topology_score(graph, aid)

        # ── Embedding similarity component ──
        emb_sim = 0.0
        if query_embedding and anchor.embedding:
            emb_sim = _cosine_sim(query_embedding, anchor.embedding)

        # ── Recency bonus ──
        recency = _compute_recency(anchor)

        # ── Combined score ──
        combined = (
            graph_weight * (topo * 0.7 + recency * 0.3) +
            embedding_weight * emb_sim
        )

        scores.append((aid, combined))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def _node_topology_score(graph, anchor_id: str) -> float:
    """Compute the graph topology score for a single node.

    Components (all normalized 0..1):
      - degree_centrality: normalized by max degree in graph
      - edge_type_diversity: Shannon entropy of edge types connected
      - causal_depth: max depth in causal chains (how central in causality)
      - community_bridge: connects nodes from different communities
    """
    anchor = graph.anchors.get(anchor_id)
    if anchor is None:
        return 0.0

    neighbors = list(graph._adjacency.get(anchor_id, set()))

    # Component 1: Degree centrality (normalized)
    max_degree = max((len(graph._adjacency.get(aid, set()))
                      for aid in graph.anchors), default=1)
    degree = len(neighbors)
    degree_score = degree / max(1, max_degree)

    # Component 2: Edge type diversity (Shannon entropy)
    edge_types: list[str] = []
    for neighbor_id in neighbors:
        edge_key = graph._key(anchor_id, neighbor_id)
        edge = graph.edges.get(edge_key)
        if edge:
            edge_types.append(getattr(edge, 'edge_type', 'topical'))

    type_counts = Counter(edge_types)
    total = len(edge_types)
    entropy = 0.0
    if total > 0:
        for count in type_counts.values():
            p = count / total
            entropy -= p * math.log(p)
        max_entropy = math.log(max(1, len(type_counts)))
        entropy_score = entropy / max(1, max_entropy)
    else:
        entropy_score = 0.0

    # Component 3: Edge type richness (weighted by type importance)
    richness = 0.0
    for etype, count in type_counts.items():
        weight = EDGE_TYPE_RICHNESS_WEIGHTS.get(etype, 0.8)
        richness += weight * count
    richness_score = min(1.0, richness / max(1, total * 1.5))

    # Component 4: Community bridge score
    bridge_score = _community_bridge_score(graph, anchor_id)

    # Component 5: Causal involvement
    causal_score = _causal_involvement_score(graph, anchor_id, edge_types)

    # Weighted combination
    weights = {
        "degree": 0.20,
        "entropy": 0.15,
        "richness": 0.25,
        "bridge": 0.15,
        "causal": 0.25,
    }

    return (
        weights["degree"] * degree_score +
        weights["entropy"] * entropy_score +
        weights["richness"] * richness_score +
        weights["bridge"] * bridge_score +
        weights["causal"] * causal_score
    )


def _community_bridge_score(graph, anchor_id: str) -> float:
    """Score how much this node bridges between communities."""
    neighbors = graph._adjacency.get(anchor_id, set())
    if len(neighbors) < 2:
        return 0.0

    # Get community IDs of neighbors
    neighbor_communities: set[str] = set()
    for nid in neighbors:
        n = graph.anchors.get(nid)
        if n and n.community_id:
            neighbor_communities.add(n.community_id)

    # More unique communities = more of a bridge
    if len(neighbor_communities) <= 1:
        return 0.0

    # Normalize: max possible is len(neighbors) different communities
    return min(1.0, len(neighbor_communities) / len(neighbors))


def _causal_involvement_score(graph, anchor_id: str,
                               edge_types: list[str]) -> float:
    """Score how involved this node is in causal chains."""
    causal_types = {"causes", "causal", "caused_by", "fixes", "resolves",
                    "depends_on", "derived_from"}

    causal_edge_count = sum(1 for et in edge_types if et in causal_types)
    total_edges = max(1, len(edge_types))

    causal_ratio = causal_edge_count / total_edges
    return min(1.0, causal_ratio * 2.0)  # boost because causal edges are rare


def _compute_recency(anchor) -> float:
    """Compute recency score (0..1) based on last activation time."""
    import time
    hours = (time.time() - anchor.last_activated_at) / 3600
    return math.exp(-hours / (7 * 24))  # 7-day half-life


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(min_len))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


# ── Graph-First Recall ─────────────────────────────────────

def graph_first_recall(graph, query_embedding: list[float],
                       top_k: int = 10,
                       max_depth: int = 2,
                       graph_weight: float = 0.6,
                       embedding_weight: float = 0.4) -> list[tuple[str, float, int]]:
    """Retrieve memories using graph structure as the primary signal.

    Algorithm:
      1. Find seed nodes via cheap embedding similarity (top 5)
      2. From each seed, walk the graph (BFS) up to max_depth
      3. Score all visited nodes by topology + embedding
      4. Return top-k with their graph distance from nearest seed

    This produces richer context than pure embedding search because it
    follows the actual relational structure of the memory graph.

    Args:
        graph: StarGraph instance
        query_embedding: Query embedding vector
        top_k: How many results to return
        max_depth: Maximum BFS depth from seeds
        graph_weight: Weight of graph topology in final score
        embedding_weight: Weight of embedding similarity

    Returns:
        List of (anchor_id, combined_score, depth_from_seed) sorted by score
    """
    from .index import ANNIndex

    # Step 1: Find seed nodes via ANN
    ann = getattr(graph, '_ann_index', None)
    if ann is None:
        ann = ANNIndex()
        for aid, a in graph.anchors.items():
            if a.embedding and a.is_retrievable:
                ann.add(aid, a.embedding)
        ann.rebuild()

    seed_results = ann.query(query_embedding, k=5)
    seed_ids = [sid for sid, _ in seed_results]
    seed_scores = [sc for _, sc in seed_results]

    if not seed_ids:
        # Fallback: scan all anchors
        scored = []
        for aid, a in graph.anchors.items():
            if a.embedding and a.is_retrievable:
                sim = _cosine_sim(query_embedding, a.embedding)
                scored.append((aid, sim))
        scored.sort(key=lambda x: -x[1])
        seed_ids = [aid for aid, _ in scored[:5]]
        seed_scores = [s for _, s in scored[:5]]

    # Step 2: BFS walk from seeds
    visited: dict[str, tuple[float, int]] = {}  # anchor_id → (max_seed_score, min_depth)

    from collections import deque
    queue = deque()
    for seed_id, seed_score in zip(seed_ids, seed_scores):
        if seed_id in graph.anchors:
            visited[seed_id] = (seed_score, 0)
            queue.append((seed_id, 0, seed_score))

    while queue:
        current_id, depth, parent_score = queue.popleft()
        if depth >= max_depth:
            continue

        for neighbor_id in graph._adjacency.get(current_id, set()):
            # Decay the score as we go deeper
            edge_key = graph._key(current_id, neighbor_id)
            edge = graph.edges.get(edge_key)
            traversal_weight = getattr(edge, 'traversal_weight', 0.5) if edge else 0.5
            propagated_score = parent_score * traversal_weight * (0.6 ** depth)

            if neighbor_id in visited:
                prev_score, prev_depth = visited[neighbor_id]
                if propagated_score > prev_score:
                    visited[neighbor_id] = (propagated_score, depth + 1)
                    queue.append((neighbor_id, depth + 1, propagated_score))
            else:
                visited[neighbor_id] = (propagated_score, depth + 1)
                queue.append((neighbor_id, depth + 1, propagated_score))

    # Step 3: Score all visited nodes by combined topology + embedding
    results: list[tuple[str, float, int]] = []

    for aid, (bfs_score, depth) in visited.items():
        anchor = graph.anchors.get(aid)
        if anchor is None or not anchor.is_retrievable:
            continue

        # Topology score
        topo = _node_topology_score(graph, aid)

        # Embedding similarity
        emb_sim = _cosine_sim(query_embedding, anchor.embedding) if anchor.embedding else 0.0

        # Combined: graph-first means topology gets more weight
        combined = (
            graph_weight * (topo * 0.6 + bfs_score * 0.4) +
            embedding_weight * emb_sim
        )

        results.append((aid, combined, depth))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]
