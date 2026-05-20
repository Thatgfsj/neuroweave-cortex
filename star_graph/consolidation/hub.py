"""Hub Abstraction Layer — cross-cortex bridges via summary nodes.

Hubs are NOT super-nodes. They are compressed, stable summary nodes that
bridge different cortices without creating O(n²) cross-connections.

Hierarchy:
    Leaf Hub (topic-level) -> Domain Hub (cortex-level) -> Global Self Hub

A hub:
- Stores a compressed summary, not raw memories
- Has pointers to source anchors (in their respective cortices)
- Is near-immune to decay (stability > 0.9)
- Is the ONLY mechanism for cross-cortex association
- Forms the "4th dimension" connecting 3D star clusters across cortices
- Can have edges to other hubs, forming a cross-domain reasoning network

v0.6: Added HubEdge for hub-to-hub multi-hop reasoning.
v0.7: Added HubShard for bounded hub partitioning with auto-split.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HubShard:
    """A bounded container partitioning hubs by topic and capacity.

    When a shard exceeds max_degree, it auto-splits into child shards
    based on embedding clustering. Prevents single-hub black-hole.
    """
    id: str
    name: str
    hub_level: str = "leaf"        # "leaf", "domain", "global"
    cortex_name: str = ""
    parent_shard_id: str = ""
    child_shard_ids: list[str] = field(default_factory=list)
    hub_ids: list[str] = field(default_factory=list)
    centroid: list[float] | None = None
    max_degree: int = 50
    _centroid_stale: bool = field(default=True, repr=False)
    created_at: float = field(default_factory=time.time)
    last_split_at: float = 0.0
    split_count: int = 0

    @property
    def degree(self) -> int:
        return len(self.hub_ids)

    @property
    def is_full(self) -> bool:
        return self.degree >= self.max_degree

    @property
    def is_leaf_shard(self) -> bool:
        return len(self.child_shard_ids) == 0

    def add_hub(self, hub_id: str) -> None:
        if hub_id not in self.hub_ids:
            self.hub_ids.append(hub_id)
            self._centroid_stale = True

    def remove_hub(self, hub_id: str) -> None:
        if hub_id in self.hub_ids:
            self.hub_ids.remove(hub_id)
            self._centroid_stale = True

    def recompute_centroid(self, hubs: dict) -> None:
        embeddings = []
        for hid in self.hub_ids:
            hub = hubs.get(hid)
            if hub and hub.embedding:
                embeddings.append(hub.embedding)
        if embeddings:
            dim = len(embeddings[0])
            self.centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
        self._centroid_stale = False

    @classmethod
    def create(cls, name: str, hub_level: str = "leaf",
               cortex_name: str = "", max_degree: int = 50,
               parent_shard_id: str = "") -> "HubShard":
        raw = f"{cortex_name}:{name}:{hub_level}"
        shard_id = hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()
        return cls(id=shard_id, name=name, hub_level=hub_level,
                   cortex_name=cortex_name, max_degree=max_degree,
                   parent_shard_id=parent_shard_id)


@dataclass
class HubNode:
    """A compressed, stable summary node bridging memory clusters.

    Hubs do not store raw memory. They store:
    - A compressed summary of the memories they abstract
    - Pointers (anchor IDs) back to source memories
    - Cross-references to other hubs (forming the hub hierarchy)
    """
    id: str
    text: str                          # compressed summary text
    hub_level: str = "leaf"            # "leaf" | "domain" | "global"
    source_anchor_ids: list[str] = field(default_factory=list)
    source_cortex: str = ""            # which cortex this hub abstracts
    shard_id: str = ""                 # which shard this hub belongs to
    embedding: list[float] | None = None
    importance: float = 0.5
    stability: float = 0.9             # hubs are highly stable
    confidence: float = 0.5
    parent_hub_id: str = ""            # link to higher-level hub
    child_hub_ids: list[str] = field(default_factory=list)  # links to lower-level hubs
    cross_ref_hubs: list[str] = field(default_factory=list) # peer hubs in other cortices
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    update_count: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        """Hubs resist decay. Stability only drops after many contradictory updates."""
        return self.stability >= 0.7

    @property
    def decay_factor(self) -> float:
        """Hubs decay extremely slowly."""
        hours = (time.time() - self.last_updated_at) / 3600
        # 10x slower decay than regular anchors (half-life ~300 days vs 30)
        return math.exp(-hours / (7200 + 1e-8))

    def refresh(self, new_text: str | None = None,
                new_importance: float | None = None):
        """Update hub summary when source anchors change."""
        if new_text:
            self.text = new_text
        if new_importance is not None:
            self.importance = 0.8 * self.importance + 0.2 * new_importance
        self.last_updated_at = time.time()
        self.update_count += 1
        # Hubs become slightly more stable with each update
        self.stability = min(0.99, self.stability + 0.005)

    def add_cross_ref(self, other_hub_id: str):
        """Add a cross-reference to a hub in another cortex."""
        if other_hub_id not in self.cross_ref_hubs:
            self.cross_ref_hubs.append(other_hub_id)

    @classmethod
    def create(cls, text: str, hub_level: str = "leaf",
               source_anchor_ids: list[str] | None = None,
               source_cortex: str = "",
               importance: float = 0.5,
               **kwargs) -> HubNode:
        """Create a hub with a deterministic ID based on text content."""
        hash_input = f"{hub_level}:{text}:{source_cortex}"
        hub_id = hashlib.blake2b(
            hash_input.encode(), digest_size=10).hexdigest()
        return cls(
            id=hub_id,
            text=text,
            hub_level=hub_level,
            source_anchor_ids=source_anchor_ids or [],
            source_cortex=source_cortex,
            importance=importance,
            **kwargs,
        )


@dataclass
class HubEdge:
    """An edge between two hub nodes in the HubSphere.

    Hub edges enable multi-hop reasoning across domains:
    "Python async" (DevCortex hub) -> "performance optimization" (InfraCortex hub)
    """
    id: str
    source_hub_id: str
    target_hub_id: str
    weight: float = 0.5
    edge_type: str = "cross_domain"  # "cross_domain", "causal", "analogical", "temporal"
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0

    def reinforce(self, delta: float = 0.05):
        self.weight = min(1.0, self.weight + delta)
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + delta * 0.5)

    @classmethod
    def create(cls, source_id: str, target_id: str,
               weight: float = 0.5, edge_type: str = "cross_domain") -> HubEdge:
        eid = hashlib.blake2b(
            f"{source_id}:{target_id}:{edge_type}".encode(), digest_size=8
        ).hexdigest()
        return cls(id=eid, source_hub_id=source_id, target_hub_id=target_id,
                   weight=weight, edge_type=edge_type)


class HubLayer:
    """Manages the hierarchy of hub nodes across cortices.

    Three-level hierarchy:
    - Leaf hubs: compress a single topic cluster (e.g., "Python async patterns")
    - Domain hubs: aggregate leaf hubs within a cortex (e.g., "Developer identity")
    - Global hubs: aggregate across all cortices (e.g., "Global self model")

    Sharding (v0.7):
    - Each hub is assigned to a HubShard bounded by max_degree_per_shard.
    - When a shard exceeds capacity, it auto-splits into child shards via
      embedding clustering, preventing any single hub from becoming a black-hole.
    - Shard routing enables O(log N) lookup instead of O(N) traversal.

    Usage:
        layer = HubLayer(max_degree_per_shard=50)
        leaf = layer.create_leaf("Python async patterns", source_anchors, "dev")
        domain = layer.create_domain("Developer identity", [leaf.id], "dev")
        global_hub = layer.create_global("Global self", [domain.id])
    """

    def __init__(self, max_degree_per_shard: int = 50):
        self.hubs: dict[str, HubNode] = {}
        self.edges: dict[str, HubEdge] = {}  # hub-to-hub edges
        # Index: cortex name -> hub IDs in that cortex
        self._cortex_index: dict[str, list[str]] = {}
        # Adjacency for hub-to-hub traversal
        self._hub_adjacency: dict[str, set[str]] = {}
        # Shard management (v0.7)
        self.shards: dict[str, HubShard] = {}
        self.max_degree_per_shard: int = max_degree_per_shard
        self._shard_adjacency: dict[str, set[str]] = {}

    # ── Hub creation ─────────────────────────────────────

    def create_leaf(self, text: str,
                    source_anchor_ids: list[str],
                    cortex_name: str,
                    importance: float = 0.5,
                    embedding: list[float] | None = None) -> HubNode:
        """Create a leaf-level hub from a topic cluster."""
        hub = HubNode.create(
            text=text,
            hub_level="leaf",
            source_anchor_ids=source_anchor_ids,
            source_cortex=cortex_name,
            importance=importance,
        )
        if embedding:
            hub.embedding = embedding

        # Register hub before shard assignment so auto_split can find it
        self._add(hub, cortex_name)

        # Shard assignment
        shard = self._find_or_create_shard(
            name=f"leaf_{cortex_name}",
            hub_level="leaf",
            cortex_name=cortex_name,
            embedding=hub.embedding,
        )
        hub.shard_id = shard.id
        shard.add_hub(hub.id)
        self._check_and_split(shard)

        return hub

    def create_domain(self, text: str,
                      child_hub_ids: list[str],
                      cortex_name: str,
                      importance: float = 0.5) -> HubNode | None:
        """Create a domain-level hub aggregating leaf hubs.

        Only leaf hubs from the SAME cortex can be aggregated into a domain hub.
        Cross-cortex aggregation happens at the global level.
        """
        # Validate all child hubs are in the same cortex
        for child_id in child_hub_ids:
            child = self.hubs.get(child_id)
            if child is None or child.source_cortex != cortex_name:
                return None

        hub = HubNode.create(
            text=text,
            hub_level="domain",
            source_cortex=cortex_name,
            importance=importance,
        )

        # Link parents
        for child_id in child_hub_ids:
            child = self.hubs[child_id]
            child.parent_hub_id = hub.id
            hub.child_hub_ids.append(child_id)
            # Domain hubs aggregate importance from children
            hub.importance = max(hub.importance, child.importance)

        # Register hub before shard assignment
        self._add(hub, cortex_name)

        # Shard assignment
        shard = self._find_or_create_shard(
            name=f"domain_{cortex_name}",
            hub_level="domain",
            cortex_name=cortex_name,
        )
        hub.shard_id = shard.id
        shard.add_hub(hub.id)
        self._check_and_split(shard)

        return hub

    def create_global(self, text: str,
                      domain_hub_ids: list[str],
                      importance: float = 0.5) -> HubNode:
        """Create a global hub aggregating domain hubs from multiple cortices."""
        hub = HubNode.create(
            text=text,
            hub_level="global",
            importance=importance,
        )

        for child_id in domain_hub_ids:
            child = self.hubs.get(child_id)
            if child:
                child.parent_hub_id = hub.id
                hub.child_hub_ids.append(child_id)
                hub.importance = max(hub.importance, child.importance)

        # Register hub before shard assignment
        self._add(hub, "__global__")

        # Shard assignment
        shard = self._find_or_create_shard(
            name="global_self",
            hub_level="global",
            cortex_name="__global__",
        )
        hub.shard_id = shard.id
        shard.add_hub(hub.id)
        self._check_and_split(shard)

        return hub

    # ── Cross-cortex bridging ────────────────────────────

    def bridge(self, hub_a_id: str, hub_b_id: str) -> bool:
        """Create a cross-reference between hubs in different cortices.

        This is the ONLY mechanism for cross-cortex association.
        Direct anchor-to-anchor edges across cortices are forbidden.
        """
        hub_a = self.hubs.get(hub_a_id)
        hub_b = self.hubs.get(hub_b_id)
        if not hub_a or not hub_b:
            return False
        if hub_a.source_cortex == hub_b.source_cortex:
            return False  # Same cortex — not a cross-cortex bridge

        hub_a.add_cross_ref(hub_b_id)
        hub_b.add_cross_ref(hub_a_id)
        return True

    # ── Hub-to-hub edges (cross-domain reasoning network) ─

    def add_hub_edge(self, source_id: str, target_id: str,
                     weight: float = 0.5,
                     edge_type: str = "cross_domain") -> HubEdge | None:
        """Create an edge between two hub nodes.

        Hub edges form a cross-domain reasoning network:
        "Python async" hub -> "performance optimization" hub -> "cost reduction" hub

        v0.7: Registers cross-shard bridges in _shard_adjacency when hubs
        belong to different shards.
        """
        if source_id not in self.hubs or target_id not in self.hubs:
            return None
        if source_id == target_id:
            return None

        edge = HubEdge.create(source_id, target_id, weight, edge_type)
        self.edges[edge.id] = edge

        # Update hub adjacency
        if source_id not in self._hub_adjacency:
            self._hub_adjacency[source_id] = set()
        self._hub_adjacency[source_id].add(target_id)

        # Register cross-shard bridge if hubs are in different shards
        src_shard = self.hubs[source_id].shard_id
        tgt_shard = self.hubs[target_id].shard_id
        if src_shard and tgt_shard and src_shard != tgt_shard:
            if src_shard not in self._shard_adjacency:
                self._shard_adjacency[src_shard] = set()
            self._shard_adjacency[src_shard].add(tgt_shard)
            if tgt_shard not in self._shard_adjacency:
                self._shard_adjacency[tgt_shard] = set()
            self._shard_adjacency[tgt_shard].add(src_shard)

        return edge

    def traverse_hubs(self, start_hub_id: str, max_hops: int = 2,
                      max_results: int = 5,
                      shard_ids: list[str] | None = None) -> list[HubNode]:
        """Multi-hop traversal on the hub-to-hub reasoning network.

        From a starting hub, follow edges up to max_hops to discover
        connected hubs in other domains. This enables cross-domain inference.

        v0.7: Optional shard_ids parameter restricts search to hubs within
        the specified shards.
        """
        if start_hub_id not in self.hubs:
            return []

        shard_set: set[str] | None = set(shard_ids) if shard_ids else None
        visited: set[str] = {start_hub_id}
        frontier: list[str] = [start_hub_id]
        results: list[HubNode] = []

        for _ in range(max_hops):
            next_frontier: list[str] = []
            for hub_id in frontier:
                for neighbor_id in self._hub_adjacency.get(hub_id, set()):
                    if neighbor_id not in visited:
                        neighbor = self.hubs.get(neighbor_id)
                        if neighbor is None:
                            continue
                        # Shard filter
                        if shard_set and neighbor.shard_id not in shard_set:
                            continue
                        visited.add(neighbor_id)
                        results.append(neighbor)
                        next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        # Sort by importance
        results.sort(key=lambda h: -h.importance)
        return results[:max_results]

    # ── Shard management (v0.7) ──────────────────────────

    def _find_or_create_shard(self, name: str, hub_level: str = "leaf",
                               cortex_name: str = "",
                               embedding: list[float] | None = None) -> HubShard:
        """Deterministic shard lookup or creation.

        Creates a shard from (cortex_name, name, hub_level). If the shard
        already exists and has children (was previously split), routes to
        the best-matching child shard based on embedding similarity.
        """
        shard = HubShard.create(name=name, hub_level=hub_level,
                                cortex_name=cortex_name,
                                max_degree=self.max_degree_per_shard)

        if shard.id in self.shards:
            existing = self.shards[shard.id]
            existing.max_degree = self.max_degree_per_shard
            # Route to child if shard was previously split
            if existing.child_shard_ids and embedding:
                child = self._route_to_best_child(existing, embedding)
                if child:
                    return child
            return existing

        self.shards[shard.id] = shard
        return shard

    def _route_to_best_child(self, parent: HubShard,
                              embedding: list[float]) -> HubShard | None:
        """Find the child shard whose centroid is closest to the embedding."""
        children = [self.shards[cid] for cid in parent.child_shard_ids
                    if cid in self.shards]
        if not children:
            return None

        # Ensure centroids are fresh
        for c in children:
            if c._centroid_stale:
                c.recompute_centroid(self.hubs)

        best_child = None
        best_sim = -1.0
        for c in children:
            if c.centroid:
                sim = self._cosine_sim(embedding, c.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_child = c

        # Fallback: pick the child with the fewest hubs
        if best_child is None and children:
            best_child = min(children, key=lambda c: c.degree)

        return best_child

    def _check_and_split(self, shard: HubShard) -> bool:
        """Check if a shard is full and auto-split if needed.

        Returns True if a split was performed.
        """
        if shard.is_full:
            self._auto_split(shard)
            return True
        return False

    def _auto_split(self, shard: HubShard) -> list[HubShard]:
        """Split a shard into child shards by clustering hub embeddings.

        Each child receives a roughly equal share of hubs, determined by
        embedding similarity. The parent shard becomes a routing node.
        Returns the list of newly created child shards.
        """
        hubs = [self.hubs[hid] for hid in shard.hub_ids if hid in self.hubs]
        if len(hubs) < 3:
            return []

        k = min(3, shard.max_degree // 16, len(hubs))
        k = max(2, k)  # at least 2 clusters
        clusters = self._detect_clusters(hubs, k)

        child_shards = []
        for i, cluster in enumerate(clusters):
            child_name = f"{shard.name}_c{i}"
            child = HubShard.create(
                name=child_name,
                hub_level=shard.hub_level,
                cortex_name=shard.cortex_name,
                max_degree=shard.max_degree,
                parent_shard_id=shard.id,
            )
            # If child already exists (deterministic ID collision), reuse it
            if child.id in self.shards:
                child = self.shards[child.id]
            else:
                self.shards[child.id] = child

            for hub in cluster:
                hub.shard_id = child.id
                child.hub_ids.append(hub.id)

            child.recompute_centroid(self.hubs)
            if child.id not in shard.child_shard_ids:
                shard.child_shard_ids.append(child.id)
            child_shards.append(child)

        # Clear parent hub_ids — parent is now a routing node
        shard.hub_ids.clear()
        shard.last_split_at = time.time()
        shard.split_count += 1

        return child_shards

    def _detect_clusters(self, hubs: list[HubNode],
                          k: int) -> list[list[HubNode]]:
        """Cluster hubs into k groups using iterative k-means on embeddings.

        Hubs without embeddings are distributed evenly across clusters.
        """
        if k <= 1 or len(hubs) <= k:
            return [list(hubs)]

        valid_hubs = [h for h in hubs if h.embedding]
        if len(valid_hubs) < k:
            # Not enough embedding data; split evenly
            n = len(hubs)
            size = max(1, n // k)
            result = []
            for i in range(0, k):
                chunk = hubs[size * i: size * (i + 1) if i < k - 1 else n]
                if chunk:
                    result.append(chunk)
            if not result:
                result.append(list(hubs))
            return result

        dim = len(valid_hubs[0].embedding)
        rng = random.Random(42)

        # Initialize centroids via k-means++ (first centroid randomly, rest weighted by distance)
        indices = list(range(len(valid_hubs)))
        first = rng.choice(indices)
        centroids: list[list[float]] = [valid_hubs[first].embedding[:]]
        centroid_indices = [first]

        for _ in range(k - 1):
            # Compute distances from each point to nearest centroid
            distances = []
            for i, h in enumerate(valid_hubs):
                if i in centroid_indices:
                    distances.append(0.0)
                    continue
                d = min(self._cosine_dist(h.embedding, c) for c in centroids)
                distances.append(max(d, 1e-8))
            total = sum(distances)
            if total <= 0:
                break
            # Weighted random selection
            r = rng.random() * total
            cumsum = 0.0
            chosen = 0
            for i, d in enumerate(distances):
                cumsum += d
                if cumsum >= r:
                    chosen = i
                    break
            centroids.append(valid_hubs[chosen].embedding[:])
            centroid_indices.append(chosen)

        # Pad centroids to k if needed
        while len(centroids) < k:
            # Pick random hub not yet used as centroid
            available = [i for i in indices if i not in centroid_indices]
            if available:
                chosen = available[0]
                centroids.append(valid_hubs[chosen].embedding[:])
                centroid_indices.append(chosen)
            else:
                # Duplicate an existing centroid with slight perturbation
                base = centroids[rng.randint(0, len(centroids) - 1)]
                perturbed = [v + rng.uniform(-0.001, 0.001) for v in base]
                centroids.append(perturbed)

        # Iterative refinement
        clusters: list[list[HubNode]] = [[] for _ in range(k)]
        for _ in range(30):
            # Assign points to nearest centroid
            clusters = [[] for _ in range(k)]
            for h in valid_hubs:
                best_c = 0
                best_sim = -2.0
                for ci, centroid in enumerate(centroids):
                    sim = self._cosine_sim(h.embedding, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_c = ci
                clusters[best_c].append(h)

            # Recompute centroids
            new_centroids = []
            for idx, cluster in enumerate(clusters):
                if cluster:
                    avg = [0.0] * dim
                    for h in cluster:
                        emb = h.embedding
                        for i in range(dim):
                            avg[i] += emb[i]
                    inv = 1.0 / len(cluster)
                    for i in range(dim):
                        avg[i] *= inv
                    new_centroids.append(avg)
                else:
                    # Empty cluster: keep previous centroid
                    new_centroids.append(centroids[idx][:])

            # Check convergence
            delta = sum(
                self._cosine_dist(centroids[i], new_centroids[i])
                for i in range(len(centroids))
            )
            centroids = new_centroids
            if delta < 0.001:
                break

        # Assign hubs without embeddings evenly across clusters
        unassigned = [h for h in hubs if h not in valid_hubs]
        for i, h in enumerate(unassigned):
            clusters[i % k].append(h)

        # Filter out empty clusters
        return [c for c in clusters if c]

    def route_to_shard(self, query_embedding: list[float],
                        top_k: int = 2) -> list[HubShard]:
        """Route a query embedding to the most relevant shards.

        Uses cosine similarity between the query embedding and each
        leaf shard's centroid. Returns top_k best-matching shards.
        """
        leaf_shards = [s for s in self.shards.values()
                       if s.is_leaf_shard and s.centroid]
        if not leaf_shards:
            return []

        scored = []
        for s in leaf_shards:
            sim = self._cosine_sim(query_embedding, s.centroid)
            scored.append((s, sim))

        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:top_k]]

    def expand_to_sibling_shards(self, shard: HubShard) -> list[HubShard]:
        """Get sibling shards that share the same parent."""
        if not shard.parent_shard_id:
            return []
        parent = self.shards.get(shard.parent_shard_id)
        if not parent:
            return []
        return [self.shards[cid] for cid in parent.child_shard_ids
                if cid in self.shards and cid != shard.id]

    def get_hubs_for_shard(self, shard_id: str) -> list[HubNode]:
        """Get all hub nodes belonging to a specific shard."""
        shard = self.shards.get(shard_id)
        if not shard:
            return []
        return [self.hubs[hid] for hid in shard.hub_ids if hid in self.hubs]

    def search_with_shard_routing(self, query_embedding: list[float],
                                   max_hops: int = 2,
                                   max_results: int = 10) -> list[HubNode]:
        """Combine shard routing with hub traversal for efficient search.

        1. Route query embedding to top-k shards by centroid similarity
        2. Collect hubs from matched shards
        3. Expand via hub-to-hub traversal from top-matching hubs
        4. Return results sorted by importance

        This provides O(log N) indexing with fallback multi-hop reasoning.
        """
        top_k = max(2, min(5, self.max_degree_per_shard // 10))
        shards = self.route_to_shard(query_embedding, top_k=top_k)

        if not shards:
            return []

        # Collect direct shard hits
        hub_map: dict[str, HubNode] = {}
        for shard in shards:
            for hid in shard.hub_ids:
                if hid in self.hubs:
                    hub_map[hid] = self.hubs[hid]

        # Expand via traversal from top-matching hubs
        start_hubs = list(hub_map.keys())[:3]
        shard_id_set = {s.id for s in shards}
        for hid in start_hubs:
            traversed = self.traverse_hubs(
                hid,
                max_hops=max_hops,
                max_results=max_results,
            )
            for h in traversed:
                if h.id not in hub_map:
                    hub_map[h.id] = h

        # Sort by importance (weighted: importance 0.7 + stability 0.3)
        def _score(h: HubNode) -> float:
            return 0.7 * h.importance + 0.3 * h.stability

        sorted_results = sorted(hub_map.values(), key=_score, reverse=True)
        return sorted_results[:max_results]

    # ── Queries ──────────────────────────────────────────

    def get_hubs_for_cortex(self, cortex_name: str,
                            level: str | None = None) -> list[HubNode]:
        """Get all hubs for a cortex, optionally filtered by level."""
        ids = self._cortex_index.get(cortex_name, [])
        hubs = [self.hubs[hid] for hid in ids if hid in self.hubs]
        if level:
            hubs = [h for h in hubs if h.hub_level == level]
        return sorted(hubs, key=lambda h: -h.importance)

    def get_parent_chain(self, hub_id: str) -> list[HubNode]:
        """Get the chain of parent hubs from leaf -> global."""
        chain: list[HubNode] = []
        current = self.hubs.get(hub_id)
        while current:
            chain.append(current)
            current = self.hubs.get(current.parent_hub_id) if current.parent_hub_id else None
        return chain

    def get_cross_references(self, hub_id: str) -> list[HubNode]:
        """Get all hubs in OTHER cortices that this hub references."""
        hub = self.hubs.get(hub_id)
        if not hub:
            return []
        return [self.hubs[rid] for rid in hub.cross_ref_hubs
                if rid in self.hubs]

    # ── Internal ─────────────────────────────────────────

    def _add(self, hub: HubNode, cortex_name: str):
        self.hubs[hub.id] = hub
        if cortex_name not in self._cortex_index:
            self._cortex_index[cortex_name] = []
        if hub.id not in self._cortex_index[cortex_name]:
            self._cortex_index[cortex_name].append(hub.id)

    # ── Math helpers ─────────────────────────────────────

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a))
        nb = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb + 1e-8)

    @staticmethod
    def _cosine_dist(a: list[float], b: list[float]) -> float:
        return 1.0 - HubLayer._cosine_sim(a, b)

    # ── Health ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        levels = {"leaf": 0, "domain": 0, "global": 0}
        for hub in self.hubs.values():
            if hub.hub_level in levels:
                levels[hub.hub_level] += 1
        edge_types: dict[str, int] = {}
        for e in self.edges.values():
            edge_types[e.edge_type] = edge_types.get(e.edge_type, 0) + 1

        # Shard stats
        total_shards = len(self.shards)
        leaf_shards = sum(1 for s in self.shards.values() if s.is_leaf_shard)
        split_shards = sum(1 for s in self.shards.values() if s.split_count > 0)
        full_shards = sum(1 for s in self.shards.values() if s.is_full)
        cross_shard_bridges = sum(
            len(neighbors) for neighbors in self._shard_adjacency.values()
        ) // 2

        return {
            "total_hubs": len(self.hubs),
            "by_level": levels,
            "total_edges": len(self.edges),
            "edge_types": edge_types,
            "cortices_indexed": len(self._cortex_index),
            "cross_cortex_bridges": sum(
                len(h.cross_ref_hubs) for h in self.hubs.values()) // 2,
            "shards": {
                "total": total_shards,
                "leaf": leaf_shards,
                "split": split_shards,
                "full": full_shards,
            },
            "cross_shard_bridges": cross_shard_bridges,
        }
