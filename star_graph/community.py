"""Community Detection — label-propagation-based graph partitioning.

Detects communities in the star graph via weighted label propagation,
computes centroids, identifies bridge nodes, and tracks community health.

Pattern: dataclass + engine class + numpy-based computation (follows abstraction.py).

v0.6 — community-aware retrieval foundation.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .anchor import Anchor
from .config import Config


@dataclass
class Community:
    """A detected community of related anchors.

    Communities are discovered via label propagation on the graph adjacency.
    Bridge nodes (anchors connecting multiple communities) are tracked via
    secondary_community_ids on the Anchor itself.
    """

    id: str
    anchor_ids: list[str]
    centroid_embedding: list[float] = field(default_factory=list)
    topic_label: str = ""
    size: int = 0
    density: float = 0.0                # actual_edges / possible_edges
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    primary_tag: str = ""

    @property
    def is_dense(self) -> bool:
        return self.density > 0.3

    @property
    def is_large(self) -> bool:
        return self.size > 100


@dataclass
class CommunityHealth:
    """Health metrics for the community structure of a graph."""

    modularity: float = 0.0
    num_communities: int = 0
    size_distribution: dict[str, int] = field(default_factory=dict)
    avg_degree_by_community: dict[str, float] = field(default_factory=dict)
    bridge_node_count: int = 0
    singletons: int = 0
    largest_community_fraction: float = 0.0

    @property
    def is_healthy(self) -> bool:
        """Communities are healthy if modularity is positive and no single
        community dominates."""
        return (
            self.modularity > 0.1
            and self.largest_community_fraction < 0.7
            and self.num_communities >= 2
        )

    def summary(self) -> str:
        return (
            f"Communities: {self.num_communities} | "
            f"Modularity: {self.modularity:.3f} | "
            f"Bridges: {self.bridge_node_count} | "
            f"Singletons: {self.singletons} | "
            f"Largest: {self.largest_community_fraction:.1%}"
        )


class CommunityDetection:
    """Label-propagation community detection for star graphs.

    Uses weighted label propagation (edge weights from graph.edges) to
    partition the graph into communities. Supports splitting oversized
    communities, filtering small ones, bridge detection, and modularity
    computation.

    Usage:
        detector = CommunityDetection()
        communities = detector.detect(graph)
        health = detector.health_metrics()
    """

    def __init__(self,
                 max_community_size: int | None = None,
                 min_community_size: int | None = None,
                 max_iterations: int | None = None):
        c = Config.get().community if hasattr(Config.get(), 'community') else None
        self.max_community_size = (
            max_community_size
            if max_community_size is not None
            else getattr(c, 'max_community_size', 200) if c else 200
        )
        self.min_size = (
            min_community_size
            if min_community_size is not None
            else getattr(c, 'min_community_size', 3) if c else 3
        )
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else getattr(c, 'max_iterations', 50) if c else 50
        )
        self.communities: list[Community] = []
        self._graph = None          # reference to last-used graph (for health_metrics)
        self._labels: dict[str, str] = {}  # anchor_id -> community_id
        self._counter = 0

    # ── Main pipeline ──────────────────────────────────────

    def detect(self, graph) -> list[Community]:
        """Run full community detection pipeline on a graph.

        Pipeline: label propagation -> split oversized -> filter small
        -> compute centroids -> detect bridges -> build Community objects.
        """
        self._graph = graph

        if not graph.anchors:
            self.communities = []
            self._labels = {}
            return []

        # 1. Label propagation
        labels = self._label_propagation(graph)

        # 2. Split oversized communities
        labels = self._split_oversized(labels, graph)

        # 3. Filter / reassign small communities
        labels = self._filter_small(labels, graph)

        self._labels = dict(labels)

        # 4. Compute centroids
        centroids = self._compute_centroids(labels, graph.anchors)

        # 5. Detect bridge nodes
        bridges = self._detect_bridge_nodes(labels, graph)

        # 6. Build Community objects
        community_map: dict[str, list[str]] = defaultdict(list)
        for aid, cid in labels.items():
            community_map[cid].append(aid)

        now = time.time()
        communities: list[Community] = []

        for cid, member_ids in sorted(community_map.items()):
            anchor_list = [graph.anchors[aid] for aid in member_ids
                           if aid in graph.anchors]
            centroid = centroids.get(cid, [])
            topic_label = self._generate_topic_label(anchor_list)
            size = len(member_ids)

            # Compute density: actual_edges / possible_edges
            member_set = set(member_ids)
            actual_edges = 0
            for aid in member_ids:
                for neighbor in graph._adjacency.get(aid, set()):
                    if neighbor in member_set and aid < neighbor:
                        actual_edges += 1
            possible_edges = size * (size - 1) / 2.0 if size > 1 else 1.0
            density = actual_edges / possible_edges if possible_edges > 0 else 0.0

            # Primary tag
            all_tags: list[str] = []
            for a in anchor_list:
                all_tags.extend(a.tags)
            tag_counter = Counter(all_tags)
            primary_tag = tag_counter.most_common(1)[0][0] if tag_counter else ""

            community = Community(
                id=cid,
                anchor_ids=member_ids,
                centroid_embedding=centroid,
                topic_label=topic_label,
                size=size,
                density=density,
                created_at=now,
                last_updated=now,
                primary_tag=primary_tag,
            )
            communities.append(community)

        # Update anchor community fields
        for aid, cid in labels.items():
            if aid in graph.anchors:
                graph.anchors[aid].community_id = cid

        # Update bridge info on anchors
        for aid, other_cids in bridges.items():
            if aid in graph.anchors:
                graph.anchors[aid].secondary_community_ids = list(other_cids)

        self.communities = communities
        return communities

    # ── Label propagation ──────────────────────────────────

    def _label_propagation(self, graph) -> dict[str, str]:
        """Iterative weighted label propagation.

        Each node adopts the most common label among its neighbors,
        weighted by edge weights. Converges when no label changes,
        or after max_iterations.
        """
        anchor_ids = list(graph.anchors.keys())
        if not anchor_ids:
            return {}

        # Initialize: each node has its own ID as label
        labels = {aid: aid for aid in anchor_ids}

        prev_labels: dict[str, str] = {}

        for iteration in range(self.max_iterations):
            # Check convergence
            if labels == prev_labels and iteration > 0:
                break
            prev_labels = dict(labels)

            changed = False
            # Process nodes in consistent order (sorted for determinism)
            for node_id in sorted(anchor_ids):
                label_weights: dict[str, float] = defaultdict(float)

                for neighbor_id in graph._adjacency.get(node_id, set()):
                    edge_key = graph._key(node_id, neighbor_id)
                    edge = graph.edges.get(edge_key)
                    weight = edge.weight if edge else 0.5
                    neighbor_label = labels.get(neighbor_id)
                    if neighbor_label:
                        label_weights[neighbor_label] += weight

                if not label_weights:
                    continue

                # Pick the most common (heaviest) label
                best_label = max(label_weights, key=lambda k: label_weights[k])

                if labels[node_id] != best_label:
                    labels[node_id] = best_label
                    changed = True

            if not changed:
                break

        return labels

    # ── Split oversized communities ────────────────────────

    def _split_oversized(self, labels: dict[str, str],
                         graph, max_size: int | None = None) -> dict[str, str]:
        """Recursively split communities larger than max_community_size.

        For each oversized community, runs sub-label-propagation on just
        that community's subgraph to subdivide it further.
        """
        if max_size is None:
            max_size = self.max_community_size

        result = dict(labels)
        split_idx = 0

        # Prevent infinite loops: allow at most 5 rounds of splitting
        for _round in range(5):
            size_counter = Counter(result.values())
            oversized = [(cid, sz) for cid, sz in size_counter.items()
                         if sz > max_size]
            if not oversized:
                break

            # Process the largest oversized community first
            cid, _ = oversized[0]
            members = [aid for aid, c in result.items() if c == cid]

            if len(members) <= max_size:
                break

            member_set = set(members)

            # Sub-label propagation on just this community
            sub_labels: dict[str, str] = {aid: aid for aid in members}
            sub_prev: dict[str, str] = {}

            for _ in range(min(20, self.max_iterations)):
                if sub_labels == sub_prev:
                    break
                sub_prev = dict(sub_labels)
                sub_changed = False

                for aid in sorted(members):
                    label_weights: dict[str, float] = defaultdict(float)
                    for neighbor in graph._adjacency.get(aid, set()):
                        if neighbor not in member_set:
                            continue
                        edge_key = graph._key(aid, neighbor)
                        edge = graph.edges.get(edge_key)
                        weight = edge.weight if edge else 0.5
                        nl = sub_labels.get(neighbor)
                        if nl:
                            label_weights[nl] += weight
                    if label_weights:
                        best = max(label_weights, key=lambda k: label_weights[k])
                        if sub_labels[aid] != best:
                            sub_labels[aid] = best
                            sub_changed = True

                if not sub_changed:
                    break

            # Group sub-results
            groups: dict[str, list[str]] = defaultdict(list)
            for aid in members:
                groups[sub_labels[aid]].append(aid)

            if len(groups) <= 1:
                # Cannot split further
                break

            # Keep the largest subgroup with original label,
            # assign new labels to remaining subgroups
            sorted_groups = sorted(groups.values(), key=len, reverse=True)
            for group in sorted_groups[1:]:
                # Only split if the subgroup is large enough to be a community
                if len(group) >= self.min_size:
                    new_cid = f"{cid}_sp{split_idx}"
                    split_idx += 1
                    for aid in group:
                        result[aid] = new_cid

        return result

    # ── Filter small communities ───────────────────────────

    def _filter_small(self, labels: dict[str, str],
                      graph, min_size: int | None = None) -> dict[str, str]:
        """Reassign members of communities smaller than min_size.

        Each node in a small community is reassigned to its most common
        neighbor's community (weighted by edge weight). Truly isolated
        nodes keep their original label.
        """
        if min_size is None:
            min_size = self.min_size

        result = dict(labels)

        for _pass_idx in range(5):
            size_counter = Counter(result.values())
            small_cids = [cid for cid, sz in size_counter.items()
                          if sz < min_size]
            if not small_cids:
                break

            for cid in small_cids:
                members = [aid for aid, c in result.items() if c == cid]
                for aid in members:
                    label_weights: dict[str, float] = defaultdict(float)
                    for neighbor in graph._adjacency.get(aid, set()):
                        nl = result.get(neighbor)
                        if nl and nl != cid:
                            edge_key = graph._key(aid, neighbor)
                            edge = graph.edges.get(edge_key)
                            weight = edge.weight if edge else 0.5
                            label_weights[nl] += weight

                    if label_weights:
                        best = max(label_weights, key=lambda k: label_weights[k])
                        result[aid] = best
                    else:
                        # Isolated node: merge into largest community
                        other_cids = [(c, s) for c, s in size_counter.items()
                                      if c != cid]
                        if other_cids:
                            largest_other = max(other_cids, key=lambda x: x[1])
                            result[aid] = largest_other[0]

        return result

    # ── Centroids ──────────────────────────────────────────

    def _compute_centroids(self, labels: dict[str, str],
                           anchors: dict[str, Anchor]) -> dict[str, list[float]]:
        """Compute mean embedding per community."""
        groups: dict[str, list[list[float]]] = defaultdict(list)
        for aid, cid in labels.items():
            anchor = anchors.get(aid)
            if anchor and anchor.embedding:
                groups[cid].append(anchor.embedding)

        centroids: dict[str, list[float]] = {}
        for cid, embeddings in groups.items():
            if embeddings:
                centroids[cid] = np.mean(np.array(embeddings), axis=0).tolist()
            else:
                centroids[cid] = []

        return centroids

    # ── Topic label generation ─────────────────────────────

    def _generate_topic_label(self, community_anchors: list[Anchor]) -> str:
        """Generate a human-readable topic label from a community's anchors.

        Strategy: most common tags first, then top keywords from text.
        """
        if not community_anchors:
            return "Empty Community"

        # 1. Most common tags
        all_tags: list[str] = []
        for a in community_anchors:
            all_tags.extend(a.tags)
        tag_counter = Counter(all_tags)
        top_tags = [tag for tag, _ in tag_counter.most_common(3)]
        if top_tags:
            return " / ".join(top_tags)

        # 2. Top keywords from text
        all_words: list[str] = []
        for a in community_anchors:
            words = [w.lower() for w in a.text.split() if len(w) > 3]
            all_words.extend(words)

        # Filter common stopwords
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'been', 'were',
            'they', 'their', 'about', 'which', 'when', 'what', 'them',
            'then', 'than', 'some', 'these', 'those', 'would', 'could',
            'should', 'there', 'into', 'over', 'after', 'before',
        }
        word_counter = Counter(w for w in all_words if w not in stopwords)
        top_words = [w for w, _ in word_counter.most_common(5)]

        if top_words:
            return " ".join(top_words[:3]).title()

        return f"Community ({len(community_anchors)} nodes)"

    # ── Bridge detection ───────────────────────────────────

    def _detect_bridge_nodes(self, labels: dict[str, str],
                             graph) -> dict[str, set[str]]:
        """Find anchors that connect to nodes in other communities.

        Returns: {anchor_id: {other_community_ids}}
        """
        bridges: dict[str, set[str]] = {}
        for aid, cid in labels.items():
            other_communities: set[str] = set()
            for neighbor in graph._adjacency.get(aid, set()):
                nl = labels.get(neighbor)
                if nl and nl != cid:
                    other_communities.add(nl)
            if other_communities:
                bridges[aid] = other_communities

        return bridges

    # ── Modularity ─────────────────────────────────────────

    def _modularity(self, graph, labels: dict[str, str]) -> float:
        """Newman-Girvan weighted modularity.

        Q = (1/2m) * sum_ij [w_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
        """
        if not graph.edges:
            return 0.0

        total_weight = sum(e.weight for e in graph.edges.values())
        if total_weight == 0:
            return 0.0

        # Weighted degrees
        degrees: dict[str, float] = defaultdict(float)
        for (a, b), edge in graph.edges.items():
            degrees[a] += edge.weight
            degrees[b] += edge.weight

        Q = 0.0
        for (a, b), edge in graph.edges.items():
            if labels.get(a) == labels.get(b):
                expected = degrees[a] * degrees[b] / (2.0 * total_weight)
                Q += (edge.weight - expected) / (2.0 * total_weight)

        return Q

    # ── Health metrics ─────────────────────────────────────

    def health_metrics(self) -> CommunityHealth:
        """Compute community health metrics from the last detection run."""
        if not self.communities or self._graph is None:
            return CommunityHealth()

        sizes = [c.size for c in self.communities]
        total_size = sum(sizes)

        # Size distribution buckets
        size_dist: dict[str, int] = {}
        for s in sizes:
            if s < 10:
                bucket = "1-9"
            elif s < 25:
                bucket = "10-24"
            elif s < 50:
                bucket = "25-49"
            elif s < 100:
                bucket = "50-99"
            elif s < 200:
                bucket = "100-199"
            else:
                bucket = "200+"
            size_dist[bucket] = size_dist.get(bucket, 0) + 1

        # Bridge nodes
        bridge_count = sum(
            1 for a in self._graph.anchors.values()
            if a.secondary_community_ids
        )

        # Singletons
        singletons = sum(1 for c in self.communities if c.size == 1)

        # Largest community fraction
        largest_frac = max(sizes) / total_size if total_size > 0 else 0.0

        # Average degree by community
        avg_degree_by_comm: dict[str, float] = {}
        for c in self.communities:
            degrees: list[int] = []
            for aid in c.anchor_ids:
                degree = len(self._graph._adjacency.get(aid, set()))
                degrees.append(degree)
            avg_degree_by_comm[c.id] = (
                sum(degrees) / len(degrees) if degrees else 0.0
            )

        # Modularity
        modularity = self._modularity(self._graph, self._labels)

        return CommunityHealth(
            modularity=modularity,
            num_communities=len(self.communities),
            size_distribution=size_dist,
            avg_degree_by_community=avg_degree_by_comm,
            bridge_node_count=bridge_count,
            singletons=singletons,
            largest_community_fraction=largest_frac,
        )

    # ── Incremental update ─────────────────────────────────

    def refresh(self, graph) -> list[Community]:
        """Re-run detection and match new communities to existing ones.

        Preserves community identity across runs by matching old and new
        communities via Jaccard overlap of anchor membership.
        """
        old_communities = {c.id: c for c in self.communities}
        new_communities = self.detect(graph)

        if not old_communities or not new_communities:
            return new_communities

        # Match new communities to old by anchor overlap
        for new_c in new_communities:
            new_set = set(new_c.anchor_ids)
            best_overlap = 0.0
            best_old: Community | None = None
            for old_id, old_c in old_communities.items():
                old_set = set(old_c.anchor_ids)
                intersection = len(new_set & old_set)
                union = len(new_set | old_set)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard > best_overlap and jaccard > 0.4:
                    best_overlap = jaccard
                    best_old = old_c
            if best_old is not None:
                new_c.created_at = best_old.created_at
                new_c.id = best_old.id  # preserve identity

        return new_communities

    # ── Community neighbors ────────────────────────────────

    def get_neighboring_communities(self, community_id: str,
                                    max_neighbors: int = 3
                                    ) -> list[tuple[str, float]]:
        """Find communities connected by edges to the given community.

        Returns list of (community_id, total_edge_weight) sorted descending.
        """
        if not self.communities or self._graph is None:
            return []

        # Find the community's anchor set
        comm_anchors: set[str] = set()
        for c in self.communities:
            if c.id == community_id:
                comm_anchors = set(c.anchor_ids)
                break

        if not comm_anchors:
            return []

        # Aggregate edge weights to other communities
        neighbor_weights: dict[str, float] = defaultdict(float)
        for aid in comm_anchors:
            for neighbor in self._graph._adjacency.get(aid, set()):
                if neighbor in comm_anchors:
                    continue
                neighbor_anchor = self._graph.anchors.get(neighbor)
                if neighbor_anchor is None:
                    continue
                nc = neighbor_anchor.community_id
                if not nc or nc == community_id:
                    continue
                edge_key = self._graph._key(aid, neighbor)
                edge = self._graph.edges.get(edge_key)
                weight = edge.weight if edge else 0.5
                neighbor_weights[nc] += weight

        sorted_neighbors = sorted(neighbor_weights.items(),
                                  key=lambda x: -x[1])
        return sorted_neighbors[:max_neighbors]
