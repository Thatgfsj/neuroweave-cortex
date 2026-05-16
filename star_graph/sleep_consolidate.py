"""Post-sleep consolidation mixin — merge/fuse nodes, rewire the graph, bridge
distant constellations, prune anchors/edges, cross-cortex hub linking, thermal
forgetting, and index rebuild.

Provides the structural reorganisation that follows NREM/REM replay phases.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict


class SleepConsolidate:
    """Mixin: post-replay graph consolidation and maintenance routines."""

    # ── Phase 4c: Sleep Rebuild ─────────────────────────

    def _sleep_rebuild(self) -> dict:
        """Restructure the entire graph — not just compress, but rebuild.

        Four-step process:
        1. Multi-node fusion: cluster 3+ related anchors → single fused anchor
        2. Graph rewiring: drop dead edges, strengthen success paths, transitive closure
        3. Dynamic rewiring: RL-based edge updates from success/failure history
        4. Abstractive memory: concrete events → pattern memory with faster source decay

        Returns a dict with per-step metrics.
        """
        t0 = time.time()
        fused = self._rebuild_fuse_nodes()
        t1 = time.time()
        rewired = self._rebuild_rewire_graph()
        t2 = time.time()
        dynamic = self._dynamic_rewire()
        t3 = time.time()
        abstracted = self._rebuild_abstractive_memory()
        t4 = time.time()

        result = {
            "fused_nodes": fused,
            "rewired_edges": rewired,
            "dynamic_rewire": dynamic,
            "abstracted_patterns": abstracted,
            "fuse_ms": (t1 - t0) * 1000,
            "rewire_ms": (t2 - t1) * 1000,
            "dynamic_ms": (t3 - t2) * 1000,
            "abstract_ms": (t4 - t3) * 1000,
        }

        parts = []
        if fused:
            parts.append(f"fused {fused} node clusters")
        dropped = rewired.get("dropped", 0)
        strengthened = rewired.get("strengthened", 0)
        transitive = rewired.get("transitive_added", 0)
        if dropped or strengthened or transitive:
            parts.append(f"rewired edges (-{dropped} +{strengthened} +{transitive}t)")
        dyn_boosted = dynamic.get("boosted", 0)
        dyn_weakened = dynamic.get("weakened", 0)
        dyn_formed = dynamic.get("clusters_formed", 0)
        if dyn_boosted or dyn_weakened or dyn_formed:
            parts.append(f"dynamic (+{dyn_boosted} -{dyn_weakened} c{dyn_formed})")
        if abstracted:
            parts.append(f"abstracted {abstracted} patterns")
        if parts:
            self._log_event("Sleep Rebuild: " + ", ".join(parts))

        return result

    def _rebuild_fuse_nodes(self) -> int:
        """Multi-node fusion: cluster 3+ related anchors into a single abstraction.

        Uses community detection to find semantic clusters, then within each
        cluster runs hierarchical merging on embedding similarity. Anchors that
        are near-identical (cosine > 0.85) get fused into the oldest anchor.
        This goes beyond pair-wise merge — it handles chains like:
          "try-except in Python" + "python异常处理" + "错误捕获"
          → fused into "Python Error Handling"

        Returns number of nodes fused (removed from graph).
        """
        fused_count = 0
        threshold = getattr(self.cfg.sleep, 'rebuild_fuse_threshold', 0.85)
        min_cluster = getattr(self.cfg.sleep, 'rebuild_min_cluster', 3)

        # Step 1: detect communities to scope the fusion search
        try:
            from .community import CommunityDetection
            detector = CommunityDetection(min_community_size=min_cluster)
            communities = detector.detect(self.graph)
        except Exception:
            communities = []

        # If community detection fails, fall back to tag-based grouping
        if not communities:
            tag_groups: dict[str, list[str]] = {}
            for aid, a in self.graph.anchors.items():
                for tag in a.tags:
                    if tag not in tag_groups:
                        tag_groups[tag] = []
                    tag_groups[tag].append(aid)
            # Build pseudo-communities from tag groups with 3+ anchors
            communities = []
            for tag, ids in tag_groups.items():
                if len(ids) >= min_cluster:
                    from .community import Community
                    communities.append(Community(
                        id=f"tag_{tag}", anchor_ids=ids,
                        topic_label=tag, size=len(ids),
                    ))

        processed: set[str] = set()

        for community in communities:
            # Get anchors in this community that have embeddings
            community_anchors = {
                aid: self.graph.anchors[aid]
                for aid in community.anchor_ids
                if aid in self.graph.anchors
                and aid not in processed
                and self.graph.anchors[aid].embedding
            }
            if len(community_anchors) < min_cluster:
                continue

            # Step 2: compute pairwise cosine similarity within community
            ids = list(community_anchors.keys())
            n = len(ids)
            # Build adjacency of high-similarity pairs
            pairs: list[tuple[int, int, float]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    a = community_anchors[ids[i]]
                    b = community_anchors[ids[j]]
                    sim = self._embedding_similarity(a.embedding, b.embedding)
                    if sim > threshold:
                        pairs.append((i, j, sim))

            if not pairs:
                continue

            # Step 3: union-find to group transitively similar anchors
            parent = list(range(n))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            for i, j, _ in pairs:
                union(i, j)

            # Step 4: collect groups of 3+ anchors
            groups: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                groups[find(i)].append(i)

            for root, indices in groups.items():
                if len(indices) < min_cluster:
                    continue

                group_aids = [ids[i] for i in indices]
                group_anchors = [community_anchors[aid] for aid in group_aids]

                # Fuse into the oldest anchor (the "core")
                core = min(group_anchors, key=lambda a: a.created_at)
                others = [a for a in group_anchors if a.id != core.id]

                # Merge properties into core
                for other in others:
                    core.vector.importance = max(core.vector.importance, other.vector.importance)
                    core.vector.frequency = (core.vector.frequency + other.vector.frequency) / 2
                    core.vector.stability = min(1.0, core.vector.stability + 0.05)
                    core.vector.emotional_valence = (
                        core.vector.emotional_valence + other.vector.emotional_valence
                    ) / 2
                    core.tags = list(set(core.tags + other.tags))
                    # Fuse text: append key phrases from other
                    if len(core.text) < 500 and other.text not in core.text:
                        core.text = core.text + "; " + other.text[:200]

                    # Transfer edges from other → core
                    for neighbor in list(self.graph._adjacency.get(other.id, set())):
                        if neighbor == core.id:
                            continue
                        k = self.graph._key(other.id, neighbor)
                        old_edge = self.graph.edges.pop(k, None)
                        if old_edge:
                            nk = self.graph._key(core.id, neighbor)
                            if nk not in self.graph.edges:
                                new_edge = self._transfer_edge(old_edge, nk)
                                self.graph.edges[nk] = new_edge
                                self.graph._adjacency[core.id].add(neighbor)
                                self.graph._adjacency[neighbor].add(core.id)
                            else:
                                # Both core and other connected to same neighbor — reinforce
                                existing = self.graph.edges[nk]
                                existing.strengthen(0.03)

                    # Remove other from adjacency
                    if other.id in self.graph._adjacency:
                        for neighbor in list(self.graph._adjacency[other.id]):
                            self.graph._adjacency[neighbor].discard(other.id)
                        del self.graph._adjacency[other.id]

                    self.graph.remove_anchor(other.id)
                    processed.add(other.id)

                # Mark core as abstractive-fused
                core.tags.append("sleep_rebuilt")
                core.vector.stability = min(1.0, core.vector.stability + 0.1)
                processed.add(core.id)
                fused_count += len(others)

        return fused_count

    def _rebuild_rewire_graph(self) -> dict:
        """Active graph rewiring: drop dead edges, strengthen success paths,
        add transitive closure edges for strong two-hop paths.

        Three operations:
        1. Drop: edges with weight < 0.05 AND co_activation_count == 0
        2. Strengthen: edges with high co-activation (>=3) get a boost
        3. Transitive closure: if A→B and B→C are both strong (weight>0.6),
           create A→C with weight = w_ab * w_bc * 0.7

        Returns dict with counts: {dropped, strengthened, transitive_added}.
        """
        dropped = 0
        strengthened = 0
        transitive_added = 0
        drop_threshold = getattr(self.cfg.sleep, 'rewire_drop_threshold', 0.05)
        strengthen_coactivation = getattr(self.cfg.sleep, 'rewire_strengthen_min_coact', 3)
        transitive_min_weight = getattr(self.cfg.sleep, 'rewire_transitive_min_weight', 0.6)
        transitive_decay = getattr(self.cfg.sleep, 'rewire_transitive_decay', 0.7)

        # ── Step 1: Drop dead edges ──
        dead_keys = []
        for key, edge in self.graph.edges.items():
            if edge.weight < drop_threshold and edge.co_activation_count == 0:
                dead_keys.append(key)

        for key in dead_keys:
            a, b = key
            self.graph._adjacency[a].discard(b)
            self.graph._adjacency[b].discard(a)
            del self.graph.edges[key]
        dropped = len(dead_keys)

        # ── Step 2: Strengthen high-utility edges ──
        for edge in self.graph.edges.values():
            if edge.co_activation_count >= strengthen_coactivation:
                boost = min(0.15, edge.co_activation_count * 0.02)
                edge.strengthen(boost)
                strengthened += 1

        # ── Step 3: Transitive closure for strong two-hop paths ──
        new_edges: list[tuple[str, str, float]] = []
        for a_id in list(self.graph._adjacency.keys()):
            a_neighbors = self.graph._adjacency.get(a_id, set())
            for b_id in a_neighbors:
                ab_key = self.graph._key(a_id, b_id)
                ab_edge = self.graph.edges.get(ab_key)
                if not ab_edge or ab_edge.weight < transitive_min_weight:
                    continue
                b_neighbors = self.graph._adjacency.get(b_id, set())
                for c_id in b_neighbors:
                    if c_id == a_id:
                        continue
                    bc_key = self.graph._key(b_id, c_id)
                    bc_edge = self.graph.edges.get(bc_key)
                    if not bc_edge or bc_edge.weight < transitive_min_weight:
                        continue
                    # A→B strong, B→C strong — check if A→C already exists
                    ac_key = self.graph._key(a_id, c_id)
                    if ac_key in self.graph.edges:
                        continue  # already connected
                    # Don't create self-loops
                    if a_id == c_id:
                        continue
                    transitive_weight = ab_edge.weight * bc_edge.weight * transitive_decay
                    if transitive_weight > 0.15:  # minimum useful weight
                        new_edges.append((a_id, c_id, transitive_weight))

        # Deduplicate and add
        seen = set()
        for a_id, c_id, w in new_edges:
            key = self.graph._key(a_id, c_id)
            if key not in seen and key not in self.graph.edges:
                seen.add(key)
                self.graph.add_edge(a_id, c_id, weight=w, edge_type="topical",
                                    relation="transitive_closure", source_type="inferred")
                transitive_added += 1

        return {
            "dropped": dropped,
            "strengthened": strengthened,
            "transitive_added": transitive_added,
        }

    def _dynamic_rewire(self) -> dict:
        """RL-based dynamic neural rewiring using success/failure history on edges.

        Three operations:
        1. Boost: edges with high success_rate (>=0.7) get strengthened
        2. Weaken: edges with low success_rate (<0.3) with enough trials get weakened
        3. Cluster: high co-activation edges form community clusters

        This is the self-evolving mechanism — the graph learns which connections
        are useful and which are not through reinforcement.

        Returns dict with counts: {boosted, weakened, clusters_formed}.
        """
        boosted = 0
        weakened = 0
        clusters_formed = 0
        min_trials = getattr(self.cfg.sleep, 'rewire_strengthen_min_coact', 3)
        success_boost_threshold = 0.7
        failure_weaken_threshold = 0.3
        cluster_coact_threshold = getattr(self.cfg.sleep, 'dynamic_cluster_coact', 5)

        for key, edge in list(self.graph.edges.items()):
            total_trials = edge.success_count + edge.failure_count

            if total_trials >= min_trials:
                rate = edge.success_rate

                if rate >= success_boost_threshold and edge.success_count >= 2:
                    # Successful reasoning chain — strengthen
                    boost = min(0.1, edge.success_count * 0.015)
                    edge.strengthen(boost)
                    boosted += 1

                elif rate < failure_weaken_threshold and total_trials >= 5:
                    # Failed reasoning chain — significantly weaken
                    penalty = min(0.1, edge.failure_count * 0.01)
                    edge.weaken(penalty)
                    weakened += 1

            # Reset counters after processing (avoid unbounded accumulation)
            edge.success_count = max(0, edge.success_count - 1)  # gradual decay
            edge.failure_count = max(0, edge.failure_count - 1)

            # Form clusters from high co-activation edges
            if edge.co_activation_count >= cluster_coact_threshold:
                # Mark both anchors for community formation
                a, b = key
                if a in self.graph.anchors and b in self.graph.anchors:
                    anchor_a = self.graph.anchors[a]
                    anchor_b = self.graph.anchors[b]
                    # Tag with shared cluster label
                    cluster_tag = f"cluster_{a[:8]}_{b[:8]}"
                    if cluster_tag not in anchor_a.tags:
                        anchor_a.tags.append(cluster_tag)
                    if cluster_tag not in anchor_b.tags:
                        anchor_b.tags.append(cluster_tag)
                    if anchor_a.community_id and anchor_b.community_id:
                        # Both already in communities — bridge them
                        if anchor_a.community_id != anchor_b.community_id:
                            anchor_a.secondary_community_ids.append(anchor_b.community_id)
                            anchor_b.secondary_community_ids.append(anchor_a.community_id)
                    elif anchor_a.community_id:
                        anchor_b.community_id = anchor_a.community_id
                    elif anchor_b.community_id:
                        anchor_a.community_id = anchor_b.community_id
                    clusters_formed += 1

        return {
            "boosted": boosted,
            "weakened": weakened,
            "clusters_formed": clusters_formed,
        }

    def _rebuild_abstractive_memory(self) -> int:
        """Convert groups of concrete events into pattern memory.

        Finds anchors sharing the same schema_ref or tag group, generates a
        higher-level "pattern" anchor, and links concrete anchors to it with
        "instance_of" edges. The concrete anchors get their retention reduced
        (faster decay), while the pattern is stable.

        Example: "chromedriver fix failed on v124" + "chromedriver fix failed on v125"
                 → Pattern: "Browser Driver Version Compatibility Issue"

        Returns number of patterns created.
        """
        abstracted = 0
        min_group_size = getattr(self.cfg.sleep, 'abstractive_min_group', 4)
        decay_factor = getattr(self.cfg.sleep, 'abstractive_decay_factor', 0.6)

        # Step 1: group anchors by tag-based topics
        tag_groups: dict[str, list[str]] = defaultdict(list)
        for aid, a in self.graph.anchors.items():
            if a.state.name in ('DORMANT', 'CONSOLIDATING'):
                for tag in a.tags:
                    if tag not in ('dormant', 'consolidating', 'ghost', 'sleep_rebuilt'):
                        tag_groups[tag].append(aid)

        # Also group by schema_ref
        schema_groups: dict[str, list[str]] = defaultdict(list)
        for aid, a in self.graph.anchors.items():
            if a.schema_ref:
                schema_groups[a.schema_ref].append(aid)

        # Merge tag_groups and schema_groups into unified topic groups
        all_groups: dict[str, set[str]] = {}
        for tag, ids in tag_groups.items():
            if len(ids) >= min_group_size:
                all_groups[f"tag:{tag}"] = set(ids)
        for schema_ref, ids in schema_groups.items():
            if len(ids) >= min_group_size:
                key = f"schema:{schema_ref}"
                if key in all_groups:
                    all_groups[key] |= set(ids)
                else:
                    all_groups[key] = set(ids)

        for group_key, anchor_ids in all_groups.items():
            if len(anchor_ids) < min_group_size:
                continue

            # Get the actual anchors
            group_anchors = [
                self.graph.anchors[aid] for aid in anchor_ids
                if aid in self.graph.anchors and self.graph.anchors[aid].embedding
            ]
            if len(group_anchors) < min_group_size:
                continue

            # Compute centroid embedding
            dim = len(group_anchors[0].embedding)
            centroid = [0.0] * dim
            for a in group_anchors:
                for i in range(dim):
                    centroid[i] += a.embedding[i]
            for i in range(dim):
                centroid[i] /= len(group_anchors)

            # Generate pattern label from tags
            all_tags: list[str] = []
            for a in group_anchors:
                all_tags.extend(a.tags)
            tag_counts = Counter(all_tags)
            top_tags = [t for t, _ in tag_counts.most_common(3)
                       if t not in ('dormant', 'consolidating', 'ghost')]
            pattern_label = " + ".join(top_tags) if top_tags else group_key

            # Generate pattern description from the shortest and longest anchors
            sorted_by_len = sorted(group_anchors, key=lambda a: len(a.text))
            short_desc = sorted_by_len[0].text[:200]
            long_desc = sorted_by_len[-1].text[:300]

            # Create the pattern anchor
            pattern_id = f"pattern_{group_key.replace(':', '_')}_{self._cycle_count}"
            pattern = Anchor.create(
                text=f"[Pattern] {pattern_label}: {short_desc} ... {long_desc}"[:800],
                tags=top_tags + ["pattern", "abstractive_memory"],
                importance=sum(a.vector.importance for a in group_anchors) / len(group_anchors),
                emotional_valence=sum(a.vector.emotional_valence for a in group_anchors) / len(group_anchors),
            )
            pattern.embedding = centroid
            pattern.vector.stability = 0.8
            pattern.vector.recency = max(a.vector.recency for a in group_anchors)
            pattern.id = pattern_id
            pattern.schema_ref = group_key

            self.graph.add_anchor(pattern)

            # Link pattern to concrete anchors with "instance_of" edges
            for a in group_anchors:
                self.graph.add_edge(
                    a.id, pattern_id,
                    weight=0.7, edge_type="topical",
                    relation="instance_of", source_type="explicit",
                    confidence=0.8,
                )
                # Reduce retention on concrete anchors — pattern survives
                a.vector.stability *= decay_factor
                a.vector.importance *= decay_factor

            abstracted += 1

        # Step 2: cross-session pattern detection via AbstractiveMemoryEngine
        try:
            from .abstraction import AbstractiveMemoryEngine
            if not hasattr(self, '_abstractive_engine'):
                self._abstractive_engine = AbstractiveMemoryEngine(
                    min_occurrences=min_group_size,
                    similarity_threshold=0.75,
                )
            engine = self._abstractive_engine

            # Extract new cross-session patterns
            new_patterns = engine.extract_patterns(self.graph)
            if new_patterns:
                # Promote stable patterns to abstract nodes
                promoted = engine.promote_stable_patterns(self.graph)
                # Consolidate: match existing patterns against all anchors
                engine.consolidate_existing_patterns(self.graph)
                abstracted += len(promoted)
                if promoted:
                    self._log_event(
                        f"Abstractive Memory: promoted {len(promoted)} cross-session "
                        f"patterns (engine: {engine.stats['total_patterns']} total, "
                        f"{engine.stats['recurring']} recurring)"
                    )
        except ImportError:
            pass

        return abstracted

    # ── Phase 5: Merge Similar ──────────────────────────

    def _merge_similar(self, threshold: float) -> int:
        """Merge near-duplicate anchors using embedding similarity.

        Uses ANN-index pre-filtering to avoid O(n²) pair enumeration.
        Only checks anchor pairs within cosine-distance reach (top-k per anchor).
        Falls back to O(n²) text-overlap scan for anchors without embeddings.
        """
        merged = 0
        processed: set[str] = set()
        ids = list(self.graph.anchors.keys())
        n = len(ids)
        min_tag_overlap = getattr(self.cfg.sleep.merge, 'min_tag_overlap', 1)

        # Build candidate pairs via ANN pre-filter (O(n log n) instead of O(n²))
        candidate_pairs: set[tuple[str, str]] = set()
        ann = self.graph._get_ann_index()
        ann_k = min(20, max(5, n // 4))

        # Anchors with embeddings: use ANN for candidate discovery
        embed_ids = [aid for aid in ids if aid in self.graph.anchors
                     and self.graph.anchors[aid].embedding]
        no_embed_ids = [aid for aid in ids if aid in self.graph.anchors
                        and not self.graph.anchors[aid].embedding]

        for aid in embed_ids:
            anchor = self.graph.anchors.get(aid)
            if anchor is None or aid in processed:
                continue
            neighbors = ann.query(anchor.embedding, k=ann_k)
            for nid, sim in neighbors:
                if nid != aid and sim > threshold * 0.8:  # pre-filter: near threshold
                    key = (aid, nid) if aid < nid else (nid, aid)
                    candidate_pairs.add(key)

        # Anchors without embeddings: add text-overlap candidates against all others
        for i, aid_a in enumerate(no_embed_ids):
            if aid_a in processed:
                continue
            for aid_b in ids:
                if aid_b in processed or aid_b == aid_a:
                    continue
                # Only include if the other anchor has no embedding either,
                # or skip — embedding anchors are handled above via ANN
                b = self.graph.anchors.get(aid_b)
                if b and b.embedding:
                    continue  # was already covered by ANN scan
                key = (aid_a, aid_b) if aid_a < aid_b else (aid_b, aid_a)
                candidate_pairs.add(key)

        # Process candidate pairs
        for aid_a, aid_b in candidate_pairs:
            if aid_a in processed or aid_b in processed:
                continue
            a = self.graph.anchors.get(aid_a)
            b = self.graph.anchors.get(aid_b)
            if a is None or b is None:
                continue

            # Prefer embedding similarity, fall back to bigrams
            if a.embedding and b.embedding:
                overlap = self._embedding_similarity(a.embedding, b.embedding)
            else:
                overlap = self._text_overlap(a.text, b.text)

            # Gate: require tag overlap to prevent cross-topic cascade merging
            tag_overlap = len(set(a.tags) & set(b.tags))

            if overlap > threshold and tag_overlap >= min_tag_overlap:
                core, variant = (a, b) if a.created_at < b.created_at else (b, a)
                core.vector.importance = max(core.vector.importance, variant.vector.importance)
                core.vector.frequency = (core.vector.frequency + variant.vector.frequency) / 2
                core.vector.stability = min(1.0, core.vector.stability + self.cfg.sleep.merge.stability_boost)
                core.tags = list(set(core.tags + variant.tags))
                for neighbor in list(self.graph._adjacency.get(variant.id, set())):
                    k = self.graph._key(variant.id, neighbor)
                    old = self.graph.edges.pop(k, None)
                    if old:
                        nk = self.graph._key(core.id, neighbor)
                        if nk not in self.graph.edges:
                            new_edge = self._transfer_edge(old, nk)
                            self.graph.edges[nk] = new_edge
                            self.graph._adjacency[core.id].add(neighbor)
                            self.graph._adjacency[neighbor].add(core.id)
                self.graph.remove_anchor(variant.id)
                processed.add(variant.id)
                merged += 1

        if merged:
            self._log_event(f"Merge: fused {merged} duplicate anchor pairs")
        return merged

    # ── Phase 6: Adaptive Prune ─────────────────────────

    def _prune_anchors(self, threshold: float) -> list[str]:
        """Interference-aware pruning: contradiction penalties + ghosts."""
        contradictions = self.graph.find_contradictions()
        penalties: dict[str, float] = defaultdict(float)
        for aid_a, aid_b, _ in contradictions:
            a = self.graph.anchors.get(aid_a)
            b = self.graph.anchors.get(aid_b)
            if a and b:
                if a.retention_score < b.retention_score:
                    penalties[aid_a] += self.cfg.sleep.prune.contradiction_penalty
                else:
                    penalties[aid_b] += self.cfg.sleep.prune.contradiction_penalty

        candidates = []
        for aid in self.graph.get_prune_candidates(threshold):
            score = self.graph.anchors[aid].retention_score - penalties.get(aid, 0.0)
            if score < threshold:
                candidates.append(aid)

        self._ghost_count = 0
        for aid in candidates:
            if aid in self.graph.anchors:
                anchor = self.graph.anchors[aid]
                # Collect residual edges before removal
                residual_edges = {}
                for neighbor in self.graph._adjacency.get(aid, set()):
                    key = self.graph._key(aid, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        residual_edges[neighbor] = edge.weight * self.cfg.sleep.prune.residual_edge_factor

                # Create rich ghost via ghost subsystem
                self.graph._ghost_subsystem.create(anchor, residual_edges)

                anchor.transition('prune')
                self.graph.remove_anchor(aid)
                self._ghost_count += 1

        # Decay ghosts via subsystem
        stale_count, _ = self.graph._ghost_subsystem.decay_all()

        if candidates:
            self._log_event(f"Adaptive Prune: removed {len(candidates)} anchors "
                            f"({self._ghost_count} ghosts, {stale_count} stale ghosts cleared)")
        return candidates

    def _prune_edges(self, threshold: float) -> list[tuple[str, str]]:
        candidates = self.graph.get_dormant_edges(threshold)
        for key in candidates:
            if key in self.graph.edges:
                del self.graph.edges[key]
                a, b = key
                self.graph._adjacency[a].discard(b)
                self.graph._adjacency[b].discard(a)
        if candidates:
            self._log_event(f"Edge Prune: removed {len(candidates)} dormant edges")
        return candidates

    # ── Phase 7: Bridge Distant ─────────────────────────

    def _bridge_distant(self) -> int:
        from .resonance import Resonator
        resonator = Resonator(self.graph)

        visited: set[str] = set()
        constellations: list[Constellation] = []
        for aid in self.graph.anchors:
            if aid not in visited:
                c = self.graph.find_constellation(aid)
                if c.anchors:
                    constellations.append(c)
                    for a in c.anchors:
                        visited.add(a.id)

        bridges = 0
        for i, c_a in enumerate(constellations):
            for c_b in constellations[i + 1:]:
                score = resonator.bridge_score(c_a, c_b)
                if score > self.cfg.sleep.bridge.min_score:
                    rep_a = c_a.anchors[0]
                    rep_b = c_b.anchors[0]
                    existing = self.graph.edges.get(self.graph._key(rep_a.id, rep_b.id))
                    if not existing or existing.weight < self.cfg.sleep.bridge.default_weight:
                        self.graph.add_edge(rep_a.id, rep_b.id, weight=self.cfg.sleep.bridge.default_weight,
                                            edge_type="bridge")
                        bridges += 1

        if bridges:
            self._log_event(f"Bridge: created {bridges} cross-constellation connections")
        return bridges

    # ── Phase 6: Hub Connection ────────────────────────────

    def _connect_cross_cortex_hubs(self, hublayer, cortices: list) -> int:
        """Detect cross-cortex patterns and create hub links.

        Compares compressed segments across cortices. When two segments have
        similar centroids (cosine > 0.6), auto-creates leaf hubs from segments
        if none exist, then creates a hub-to-hub edge for cross-domain reasoning.
        """
        connections = 0

        def _ensure_segment_hub(seg, cortex_name: str):
            """Get or create a leaf hub for a segment."""
            if seg.hub_links:
                return seg.hub_links[0]
            if not seg.centroid or not seg.node_ids:
                return None
            topic = seg.summary or f"{cortex_name}_cluster"
            hub = hublayer.create_leaf(
                text=f"[{cortex_name}] {topic}",
                source_anchor_ids=list(seg.node_ids),
                cortex_name=cortex_name,
                importance=seg.importance,
                embedding=seg.centroid,
            )
            seg.link_hub(hub.id)
            return hub.id

        for i, ctx_a in enumerate(cortices):
            seg_a = ctx_a.get_segment_for_hub("compressed")
            if not seg_a or not seg_a.centroid:
                continue
            for ctx_b in cortices[i + 1:]:
                seg_b = ctx_b.get_segment_for_hub("compressed")
                if not seg_b or not seg_b.centroid:
                    continue
                sim = _cosine_sim_sleep(seg_a.centroid, seg_b.centroid)
                if sim > 0.6:
                    hub_a_id = _ensure_segment_hub(seg_a, ctx_a.config.name)
                    hub_b_id = _ensure_segment_hub(seg_b, ctx_b.config.name)
                    if hub_a_id and hub_b_id:
                        hublayer.add_hub_edge(hub_a_id, hub_b_id, weight=sim, edge_type="cross_domain")
                        connections += 1
        return connections

    # ── Phase 7: Thermal Forgetting ────────────────────────

    def _apply_reinforcement_decay(self) -> dict:
        """Adjust anchor decay rates based on success/feedback history.

        Reinforcement-adjusted decay formula:
          decay_rate = base_decay × (1 - success_feedback × 0.5) × (1 - reinforcement × 0.3)

        Anchors with high success_feedback decay slower (better retention).
        Anchors with low success_feedback decay faster (adaptive forgetting).
        Anchors with high confidence resist decay more strongly.

        Returns stats: {adjusted, boosted, penalized}.
        """
        adjusted = 0
        boosted = 0
        penalized = 0

        for anchor in self.graph.anchors.values():
            v = anchor.vector
            # Base decay rate from config or anchor's own rate
            base_rate = getattr(v, 'decay_rate', 0.01)

            # Success feedback: 0..1, higher = slower decay
            success_damping = 1.0 - v.success_feedback * 0.5

            # Confidence: higher confidence = slower decay
            confidence_damping = 1.0 - v.confidence * 0.2

            # Reinforcement from access patterns
            reinforcement = getattr(anchor, 'replay_count', 0) / max(1, self._cycle_count)
            reinforcement_damping = 1.0 - min(0.3, reinforcement * 0.3)

            # Stability slows decay further
            stability_damping = 1.0 - v.stability * 0.4

            # New effective decay rate
            effective_rate = base_rate * success_damping * confidence_damping
            effective_rate *= reinforcement_damping * stability_damping

            # Clamp
            effective_rate = max(0.001, min(0.5, effective_rate))

            if abs(effective_rate - base_rate) > 0.001:
                if effective_rate < base_rate:
                    boosted += 1
                else:
                    penalized += 1
                adjusted += 1

            v.decay_rate = effective_rate

        return {"adjusted": adjusted, "boosted": boosted, "penalized": penalized}

    def _apply_thermal_forgetting(self) -> dict:
        """Apply thermal lifecycle degradation.

        Five-level thermal downgrade:
        - HOT→WARM: retention dropped below 0.4, reduce priority
        - WARM→COLD: long-unaccessed, retention < 0.15, offload to index
        - COLD→FROZEN: retention < 0.06, disk-only archive tier
        - FROZEN→DEAD: retention < 0.01, hash-only audit trail
        """
        stats = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0, "dead": 0,
                 "downgraded": 0, "finalized": 0}
        import time as _time
        now = _time.time()

        from .anchor import ThermalState, MemoryState as MS
        for anchor in self.graph.anchors.values():
            ts = anchor.thermal_state
            stats[ts.value] = stats.get(ts.value, 0) + 1

            if ts == ThermalState.HOT:
                # HOT→WARM: idle > 72h and retention below 0.4
                hours_idle = (now - anchor.last_activated_at) / 3600
                if hours_idle > 72 and anchor.retention_score < 0.4:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.05)
                    stats["downgraded"] += 1

            elif ts == ThermalState.WARM:
                # WARM→COLD: very long idle or low retention
                hours_idle = (now - anchor.last_activated_at) / 3600
                if hours_idle > 720 or anchor.retention_score < 0.15:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.03)
                    anchor.vector.recency *= 0.5
                    stats["downgraded"] += 1

            elif ts == ThermalState.COLD:
                # COLD→FROZEN: retention below 0.06
                if anchor.retention_score < 0.06:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.02)
                    stats["downgraded"] += 1

            elif ts == ThermalState.FROZEN:
                # FROZEN→DEAD: retention below 0.01
                if anchor.retention_score < 0.01:
                    anchor.state = MS.GHOST
                    anchor._ghost_reactivation_prob = 0.005
                    stats["finalized"] += 1

        return stats

    # ── Phase 8: Index Rebuild ─────────────────────────────

    def _rebuild_ann_index(self):
        """Rebuild the ANN search index from current anchors."""
        try:
            from .index import ANNIndex
            index = ANNIndex(self.graph)
            index.build()
        except Exception:
            pass  # Index rebuild is best-effort

    # ── Helpers ─────────────────────────────────────────

    def _refresh_cortical_index(self) -> None:
        """Rebuild cortical index, excluding FROZEN and DEAD anchors."""
        from .anchor import ThermalState
        self.graph.cortical_index = [
            (a.embedding, a.id)
            for a in self.graph.anchors.values()
            if a.embedding and a.is_cortical
            and a.thermal_state not in (ThermalState.FROZEN, ThermalState.DEAD)
        ]
        # Sync ANN index — exclude FROZEN/DEAD
        ann = self.graph._get_ann_index() if self.graph._ann_index is not None else None
        if ann is not None:
            ann.clear()
            for a in self.graph.anchors.values():
                if a.embedding and a.thermal_state not in (ThermalState.FROZEN, ThermalState.DEAD):
                    ann.add(a.id, a.embedding)
            ann.rebuild()

    @staticmethod
    def _embedding_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(min_len))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Fallback: character bigram Jaccard (only when embeddings unavailable)."""
        def bigrams(s):
            return {s[i:i + 2] for i in range(len(s) - 1)}
        ba, bb = bigrams(a), bigrams(b)
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)

    @staticmethod
    def _transfer_edge(old: Edge, new_key: tuple[str, str]) -> Edge:
        """Transfer an edge to a new key, preserving RichEdge properties."""
        from .graph import RichEdge
        if isinstance(old, RichEdge):
            return RichEdge(
                source=new_key[0], target=new_key[1],
                weight=old.weight, edge_type=old.edge_type,
                confidence=old.confidence, source_type=old.source_type,
                reinforcement_count=old.reinforcement_count,
                decay_rate=old.decay_rate, is_stale=old.is_stale,
                stale_since=old.stale_since, replaced_by=old.replaced_by,
                version_history=list(old.version_history),
                success_count=old.success_count,
                failure_count=old.failure_count,
                last_success_at=old.last_success_at,
                last_failure_at=old.last_failure_at,
            )
        return Edge(source=new_key[0], target=new_key[1],
                    weight=old.weight, edge_type=old.edge_type)

    @staticmethod
    def _infer_edge_type(a: Anchor, b: Anchor) -> str:
        if a.source_session == b.source_session:
            return "temporal"
        if set(a.tags) & set(b.tags):
            return "topical"
        return "topical"


def _cosine_sim_sleep(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
