# Complexity Analysis

## Storage Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Node insertion | O(1) | Hash map append |
| Edge insertion | O(1) | Adjacency list append |
| Node lookup | O(1) | Hash map get |
| Edge lookup | O(1) | Sorted tuple key |
| Graph memory | O(V + E) | V=anchors, E=edges |
| Activation level per node | O(1) | Float field |

## Retrieval Complexity

### Domain Routing (Layer 1)
- **Best case**: O(1) — direct domain match
- **Worst case**: O(k·d) — k = top-k domains, d = depth
- **Reduces** full O(N) search to O(N/domains)

### Activation Propagation (Layer 2)
- **BFS spread**: O(V + E) per query
- **With depth limit D**: O(b^D) where b = avg branching factor
- **With node cap C**: O(C·b) — capped at C=50 nodes

### Traditional Vector Search
- **Flat scan**: O(N) — all N anchors
- **ANN index**: O(log N) — but lower recall

### Retrieval Cost Comparison
| Method | Query Complexity | Recall |
|--------|-----------------|--------|
| Flat vector search | O(N) | Baseline |
| ANN index | O(log N) | -3-5% |
| Domain routing + flat | O(N/domains) | Same |
| **NWC full pipeline** | **O(C·b + N/domains)** | **+16.2pp over vector** |

## Sleep Consolidation Complexity

| Phase | Complexity | Trigger |
|-------|-----------|---------|
| N1 Replay | O(R) — R recent anchors | Every sleep |
| N2 Merge | O(k²) — k clusters | Every sleep |
| N2b Conflict | O(M²) — M high-sim pairs | Every sleep |
| N3 Compression | O(T·log T) — T topic groups | Weekly |
| N4 Dim | O(V) — all anchors | Nightly |
| Edge decay | O(E) — all edges | Nightly |

### Scalability
- **<10K anchors**: O(N) vector scan acceptable, ~50ms/query
- **10K-100K anchors**: Domain routing reduces effective search to O(N/10)
- **>100K anchors**: ANN index recommended (O(log N))

## Update Complexity

| Operation | Complexity | Frequency |
|-----------|------------|-----------|
| Memory write (create) | O(1) | Per interaction |
| Edge update | O(1) | Per co-activation |
| Activation update | O(1) | Per retrieval |
| Sleep consolidation | O(V²·E) worst | Nightly |
| Decay sweep | O(V + E) | Nightly |

## Theorem: Activation Propagation Bounds

**Theorem 1**: For a query q, activation propagation with depth limit D and branching factor b explores at most O(b^D) nodes.

*Proof*: At level 0, we have 1 seed node. At level 1, at most b neighbors. At level 2, at most b², etc. Total = 1 + b + b² + ... + b^D = O(b^D).

**Theorem 2**: The capped propagation (max_nodes=C) runs in O(C·b) time.

*Proof*: Each of at most C explored nodes expands at most b neighbors. Total edge evaluations ≤ C·b.

## Empirical Validation

| Dataset | Anchors | Edges | Query Time | Memory |
|---------|---------|-------|-----------|--------|
| LoCoMo-10 (conv) | 419 | 6 | 94ms | ~8MB |
| LoCoMo-10 (10 convs) | 4,190 | ~60 | — | ~80MB |
| Estimated 100K anchors | 100,000 | ~1.6M | ~200ms | ~2GB |
