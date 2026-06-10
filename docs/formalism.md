# Mathematical Formalism — NeuroWeave Cortex

## Notation

Let:
- `M` = set of all memories in the system
- `A(m)` = activation level of memory `m ∈ M`
- `I(m)` = importance of memory `m`
- `S(m)` = stability of memory `m`
- `G = (V, E)` = memory graph where `V = M` and `E` = typed weighted edges
- `N(m)` = neighborhood of `m` in `G`

---

## 1. Memory Lifecycle

### State Machine

A memory `m` has state `σ(m) ∈ {ACTIVE, REHEARSING, CONSOLIDATING, INACTIVE, DORMANT, REACTIVATED}`.

Transitions are governed by:

```
σₜ₊₁ = T(σₜ, activation_level, elapsed_time)
```

where `T` is a deterministic state machine with allowed transitions:

| From | To | Trigger |
|------|----|---------|
| ACTIVE | REHEARSING | Sleep N1 begins |
| REHEARSING | CONSOLIDATING | Replay count ≥ threshold |
| CONSOLIDATING | INACTIVE | Importance stable for 2 cycles |
| INACTIVE | DORMANT | No access for `t_dormant` hours |
| DORMANT | REACTIVATED | Similarity with new input > θ_react |
| REACTIVATED | ACTIVE | Retrieval/use |

### Thermal Correspondence

Each state maps to a thermal tier `τ(m) ∈ {HOT, WARM, COLD, FROZEN, DEAD}`:

```
τ(m) = φ(σ(m))
```

| σ(m) | τ(m) |
|------|------|
| ACTIVE | HOT |
| CONSOLIDATING | WARM |
| INACTIVE | COLD |
| DORMANT | FROZEN |
| GHOST | DEAD |

Retrieval cost `c(m)` scales inversely with thermal tier:

```
c(m) = base_cost · (1 + 2⁻ᵏ)   where k = rank(HOT) - rank(τ(m))
```

---

## 2. Importance Function

### Signal Vector

The importance signal vector `v(m)` has components:

```
v(m) = [e(m), r(m), g(m), c(m), f(m)]ᵀ
```

where:
- `e(m)` ∈ [0,1] = emotional salience = |valence| · 0.5 + intent_weight · 0.5
- `r(m)` ∈ [0,1] = repetition signal = max(candidates)/50 (capped)
- `g(m)` ∈ [0,1] = goal relevance = word_overlap · 0.4 + concept_overlap · 0.6
- `c(m)` ∈ [0,1] = content richness = entity_density · 0.35 + specificity · 0.25 + actionability · 0.25 + density · 0.15
- `f(m)` ∈ [0,1] = retrieval feedback = EMA of (count, success_rate)

### Composite Score

```
I(m) = Wᵀ · v(m)
     = wₑ·e + wᵣ·r + w_g·g + w_c·c + w_f·f
```

### Noise Gate

Memories with `I(m) < θ_discard` are filtered at write time:

```
store(m) = I(m) ≥ θ_discard
```

where `θ_discard = 0.08` (determined by content richness analysis).

---

## 3. Multi-Dimensional Decay

### Recency Decay (Temporal)

```
r(t) = r₀ · e^(-λ_t · t)
```

where:
- `r₀` = initial recency (set to 1.0 on activation)
- `λ_t = λ_base / halflife` = temporal decay rate
- `t` = hours since last activation

Modulated by stability:

```
λ_t' = λ_t · (1 - S(m) · 0.6)
```

### Activation Decay

```
a(t) = a_base + (a₀ - a_base) · e^(-λ_a · t)
```

where:
- `a₀` = activation at t=0
- `a_base = 0.1` = residual activation floor
- `λ_a` = activation decay rate

### Utility Decay

```
f(t) = f₀ · e^(-λ_u · t)
```

where:
- `f₀` = frequency at t=0
- `λ_u` = utility decay rate

### Combined Retention

```
retention(m, t) = r(t) · (a(t)/a₀) · f(t)/f₀
```

This is an upper bound: actual retrieval depends on cue strength.

---

## 4. Spreading Activation

### Activation Propagation

Given a seed set `Q = {q₁, q₂, ...}` with initial activations `A(q) = 1.0`:

```
A⁽ᵏ⁺¹⁾(j) = A⁽ᵏ⁾(j) + Σᵢ A⁽ᵏ⁾(i) · w(i,j) · αᵏ
```

where:
- `w(i,j)` = edge weight (including type multiplier)
- `α` = decay factor per hop (default 0.85)
- `k` = hop distance from seed

### Convergence

Activation converges when `||A⁽ᵏ⁺¹⁾ - A⁽ᵏ⁾|| < ε`. Typically within 3-5 hops.

### Retrieval Score

The final retrieval score for node `j` is:

```
score(j) = β · A(j) + (1-β) · sim(q_emb, emb(j))
```

where `β` controls graph vs. embedding weight.

---

## 5. Edge Update

### Reinforcement

On co-activation:

```
w'(i,j) = min(1.0, w(i,j) + δ_strengthen)
```

### Decay

Over time without reinforcement:

```
w'(i,j) = w(i,j) · e^(-λ_e · t · (1 - stability(i,j) · 0.7))
```

### Staleness

When superseded:

```
w'(i,j) = w(i,j) · 0.3   (marked stale)
```

---

## 6. Sparticipation Gate (Write)

### Deciding When to Store

A memory `m` passes the sparticipation gate if:

```
I(m) > θ_write OR
richness(m).entity_count > 0 OR
richness(m).technical_term_count > 2
```

The gate prevents storage of purely casual/small-talk content while
preserving all substantive interactions.

---

## 7. Ghost Trace

### Ghost Creation

For a pruned memory `m`, a ghost `g` is created:

```
g = ⟨e_compressed(m), s_semantic(m), e_trace(m), i_residual(m)⟩
```

where:
- `e_compressed(m) = top_k(embedding(m))` — preserve k most salient dimensions
- `s_semantic(m)` — set of high-TF-IDF keywords
- `e_trace(m) = [valence, arousal, dominance]` — emotional vector
- `i_residual(m) = I(m) · 0.3` — residual importance

### Ghost Resonance

Ghost `g` resonates with query `q` when:

```
sim(e_compressed(g), embedding(q)) > θ_ghost
```

Resonance triggers partial recall; sustained resonance across multiple
queries may trigger full revival.

### Savings Effect

The savings effect for revived ghost `g` is:

```
savings(g) = (strength_after - strength_before) / strength_before
```

Aggregate savings across all revived ghosts:

```
S_avg = 1/|R| · Σ_{g∈R} savings(g)
```

where `R` is the set of revived ghosts.

---

## 8. Schema Matching

### Schema Definition

A schema `s` is a tuple:

```
s = ⟨template, slots, instances, confidence⟩
```

### Match Score

Memory `m` matches schema `s` with score:

```
match(m, s) = 0.6 · |W_template ∩ W_m| / |W_template| + 0.4 · sim(emb(m), emb(s))
```

where `W_x` is the word set of `x`.

### Schema Update

When matched:

```
confidence'(s) = min(1.0, confidence(s) + δ_confirm)
instances'(s) = instances(s) ∪ {id(m)}
```

---

## 9. Complexity Analysis

### Time Complexity

| Operation | Average | Worst | Frequency |
|-----------|---------|-------|-----------|
| Write (remember) | O(log n) | O(n) | Per interaction |
| Read (recall) | O(k·log n) | O(n) | Per user query |
| Sleep consolidation | O(n²) | O(n²) | Periodic |
| Spreading activation | O(n + m) | O(n²) | Per recall |
| Ghost revival | O(g·d) | O(g·n) | Per recall |

where `n` = anchor count, `m` = edge count, `g` = ghost count,
`k` = retrieved items, `d` = embedding dimension.

### Space Complexity

| Tier | Space | Description |
|------|-------|-------------|
| HOT anchors | O(n_h) | Full memory objects in memory |
| WARM anchors | O(n_w) | Memory objects, lazy embedding |
| COLD anchors | O(n_c) | Serialized to storage |
| FROZEN anchors | O(n_f) | On disk, meta in memory |
| Ghosts | O(g·k) | Compressed traces |

Total: `O(n·d + m + g·k)` where `d` = embedding dimension.

### Read Scaling

Retrieval latency scales sub-linearly with total anchors:

```
latency(n) = log(n) · (c_bm25 + c_vector + c_graph)
```

The ANN index (HNSW) keeps vector search at O(log n), while the graph
is bounded by the active subgraph typically containing < 1% of total
anchors.

---

## 10. Benchmark Metrics

### Retrieval Quality

```
Recall@K = |relevant ∩ retrieved| / |relevant|
MRR = 1/|Q| · Σ_{q∈Q} 1/rank_q
NDCG = 1/|Q| · Σ_{q∈Q} DCG_q / IDCG_q
has_answer = |{q: relevant_memory ∩ retrieved ≠ ∅}| / |Q|
```

### Efficiency

```
latency = T_response / T_interaction
throughput = queries_per_second
storage_cost = bytes / memory
compression_ratio = raw_tokens / stored_tokens
```
