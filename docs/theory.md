# Cognitive Memory Theory — NeuroWeave Cortex

## Overview

NeuroWeave Cortex (NWC) implements a **lifecycle-aware cognitive memory architecture**
inspired by human memory consolidation, forgetting, and reconsolidation processes.
Unlike conventional vector databases or retrieval-augmented generation (RAG) systems
that treat all memories uniformly, NWC models memory as a **dynamic, evolving system**
with distinct states, multi-dimensional importance, and autonomous maintenance cycles.

This document describes the cognitive theory underpinning NWC's architecture.

---

## 1. Memory as an Evolving System

### 1.1 The Lifecycle Principle

Every memory in NWC passes through a well-defined lifecycle:

```
ACTIVE → REHEARSING → CONSOLIDATING → INACTIVE → DORMANT → REACTIVATED
```

This mirrors the human memory process where experiences move from
working memory (ACTIVE), through sleep-based consolidation (REHEARSING →
CONSOLIDATING), into long-term storage (INACTIVE), and potentially to
near-permanent latent traces (DORMANT), from which they can be
reactivated when triggered by related cues (REACTIVATED).

**Key axiom**: Memories are NEVER deleted. Only their accessibility changes.
This is a fundamental departure from cache-based systems where eviction is permanent.

### 1.2 Thermal Analogy for Accessibility

Memory accessibility follows a thermal metaphor:

| State | Thermal | Retrieval Cost | Description |
|-------|---------|---------------|-------------|
| ACTIVE | HOT | Instant | Recently accessed, fully available |
| CONSOLIDATING | WARM | Low | Being strengthened during sleep |
| INACTIVE | COLD | Medium | Not recently used but still accessible |
| DORMANT | FROZEN | High | Very low activation, requires strong cue |
| (evicted) | DEAD | N/A | Only compressed residual trace (Ghost) remains |

---

## 2. Multi-Dimensional Importance

### 2.1 Importance Signal Vector

Memory importance is not a single scalar but a **multi-dimensional signal**:

```
I(m) = f(e, r, g, c, f)
```

Where:
- **e** = **Emotional salience**: intensity (not polarity) of emotional content
- **r** = **Repetition frequency**: how often the topic recurs across sessions
- **g** = **Goal relevance**: alignment with active and historical goals
- **c** = **Content richness**: information density, entity count, specificity
- **f** = **Retrieval feedback**: past retrieval frequency × success rate

### 2.2 Composite Scoring

The composite importance score is a weighted linear combination:

```
score = W_emo · e + W_rep · r + W_goal · g + W_rich · c + W_fb · f
```

Default weights (tunable per application):
- W_emo = 0.18, W_rep = 0.22, W_goal = 0.28, W_rich = 0.12, W_fb = 0.12

Goal relevance carries the highest weight, reflecting NWC's focus on
**task-relevant memory**.

### 2.3 Retrieval Feedback Loop

Each successful retrieval reinforces the importance signal for that
memory's semantic signature:

```
retrieval_feedback(key) = α · count + (1-α) · success_rate
```

where α = 0.3 and the success_rate is updated via exponential moving average:

```
success_rate_{t+1} = 0.7 · success_rate_t + 0.3 · result_{t}
```

This creates a self-reinforcing loop: frequently retrieved memories
become more important, making them more likely to be retrieved again.

---

## 3. Multi-Dimensional Decay

### 3.1 Three Dimensions of Forgetting

NWC models forgetting as a **three-dimensional process**:

| Dimension | Variable | Formula | Interpretation |
|-----------|----------|---------|----------------|
| Temporal | recency | r·e^(-λ_t · t) | Time since last access |
| Activation | activation_level | a·e^(-λ_a · t) | Retrieval readiness |
| Utility | frequency | f·e^(-λ_u · t) | Long-term usefulness |

Each dimension decays independently, allowing different forgetting rates
for different aspects of memory.

### 3.2 Stability-Modulated Decay

Decay rates are modulated by **memory stability**:

```
λ_effective = λ_base · (1 - stability · 0.6)
```

High-stability memories (reinforced across multiple sessions) decay
significantly slower than low-stability ones. This reflects the
psychological finding that well-consolidated memories are more
resistant to forgetting.

### 3.3 State-Dependent Decay

Different memory states have different half-lives:

| State | Half-Life Multiplier | Effect |
|-------|---------------------|--------|
| ACTIVE | 1.0× | Baseline decay |
| DORMANT | 3.0× | Decay slows significantly |
| GHOST | 0.5× | Decay accelerates (residual only) |
| REACTIVATED | 2.0× | Moderate protection |

---

## 4. Spreading Activation

### 4.1 Activation Propagation

When a memory is retrieved, activation spreads through the graph along
edges, following:

```
A(j) = A(i) · w_ij · d_ij
```

Where:
- A(i) = activation at source node i
- w_ij = weight of edge between i and j
- d_ij = distance attenuation (1/hops)

### 4.2 Edge-Type Modulation

Different edge types carry different traversal weights:

| Edge Type | Traversal Weight | Category |
|-----------|-----------------|----------|
| causes, causal | 1.5 | Causal — strongest |
| fixes, caused_by | 1.4 | Causal |
| depends_on | 1.3 | Structural dependency |
| supports, derived_from | 1.2 | Logical support |
| resolves | 1.3 | Resolution |
| task_flow | 1.4 | Workflow continuity |
| before, after | 1.0 | Temporal ordering |
| contradicts | 0.5 | Contradiction — suppressed |
| superseded_by | 0.4 | Outdated knowledge |
| invalidated_by | 0.3 | Disproven — very low |

---

## 5. Consolidation (Sleep)

### 5.1 Sleep Phases

NWC's sleep consolidation mirrors human sleep architecture:

| Phase | Name | Function |
|-------|------|----------|
| N1 | Replay | Replay recent experiences at accelerated speed |
| N2a | Importance Update | Re-score all memories based on new evidence |
| N2b | Conflict Detection | Identify contradictions between memories |
| N2c | Memory Revision | Revise low-confidence / high-surprise memories |
| N3 | Schema Formation | Extract abstract patterns from concrete episodes |
| REM | Emotional Consolidation | Integrate emotional valence, strengthen salient memories |
| N4 | Ghost Pruning | Compress low-importance memories to residual traces |
| Phase 8 | Index Rebuild | Rebuild ANN index and cognitive caches |

### 5.2 Ghost Creation (Compressed Residuals)

When a memory transitions to DORMANT or is pruned, a **Ghost** is created:

```
Ghost(m) = ⟨e_compressed, s_semantic, e_trace, i_residual⟩
```

Where:
- e_compressed = compressed embedding (preserved top-k dimensions)
- s_semantic = semantic shadow (keywords that defined the memory)
- e_trace = emotional trace (valence, arousal)
- i_residual = residual importance score

Ghosts enable **partial recall** without full memory restoration,
and support the **savings effect** for re-learning.

---

## 6. Savings Effect

### 6.1 Definition

The savings effect measures the efficiency gain when re-learning a
previously known memory:

```
savings = (strength_after_revival - strength_before) / strength_before
```

A positive savings value indicates that the revived memory regains
strength faster than an entirely new memory. This mirrors the
Ebbinghaus savings effect in human memory.

### 6.2 Implementation via Ghost Traces

When a memory is revived from its ghost trace:

1. The ghost's compressed embedding provides a head start in
   similarity matching
2. The semantic shadow primes the re-encoding process
3. The revival takes fewer repetitions to reach full strength
4. The savings effect is tracked per ghost and aggregated globally

---

## 7. Schema Formation

### 7.1 Abstraction Mechanism

Schemas are formed through repeated exposure to similar patterns:

```
Schema(s) = ⟨template, slots, instances, confidence⟩
```

Where:
- template = abstract pattern description
- slots = placeholders for variable details
- instances = concrete memories that instantiate this schema
- confidence = 0..1, grows with each confirming instance

### 7.2 Schema Matching

A new memory matches a schema when:

```
match_score = 0.6 · keyword_overlap + 0.4 · embedding_similarity
```

High match scores trigger schema application, which guides encoding
by filling schema slots from the input.
