# Star Graph Memory · Cognitive Memory Engine

A hippocampal-inspired long-term memory runtime for AI agents — not a vector database wrapper.

Conversations compress into **anchor points** connected in a navigable **star graph**. Retrieval uses **oscillatory phase-locking resonance** with real phase synchronization. Consolidation happens during **9-phase sleep cycles** that change graph topology. Ghost traces enable **fuzzy recall** ("I seem to remember..."). The system discovers **emergent abstractions** from related episodes.

```
ACTIVE → REHEARSING → CONSOLIDATING → DORMANT → GHOST → REACTIVATED
```

## Mathematical Definitions

### Resonance Score

Retrieval is not cosine similarity. It is phase-locked resonance:

```
Res(q, m) = (1 - w) × |z_q · z_m| / (‖z_q‖ ‖z_m‖) + w × cos(Δφ)

where:
  z_q = query phasor (magnitude from embedding variance, phase from principal angles)
  z_m = memory phasor (magnitude = retention_score, phase = oscillator.phase_offset)
  Δφ  = |atan2(z_q) - atan2(z_m)|, normalized to [0, π]
  w   = phase_weight (default 0.3) — how much phase matters vs magnitude
```

### Phase Derivation (not random)

```
θ_anchor = f(t_created, importance, emotional_valence, semantic_direction)

  t_created → temporal_phase = (timestamp % 86400) / 86400 × 2π
  importance → alignment_shift = (1 - importance) × π × 0.5
  emotion    → emotion_shift = emotional_valence × π × 0.25
  embedding  → semantic_shift = atan2(emb[1], emb[0]) × 0.3

  θ = (temporal + importance + emotion + semantic) mod 2π
```

### Prioritized Experience Replay (SWR)

```
priority = |valence| × 0.25 + surprise × 0.25 + frequency × 0.20
         + centrality × 0.15 + |1 - stability| × 0.15

Replay: top 50% always replayed, remainder sampled by priority
```

### Memory State Transitions

```
                    create
    ┌──────────────────────────────┐
    │                              ▼
    │    ┌─────────┐  replay  ┌────────────┐
    │    │  ACTIVE  │─────────▶│ REHEARSING │
    │    └────┬─────┘          └─────┬──────┘
    │         │  consolidate   consolidate
    │         │  ┌───────────────┘
    │         ▼  ▼
    │    ┌──────────────┐
    │    │ CONSOLIDATING │◀──── consolidate
    │    └──────┬───────┘
    │           │ stabilize
    │    ┌──────▼───────┐  retrieve
    │    │   DORMANT    │──────────┐
    │    └──────┬───────┘          │
    │           │ prune            │
    │    ┌──────▼───────┐          │
    │    │    GHOST     │          │
    │    └──────┬───────┘          │
    │           │ revive           │
    │    ┌──────▼───────┐          │
    │    │ REACTIVATED  │──────────┘
    │    └──────────────┘
    └───────────────────────────────── retrieve from DORMANT → ACTIVE
```

## Biological Mapping

| Neuroscience | Brain Region | Algorithm | Code |
|---|---|---|---|
| Phase precession | Hippocampus (CA1) | Oscillator.phase_offset encodes temporal position | `anchor.py:Oscillator` |
| Theta-gamma coupling | Entorhinal-HPC | natural_frequency (theta) + edge coupling (gamma) | `retriever.py:_resonance_score` |
| SWR replay (~20×) | Hippocampus | Prioritized replay with stochastic sampling | `sleep.py:_swr_replay` |
| Systems consolidation | HPC → Neocortex | hippocampal_dependency = exp(-replay/τ) | `sleep.py:_systems_consolidation` |
| Emotional modulation | Amygdala → HPC | |valence| boosts encoding, decays during sleep | `sleep.py:_emotional_stripping` |
| Schema formation | mPFC | Embedding-cluster → AbstractNode | `abstraction.py:AbstractionEngine` |
| Savings effect | Whole-brain | GhostNode with compressed embedding, fuzzy recall | `ghost.py:GhostSubsystem` |
| Reconsolidation | Hippocampus | activate() → stability*=0.7, consolidate(error) | `anchor.py:Anchor.consolidate` |
| Synaptic homeostasis | Cortex | Global weight scaling toward mean 0.3 | `sleep.py:_synaptic_homeostasis` |
| Interference forgetting | Prefrontal-HPC | Competitive retrieval, new-inhibits-old | `competition.py:MemoryCompetition` |
| Hebbian plasticity | Synapses | Co-activated edges strengthen, dormant weaken | `sleep.py:_hebbian_update` |
| Pattern completion | CA3 | Spreading activation from partial cues | `graph.py:spread_activation` |

## Quick Start

```python
from star_graph import StarGraph, Anchor, SleepCycle, get_embedder
from star_graph.retriever import OscillationResonanceRetriever

graph = StarGraph()

# Create anchors — embeddings auto-encoded via sentence-transformers
a1 = Anchor.create("Discussed microservices deployment patterns",
                    tags=["architecture", "backend"], emotional_valence=0.6)
a2 = Anchor.create("User prefers Python for automation, Rust for performance",
                    tags=["preferences", "tech-stack"])
a3 = Anchor.create("Decided to use Postgres with connection pooling",
                    tags=["architecture", "database"], emotional_valence=0.8)

for a in [a1, a2, a3]:
    graph.add_anchor(a)

# Phase-locking resonance retrieval
ret = OscillationResonanceRetriever(graph)
result = ret.retrieve("database connection pooling best practices")
for c in result.constellations:
    for a in c.anchors:
        print(f"[{a.state.value}] {a.text}")

# Predictive retrieval with action decision
constellation, action = resonator.predictive_retrieve("database connection pooling for APIs")
print(f"Action: {action}")  # 'confirm', 'update', or 'novel'

# Retrieval trace / explainability (LoCoMo-style debugging)
from star_graph.retriever import OscillationResonanceRetriever

result = OscillationResonanceRetriever(graph).retrieve("When did Alice visit Hawaii?")
print(result.retrieval_trace)
# {
#   "query": "When did Alice visit Hawaii?",
#   "method": "OscillationResonance",
#   "retrieved_memories": [
#     {
#       "memory_id": "...",
#       "score": 0.91,
#       "reason": "temporal_match + entity_match + phase_match"
#     }
#   ]
# }

# 9-phase sleep cycle — changes graph topology
cycle = SleepCycle(graph)
result = cycle.run()
# result: merged, pruned_anchors, ghosts_created, schemas_formed, bridges_created
```

## Deterministic Mode

For reproducible benchmarks:

```python
from star_graph import seed_everything

seed_everything(42)  # all RNG, replay order, pruning, phase init
```

## Emergent Behavior Demo

```bash
python examples/emergence_demo.py

# Or use the CLI:
# sg-add "Discussed microservices deployment patterns" --tags architecture --emotional 0.6
# sg-query "database connection pooling best practices"
# sg-query --trace "When did Alice visit Hawaii?"
# sg-stats --schemas --ghosts
# sg-sleep --retention 0.15 --edge-prune 0.1
```

5-stage cognitive lifecycle:
1. 10 anchors encoded (Python, Flask, FastAPI, Redis, etc.)
2. Semantic graph forms via embedding similarity
3. Sleep → abstraction emerges: "python + backend + fastapi"
4. Weak anchor pruned → ghost with fuzzy recall
5. Related query → ghost resonates → **REVIVED!** (REACTIVATED state)

## Current Limitations

- **Resonance recall without embeddings**: When `sentence-transformers` is unavailable, TF-IDF or hash fallback gives reduced retrieval quality. Real embeddings are essential for the phase-locking mechanism.
- **Abstraction threshold sensitivity**: The `similarity_threshold` (default 0.55) heavily affects abstraction emergence. Too high → no abstractions form. Too low → false abstractions.
- **Ghost compressed embedding fidelity**: 16-dim compression from 384-dim loses information. Ghost resonance scores are typically 0.3-0.4 vs 0.5-0.8 for full anchors.
- **Linear scan fallback**: ANN index requires embeddings on all anchors. Cold start or embeddingless anchors fall back to O(n) scan.
- **Schema persistence**: Schemas formed in one sleep cycle may be overwritten in the next. Schema stability across cycles is not yet guaranteed.
- **No streaming consolidation**: Full sleep processes all anchors. Incremental/streaming consolidation for large graphs is planned.

## Cognitive Metrics

```python
from star_graph import CognitiveMetrics

metrics = CognitiveMetrics(graph)
metrics.snapshot()            # take baseline
# ... run operations, sleep cycles ...
metrics.snapshot()            # take follow-up
print(metrics.report())
# Memory Stability:          0.72
# Recall Plasticity:         0.58
# Compression Ratio:         0.15
# Semantic Drift Resistance: 0.81
# Abstraction Emergence:     1.0/cycle
# Ghost Reactivation:        1.00
```

## Architecture Layers

```
Layer 3 (Behavior): retrieval policy, forgetting policy, adaptive replay
                   (seed.py, embedding.py)
Layer 2 (Cognitive): resonance, abstraction, replay, consolidation
                    (retriever.py, sleep.py, resonance.py, abstraction.py,
                     competition.py, ghost.py, metrics.py)
Layer 1 (Storage): CRUD, persistence, indexing
                  (graph.py, index.py, anchor.py, storage.py)
```

## Ablation Testing

To prove each component contributes:

```bash
# Baseline: no sleep, no oscillation, no emotion, no ghosts
python -c "
from star_graph import seed_everything, StarGraph, Anchor
seed_everything(42)
# ... run benchmark with minimal config
"
```

| Component Disabled | Expected Impact |
|---|---|
| No oscillation (phase_weight=0) | Temporal sequence retrieval drops |
| No emotion (emotional_valence=0) | Prioritized replay loses salience weighting |
| No sleep (skip cycle.run) | No abstraction, no merge, no consolidation |
| No ghosts (retention_threshold=0) | Lost memories cannot be revived |

## Installation

```bash
pip install sentence-transformers scikit-learn numpy
git clone https://github.com/Thatgfsj/star-graph-memory.git
cd star-graph-memory
python examples/emergence_demo.py
```

## License

MIT
