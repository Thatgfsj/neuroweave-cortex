# NeuroWeave Cortex (NWC) — Cognitive Memory Runtime

An infrastructure-grade memory system for AI agents. Not a vector database. Not a graph database. A **memory runtime** — it remembers, forgets, strengthens, connects, abstracts, and evolves, the way biological memory does.

```
Anchors: 131   Edges: 211   Ghosts: 0   Schemas: 9   Abstracts: 1
Memory Stability: 0.72   Recall Plasticity: 0.58   Compression: 1.6x
```

## What makes it different

Vector databases retrieve. Graph databases traverse. NeuroWeave Cortex runs a **cognitive lifecycle**:

| Capability | Vector DB | Graph DB | NeuroWeave Cortex |
|---|---|---|---|
| Semantic retrieval | yes | no | yes |
| Graph traversal | no | yes | yes |
| Automatic forgetting (survival decay) | no | no | yes |
| Memory strengthening (rehearsal) | no | no | yes |
| Conflict detection (contradiction edges) | no | no | yes |
| Fuzzy recall ("I think I remember...") | no | no | yes |
| Emergent abstraction (pattern discovery) | no | no | yes |
| Temporal context (TimeSpine-indexed) | no | no | yes |
| 8-phase sleep consolidation | no | no | yes |
| Ghost revival (savings effect) | no | no | yes |
| Autobiographical self-model | no | no | yes |

## Quick Start

```python
from star_graph import MemoryManager
from star_graph.scheduler import AgentContext

# One-line setup
mgr = MemoryManager()

# Remember things
mgr.remember("User debugs Redis connection timeout — pool was 10, fixed to 20",
             tags=["redis", "debug", "timeout"])
mgr.remember("User knows Python async programming with asyncio",
             tags=["python", "knowledge"])
mgr.remember("User prefers type hints and concise code",
             tags=["preference", "style"])

# Working memory — fast, ephemeral buffer for active context
mgr.remember_working("Currently debugging auth middleware timeout",
                     tags=["debug", "auth"])

# Context-aware recall — working memory gets retrieval priority
ctx = AgentContext(task_type="debugging", active_goals=["fix Redis connection"])
memories = mgr.recall("Redis connection pool config", context=ctx)
print(memories.memory_summary)

# System-2 deep recall for exhaustive or low-confidence queries
memories = mgr.dual_recall("list all Redis-related issues", context=ctx)

# Micro-consolidation — incremental, non-blocking
mgr.micro_consolidate()

# Let the system sleep — 8-phase consolidation
report = mgr.sleep()
print(report)

# Persist
mgr.save("agent_memory.json")
mgr.load("agent_memory.json")
```

## Architecture

Three-layer design. Each layer depends only on the one below.

```
Layer 3: Behavior    │  Cortex routing, memory gating, working memory,
    (cortex.py,      │  dual-channel retrieval, adaptive replay, autobiographical memory
     router.py,      │  "What should I recall right now, at what detail level?"
     gate.py,        │
     working_memory.py,│
     scheduler.py,   │
     autobiography.py)│
                     │
Layer 2: Cognitive   │  Hub abstraction, cascade recall, TimeSpine temporal index,
    (retriever.py,   │  sleep consolidation, evolution, ghost revival,
     sleep.py,       │  abstraction, community detection, competition
     evolution.py,   │  "How do memories connect, strengthen, and fade?"
     ghost.py,       │
     abstraction.py, │
     community.py,   │
     competition.py, │
     timespine.py,   │
     cascade.py,     │
     hub.py)         │
                     │
Layer 1: Storage     │  CRUD, persistence, ANN indexing, tiered storage,
    (graph.py,       │  BM25 keyword index, multi-level caching
     anchor.py,      │  "Where is this memory stored?"
     storage.py,     │
     sqlite_storage.py,│
     index.py,       │
     bm25.py,        │
     cognitive_cache.py,│
     tier.py)        │
```

### Core modules

| Module | Role |
|---|---|
| `manager.py` | High-level facade — `remember()`, `recall()`, `sleep()`, `save()` |
| `runtime.py` | Dependency container — manages all subsystem lifecycles |
| `retrieval_pipeline.py` | 5-layer dimensional descent (L0→L4) with automatic degradation |
| `scheduler.py` | Context-aware retrieval with memory type selection |
| `working_memory.py` | Short-term buffer (15-item, 1h TTL) — auto-promotes to long-term |
| `sleep.py` | 8-phase sleep: N1_Replay → N2_Merge → N3_Compression → N3b_AtomFacts → REM_Emotion → N4_Prune → N5_HubConnect → N6_IndexRebuild |
| `evolution.py` | Survival-based decay (Ebbinghaus/Power-law/Exponential), belief transitions, interference |
| `retriever.py` | HybridFusion + OscillationResonance + VectorSimilarity + Personalized PageRank + explainable scores |
| `dual_channel.py` | System-1 (fast) + System-2 (deep) dual-channel retrieval with auto-trigger |
| `bm25.py` | Sparse keyword retrieval (BM25) with reciprocal rank fusion for hybrid search |
| `ghost.py` | Latent memory traces with fuzzy recall and contradiction tracking (NegativeGhost) |
| `abstraction.py` | Emergent category discovery from anchor clusters |
| `community.py` | Louvain community detection with centroid routing |
| `anchor.py` | Memory unit with 6-state lifecycle, 10-dim AnchorVector, multiplicative retention |
| `graph.py` | Star graph with RichEdge (temporal, causal, state-transition), Schema, ReflectionNode |
| `timespine.py` | Temporal index for O(days×buckets) time-scoped retrieval |
| `cascade.py` | Causal chain traversal across connected memory sequences |
| `hub.py` | Hierarchical hub-and-spoke abstraction (leaf→domain→global) |
| `cortex.py` | Partitioned memory cortices with independent sleep and retrieval |
| `cognitive_cache.py` | Multi-level cache (query/session/topic/activation) + exact-match entity lookup |
| `tier.py` | STM/MTM/LTM/Core cognitive tiering + HOT/WARM/COLD storage tiers |
| `autobiography.py` | Self-narrative formation and autobiographical memory |
| `atom_facts.py` | LLM-powered atomic fact extraction from memory clusters |
| `survival.py` | Pluggable survival functions (Ebbinghaus, Power-law, Exponential, Custom) |
| `compression.py` | Multi-level session compression (episodic/strategic/meta) |
| `resonance.py` | Phase-locked oscillation resonance for temporal-coherent retrieval |
| `streaming.py` | Streaming memory buffer with backpressure |
| `benchmark.py` | Built-in benchmark suite (5 categories) |
| `config.py` | Centralized YAML config with schema validation, dot-path access, overrides |

## Retrieval Benchmarks

### LoCoMo Benchmark (real-world conversations)

Evaluated on the [LoCoMo-10 dataset](https://github.com/snap-research/LoCoMo): 10 conversations, 5,882 turns across 272 sessions, 1,986 QA pairs across 5 categories. Pure retrieval (no LLM generation).

| Method | has_answer | F1 | Latency |
|---|---|---|---|
| VectorSimilarity | 13.1% | 0.026 | 122.2ms |
| OscillationResonance | 11.9% | 0.026 | 110.4ms |
| **HybridFusion + BM25** | **15.3%** | **0.025** | <200ms |

Per-category has_answer:

| Category | #QA | VecSim | OscRes | HybFus |
|---|---|---|---|---|
| Temporal (1) | 282 | 3.5% | 2.8% | 4.3% |
| Short Memory (2) | 321 | 1.9% | 2.2% | 1.9% |
| Long Memory (3) | 96 | 2.1% | 2.1% | 3.1% |
| Composite (4) | 841 | 18.0% | 16.4% | 21.0% |
| Adversarial (5) | 446 | 20.4% | 18.4% | 23.5% |

### Internal Benchmark (synthetic multi-session)

6 sessions × 80 turns, 5 categories. 1.7x compression (6,708 → 3,982 tokens) with maintained or improved recall.

| Method | C-R@3 | C-R@5 | Interf |
|---|---|---|---|
| VectorSimilarity | 0.933 | 0.933 | N/A |
| **OscillationResonance** | **0.967** | **0.967** | 0.667 |
| HybridFusion | 0.900 | 0.900 | 0.667 |

## Memory Lifecycle

Every anchor moves through 6 states:

```
ACTIVE → REHEARSING → CONSOLIDATING → DORMANT → GHOST → REACTIVATED
```

- **Active**: Just created or recently accessed — fully plastic, easy to update
- **Rehearsing**: Being replayed during sleep — temporarily elevated importance
- **Consolidating**: Transferring from hippocampal to cortical — increasing stability
- **Dormant**: Stable, low-activity — read-only, cortical retrieval
- **Ghost**: Pruned but with residual trace — can partially recall or fully revive
- **Reactivated**: Ghost revived by new relevant experience — reduced stability, high plasticity

Paired with **ThermalState** (HOT → WARM → COLD → DEAD) for storage tier switching:
- HOT: in-memory, fully accessible
- WARM: in-memory, periodically flushed to disk
- COLD: disk-only, transparent thaw on access

## Sleep Consolidation

Sleep is not cleanup. Sleep **changes the graph**:

1. **N1_Replay** — prioritizes surprising and emotional memories for replay via SWR scoring
2. **N2_Merge** — fuses near-duplicate anchors (ANN-accelerated, O(n×k)), bridges constellations
3. **N3_Compression** — transfers memories from hippocampal to cortical, forms schemas
4. **N3b_AtomFacts** — LLM extraction of atomic facts from compressed clusters
5. **REM_Emotion** — strips emotional charge from consolidated memories
6. **N4_Prune** — removes weak anchors/edges, creates ghost traces for savings effect
7. **N5_HubConnect** — cross-cortex hub bridge formation
8. **N6_IndexRebuild** — refreshes ANN, BM25, and community indices

## Dual-Channel Retrieval

System-1 (fast, embedding + BM25 hybrid) and System-2 (deep, hierarchical traversal) with automatic triggering:

- Low-confidence System-1 results (<0.35) automatically trigger System-2
- Structural keywords ("all", "every", "list", "which", "before", "last") trigger exhaustive search
- Results merged via weighted reciprocal rank fusion

## Configuration

```python
from star_graph.config import config, override, load_config

# Dot-path access
threshold = config.sleep.merge.default_threshold  # 0.85

# Programmatic override
override('sleep.merge.default_threshold', 0.75)
override('gate.k', 30)

# Schema validation
warnings = config.validate()  # type, range, and cross-section checks

# Load custom YAML
cfg = load_config("my_params.yaml")
```

See `star_graph/defaults.yaml` for all 300+ tunable parameters.

## Installation

```bash
git clone https://github.com/Thatgfsj/neuroweave-cortex.git
cd neuroweave-cortex

# Install in editable mode
pip install -e .

# Optional: for SQLite storage backend
pip install aiosqlite

# Run demo
python examples/emergence_demo.py
```

**Note:** NeuroWeave Cortex is not published on PyPI. Install from source as shown above.

## Interactive Demo

```bash
python examples/emergence_demo.py

# Or use the CLI:
# sg-add "Discussed microservices deployment patterns" --tags architecture --emotional 0.6
# sg-query "database connection pooling best practices"
# sg-query --trace "When did Alice visit Hawaii?"
# sg-stats --schemas --ghosts
# sg-sleep --retention 0.15 --edge-prune 0.1
```

## Benchmarks

```bash
python examples/memory_benchmark.py --quick   # 4 sessions, ~200 turns
python examples/memory_benchmark.py --full    # 12 sessions, ~5000 turns
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned work.

## License

MIT
