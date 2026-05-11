# Star Graph Memory — Cognitive Memory Runtime

An infrastructure-grade memory system for AI agents. Not a vector database. Not a graph database. A **memory runtime** — it remembers, forgets, strengthens, connects, abstracts, and evolves, the way biological memory does.

```
Anchors: 131   Edges: 211   Ghosts: 0   Schemas: 9   Abstracts: 1
Memory Stability: 0.72   Recall Plasticity: 0.58   Compression: 1.6x
```

## What makes it different

Vector databases retrieve. Graph databases traverse. Star Graph runs a **cognitive lifecycle**:

| Capability | Vector DB | Graph DB | Star Graph |
|---|---|---|---|
| Semantic retrieval | yes | no | yes |
| Graph traversal | no | yes | yes |
| Automatic forgetting | no | no | yes |
| Memory strengthening (rehearsal) | no | no | yes |
| Conflict detection (belief tracking) | no | no | yes |
| Fuzzy recall ("I think I remember...") | no | no | yes |
| Emergent abstraction (pattern discovery) | no | no | yes |
| Temporal context (when did I learn this?) | no | no | yes |
| Sleep consolidation (restructure during rest) | no | no | yes |

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

# Context-aware recall — knows what you're doing and picks the right memories
ctx = AgentContext(task_type="debugging", active_goals=["fix Redis connection"])
memories = mgr.recall("Redis connection pool config", context=ctx)
print(memories.memory_summary)

# Let the system sleep — it merges, prunes, abstracts, and forms schemas
mgr.sleep()

# Persist
mgr.save("agent_memory.json")
```

## Architecture

Three-layer design. Each layer depends only on the one below.

```
Layer 3: Behavior    │  Retrieval policy, forgetting policy, adaptive replay
    (scheduler.py,   │  "What should I recall right now, at what detail level?"
     embedding.py)   │
                     │
Layer 2: Cognitive   │  Resonance, abstraction, sleep consolidation, evolution
    (retriever.py,   │  "How do memories connect, strengthen, and fade?"
     sleep.py,       │
     evolution.py,   │
     ghost.py,       │
     abstraction.py, │
     competition.py) │
                     │
Layer 1: Storage     │  CRUD, persistence, indexing, ANN lookup
    (graph.py,       │  "Where is this memory stored?"
     anchor.py,      │
     storage.py)     │
```

### Core modules

| Module | Role |
|---|---|
| `manager.py` | High-level facade — `remember()`, `recall()`, `sleep()`, `save()` |
| `scheduler.py` | Context-aware retrieval with memory type selection and compression |
| `sleep.py` | 5-phase sleep: replay → merge → compress → emotional-decouple → prune |
| `evolution.py` | Time decay, frequency boost, conflict resolution, interference |
| `retriever.py` | Hybrid fusion retrieval + Personalized PageRank + explainable scores |
| `ghost.py` | Latent memory traces with fuzzy recall ("I seem to remember...") |
| `abstraction.py` | Emergent category discovery from anchor clusters |
| `anchor.py` | Memory unit with 6-state lifecycle (active → rehearsing → ... → reactivated) |

## Retrieval Benchmarks

### Internal Benchmark (synthetic multi-session conversations)

6 sessions × 80 turns, 5 categories of 6 queries each. Content Recall@k checks if retrieved memory text contains required keywords.

| Method | C-R@3 | C-R@5 | Interf | Note |
|---|---|---|---|---|
| Raw History (full context) | N/A | N/A | N/A | 6,708 tokens, upper bound |
| TF-IDF Vector | N/A | N/A | N/A | keyword baseline |
| VectorSimilarity | 0.933 | 0.933 | N/A | pure embedding similarity |
| **OscillationResonance** | **0.967** | **0.967** | 0.667 | embedding + phase coherence |
| HybridFusion | 0.900 | 0.900 | 0.667 | multi-signal fusion |

Per-category Content Recall@3:

| Category | OscRes | VecSim | HybFus |
|---|---|---|---|
| Long Context | 1.000 | 1.000 | 1.000 |
| Cross-Session | 1.000 | 0.833 | 0.833 |
| Compression | 0.833 | 0.833 | 0.667 |
| Forgetting | 1.000 | 1.000 | 1.000 |
| Interference | 1.000 | 1.000 | 1.000 |

1.7x compression (6,708 → 3,982 tokens) with maintained or improved recall.

### LoCoMo Benchmark (real-world conversations)

Evaluated on the [LoCoMo-10 dataset](https://github.com/snap-research/LoCoMo): 10 conversations, 5,882 turns across 272 sessions, 1,986 QA pairs across 5 categories. Pure retrieval (no LLM generation) — metrics check if retrieved memory text contains the ground-truth answer tokens.

| Method | has_answer | F1 | Latency |
|---|---|---|---|
| VectorSimilarity | 13.1% | 0.026 | 122.2ms |
| OscillationResonance | 11.9% | 0.026 | 110.4ms |
| **HybridFusion** | **15.3%** | **0.025** | 805.9ms |

Per-category has_answer:

| Category | #QA | VecSim | OscRes | HybFus |
|---|---|---|---|---|
| Temporal (1) | 282 | 3.5% | 2.8% | 4.3% |
| Short Memory (2) | 321 | 1.9% | 2.2% | 1.9% |
| Long Memory (3) | 96 | 2.1% | 2.1% | 3.1% |
| Composite (4) | 841 | 18.0% | 16.4% | 21.0% |
| Adversarial (5) | 446 | 20.4% | 18.4% | 23.5% |

Key findings:
- **HybridFusion** leads on overall retrieval quality (15.3% has_answer), 16.7% above VecSim
- Strongest on **Composite** (21.0%) and **Adversarial** (23.5%) — questions with longer, multi-fact answers
- Weakest on **Short Memory** (1.9%) and **Temporal** (2.8%) — questions with single-token answers (dates, names)
- These are retrieval-only baselines; the LoCoMo paper reports LLM-augmented results of 30-60% F1
- Compression: ~588 turns → ~600 anchors (~1:1, semantic structure instead of raw text)

## Memory Lifecycle

Every anchor moves through 6 states:

```
ACTIVE → REHEARSING → CONSOLIDATING → DORMANT → GHOST → REACTIVATED
```

- **Active**: Just created or recently accessed — fully plastic, easy to update
- **Rehearsing**: Being replayed during sleep — temporarily elevated importance
- **Consolidating**: Transferring from short-term to long-term — increasing stability
- **Dormant**: Stable, low-activity — read-only, cortical retrieval
- **Ghost**: Pruned but with residual trace — can partially recall or fully revive
- **Reactivated**: Ghost revived by new relevant experience — reduced stability, high plasticity

Ghosts are the key innovation. When a memory is pruned, it doesn't disappear — it leaves a compressed trace. If something similar appears later, the ghost resonates and can partially recall ("I seem to remember something about Redis...") or fully revive.

## Sleep Consolidation

Sleep is not cleanup. Sleep **changes the graph**:

1. **N1 Replay Indexing** — prioritizes surprising and emotional memories for replay
2. **N2 Weak Merge** — fuses near-duplicate anchors, bridges distant constellations
3. **N3 Compression** — transfers memories from hippocampal to cortical, forms schemas
4. **REM Emotional Decoupling** — strips emotional charge from consolidated memories
5. **Wake-prep** — prunes weak anchors and edges, refreshes search indices

Each phase produces metrics captured in a human-readable `SleepReport`.

## Edge Versioning

Edges aren't static. They track:

```python
@dataclass
class RichEdge:
    confidence: float          # explicit=0.95, implicit=0.42, inferred=0.3
    source_type: str           # "explicit", "implicit", "inferred"
    reinforcement_count: int   # how many times this connection was confirmed
    decay_rate: float          # how fast it weakens without use
    is_stale: bool             # has this been superseded?
    replaced_by: str           # which edge replaced it
    version_history: list      # full change log
```

Connections strengthen with use and decay with neglect — exactly like real memory.

## Memory Evolution

The evolution engine runs continuously (or on-demand):

- **Time decay**: older memories fade unless accessed (w_t = w_0 * e^(-λt))
- **Frequency boost**: repeatedly recalled memories get stronger
- **Conflict resolution**: detects changing beliefs, creates contradiction edges instead of overwriting
- **Interference**: similar memories compete — proactive (old inhibits new) + retroactive (new inhibits old)

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

5-stage cognitive lifecycle:
1. Encode 10 anchors (Python, Flask, FastAPI, Redis, ...)
2. Semantic graph forms via embedding similarity
3. Sleep → abstraction emerges: "Python + backend + FastAPI"
4. Weak anchor pruned → ghost with fuzzy recall
5. Related query → ghost resonates → **revived**

## Benchmarks

```bash
python examples/memory_benchmark.py --quick   # 4 sessions, ~200 turns
python examples/memory_benchmark.py --full    # 12 sessions, ~5000 turns
```

## Installation

```bash
git clone https://github.com/Thatgfsj/star-graph-memory.git
cd star-graph-memory

# Install in editable mode (required for the Quick Start imports)
pip install -e .

# Optional: for SQLite storage backend
pip install aiosqlite

python examples/emergence_demo.py
```

**Note:** `star-graph-memory` is not published on PyPI. Install from source as shown above.

## Current limitations

- **No streaming consolidation**: sleep processes all anchors at once. Incremental mode planned.
- **Abstraction threshold sensitivity**: too strict → no abstractions form. Too loose → false patterns.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned work.

## License

MIT
