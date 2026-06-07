# NeuroWeave Cortex (NWC) — 星图记忆引擎 · Star-Graph Cognitive Engine

Not a vector database. Not a graph database. Not RAG. A **cognitive character** — an observatory that illuminates a star field of memories, each star's brightness depending on **where you stand and what you're looking for**. It forgets, distorts, has emotional judgment, makes free associations, and walks discovery paths through a multi-dimensional star graph — the way biological cognition does.

```
v1.2.11 | 134 modules | Observation-Driven Retrieval | Cognitive Character
```

## What makes it different

Vector databases retrieve. Graph databases traverse. NeuroWeave Cortex runs a **full cognitive lifecycle**:

| Capability | Vector DB | Graph DB | NW Cortex |
|---|---|---|---|
| Semantic retrieval | yes | no | yes |
| Graph traversal | no | yes | yes |
| Automatic forgetting (survival decay) | no | no | yes |
| Memory strengthening (rehearsal) | no | no | yes |
| Conflict detection & resolution | no | no | yes |
| Multi-strategy RRF retrieval fusion | no | no | yes |
| **Observatory: observer-dependent luminosity** | no | no | **yes** |
| **Gravity wells: query inertia & momentum** | no | no | **yes** |
| **Source attribution & trust scores** | no | no | **yes** |
| **Event-anchored timeline** | no | no | **yes** |
| Cross-Encoder reranking | no | no | yes |
| Explainable reasoning paths | no | no | yes |
| Memory revision engine (sleep) | no | no | yes |
| Hot/Warm/Cold memory tiering | no | no | yes |
| Multimodal (text+image+audio) | no | no | yes |
| Fuzzy recall ("I think I remember...") | no | no | yes |
| Emergent abstraction (pattern discovery) | no | no | yes |
| Temporal context (TimeSpine-indexed) | no | no | yes |
| 8-phase sleep consolidation | no | no | yes |
| Ghost revival (savings effect) | no | no | **world-first** |
| Hebbian edge learning | no | no | yes |
| 4-layer memory pyramid | no | no | yes |
| Goal-driven inference | no | no | yes |
| **Working Memory Workspace** | no | no | **yes** |
| **Spreading Activation Engine** | no | no | **yes** |
| **Salience-based attention** | no | no | **yes** |
| **Self Model → LLM prompt injection** | no | no | **yes** |
| **Autonomous reasoning loop** | no | no | **yes** |
| **Full memory lifecycle (9 stages)** | no | no | **yes** |

## Quick Start

### Classic API (Phase 1-5)

```python
from star_graph import MemoryManager
from star_graph.scheduler import AgentContext

mgr = MemoryManager()

# Remember
mgr.remember("User debugs Redis connection timeout — pool was 10, fixed to 20",
             tags=["redis", "debug", "timeout"])
mgr.remember("User prefers type hints and concise code",
             tags=["preference", "style"])

# Working memory
mgr.remember_working("Currently debugging auth middleware timeout",
                     tags=["debug", "auth"])

# Context-aware recall
ctx = AgentContext(task_type="debugging", active_goals=["fix Redis connection"])
memories = mgr.recall("Redis connection pool config", context=ctx)
print(memories.memory_summary)

# Sleep consolidation
report = mgr.sleep()

# Persist
mgr.save("agent_memory.json")
```

### Cognitive Cortex API (Phase 6)

```python
from star_graph import (
    PerceptionLayer, CognitiveWorkspace, ConceptCortex,
    ActivationEngine, GoalSystem, SalienceEngine, SelfModel
)

# ── Full cognitive pipeline ──
pl = PerceptionLayer()
ws = CognitiveWorkspace(max_items=20)
cc = ConceptCortex()
ae = ActivationEngine(graph=graph)
gs = GoalSystem()
se = SalienceEngine()

# 1. Perceive user input
frame = pl.perceive("I need to debug a Redis memory leak urgently!")
# → PerceptionFrame(intent="command", valence=-0.4, concepts=["redis","debug","leak"])

# 2. Admit to working memory workspace
ws.on_perception(frame)

# 3. Activate concepts
for concept in frame.extracted_concepts:
    cc.get_or_create_concept(concept)

# 4. Spread activation through memory graph
result = ae.activate_from_perception(frame)

# 5. Track goals
goal = gs.add_goal("Fix Redis pipeline memory leak", priority=0.9)
gs.activate_goal(goal.id)

# 6. Compute salience — decide what enters attention
signals = [se.compute_salience(item.id, "thought",
    context={"task_relevance": item.relevance_to_current_task})
    for item in ws.get_items()]
winners = se.compete(signals, max_winners=5)

# 7. Build self-model → LLM prompt injection
sm = SelfModel(workspace=ws, goal_system=gs, concept_cortex=cc, salience_engine=se)
prompt = sm.get_prompt_injection()
# → "# Cognitive State Summary\n## Current Focus: ...\n## Active Goals: ..."

# This is what the LLM sees — NOT raw memory dumps!
```

## Architecture

Four-layer cognitive design. Phase 6 adds the Cortex layer above Behavior.

```
Layer 4: Cortex      │  Perception, Working Memory Workspace, Concept Cortex,
    (Phase 6)        │  Spreading Activation, Goal System, Salience Engine,
    perception.py,   │  Self Model, Autonomous Reasoning, Cognitive Compression,
    cognitive_workspace.py,│  Memory Lifecycle Manager
    concept_cortex.py,│  "What am I thinking right now? What should I focus on?"
    activation_engine.py, │
    goal_system.py,  │
    salience.py,     │
    self_model.py,   │
    autonomous_reasoning.py,│
    cognitive_compression.py,│
    memory_lifecycle.py)│
                     │
Layer 3: Behavior    │  Cortex routing, memory gating, working memory,
    (cortex.py,      │  dual-channel retrieval, scheduler, quality scoring,
     router.py,      │  memory budget, stability control, cognitive priority
     gate.py,        │  "What should I recall right now, at what detail level?"
     working_memory.py,│
     scheduler.py,   │
     quality_score.py,│
     memory_budget.py,│
     stability_control.py,│
     cognitive_priority.py)│
                     │
Layer 2: Cognitive   │  Hub abstraction, cascade recall, TimeSpine temporal index,
    (retriever.py,   │  sleep consolidation, evolution, ghost revival,
     sleep.py,       │  abstraction, community detection, competition,
     evolution.py,   │  conflict detection, memory revision, cross-encoder,
     ghost.py,       │  Hebbian learning, domain routing, context routing
     abstraction.py, │  "How do memories connect, strengthen, and fade?"
     community.py,   │
     competition.py, │
     hebbian_learning.py,│
     domain_graph.py,│
     context_routing.py)│
                     │
Layer 1: Storage     │  CRUD, persistence, ANN indexing, tiered storage,
    (graph.py,       │  BM25 keyword index, multi-level caching, typed memory,
     anchor.py,      │  4-layer memory pyramid, abstraction chain
     storage.py,     │  "Where is this memory stored?"
     index.py,       │
     bm25.py,        │
     typed_memory.py,│
     memory_layers.py,│
     abstraction_chain.py)│
```

### Core modules (Phase 1-4 foundation)

| Module | Role |
|---|---|
| `manager.py` | High-level facade — `remember()`, `recall()`, `sleep()`, `save()` |
| `runtime.py` | Dependency container — manages all subsystem lifecycles |
| `scheduler.py` | Context-aware retrieval with memory type selection |
| `working_memory.py` | Short-term buffer (15-item, 1h TTL) — auto-promotes to long-term |
| `sleep.py` | 8-phase sleep: N1_Replay → N2_Merge → N2b_Conflict → N2c_Revision → N3_Compression → N3d_Rebuild → REM_Emotion → N4_Prune |
| `evolution.py` | Survival-based decay (Ebbinghaus/Power-law/Exponential), belief transitions |
| `retriever.py` | HybridFusion + OscillationResonance + VectorSimilarity + Personalized PageRank |
| `bm25.py` | Sparse keyword retrieval with RRF fusion |
| `cross_encoder.py` | Cross-Encoder reranking for precision improvement |
| `ghost.py` | Ghost traces, fuzzy recall, savings effect revival |
| `abstraction.py` | Emergent category discovery from anchor clusters |
| `community.py` | Louvain community detection with centroid routing |
| `anchor.py` | Memory unit with 6-state lifecycle, 10-dim AnchorVector |
| `graph.py` | Star graph with RichEdge, Schema, ReflectionNode |
| `compression.py` | Multi-level session compression (episodic/strategic/meta) |
| `forget_certificate.py` | Ed25519-signed JWS deletion certificates — GDPR Article 17 |
| `multimodal.py` | Cross-modal text/image/audio memory |
| `zero_llm_pipeline.py` | 7-stage zero-LLM ingestion pipeline |
| `batch_vectorizer.py` | Deferred batch embedding writes with SQLite WAL |
| `markdown_export.py` | GBrain-aligned Markdown memory export |

### Phase 5 — Cognitive Depth

| Module | Role |
|---|---|
| `memory_budget.py` | Token + anchor budget management with eviction policies |
| `quality_score.py` | 7-dimension memory quality scoring (usage, reasoning, feedback, etc.) |
| `stability_control.py` | Long-term stability with exponential/linear decay, drift monitoring |
| `memory_layers.py` | 4-layer pyramid: Working → Episodic → Semantic → Core Identity |
| `typed_memory.py` | 7 memory types (code/task/dialogue/tool_call/knowledge/event/preference) |
| `abstraction_chain.py` | Event → Summary → Pattern → Identity abstraction pipeline |
| `domain_graph.py` | Domain-based graph partitioning with soft isolation |
| `context_routing.py` | 6-dimension context-aware retrieval routing |
| `hebbian_learning.py` | Hebbian edge learning — "neurons that fire together wire together" |
| `agent_state.py` | Agent state memory: goal tree, tool calls, checkpoints |
| `cognitive_closure.py` | Closed-loop feedback: recall → use → learn → improve |
| `cognitive_priority.py` | 5-level priority assignment with forced injection for core identity |

### Phase 6 — Cognitive Cortex

| Module | Role |
|---|---|
| `thought_object.py` | Unified cognitive base class — activated nodes, not static memory |
| `perception.py` | Raw text → structured PerceptionFrame (intent, emotion, goals, concepts) |
| `cognitive_workspace.py` | **Core**: Working memory workspace with reasoning chains, attention, TTL decay |
| `concept_cortex.py` | Concept network: activation, fusion, competition, 10 built-in core concepts |
| `activation_engine.py` | Multi-source spreading activation (query/goal/concept/emotion seeds) |
| `goal_system.py` | Goal hierarchy, conflict detection, goal-driven inference |
| `salience.py` | 10-component attention competition with lateral inhibition |
| `cognitive_compression.py` | Events → Concepts → Identity → World Model compression pipeline |
| `self_model.py` | System self-model → compressed CognitiveState → LLM prompt injection |
| `autonomous_reasoning.py` | Contradiction → Activation → Resolution loop (no LLM required) |
| `memory_lifecycle.py` | Unified 9-stage lifecycle: Perception → Working → ... → Ghost → Dead |

## Retrieval Benchmarks

### LoCoMo Benchmark (real-world conversations)

Evaluated on the [LoCoMo-10 dataset](https://github.com/snap-research/LoCoMo): 10 conversations, 5,882 turns across 272 sessions, 1,986 QA pairs across 5 categories. Pure retrieval (no LLM generation). Token-matching has_answer & F1 following the LoCoMo paper methodology.

| Method | has_answer | F1 | Latency |
|---|---|---|---|
| VectorSimilarity | 25.3% | 0.016 | 57.4ms |
| OscillationResonance | 24.8% | 0.016 | 143.9ms |
| **HybridFusion + BM25** | **31.1%** | **0.017** | 297.2ms |

Per-category has_answer (HybridFusion):

| Category | #QA | Description | has_answer |
|---|---|---|---|
| 1 (Temporal) | 282 | Time/date/sequence questions | 11.0% |
| 2 (Short Memory) | 321 | Recent fact recall | 3.1% |
| 3 (Long Memory) | 96 | Distant fact recall | 7.3% |
| 4 (Composite) | 841 | Multi-step reasoning | 44.0% |
| 5 (Adversarial) | 446 | Misleading/distractor queries | 44.8% |

**Key observations:**
- Category 1 (temporal) and 2-3 (short/long memory) remain weak — these require exact token matching of dates and entities, which pure embedding struggles with
- Category 4 (composite) and 5 (adversarial) show reasonable performance — the multi-strategy retrieval handles these well
- LLM-as-judge evaluation (MiniMax M2.7): HybridFusion achieves **llm_judge_score=0.05** (vs 0.025 for VectorSimilarity, 0.075 for OscillationResonance), indicating that retrieval results often lack the full context needed for answer generation
- **Next target**: Improve category 1-3 via BM25+embedding RRF fusion and query expansion (see [plan.md](plan.md))

## Memory Lifecycle (9 stages)

Every memory moves through a full 9-stage cognitive lifecycle (Phase 6.10 `memory_lifecycle.py`):

```
PERCEPTION → WORKING → SHORT_TERM → LONG_TERM → CONSOLIDATED → ARCHIVED → DORMANT → GHOST → DEAD
```

| Stage | Description | Duration |
|-------|-------------|----------|
| **Perception** | Initial intake, not yet stored | seconds |
| **Working** | In cognitive workspace, high activation | minutes~hours |
| **Short-term** | Stored, not yet consolidated | hours~days |
| **Long-term** | Consolidated, actively retrievable | days~months |
| **Consolidated** | Deeply integrated, high stability | months~years |
| **Archived** | Cold storage, low access frequency | indefinite |
| **Dormant** | Very low access, near ghost | indefinite |
| **Ghost** | Forgotten but potentially revivable | 90 days |
| **Dead** | Permanently removed | — |

Paired with **ThermalState** (HOT → WARM → COLD → DEAD) for storage tier switching and **4-layer pyramid** (Working → Episodic → Semantic → Core Identity).

## Sleep Consolidation (8 phases)

Sleep is not cleanup. Sleep **changes the graph**:

1. **N1_Replay** — SWR-scored replay of surprising/emotional memories
2. **N2_Merge** — ANN-accelerated near-duplicate fusion + constellation bridging
3. **N2b_Conflict** — Semantic contradiction detection (overwrite/coexist/deprecate)
4. **N2c_Revision** — Surprise-prioritized memory revision engine
5. **N3_Compression** — Hippocampal→cortical transfer, schema formation
6. **N3d_Rebuild** — Multi-node fusion, edge rewiring, abstractive pattern discovery
7. **REM_Emotion** — Emotional decoupling from consolidated memories
8. **N4_Prune** — Weak anchor/edge removal, ghost trace creation (savings effect)

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
# From PyPI (package name: NWcortex, import as: star_graph)
pip install NWcortex

# With sentence-transformers for semantic embeddings
pip install "NWcortex[embeddings]"

# With MCP server support (for AI agent integration)
pip install "NWcortex[mcp]"

# Everything
pip install "NWcortex[all]"

# Run demo
python examples/emergence_demo.py
```

**Note:** Without `[embeddings]`, the system uses a lightweight TF-IDF fallback for text encoding. Install `sentence-transformers` only if you need semantic-quality embeddings.

## MCP Server — AI Agent Integration

NeuroWeave Cortex ships with a [Model Context Protocol](https://modelcontextprotocol.io) server. Connect any MCP-compatible AI agent (Claude Desktop, OpenClaw, Cursor, etc.) for persistent cognitive memory across conversations.

### Quick start

```bash
# Install with MCP support
pip install "NWcortex[mcp]"

# Launch the MCP server on stdio
nwc-mcp

# With persistent storage
nwc-mcp --storage agent_memory.json

# Load a previously saved memory graph
nwc-mcp --load my_memories.json
```

### Tool reference (12 tools)

| Tool | Description |
|------|-------------|
| `remember` | Store a long-term memory (tags, importance, emotional valence) |
| `remember_working` | Store in fast ephemeral working-memory buffer (current task context) |
| `recall` | Context-aware semantic retrieval with task-type routing |
| `forget` | Remove a memory, creating a ghost trace for potential fuzzy recall |
| `sleep` | Run 5-phase sleep consolidation — merges, prunes, forms schemas |
| `consolidate` | Micro-consolidation — incremental, non-blocking |
| `stats` | Memory system statistics (anchors, edges, ghosts, schemas, cognitive health) |
| `fuzzy_recall` | Low-confidence recall from ghost traces ("I seem to remember...") |
| `get_profile` | Inferred user profile from accumulated memories |
| `evolve` | Memory evolution cycle (decay, boost, conflict resolution) |
| `save` | Persist memory graph to disk (JSON) |
| `load` | Load memory graph from disk (JSON) |

### Claude Desktop config

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc-mcp",
      "args": ["--storage", "/path/to/agent_memory.json"]
    }
  }
}
```

### OpenClaw config

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc-mcp",
      "args": ["--storage", "/path/to/agent_memory.json"]
    }
  }
}
```

## REST API Server

```bash
# Start the REST server
nwc-server --port 8420

# Or via module
python -m star_graph.server --port 8420
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with anchor/edge counts |
| GET | `/metrics` | Prometheus-format metrics |
| GET | `/stats` | Full memory system statistics |
| POST | `/remember` | Store a memory `{"text": "...", "tags": [...]}` |
| POST | `/recall` | Retrieve memories `{"query": "...", "max_items": 10}` |
| POST | `/sleep` | Run sleep consolidation |
| POST | `/consolidate` | Run micro-consolidation |

## Interactive Demo

```bash
python examples/emergence_demo.py

# Or use the CLI:
# sg-add "Discussed microservices deployment patterns" --tags architecture --emotional 0.6
# sg-query "database connection pooling best practices"
# sg-query --trace "When did Alice visit Hawaii?"
# sg-stats --schemas --ghosts
# sg-sleep --retention 0.15 --edge-prune 0.1
# nwc-export --output memories/ --organize both
# nwc-forget <anchor_id> --certificate --reason gdpr_art17
# nwc-verify certificates/forget-<id>.jws
```

## Benchmarks

```bash
python examples/memory_benchmark.py --quick   # 4 sessions, ~200 turns
python examples/memory_benchmark.py --full    # 12 sessions, ~5000 turns
```

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v

# With coverage report
pytest tests/ --cov=star_graph --cov-report=term
```

**Status:** v1.2.11 | 134 modules | Observation-Driven Retrieval | Cognitive Character. See [plan.md](plan.md) for latest roadmap.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned work.

## License

MIT
