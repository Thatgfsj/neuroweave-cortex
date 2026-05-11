# Star Graph Memory · 星图记忆

A hippocampal-inspired long-term memory system for AI agents.

Conversations are compressed into **anchor points** (≤200 chars) connected in a navigable **star graph**. Retrieval uses **oscillatory phase-locking resonance** — not keyword search. Consolidation happens during **9-phase sleep cycles**.

## Theory

The system is grounded in seven neuroscience principles:

| Principle | Mechanism |
|-----------|-----------|
| **Sharp-Wave Ripples** | Compressed (~15×) replay of prioritized recent memories during sleep |
| **Memory Reconsolidation** | Prediction error gates: confirm / update / create-new |
| **Phase Precession** | Theta-gamma oscillatory resonance encodes sequences in phase offsets |
| **Schema Formation** | Abstraction of common patterns across episodes (mPFC analog) |
| **Predictive Coding** | Free energy minimization unifies encoding, retrieval, and sleep |
| **Emotional Modulation** | Two-factor tagging (NE+CORT) boosts encoding; REM strips charge |
| **Adaptive Forgetting** | Precision-weighted decay; ghosts enable savings effect on relearning |

## Architecture

```
                  ┌─────────────────────────┐
                  │   New Experience / Query  │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │   RESONANCE ENGINE        │
                  │   - Oscillatory phase-    │
                  │     locking (hippocampal) │
                  │   - Cortical direct       │
                  │     lookup                │
                  │   - Prediction error      │
                  │     minimization          │
                  └───────────┬─────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
   ┌────────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
   │  CONFIRM       │  │  UPDATE      │  │  NOVEL        │
   │  Strengthen    │  │  Reconsolid. │  │  New Anchor   │
   └────────┬──────┘  └──────┬──────┘  └───────┬──────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │   STAR GRAPH              │
                  │   - Anchors + Oscillators │
                  │   - Ghosts (savings)      │
                  │   - Schemas               │
                  │   - Cortical Index        │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │   9-PHASE SLEEP ENGINE    │
                  │   1. SWR Replay           │
                  │   2. Systems Consolid.    │
                  │   3. Emotional Stripping  │
                  │   4. Schema Extraction    │
                  │   5. Merge Similar        │
                  │   6. Adaptive Prune+Ghosts│
                  │   7. Bridge Constellations│
                  │   8. Hebbian Update       │
                  │   9. Synaptic Homeostasis │
                  └───────────────────────────┘
```

## Installation

```bash
pip install star-graph-memory
# or from source:
git clone https://github.com/Thatgfsj/star-graph-memory.git
cd star-graph-memory
pip install -e .
```

## Quick Start

```python
from star_graph import StarGraph, Anchor
from star_graph.sleep import SleepCycle
from star_graph.resonance import Resonator
from star_graph.storage import Storage

graph = StarGraph()

# Create anchors with emotional tags
a1 = Anchor.create("Discussed microservices architecture and deployment patterns",
                    tags=["architecture", "backend"], emotional_valence=0.6)
a2 = Anchor.create("User prefers Python for automation, Rust for performance-critical work",
                    tags=["preferences", "tech-stack"])
a3 = Anchor.create("Decided to use Postgres with connection pooling for the API service",
                    tags=["architecture", "database"], emotional_valence=0.8)
graph.add_anchor(a1)
graph.add_anchor(a2)
graph.add_anchor(a3)

# Connect related anchors
graph.add_edge(a1.id, a3.id, weight=0.9, edge_type="topical")

# Resonance retrieval (hippocampal pathway)
resonator = Resonator(graph)
constellations = resonator.resonate("microservices deployment architecture")
for c in constellations:
    for a in c.anchors:
        print(f"[{a.retention_score:.2f}] {a.text}")

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

# Run 9-phase sleep cycle
cycle = SleepCycle(graph)
result = cycle.run()
print(f"Merged: {result['merged']}, Pruned: {result['pruned_anchors']}")
print(f"Ghosts: {result['ghosts_created']}, Schemas: {result['schemas_formed']}")

# Persist (includes ghosts, schemas, cortical index)
store = Storage()
store.save(graph)
```

## CLI Commands

```bash
sg-add "Discussed microservices deployment patterns" --tags architecture --emotional 0.6
sg-query "database connection pooling best practices"
sg-query --trace "When did Alice visit Hawaii?"
sg-stats --schemas --ghosts
sg-sleep --retention 0.15 --edge-prune 0.1
```

## Sleep Daemon

```powershell
# Install nightly task (2 AM)
powershell -ExecutionPolicy Bypass -File scripts/install_sleep_task.ps1
```

```bash
# Idle-watch mode
python scripts/sleep_daemon.py --mode idle --idle-threshold 15
```

## Deeper Reading

- `docs/research.md` — full theoretical framework and neuroscience references
- See the research agent's output for literature citations across all seven principles

## License

MIT
