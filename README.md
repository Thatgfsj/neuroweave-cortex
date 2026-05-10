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
a1 = Anchor.create("Discussed Genette's narrative layering theory",
                    tags=["narratology", "literature"], emotional_valence=0.6)
a2 = Anchor.create("User prefers Python for crawlers, Rust for large programs",
                    tags=["preferences", "tech-stack"])
a3 = Anchor.create("Applied narrative levels to Mo Yan's Carpenter and Dog analysis",
                    tags=["narratology", "thesis"], emotional_valence=0.8)
graph.add_anchor(a1)
graph.add_anchor(a2)
graph.add_anchor(a3)

# Connect related anchors
graph.add_edge(a1.id, a3.id, weight=0.9, edge_type="topical")

# Resonance retrieval (hippocampal pathway)
resonator = Resonator(graph)
constellations = resonator.resonate("莫言叙事层次分析")
for c in constellations:
    for a in c.anchors:
        print(f"[{a.retention_score:.2f}] {a.text}")

# Predictive retrieval with action decision
constellation, action = resonator.predictive_retrieve("莫言小说中的叙事者")
print(f"Action: {action}")  # 'confirm', 'update', or 'novel'

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
sg-add "Discussed Genette's narrative layering" --tags narratology --emotional 0.6
sg-query "莫言小说叙事结构"
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
