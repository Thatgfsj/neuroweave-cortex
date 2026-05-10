# Star Graph Memory

A hippocampal-inspired long-term memory system for AI agents. Conversations are compressed into **anchor points** (≤200 chars) connected in a navigable **star graph**. Consolidation happens during **sleep cycles** — replay, merge, prune, and bridge.

## Core Concepts

### Anchor Points
Each anchor is a ≤200-character summary of a conversation turn or session, augmented with a dynamic vector (importance, frequency, recency, emotional valence, stability, surprise). Anchors are not static records — they evolve as they're used or forgotten.

### Star Graph
Anchors are nodes. Edges are typed (temporal, topical, causal, bridge) and weighted. Connected subgraphs form **constellations** — linked strings of related memories that are retrieved together, not as isolated facts.

### Resonance (Retrieval)
Instead of keyword search or vector similarity, memories are found by **resonance**: the current context activates seed anchors, spreading activation propagates through the graph, and the constellation that lights up is what you recall. This mirrors how a smell triggers not just one memory but an entire scene.

### Sleep (Consolidation)
Nightly cycles that mimic hippocampal replay:
1. **Replay** — Recent anchors are replayed against existing memory
2. **Merge** — Near-duplicate anchors fuse into core+variant structures
3. **Prune** — Weak anchors and dormant edges are removed
4. **Bridge** — Surprising connections between distant constellations are discovered
5. **Hebbian update** — Co-activated edges strengthen; dormant ones fade

## Installation

```bash
pip install star-graph-memory
```

Or from source:

```bash
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

# Create a graph and add anchors
graph = StarGraph()
a1 = Anchor.create("Discussed narrative layering theory with user — Genette's framework")
a2 = Anchor.create("User prefers Python for crawlers, Rust for large programs")
a3 = Anchor.create("Applied Genette's narrative levels to Mo Yan's short story analysis")
graph.add_anchor(a1)
graph.add_anchor(a2)
graph.add_anchor(a3)

# Connect related anchors
graph.add_edge(a1.id, a3.id, weight=0.9, edge_type="topical")

# Retrieve by resonance
resonator = Resonator(graph)
results = resonator.resonate("莫言叙事层次分析")
for c in results:
    for anchor in c.anchors:
        print(f"[{anchor.retention_score:.2f}] {anchor.text}")

# Run sleep consolidation
cycle = SleepCycle(graph)
result = cycle.run()
print(result["stats_after"])

# Persist
store = Storage()
store.save(graph)
```

## CLI

```bash
# Add an anchor
sg-add "Discussed Genette's narrative layering with user" --tags narratology,literature

# Query by resonance
sg-query "莫言小说叙事结构"

# Trigger sleep cycle
sg-sleep --retention 0.15 --edge-prune 0.1
```

## Sleep Daemon

Install as a scheduled task to run nightly at 2 AM:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts/install_sleep_task.ps1

# Linux/macOS (cron)
# 0 2 * * * python /path/to/sleep_daemon.py --mode scheduled
```

Or run in idle-watch mode (triggers when user is away):

```bash
python scripts/sleep_daemon.py --mode idle --idle-threshold 15
```

## Architecture

```
star_graph/
├── anchor.py      # Anchor data model + dynamic vector
├── graph.py       # Star graph: nodes, edges, constellations, spreading activation
├── resonance.py   # Resonance-based retrieval + bridge scoring
├── sleep.py       # Sleep cycle: replay → merge → prune → bridge → Hebbian
├── storage.py     # JSON persistence (swap with SQLite/vector DB for production)
└── cli.py         # CLI entry points
scripts/
├── sleep_daemon.py        # Background sleep daemon
└── install_sleep_task.ps1  # Windows Task Scheduler installer
```

## Design Principles

- **Not a database** — Memories are navigated, not queried. Spreading activation over a graph, not SELECT with WHERE
- **Not a vector store** — Semantic similarity is a bridge-building tool, not the retrieval mechanism. Resonance > cosine similarity
- **Sleep is maintenance, not downtime** — Like biological sleep, consolidation is when the real work of memory happens
- **Memories evolve** — An anchor's retention score changes with use. Important memories survive; trivia fades. No manual curation needed
