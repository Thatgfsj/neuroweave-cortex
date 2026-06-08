<p align="center">
  <img src="https://img.shields.io/badge/version-1.3.4-blue?style=flat-square" alt="version"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-orange?style=flat-square" alt="python"/>
  <img src="https://img.shields.io/badge/status-beta-yellow?style=flat-square" alt="status"/>
  <img src="https://img.shields.io/github/stars/Thatgfsj/neuroweave-cortex?style=flat-square" alt="stars"/>
  <img src="https://img.shields.io/badge/LLM-routing%20free-success?style=flat-square" alt="llm-free"/>
  <img src="https://img.shields.io/badge/retrieval-activation%20graph-8A2BE2?style=flat-square" alt="activation"/>
  <img src="https://img.shields.io/badge/lifecycle-L1%E2%86%92L2%E2%86%92L3-blueviolet?style=flat-square" alt="lifecycle"/>
</p>

<h1 align="center">🧠 NeuroWeave Cortex</h1>
<h3 align="center"><i>A Hippocampal-Inspired Cognitive Memory Engine for AI Agents</i></h3>

<p align="center">
  <b>Not a vector database. Not a graph database. Not RAG.</b><br>
  <b>Not even an "LLM-powered memory system".</b><br>
  A <b>self-organizing cognitive memory</b> — memory is structured by <b>use, not by prompt</b>.<br>
  <b>No memory is ever deleted. Only accessibility changes.</b><br>
  LLM reads memory. LLM does <b>not</b> decide what memory says.
</p>

<br>

---

## ✨ Core Design Principle

> **Memory structure is decided by the memory system, not by the LLM.**

The LLM can *read* memory. The LLM can *maintain* memory (summarize, merge, archive).  
But the LLM does **not** determine retrieval paths, edge weights, or relevance scores.

**Brightness = f(historical access frequency, recency, edge strength)**  
**NOT f(LLM-generated vector at query time)**

This is the difference between a memory system that *learns* and one that *re-computes* every time.

---

## 🔬 What Makes It Different

Vector databases retrieve. Graph databases traverse. RAG systems query.  
NeuroWeave Cortex **learns from use**:

| Capability | Vector DB | Graph DB | RAG | NWC |
|------------|:---------:|:--------:|:---:|:---:|
| Semantic retrieval | ✅ | ❌ | ✅ | ✅ |
| Graph traversal | ❌ | ✅ | ❌ | ✅ |
| **Activation-based retrieval** | ❌ | ❌ | ❌ | ✅ |
| **Edge decay (no deletion)** | ❌ | ❌ | ❌ | ✅ |
| **Memory never deleted** | ❌ | ❌ | ❌ | ✅ |
| **Memory lifecycle (L1→L2→L3)** | ❌ | ❌ | ❌ | ✅ |
| **LLM-free retrieval routing** | ✅ | ✅ | ❌ | ✅ |
| **LLM for maintenance only** | ❌ | ❌ | ❌ | ✅ |
| Sleep consolidation | ❌ | ❌ | ❌ | ✅ |
| Source attribution & trust | ❌ | ❌ | ❌ | ✅ |

---

## 🧠 How Retrieval Actually Works

### The Wrong Way (what most systems do):
```
Query → LLM generates embedding → LLM decides what's relevant → Return
```
Problem: **LLM is the router.** Every retrieval is a new LLM computation. Nothing stabilizes.

### The NWC Way (v1.3.1):
```
Query → Embedding → Find seed nodes → Activation spread → Return
```
The LLM never touches the retrieval path. Retrieval is driven by:
1. **Historical activation** — how often has each memory been accessed?
2. **Edge strength** — how strongly are memories connected (reinforced by co-use)?
3. **Temporal decay** — edges weaken naturally over days of disuse (but NEVER to zero)
4. **Recency boost** — recently accessed memories activate more readily

### 🗝️ Core Design Principle: No Memory Is Ever Deleted

```
Traditional memory systems:  Store → Retrieve → Delete (when full)
NWC memory system:          Create → Activate → Dim → Dormant → Reactivate ♾️
```

Memories do NOT have a "delete" state. They transition between activation levels:
- **1.0** — currently active (just recalled)
- **0.7** — frequent access
- **0.3** — infrequent but known  
- **0.1** — dormant (not accessed in a long time)
- **0.01** — deep dormant (years old)

Even a memory with `activation_level = 0.01` is retrievable — if activation
propagation from a connected query reaches it. This mimics human memory:
you may not think about your elementary school classmates for decades,
but a single photo can bring everything flooding back.

---

## 🏗️ Memory Layers (L0–L3)

NWC organizes memory into four cognitively-grounded layers:

```
L0: Input    ─── Ephemeral, per-session only (minutes)
    ↓ promote on repeated access (>3 times) or age (>30 days)
L1: Working  ─── Recent active memories (hours-days)
    │               LLM maintains: add, merge, summarize
    │               System routes: retrieval is LLM-free
    ↓ consolidate
L2: Long-term ─── Stable facts, experiences, relationships (months-years)
    │               ONLY operations: strengthen, supplement, weaken
    │               NEVER: full reconstruction
    ↓ archive on 90d no-access or low importance
L3: Dormant  ─── Low-activation memories retrievable via propagation (years)
    │               NEVER deleted. Only activation_level approaches 0.
    ↑ reactivate on query similarity or activation propagation
```

This is **not** a flat table. Memories physically transition between layers
based on access patterns — just like human memory.

---

## 🚀 Quick Start

```bash
pip install NWcortex
```

```python
from star_graph import MemoryManager, AgentContext

mgr = MemoryManager()

# Remember
mgr.remember("User prefers type hints and concise code", tags=["preference"])
mgr.remember("Debugged Redis timeout — pool 10 → 20", tags=["redis","debug"])

# Recall (LLM-free activation spreading)
ctx = AgentContext(task_type="debugging")
memories = mgr.recall("Redis connection pool config", context=ctx)

# Sleep: consolidate, migrate layers, decay weak edges
report = mgr.sleep()

# Persist
mgr.save("agent_memory.db")
```

---

## 📦 Installation

| Command | Includes |
|---------|---------|
| `pip install NWcortex` | Core engine (138 modules) |
| `pip install "NWcortex[embeddings]"` | + sentence-transformers |
| `pip install "NWcortex[mcp]"` | + MCP Server |
| `pip install "NWcortex[all]"` | Everything |

---

## 🧪 Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

---

## 📄 Citation

```bibtex
@software{neuroweave_cortex,
  title = {NeuroWeave Cortex: A Hippocampal-Inspired Cognitive Memory System},
  author = {Thatgfsj},
  year = {2026},
  url = {https://github.com/Thatgfsj/neuroweave-cortex}
}
```

---

## ⚖️ License

MIT

---

<p align="center">
  <a href="https://github.com/Thatgfsj/neuroweave-cortex/issues">Report a bug</a> ·
  <a href="https://github.com/Thatgfsj/neuroweave-cortex/discussions">Discussion</a> ·
  <a href="https://github.com/Thatgfsj/neuroweave-cortex/blob/main/ROADMAP.md">Roadmap</a>
</p>
