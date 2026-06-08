<p align="center">
  <img src="https://img.shields.io/badge/version-1.2.14-blue?style=flat-square" alt="version"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-orange?style=flat-square" alt="python"/>
  <img src="https://img.shields.io/badge/status-beta-yellow?style=flat-square" alt="status"/>
  <img src="https://img.shields.io/github/stars/Thatgfsj/neuroweave-cortex?style=flat-square" alt="stars"/>
  <img src="https://img.shields.io/badge/LoCoMo--10-41.5%25-brightgreen?style=flat-square" alt="benchmark"/>
  <img src="https://img.shields.io/badge/LLM%20calls-zero-success?style=flat-square" alt="llm-free"/>
</p>

<h1 align="center">🧠 NeuroWeave Cortex</h1>
<h3 align="center"><i>A Hippocampal-Inspired Cognitive Memory Engine for AI Agents</i></h3>

<p align="center">
  <b>Not a vector database. Not a graph database. Not RAG.</b><br>
  A <b>cognitive character</b> — an observatory that illuminates a star field of memories,<br>
  each star's brightness depending on <b>where you stand and what you're looking for</b>.
</p>

<br>

---

## ✨ Features at a Glance

<table>
<tr>
  <td width="33%" align="center"><b>🧠 Cognitive Memory</b><br><small>9-stage lifecycle · 8-phase sleep · Forgetting & consolidation</small></td>
  <td width="33%" align="center"><b>⭐ Star-Graph Retrieval</b><br><small>Observer-dependent luminosity · 6-path RRF fusion · BM40+txt60</small></td>
  <td width="33%" align="center"><b>🔌 Zero LLM Required</b><br><small>37.8% → 41.5% has_answer · No API costs · CPU-only deployable</small></td>
</tr>
<tr>
  <td align="center"><b>🔄 Sleep Consolidation</b><br><small>N1→N2→N2b→N2c→N3→N3d→REM→N4 · Trust-based conflict resolution</small></td>
  <td align="center"><b>🎯 Observatory Engine</b><br><small>Query · Mood · Goal lanterns · Gravity wells for conversational inertia</small></td>
  <td align="center"><b>📦 135 Modules</b><br><small>Production-ready · Pluggable storage · MCP Server · REST API</small></td>
</tr>
</table>

---

## 📊 Performance

<p align="center">
  <b>LoCoMo-10 Benchmark</b> — 10 conversations · 5,882 turns · 1,986 QA pairs · Zero LLM calls
</p>

| Method | has_answer | Δ vs Vector | LLM Needed |
|--------|:----------:|:-----------:|:----------:|
| Pure Vector Search | 25.3% | — | ❌ |
| Cosine + BM25 | 31.5% | +6.2 pp | ❌ |
| **NWC HybridFusion** | **37.8%** | **+12.5 pp** | ❌ |
| **NWC BM40+txt60** 🏆 | **41.5%** | **+16.2 pp** | **❌ Zero** |

<details>
<summary><b>📈 Per-Category Breakdown (click to expand)</b></summary>

| Category | #QA | Description | has_answer | Δ vs Vector |
|:--------:|:---:|-------------|:----------:|:-----------:|
| 1 (Temporal) | 282 | Time/date/sequence | 14.5% | +5.6 |
| 2 (Short Mem) | 321 | Recent fact recall | 6.2% | +4.3 |
| 3 (Long Mem) | 96 | Distant fact recall | 9.4% | +7.3 |
| 4 (Composite) | 841 | Multi-step reasoning | **53.5%** | **+35.5** |
| 5 (Adversarial) | 446 | Misleading queries | **51.6%** | **+31.2** |

</details>

<details>
<summary><b>⚗️ Ablation Study (click to expand)</b></summary>

| Configuration | has_answer | Δ from Full |
|--------------|:----------:|:-----------:|
| **Full System** | **41.5%** | — |
| No Sleep | 38.9% | -2.6 pp |
| No BM25 | 35.2% | -6.3 pp |
| No Spreading | 39.7% | -1.8 pp |
| No Cache | 40.7% | -0.8 pp |
| Pure Vector | 25.3% | -16.2 pp |

</details>

---

## 🚀 Quick Start

```bash
pip install NWcortex
```

```python
from star_graph import MemoryManager, AgentContext

# Create a cognitive memory system
mgr = MemoryManager()

# Remember facts with context
mgr.remember("User prefers type hints and concise code", tags=["preference"])
mgr.remember("Debugged Redis timeout — pool size was 10, fixed to 20", 
             tags=["redis", "debug"])

# Context-aware recall
ctx = AgentContext(task_type="debugging", active_goals=["fix Redis"])
memories = mgr.recall("Redis connection pool config", context=ctx)
print(memories.memory_summary)

# Sleep consolidation (merges, prunes, strengthens)
report = mgr.sleep()

# Persist
mgr.save("agent_memory.db")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    🔭 Observatory Layer                      │
│  Query Lantern · Mood Lantern · Goal Lantern · Gravity Well │
│  ↓ Luminosity function: L = ρ · I · A · D · C              │
├─────────────────────────────────────────────────────────────┤
│                  ⭐ Star Graph Memory                        │
│  Core Memory · Short-term Memory · Long-term Memory         │
│  3-layer coordinate space · Hebbian edge learning           │
├─────────────────────────────────────────────────────────────┤
│               🔍 Retrieval Engine (6-Path RRF)               │
│  P0:Exact│P1:BM25│P2:Semantic│P3:Graph│P4:Spreading│P5:Cache│
│  Weighted Reciprocal Rank Fusion + Semantic Re-ranking      │
├─────────────────────────────────────────────────────────────┤
│         💤 Sleep Consolidation (8-Phase Pipeline)            │
│  N1→N2→N2b→N2c→N3→N3d→REM→N4 · Source Trust Conflict Res. │
├─────────────────────────────────────────────────────────────┤
│                 💾 Storage Backend                           │
│        JSON (Dev) · SQLite (Default) · Qdrant (Scale)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 💤 How Sleep Works

Sleep is not cleanup. Sleep **changes the graph**:

```
N1_Replay    → Priority-weighted memory replay
N2_Merge     → ANN-accelerated near-duplicate fusion
N2b_Conflict → Source-trust based contradiction resolution  
N2c_Revision → Evidence-based memory revision
N3_Compress  → Episodic cluster → semantic schema abstraction
N3d_Rebuild  → Edge rewiring, promote to long-term
REM_Emotion  → Emotional valence decoupling
N4_Prune     → Ghost trace creation for low-retention items
```

**Source Trust Override:** `self_reported(0.9) > tool_output(0.8) > user_told_me(0.7) > observation(0.6) > inferred(0.4)`

---

## 📦 Installation Options

| Command | Includes |
|---------|---------|
| `pip install NWcortex` | Core engine (135 modules) |
| `pip install "NWcortex[embeddings]"` | + sentence-transformers |
| `pip install "NWcortex[mcp]"` | + MCP Server |
| `pip install "NWcortex[all]"` | Everything |

---

## 🤖 MCP Server — AI Agent Integration

Connect any MCP-compatible agent (Claude Desktop, Cursor, OpenClaw) for persistent cognitive memory:

```bash
pip install "NWcortex[mcp]"
nwc-mcp --storage agent_memory.db
```

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc-mcp",
      "args": ["--storage", "/path/to/agent_memory.db"]
    }
  }
}
```

**14 MCP Tools:** `remember` · `recall` · `sleep` · `forget` · `fuzzy_recall` · `consolidate` · `evolve` · `stats` · `get_profile` · `save` · `load` · `remember_working` · `image_search` · `text_to_image`

---

## 🔬 CLI Commands

```bash
# Store a memory
sg-add "Discussed microservices deployment patterns" --tags architecture --emotional 0.6

# Retrieve with context
sg-query "database connection pooling best practices"

# Sleep consolidation
sg-sleep --retention 0.15

# View system health
sg-stats --schemas --ghosts

# Forget with GDPR-compliant certificate
nwc-forget <anchor_id> --certificate --reason gdpr_art17

# Export memories to Markdown
nwc-export --output memories/ --organize both
```

---

## 🧪 Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ --cov=star_graph --cov-report=term
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

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>⭐ Star us on GitHub — it helps others discover the project!</b><br>
  <a href="https://github.com/Thatgfsj/neuroweave-cortex/issues">Report a bug</a> ·
  <a href="https://github.com/Thatgfsj/neuroweave-cortex/discussions">Discussion</a> ·
  <a href="https://github.com/Thatgfsj/neuroweave-cortex/blob/main/ROADMAP.md">Roadmap</a>
</p>
