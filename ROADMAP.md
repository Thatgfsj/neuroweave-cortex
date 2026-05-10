# Star Graph Memory — Roadmap

## v0.3.0 — "Verifiable" (2026 Q3, June-July)

**Goal**: Every mechanism has a test. Every claim is measurable.

### P0 — Engineering Verifiability
- [x] End-to-end runnable example (`examples/memory_basic.py`)
- [x] 13-test suite covering sleep consolidation + oscillation retrieval
- [x] Bugfixes: merge KeyError, score clamping, anchor create TypeError
- [x] Chinese README with architecture diagram and quick start
- [ ] `sentence-transformers` integration for real embeddings
- [ ] GitHub Actions CI: `pytest` + `flake8` on push/PR
- [ ] `tune_threshold.py`: Bayesian optimization of retention/similarity/homeostasis params
- [ ] Re-run oscillation resonance recall benchmarks with real embeddings

### P1 — Core Mechanism Decoupling
- [x] `retriever.py`: Abstract `Retriever`, `OscillationResonanceRetriever`, `VectorSimilarityRetriever`
- [x] `compare_retrievers()`: side-by-side comparison framework with latency + recall@k
- [x] `docs/neuro_mapping.md`: neuroscience→algorithm mapping table
- [x] `docs/benchmark.md`: comparison vs Mem0, HippoRAG, MemGPT, LangChain
- [ ] `mem0_adapter.py`: drop-in replacement for Mem0 users (`from star_graph.adapter import Memory`)

---

## v0.4.0 — "Scalable" (2026 Q3-Q4, Aug-Oct)

**Goal**: Handle 10K+ anchors with sub-100ms retrieval.

### P2 — Ecosystem & Scale
- [ ] FAISS/HNSW embedding index for `cortical_lookup()` (currently linear scan)
- [ ] Incremental sleep: don't replay all anchors, only recent + high-surprise
- [ ] Benchmark suite: standard dataset (e.g., Multi-session Chat) with recall/precision/latency
- [ ] VS Code extension: memory visualization (constellation graph view)
- [ ] Token-aware anchor creation: embedding via local model (all-MiniLM-L6-v2)

### P2 — Documentation & Community
- [x] `ROADMAP.md` (this file)
- [ ] English README polish: architecture deep-dive, API reference
- [ ] Tutorial: "Building a Memory-Aware Chatbot in 50 Lines"
- [ ] CONTRIBUTING.md: dev setup, test conventions, PR template
- [ ] CHANGELOG.md: per-version change tracking

---

## v0.5.0 — "Integrated" (2026 Q4-Q1, Nov-Feb)

**Goal**: Drop into existing AI agent frameworks with zero friction.

### Integration Layer
- [ ] LangChain `BaseMemory` adapter
- [ ] LlamaIndex `BaseMemory` adapter
- [ ] OpenAI custom GPT action
- [ ] REST API server (`sg-server`): `POST /remember`, `POST /recall`, `POST /sleep`
- [ ] gRPC API for low-latency integration

### Advanced Mechanisms
- [ ] Cross-modal anchors: text + image + code embeddings in same graph
- [ ] Temporal edge decay curves (not just linear/log decay — power-law with plateaus)
- [ ] Hierarchical constellations: sub-constellations for nested topic structures
- [ ] Attention-weighted replay: not all anchors replayed equally during SWR

---

## v0.6.0 — "Biological" (2027 Q1-Q2, Feb-May)

**Goal**: Full biological fidelity — every known hippocampal mechanism mapped.

### Advanced Neuroscience
- [ ] Pattern separation (dentate gyrus): automatically separate similar-but-distinct anchors
- [ ] Pattern completion (CA3): partial cues retrieve full constellations
- [ ] Sharp-wave ripple content analysis: what gets replayed and why
- [ ] Noradrenergic tone modulation: global arousal state affects encoding/retrieval thresholds
- [ ] Cortical remapping: schema reorganization after contradictory evidence accumulation

### Deployment
- [ ] SQLite/postgres backend for persistent multi-process access
- [ ] Docker image with pre-configured embedding model
- [ ] Multi-agent memory: shared graph with agent-specific edge weights
- [ ] Federated sleep: coordinate consolidation across multiple agents

---

## Version History

| Version | Date | Highlights |
|---|---|---|
| **0.1.0** | 2026-05 | Core StarGraph, AnchorVector, edges, constellations, basic sleep |
| **0.2.0** | 2026-05 | Oscillators, GhostAnchor, Schema, reconsolidation, 9-phase sleep, resonance engine |
| **0.3.0-dev** | 2026-05 | Pluggable retrievers, OnlineConsolidator, test suite (13 tests), Chinese docs, bugfixes |
| **0.4.0** | target 2026-09 | FAISS index, incremental sleep, benchmark suite, VS Code viz |
| **0.5.0** | target 2026-12 | LangChain/LlamaIndex adapters, REST API, cross-modal anchors |
| **0.6.0** | target 2027-04 | Pattern separation/completion, multi-agent, federated sleep |
