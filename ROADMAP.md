# Star Graph Memory — Roadmap

## v0.4 (current — 2026-05)

Completed infrastructure for a cognitive memory runtime:

- [x] 6-state memory lifecycle (ACTIVE → REHEARSING → CONSOLIDATING → DORMANT → GHOST → REACTIVATED)
- [x] Ghost subsystem with fuzzy recall and revival
- [x] Abstraction engine (emergent categories from anchor clusters)
- [x] 5-phase systematized sleep architecture with SleepReport
- [x] Memory evolution engine (decay, boost, conflict, interference)
- [x] Cognitive Memory Scheduler (context-aware type selection + adaptive compression)
- [x] HybridFusion retriever (semantic + temporal + graph structure + Personalized PageRank)
- [x] Edge versioning with confidence, source_type, and lifecycle management
- [x] MemoryManager high-level facade (remember/recall/sleep/save/load)
- [x] SQLite storage backend with WAL mode
- [x] Explainable confidence scores
- [x] 5-category benchmark suite with content-based metrics
- [x] Async sleep with progress callbacks
- [x] 13-test suite passing

## v0.5 (planned — 2026-06)

Integration and ecosystem:

- [ ] MCP server — expose as a Model Context Protocol server for Claude, GPT, and other MCP clients
- [ ] REST API — HTTP endpoints (`POST /remember`, `POST /recall`, `POST /sleep`)
- [ ] LangChain memory adapter — drop-in replacement for ConversationBufferMemory
- [ ] Streaming sleep — incremental consolidation for graphs >10K anchors
- [ ] Fix oscillation resonance — derive phase from actual temporal context, not embedding statistics
- [ ] Multi-agent memory — shared graph with agent-specific views and access control
- [ ] Memory reflection tool — self-audit: "what do I know? what's stale? what's contradicted?"
- [ ] Production SQLite backend with migration support and backup

## v0.6 (planned — 2026-07)

Scale and robustness:

- [ ] Incremental indexing — add anchors without full index rebuild
- [ ] Configurable memory type profiles per domain (coding agent vs. chatbot vs. research agent)
- [ ] Long-term personality formation from accumulated episodic memory
- [ ] Predictive retrieval — anticipate what the agent will need next
- [ ] Memory safety — forgetting by request, access control, audit trails
- [ ] Visual graph explorer — functional, minimal, for debugging not prettiness

## v1.0 (future)

- [ ] Distributed memory — federated graphs across agents
- [ ] Cross-agent memory transfer — bootstrap new agents from consolidated memory
- [ ] Cross-modal anchors — text + code + structured data in same graph
- [ ] Production deployment guide and operator handbook

## Research questions

- **Personality emergence**: can long-running memory produce stable behavioral traits?
- **Fidelity under compression**: at what compression ratio does recall quality break?
- **Cross-agent transfer**: can one agent's consolidated memory bootstrap another?
- **Adversarial memory**: can memories be poisoned? how to detect and recover?
- **Forgetting curves**: what parameters produce the most human-like forgetting patterns?

## Version history

| Version | Date | Highlights |
|---|---|---|
| 0.1.0 | 2026-05 | Core graph, anchors, edges, constellations, basic sleep |
| 0.2.0 | 2026-05 | Oscillators, ghost anchors, schemas, reconsolidation, resonance engine |
| 0.3.0 | 2026-05 | Pluggable retrievers, online consolidator, 13-test suite, Chinese docs |
| 0.4.0 | 2026-05 | Evolution engine, scheduler, hybrid fusion, edge versioning, benchmarks, manager facade |
