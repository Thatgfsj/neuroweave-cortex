# NeuroWeave Cortex (NWC) — Roadmap

## Current (v1.0.0 — 2026-05)

- [x] 6-state memory lifecycle (ACTIVE → GHOST → REACTIVATED)
- [x] Ghost subsystem with intensity ranking + NegativeGhost contradiction tracking
- [x] Abstraction engine (emergent categories + multi-level compression)
- [x] 8-phase sleep consolidation with SleepReport
- [x] Memory evolution engine (decay, boost, conflict, interference)
- [x] 5-layer dimensional descent retrieval (Brain → Cortex → Hub → 2D Plane → Timeline)
- [x] HybridFusion retriever + OscillationResonance + VectorSimilarity
- [x] System-1 + System-2 dual-channel retrieval
- [x] Raw chunk buffer (L0 uncompressed short-term tier)
- [x] Exact match cache (KV deterministic O(1) entity-pair lookup)
- [x] Micro-sleep scheduler (incremental non-blocking consolidation)
- [x] Snapshot + WAL (crash-safe versioned state persistence)
- [x] Cortex partitioning with router + memory gating
- [x] TimeSpine temporal index with "upper-right to lower-left" priority scan
- [x] HubLayer cross-cortex abstraction bridges
- [x] CascadeRecall causal chain traversal
- [x] Configurable survival functions (Ebbinghaus / Power-law / Exponential / Custom)
- [x] Multimodal memory (CLIP joint embedding: text + image)
- [x] Streaming memory buffer with backpressure + auto-batch + dedup
- [x] Async manager + OpenTelemetry tracing
- [x] 5-category benchmark suite
- [x] MemoryManager high-level facade (remember/recall/sleep/save/load)
- [x] SQLite storage backend + JSON persistence
- [x] MCP server (Model Context Protocol)
- [x] CLI entry points (sg-sleep, sg-add, sg-query, sg-stats)
- [x] 232-test suite passing
- [x] Dependency manifest (requirements.txt)
- [x] Version unification (1.0.6 across all files)

## v1.0.7 (completed — correctness & architecture)

P0 + P1 fixes from code audit:

- [x] Ghost subsystem unification (GhostAnchor → GhostNode, single data model)
- [x] Raw Buffer priority elevation in recall() merge order
- [x] ANN index incremental maintenance (add/remove sync, no full rebuild on query)
- [x] MemoryManager split (MemoryRuntime + RetrievalPipeline + MemoryManager)
- [x] Cortex independent sleep (per-cortex consolidation cycles)
- [x] Dual-Channel auto-trigger in recall() (structural query detection + low-confidence fallback)

## v1.0.8 (completed — cognitive fidelity)

- [x] Autobiographical memory layer (SelfNarrative — "what I know about myself")
- [x] State / ThermalState unified transition matrix
- [x] Oscillation phase derivation from temporal/emotional context (not embedding stats)
- [ ] Cosine similarity deduplication (single `math_utils.py` implementation)
- [ ] Sleep phase naming standardization (N1_Replay → N6_IndexRebuild)

## v1.1.0 (completed — production readiness)

- [ ] Config access API (`cfg.get('exact_cache.auto_harvest', True)`)
- [ ] `find_contradictions()` O(n²) → O(n*k) via ANN pre-filter
- [ ] Layer 3 (2D Plane) TimeSpine-indexed scan (not full anchor linear scan)
- [ ] `retention_score` caching with dirty flag
- [ ] Test coverage ≥ 80% (currently ~48%)
- [ ] mypy/pyright static type checking
- [ ] Config schema validation
- [ ] Structured logging (replace `print()` + `self.log: list[str]`)
- [ ] README doctest in CI

## v1.2+ (completed)

All planned v1.2+ features have been delivered across v1.2–v1.6. See version history below.

See [plan.md](plan.md) for detailed implementation history and architecture decisions.

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-05 | Core graph, anchors, edges, constellations, basic sleep |
| 0.2.0 | 2026-05 | Oscillators, ghost anchors, schemas, reconsolidation, resonance engine |
| 0.3.0 | 2026-05 | Pluggable retrievers, online consolidator, 13-test suite, Chinese docs |
| 0.4.0 | 2026-05 | Evolution engine, scheduler, hybrid fusion, edge versioning, benchmarks, manager facade |
| 1.0.5 | 2026-05 | Survival functions (4 curves), ghost intensity, NegativeGhost contradiction tracking |
| 1.0.6 | 2026-05 | Multimodal CLIP, streaming buffer with backpressure, dependency manifest, 232 tests |
| 1.0.7 | 2026-05 | Ghost unification, raw buffer priority, ANN incremental, manager split, cortex sleep, dual-channel auto |
| 1.0.8 | 2026-05 | Sleep merge ANN, BM25 hybrid, PPR sparse, EmbedderRegistry, AnchorVector 10-dim, tiered storage |
| 1.0.9 | 2026-05 | Global hard cap, auto-sleep daemon, cold ghost cleanup, cortex hard rejection |
| 1.1.0 | 2026-05 | Hippocampus buffer, edge sparsification, file sharding, sleep rebuild, cortex hierarchy |
| 1.2.0 | 2026-05 | Memory tiering, decay+reinforcement, edge traversal weights, spreading activation, cognitive cache |
| 1.3.0 | 2026-05 | Domain router, edge budget, write gate, four-layer compression — 467 tests |
| 1.4.0 | 2026-05 | Spreading activation retrieval, 3-tier thermal store, continuous edge time decay — 496 tests |
| 1.5.0 | 2026-05 | Renamed to NeuroWeave Cortex. Self-organization, personality model, goal tree — 582 tests |
| 1.0.0 | 2026-05 | Official release — PyPI publication, 1,989 tests, 80% coverage |
