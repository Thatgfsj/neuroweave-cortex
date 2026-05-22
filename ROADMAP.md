# NeuroWeave Cortex (NWC) — Roadmap

## Current (v1.0.8 — 2026-05)

Phase 1-6 complete, 112 modules, 4-layer cognitive architecture.

- [x] 9-stage memory lifecycle (PERCEPTION → WORKING → ... → GHOST → DEAD)
- [x] Ghost subsystem with intensity ranking + NegativeGhost contradiction tracking
- [x] Abstraction engine (emergent categories + multi-level compression)
- [x] 8-phase sleep consolidation with SleepReport
- [x] Memory evolution engine (decay, boost, conflict, interference)
- [x] 5-layer dimensional descent retrieval (Brain → Cortex → Hub → 2D Plane → Timeline)
- [x] HybridFusion retriever + OscillationResonance + VectorSimilarity + PPR
- [x] System-1 + System-2 dual-channel retrieval with auto-trigger
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
- [x] MCP server (Model Context Protocol, 12 tools)
- [x] REST API server
- [x] CLI entry points (sg-sleep, sg-add, sg-query, sg-stats, nwc-mcp, nwc-server)
- [x] 300+ tunable YAML parameters with schema validation
- [x] Dependency manifest
- [x] Version unification

## v1.0.7 (completed — cognitive depth)

Phase 5: 12 modules adding cognitive depth to the Behavior layer.

- [x] `memory_budget.py` — Token + anchor budget management with eviction policies
- [x] `quality_score.py` — 7-dimension memory quality scoring (usage, reasoning, feedback, etc.)
- [x] `stability_control.py` — Long-term stability with exponential/linear decay, drift monitoring
- [x] `memory_layers.py` — 4-layer pyramid: Working → Episodic → Semantic → Core Identity
- [x] `typed_memory.py` — 7 memory types (code/task/dialogue/tool_call/knowledge/event/preference)
- [x] `abstraction_chain.py` — Event → Summary → Pattern → Identity abstraction pipeline
- [x] `domain_graph.py` — Domain-based graph partitioning with soft isolation
- [x] `context_routing.py` — 6-dimension context-aware retrieval routing
- [x] `hebbian_learning.py` — Hebbian edge learning — "neurons that fire together wire together"
- [x] `agent_state.py` — Agent state memory: goal tree, tool calls, checkpoints
- [x] `cognitive_closure.py` — Closed-loop feedback: recall → use → learn → improve
- [x] `cognitive_priority.py` — 5-level priority assignment with forced injection for core identity

## v1.0.8 (completed — cognitive cortex)

Phase 6: 11 modules forming Layer 4 (Cortex) — the external cognitive cortex for LLM agents.

- [x] `thought_object.py` — Unified cognitive base class (activated nodes, not static memory)
- [x] `perception.py` — Raw text → structured PerceptionFrame (intent, emotion, goals, concepts)
- [x] `cognitive_workspace.py` — Working memory workspace with reasoning chains, attention, TTL decay
- [x] `concept_cortex.py` — Concept network: activation, fusion, competition, 10 built-in core concepts
- [x] `activation_engine.py` — Multi-source spreading activation (query/goal/concept/emotion seeds)
- [x] `goal_system.py` — Goal hierarchy, conflict detection, goal-driven inference
- [x] `salience.py` — 10-component attention competition with lateral inhibition
- [x] `cognitive_compression.py` — Events → Concepts → Identity → World Model compression pipeline
- [x] `self_model.py` — System self-model → compressed CognitiveState → LLM prompt injection
- [x] `autonomous_reasoning.py` — Contradiction → Activation → Resolution loop (no LLM required)
- [x] `memory_lifecycle.py` — Unified 9-stage lifecycle: Perception → Working → ... → Ghost → Dead

## v1.1.0 (planned — production readiness)

- [ ] Cosine similarity deduplication (single `math_utils.py` implementation)
- [ ] Config access API (`cfg.get('exact_cache.auto_harvest', True)`)
- [ ] `find_contradictions()` O(n²) → O(n*k) via ANN pre-filter
- [ ] Layer 3 (2D Plane) TimeSpine-indexed scan (not full anchor linear scan)
- [ ] `retention_score` caching with dirty flag
- [ ] Test coverage ≥ 80% (currently ~48%)
- [ ] mypy/pyright static type checking
- [ ] Config schema validation
- [ ] Structured logging (replace `print()` + `self.log: list[str]`)
- [ ] README doctest in CI

## v1.2+ (future)

- [ ] LLM-integrated reasoning (optional LLM calls for deeper inference)
- [ ] Cross-agent memory sharing (shared concept cortex across instances)
- [ ] Plugin system for custom perception/activation/salience strategies
- [ ] Real-time cognitive dashboard (WebSocket-based visualization)
- [ ] Multi-language perception layer (Chinese, Japanese, etc.)

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
| 1.0.7 | 2026-05 | Phase 5 cognitive depth: 12 modules — memory budget, quality scoring, stability control, 4-layer pyramid, typed memory, domain graph, context routing, Hebbian learning, cognitive priority |
| 1.0.8 | 2026-05 | Phase 6 cognitive cortex: 11 modules — ThoughtObject, PerceptionLayer, CognitiveWorkspace, ConceptCortex, ActivationEngine, GoalSystem, SalienceEngine, SelfModel, AutonomousReasoning, MemoryLifecycle (9-stage). 4-layer architecture. 112 modules total. |
| 1.0.9 | 2026-05 | Global hard cap, auto-sleep daemon, cold ghost cleanup, cortex hard rejection |
| 1.1.0 | 2026-05 | Hippocampus buffer, edge sparsification, file sharding, sleep rebuild, cortex hierarchy |
| 1.2.0 | 2026-05 | Memory tiering, decay+reinforcement, edge traversal weights, spreading activation, cognitive cache |
| 1.3.0 | 2026-05 | Domain router, edge budget, write gate, four-layer compression — 467 tests |
| 1.4.0 | 2026-05 | Spreading activation retrieval, 3-tier thermal store, continuous edge time decay — 496 tests |
| 1.5.0 | 2026-05 | Renamed to NeuroWeave Cortex. Self-organization, personality model, goal tree — 582 tests |
| 1.0.0 | 2026-05 | Official release — PyPI publication, 1,989 tests, 80% coverage |
