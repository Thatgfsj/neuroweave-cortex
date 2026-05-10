# Benchmark Comparison: Star Graph Memory vs Alternatives

## Systems Compared

| System | Type | Retrieval | Consolidation | Open Source |
|---|---|---|---|---|
| **Star Graph Memory v0.3** | Hippocampal-inspired graph | Phase-locking + embedding | 9-phase sleep cycle | MIT |
| **Mem0** | Embedding + metadata filtering | Cosine similarity + metadata | None (ephemeral) | Apache 2.0 |
| **HippoRAG** | Knowledge graph + LLM | Graph traversal + LLM rerank | None (static) | MIT |
| **MemGPT** | LLM OS with memory tiers | LLM-generated retrieval | Context window management | Apache 2.0 |
| **LangChain Memory** | Message buffer/summary | Sequential + LLM summary | None (fixed window) | MIT |
| **Cognee** | Graph + vector hybrid | Graph traversal + embedding | None | Apache 2.0 |

## Qualitative Comparison

| Capability | Star Graph | Mem0 | HippoRAG | MemGPT | LangChain | Cognee |
|---|---|---|---|---|---|---|
| **Sequence-aware retrieval** | Phase precession | No | No | Partial (LLM) | No | No |
| **Automatic consolidation** | 9-phase sleep | No | No | No | No | No |
| **Emotion-modulated encoding** | Emotional gating | No | No | No | No | No |
| **Memory merging (dedup)** | Jaccard + merge | Embedding dedup | No | No | No | No |
| **Savings effect (ghosts)** | Ghost anchors | No | No | No | No | No |
| **Schema abstraction** | Tag-based clustering | No | No | No | No | No |
| **Predictive coding** | Prediction error gates | No | No | No | No | No |
| **Micro-consolidation (online)** | `OnlineConsolidator` | No | No | No | No | No |
| **Pluggable retrievers** | 2 backends + compare | 1 backend | 1 backend | 1 backend | 1 backend | 2 backends |
| **Chinese docs** | Complete README_CN | Limited | No | No | No | No |
| **Zero external API** | Yes (local-first) | Requires LLM | Requires LLM | Requires LLM | Requires LLM | Requires LLM |

## Retrieval Quality (Controlled Test — 100 anchors, 6 topics)

| Metric | OscillationResonance | VectorSimilarity | Notes |
|---|---|---|---|
| **Latency (p50)** | 10-30ms | 5-15ms | Oscillation adds spread activation cost |
| **Latency (p99)** | <100ms | <50ms | Within 100ms budget |
| **Recall@3** | 0.0 (w/o embed) | 0.31 (text-only) | Oscillation needs real embeddings |
| **Recall@5** | 0.0 (w/o embed) | 0.44 (text-only) | Vector with `sentence-transformers` expected 0.5-0.7 |
| **Score bounds** | [0, 1] ✓ | [0, 1] ✓ | Both guarantee normalized scores |
| **Empty graph** | ✓ returns 0 | ✓ returns 0 | Both handle gracefully |

> OscillationResonanceRetriever requires real embeddings (`sentence-transformers`) for meaningful phase derivation. Current text-only hash fallback produces random phases. The 0.0 recall is expected until embeddings are integrated.

## Consolidation Quality (50 anchors, 1 sleep cycle)

| Metric | Before Sleep | After Sleep | Delta |
|---|---|---|---|
| **Anchors** | 50 | ≤50 | Merged near-duplicates |
| **Edges** | ~25 | ≤25 | Dormant edges pruned |
| **Ghosts** | 0 | ≥0 | Weak anchors → ghosts |
| **Schemas** | 0 | ≥0 | From tag clusters ≥3 |
| **Edge mean weight** | ~0.3 | ~0.3 | Homeostasis maintained |
| **Hippocampal dep.** | ~1.0 | <1.0 | Consolidation progressed |

## Design Trade-offs

| Dimension | Star Graph Advantage | Star Graph Limitation |
|---|---|---|
| **Richness** | Multi-vector, phase, prediction, emotion | Heavier anchor model (24 fields) |
| **Complexity** | 9-phase sleep is comprehensive | Needs tuning (7 thresholds) |
| **Prerequisites** | Local-first, no LLM needed | Embedding quality limits resonance |
| **Scalability** | O(n²) naive merge, O(n) resonance | Needs indexing for >10K anchors |
| **Ecosystem** | New project (v0.3-dev) | No integrations yet (Mem0 adapter planned) |
| **Validation** | 13 automated tests | No production deployment data |

## Planned Improvements (P1-P2)

- [ ] `sentence-transformers` integration for real embeddings (→ resonance recall >0.5 expected)
- [ ] `Mem0Adapter` for zero-code migration from Mem0
- [ ] GitHub Actions CI with pytest + flake8
- [ ] Bayesian threshold tuning via `tune_threshold.py`
- [ ] Embedding index (FAISS/HNSW) for >10K anchor scale
- [ ] Streaming sleep (incremental consolidation for large graphs)
