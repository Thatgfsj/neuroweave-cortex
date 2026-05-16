# NeuroWeave Cortex (NWC) — Plan

> Last updated: 2026-05-16 | **v1.0.0** | 1,989 tests passing

## Current State

- **81 modules** in `star_graph/`, **contrib/** with 6 extracted modules
- Full S/A/B implementation: Retrieval Budget, Versioned Memory, Cluster Memory, Causal Edge Types, Episodic Memory
- Lazy imports (PEP 562) — all symbols loaded on first access
- CI pipeline with version consistency checks
- `sqlite_storage.py` exists, `async_manager.py` uses `asyncio.to_thread` as transition layer
- `embedding.py` limited to sentence-transformers + hash fallback only
- **Critical blockers**: `sleep.py` (81KB), `runtime.py` (78KB), `retrieval_pipeline.py` (39KB) — too large to modify safely

## Architecture Target

5 core packages + abstraction sub-package + extras:

| Package | Merged Modules | Core Interface |
|---------|---------------|----------------|
| `memory_core` | anchor, graph, storage, sqlite_storage, tier | `create_anchor`, `attach_vector`, `get_neighbors` |
| `retrieval_engine` | retriever, bm25, dual_channel, cognitive_cache, exact_cache | `retrieve`, `hybrid_search` |
| `embedding_provider` | **new** | `embed`, `dimension`, `max_batch_size` |
| `consolidation` | sleep (split first), evolution, ghost, compression | `consolidate`, `prune`, `ghost_pass` |
| `cortex_api` | manager, runtime (split first), scheduler, working_memory | `remember`, `recall`, `session_context` |
| `abstraction` (sub) | abstraction, hub, community, atom_facts | deferred to next cycle |
| `extras/` | resonance, autobiography, streaming + feature flags | opt-in, not core |

### Key Principles
- Internal modules swappable via DI (JSON/Redis/SQLite backends)
- All external API eventually async (`aremember`, `arecall`, `aconsolidate`), sync as thin wrappers
- `extras/` gated behind feature flags
- **merge first, then add new features** — do not add embedding_provider before splitting the monoliths

## Execution Order (corrected)

### Phase 1 — Structural Convergence (merge first)
1. Split `sleep.py` (81KB) → `sleep_rem.py`, `sleep_nrem.py`, `sleep_consolidate.py`
2. Split `runtime.py` (78KB) → `runtime_core.py`, `runtime_stats.py`, `runtime_lifecycle.py`
3. Split `retrieval_pipeline.py` (39KB) → merge relevant parts into retrieval_engine
4. Merge `cognitive_cache.py` + `exact_cache.py` (already stubbed)
5. Merge `tier.py` + `tiered.py` (if exists)
6. Form 5 core packages, move `resonance/autobiography/streaming` → `extras/`
7. Update `__init__.py` lazy imports for new package structure
8. Backward-compat re-exports from old module paths

### Phase 2 — Embedding Provider (new capability)
1. Design `EmbeddingProvider` ABC
2. Implement `LocalProvider` (sentence-transformers / ONNX)
3. Implement `OpenAIProvider` (text-embedding-3-small/large, base_url proxy, dimensions truncation)
4. Implement `ZhipuProvider` (embedding-2, semaphore rate limiting)
5. Implement `MixedProvider` (primary/fallback with dimension validation)
6. YAML config integration into existing `config.py` + `defaults.yaml`
7. Downgrade monitoring: error counters, fallback latency logging

### Phase 3 — Async Migration
1. Keep `asyncio.to_thread` wrappers as API scaffold
2. Migrate embedding calls to true async first (I/O bound)
3. Migrate storage I/O to async
4. Compute-heavy paths (graph traversal) use `run_in_executor` as transition
5. Mark old sync API as deprecated, keep for one major version

### Phase 4 — Observability & Production
1. Core metrics: `embedding_latency_seconds`, `recall_hit_rate`, `consolidation_duration_seconds`, `embedding_fallback_count`
2. `/health` and `/metrics` endpoints (Prometheus format)
3. gRPC or REST service wrapper for LangChain/LlamaIndex integration
4. Integration tests + ecosystem demos

## Deferred to vNext

- **Deferred batch vectorization** (buffer ≥32 / >30s flush): SQLite pending queue exists but crash-recovery complexity too high for Phase 1
- **abstraction sub-package** merge: wait for abstraction/atom_facts to stabilize

## Embedding Provider Design

### Unified Interface
```python
class EmbeddingProvider(ABC):
    async def embed(self, texts: List[str]) -> List[List[float]]: ...
    dimension: int
    max_batch_size: int
```

### 4 Required Providers
- **LocalProvider**: sentence-transformers (existing code as base), ONNX option
- **OpenAIProvider**: text-embedding-3-small/large, base_url proxy, dimensions param
- **ZhipuProvider**: embedding-2, semaphore concurrency control
- **MixedProvider**: primary/fallback auto-failover, dimension alignment check on init

### Config (YAML)
```yaml
embedding:
  provider: mixed
  mixed:
    primary: openai
    fallback: local
    timeout: 8
  openai:
    model: text-embedding-3-small
    dimensions: 512
  local:
    model: BAAI/bge-small-zh
```

### Risk: Dimension Alignment
Different providers output different dimensions (OpenAI 512/1536, Local 384/768, Zhipu 1024/2048). Must validate at init and on failover — mismatched dimensions corrupt the vector index.

## Cost & LLM Control
- `sleep()` LLM-dependent stages (REM_Emotion, N3b_AtomFacts) default OFF
- `skip_llm=True` parameter for stats-only consolidation (compression, merge, prune)
- AtomFacts: daily token quota + rate limiting

## Risk Notes
- **Dimension alignment**: validate provider dimensions on init and failover
- **Local model size**: do not bundle models in pip install; on-demand download
- **Async scope**: migrate I/O paths first (embedding, storage); keep compute in executor
- **Backward compat**: re-export old paths, deprecate sync API over one major version
- **Phase derivation**: current `embedding.py` phase/frequency logic (theta band) is core differentiator — preserve as mixin or standalone utility during provider refactor
