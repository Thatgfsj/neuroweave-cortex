# NeuroWeave Cortex (NWC) — Plan

> Last updated: 2026-05-21 | **v1.0.4-dev** | 1,989 tests passing

## Recently Completed (2026-05-21)

- **1.1 Multi-strategy RRF fusion** — `recall()` now uses `weighted_reciprocal_rank_fusion()` to fuse exact cache, BM25, cognitive cache, graph descent, and spreading activation (replaced sequential merge)
- **1.2 Cross-Encoder reranking** — `cross_encoder.py` module + `CrossEncoderReranker` class; lazy-loads sentence-transformers CrossEncoder; integrated into `recall()` after RRF fusion; config in `defaults.yaml` under `retrieval.rerank`
- **1.3 Explainable reasoning paths** — `explain()` API on `RetrievalCore`; `retrieval_trace` field on `MemoryContext`; per-pathway rank tracking in RRF fusion traces
- **2.1 Conflict detection & resolution** — `conflict_detection.py` module with `ConflictDetector`; `Anchor.invalid_at` + `Anchor.conflict_candidate` fields; overwrite/coexist/deprecate strategies; integrated as N2b phase in sleep cycle
- **2.2 Memory tiering (Hot/Warm/Cold)** — `Anchor.memory_tier` property; Ghost Revival hook in `ThermalStore.thaw_anchor()`; `auto_archive()` scheduled task; `tier_thresholds` config in `defaults.yaml`
- **2.3 Memory revision engine** — `memory_revision.py` module with `MemoryRevisionEngine`; template-based + optional LLM revision; surprise-prioritized candidate selection; integrated as N2c phase in sleep cycle
- **3.1 Public benchmark suite** — `benchmarks/run_benchmarks.py`; LongMemEval/LoCoMo/BEAM-style benchmarks with Recall@K, MRR, NDCG metrics; terminal dashboard; configurable scale (quick/standard/large)
- **3.2 Markdown export** — `markdown_export.py` module; operator-editable plain-text memory export (GBrain-aligned); timeline + topic organization; `nwc-export` CLI command; `MemoryManager.export_markdown()` API
- **3.3 Batch vectorization** — `batch_vectorizer.py` module; deferred embedding writes (buffer ≥32 or >30s flush); SQLite WAL journal for crash recovery; `MemoryManager.batch_remember()` + `flush_vectorizer()` API
- **4.1 Encrypted forgetting certificates** — `forget_certificate.py` module; Ed25519-signed JWS self-verifying certificates; GDPR Article 17 compliance; `nwc forget --certificate` + `nwc verify` CLI commands
- **4.2 Zero-LLM ingestion pipeline** — `zero_llm_pipeline.py` module; 7-stage algorithmic pipeline (security → embed → dedup → entity → classify → score → link); optional LLM ambiguity gate; `MemoryManager.zero_llm_ingest()` API
- **4.3 Multimodal support** — extended `multimodal.py`; `AudioEncoder` (Whisper/spectrogram) + `AudioAnchor`; `MemoryManager.remember_image()` + `remember_audio()`; unified graph for text/image/audio

## Current State

- **89 modules** in `star_graph/`, **contrib/** with 6 extracted modules
- Full S/A/B implementation: Retrieval Budget, Versioned Memory, Cluster Memory, Causal Edge Types, Episodic Memory
- Lazy imports (PEP 562) — all symbols loaded on first access
- CI pipeline with version consistency checks
- `sqlite_storage.py` exists, `async_manager.py` uses `asyncio.to_thread` as transition layer
- `embedding.py` limited to sentence-transformers + hash fallback only
- **New**: `cross_encoder.py` — CrossEncoder reranking module
- **New**: `bm25.py` — `weighted_reciprocal_rank_fusion()` for weighted multi-path fusion
- **Resolved**: `sleep.py`/`runtime.py`/`retrieval_pipeline.py` monoliths tracked for Phase 1 splitting

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

## Competitive Analysis & Optimization Roadmap (2026-05-21)

行业对标: Mem0 / Hindsight / MemForge / Zep / MAGMA / Mnemosyne / ECHOFORM

### NWC 的独特护城河（保持并强化）

| 特性 | 地位 | 策略 |
|------|------|------|
| Ghost Revival（幽灵复苏/节省效应） | **业界唯一** | 宣传核心，考虑单独出 whitepaper |
| 8-Phase Sleep Consolidation | MemForge 有 10-phase 但偏工程化；NWC 更生物化 | 强化 REM/NREM 阶段区分，匹配神经科学文献 |
| Emotional Valence (-1~+1) | 除 Second-Me 外无竞品实现 | 情感驱动的记忆权重是差异化方向 |
| Survival Decay（生存衰减） | 业界多用简单 TTL；NWC 的生存函数更深 | 保留并可视化衰减曲线 |
| Emergent Abstraction（涌现抽象） | MAGMA/Hindsight 有类似但非自发生成 | 宣传"自学能力" |

### 优先优化项

#### 🔴 Priority 1 — 检索精度（短期收益最大）

**1.1 多策略融合检索（对标 Hindsight TEMPR）**

当前: 纯图谱检索
目标: 四路并行 → RRF 融合 → Cross-Encoder 重排

```
查询 → ┌─ 语义向量 (embedding cosine)
        ├─ BM25 关键词 (倒排索引)
        ├─ 图谱多跳遍历 (Star Graph edges)
        └─ 时序推理 (causal/temporal edges)
          ↓
        RRF (Reciprocal Rank Fusion) 融合
          ↓
        Cross-Encoder 重排序
          ↓
        最终结果 + 推理路径
```

新增模块: `star_graph/bm25_index.py`, `star_graph/retrieval_fusion.py`
改动范围: `retrieval_pipeline.py` → 扩展为多通道入口
预计收益: 检索精度 +20-30%

**1.2 Cross-Encoder 重排序**

- 初筛 top-N 后过一遍 lightweight cross-encoder (sentence-transformers 内置)
- 可配置: `rerank_top_k`, `rerank_model`, `rerank_threshold`
- 仅对用户查询 + 候选记忆对打分，开销可控

**1.3 可解释推理路径（对标 MAGMA Adaptive Traversal）**

- 每次 recall 返回 `retrieval_trace`: 哪些边被遍历、各通道的贡献分
- `explain()` 新 API: "为什么这条记忆被召回"
- 图谱路径可视化（CLI 输出 ASCII graph 或 JSON trace）

#### 🔴 Priority 2 — 记忆质量（睡眠/巩固增强）

**2.1 矛盾检测与冲突解决（对标 Hindsight Background Merging）**

当前: sleep 期间做 consolidation，但没有显式冲突检测
目标: 在 NREM 阶段增加冲突检测步骤

- 新事实 vs 已有记忆的语义矛盾检测 (embedding similarity > 0.9 但 sentiment 相反)
- 冲突记忆标记 `conflict_candidate`
- 解决策略: `overwrite` (高置信度新事实) / `coexist` (观点分歧) / `deprecate` (旧事实标记 invalid_at)
- 新增: `memory.invalid_at` 时间戳字段（对标 Zep 的图节点管理）

**2.2 记忆三级分层（对标 MemForge Hot→Warm→Cold）**

- **Hot**: 未整合的原始事件 (当前 working_memory)
- **Warm**: 已 consolidate、可检索、有 score (当前长期记忆主体)
- **Cold**: 低频访问归档 (不参与检索，但可通过 Ghost Revival 复活)
- 新增: `tier_thresholds` 配置, `auto_archive` 定时任务
- Ghost Revival 在 Cold → Warm 复活时触发，完美契合

**2.3 记忆修订引擎（对标 MemForge Memory Revision）**

- 睡眠中识别低置信度记忆 (score < threshold)
- LLM 重新摘要/合并，提升记忆质量
- 高"惊吓值"（surprise）记忆优先修订

#### 🟡 Priority 3 — 工程化 & 生态

**3.1 公开 Benchmark**

- LongMemEval: 长期对话记忆保持率
- LoCoMo: 多轮对话记忆一致性
- BEAM: 大规模记忆检索精度
- 目标: 至少达到 Mem0 2026 水平 (LongMemEval ~90%)

**3.2 Markdown 导出**

- `nwc export --format markdown --output memories/`
- 按时间线或主题组织成 Markdown 文件
- 用户可直接编辑、删除、补充
- 对标 GBrain "operator-owned plain-text memory"

**3.3 批量向量化延迟写入**

- 已在 Phase 1 标记为 deferred → 提升优先级
- 目标: 减少 embedding I/O 次数，支持 32+ 条批量处理
- crash-recovery: SQLite pending queue + WAL

#### 🟢 Priority 4 — 远期差异化

**4.1 加密遗忘证书（对标 ECHOFORM）**

- Ed25519 签名的 JWS 证书，证明某条记忆已被可证明地删除
- GDPR Article 17 合规: "被遗忘权"的工程化实现
- 生成: `nwc forget --certificate --query "xxx"` → 输出签名证书
- 验证: `nwc verify --certificate <path>`

**4.2 零 LLM 调用摄入管道（对标 Mnemosyne）**

- 纯算法管线: 安全过滤 → embedding → 去重 → 实体提取 → 分类 → 评分 → 链接
- 仅在"模糊判断"环节可选调用 LLM
- 目标: 摄取成本降至 $0 / 条（当前如有 LLM 调用则 > 0）

**4.3 多模态支持（对标 LATRACE）**

- `remember_image()`, `remember_audio()` — 多模态 embedding → 统一图谱
- CLIP / Whisper 嵌入 → 同一向量空间
- 文本查询可召回相关图片/音频记忆

### 实施优先级矩阵

```
                    高影响 ───┬─── 低影响
                   ┌──────────┼──────────┐
             高    │ 1.1 多策略检索  │ 3.1 Benchmark  │
             难    │ 1.2 Cross-Encoder│ 4.1 加密遗忘    │
             度    │ 2.1 矛盾检测    │ 4.3 多模态      │
                   ├──────────┼──────────┤
             低    │ 1.3 可解释路径  │ 3.2 Markdown导出│
             难    │ 2.2 记忆分级    │ 3.3 批量向量化   │
             度    │ 2.3 记忆修订    │ 4.2 零LLM摄入    │
                   └──────────┴──────────┘
```

建议实施顺序: **1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 3.1 → 其他**

### 参考项目链接

- **Mem0** (向量检索标杆): https://github.com/mem0ai/mem0 — LongMemEval 93.4%
- **Hindsight** (多策略融合检索 SOTA): https://vectorize.io — BEAM 10M 64.1%
- **MemForge** (神经科学睡眠周期): https://github.com/salishforge/memforge — LongMemEval 93.2%
- **Zep/Graphiti** (知识图谱优先): https://github.com/getzep/graphiti — SOC 2 Type 2 合规
- **Cognee** (向量+图谱混合): https://github.com/topoteretes/cognee — Memphis 算法
- **MAGMA** (多图学术 SOTA): https://arxiv.org/abs/2601.03236 — 四正交关系图
- **Mnemosyne** (5层认知OS): https://github.com/28naem-del/mnemosyne — 零LLM摄入
- **ECHOFORM** (加密遗忘证书): https://github.com/OpenAgentic-Labs/echoform-ghost-memory
- **GBrain** (Markdown-first 个人记忆): YC CEO Garry Tan 开源 — 面向 OpenClaw
- **LATRACE** (多模态记忆): https://github.com/ZXXZ1000/LATRACE-AI
