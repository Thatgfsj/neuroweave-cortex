# NeuroWeave Cortex (NWC) — Plan

> Last updated: 2026-05-22 | **v1.0.7** | Phase 1-5 complete

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
- **Phase 1 Architecture Slimdown (2026-05-21)** — 49 flat modules merged into 5 core subpackages (`memory_core/`, `retrieval_engine/`, `consolidation/`, `cortex_api/`, `embedding_provider/`) via git mv; backward-compat root stubs `from .subpkg.mod import *`; cross-package imports fixed
- **MCP Server 稳定性修复 (2026-05-21)** — import fix (`.contrib` → `..contrib` after subpackage move) + embedder warmup on startup to prevent first-call timeout
- **Phase 5 Cognitive Depth (2026-05-22)** — 12 new modules across 4 priority areas: memory budget, quality score, stability control, 4-layer memory pyramid, typed memory, abstraction chain, domain graph, context routing, Hebbian edge learning, agent state memory, cognitive closure, cognitive priority layer | 2,000+ lines of new production code | all 12 modules verified in integration test

## Current State

- **101 modules** in `star_graph/` (49 in 5 subpackages + 52 root), **contrib/** with 7 extracted modules
- Phase 1-5 complete: RRF fusion, cross-encoder, explainability, conflict detection, memory tiering, revision engine, benchmarks, markdown export, batch vectorizer, forget certificates, zero-LLM pipeline, multimodal, architecture slimdown, MCP stability fixes, **12 new Phase 5 cognitive depth modules**
- Full S/A/B implementation: Retrieval Budget, Versioned Memory, Cluster Memory, Causal Edge Types, Episodic Memory
- Lazy imports (PEP 562) — all symbols loaded on first access
- `sqlite_storage.py` exists, `async_manager.py` uses `asyncio.to_thread` as transition layer
- `embedding.py` limited to sentence-transformers + TF-IDF + hash fallback (3-tier)
- **Phase 5 planned**: 17-item cognitive depth optimization (long-term stability, 4-layer memory, context routing, agent autonomy)

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

## Phase 5 — 认知深度优化 (2026-05-22)

基于 17 项深度审查，以下按优先级分类。已覆盖项标注状态，新项纳入实施队列。

### 已在 Phase 1-4 完成的项

| # | 项目 | 对应实现 | 状态 |
|---|------|----------|------|
| 6 | 记忆冲突检测 | 2.1 Conflict Detection & Resolution (`conflict_detection.py`) | done |
| 9 | 睡眠整理机制 | 8-Phase Sleep Consolidation (`consolidation/sleep*.py`) | done |
| 11 | 可解释性 | 1.3 Explainable reasoning paths (`retrieval_trace`, `explain()`) | done |
| 1 | 批量向量化 | 3.3 Batch Vectorizer (`batch_vectorizer.py`) | done (distillation 未实现) |
| 7 | 检索多策略融合 | 1.1 RRF fusion (`bm25.py`, `dual_channel.py`) | done (context routing 未实现) |
| 2 | 记忆分层 | 2.2 Hot/Warm/Cold tiering (`tier.py`, `tiered.py`) | done (4-layer identity 未实现) |

### 🔴 Priority 5.1 — 长期稳定性（防退化）

#### 5.1.1 Memory Budget & Token Budget

**问题**: 记忆无限增长 → 检索延迟上升、token 消耗失控、噪声累积

**方案**:
- `MemoryBudget`: 总量上限 + 各层配额 (Working/Episodic/Semantic/Core)
- `TokenBudget`: 单次 recall 最大 token 消耗 + 每日 LLM 调用上限
- 动态上下文裁剪: 低价值记忆在注入 prompt 前截断
- 与现有 `edge_budget.py` / `retrieval_budget.py` 统一

```
MemoryBudget:
  total_anchors: 50,000
  working_memory: 200
  episodic: 30,000
  semantic: 15,000
  core_identity: 5,000
```

#### 5.1.2 Memory Quality Score

**问题**: 无法区分高价值记忆 vs 噪声记忆 vs 错误记忆

**方案**:
- `MemoryQualityScore`: 使用频率 × 推理贡献 × 用户反馈 × 任务命中率
- 低分记忆自动归档 Cold tier → 定期清理
- 高分记忆强化保护（跳过 pruner）
- 质量阈值可配置

#### 5.1.3 Long-Term Stability Control

**问题**: 长期运行后记忆污染、Agent 人格漂移、召回失真

**方案**:
- `MemoryDecayCurve`: 基于 importance × recency × repetition × emotion × task_relevance
- 自动衰减（指数/线性可配）
- 自动归档到 Cold tier
- Core Identity 层记忆豁免衰减
- 定期稳定性报告: 人格漂移度 / 记忆一致性 / 检索退化率

### 🟡 Priority 5.2 — 认知结构深化

#### 5.2.1 四层记忆金字塔 (4-Layer Memory)

**当前**: Hot/Warm/Cold 三级（按温度）

**目标**: 四级认知分层:

| 层级 | 用途 | 生命周期 | 召回策略 | 压缩策略 |
|------|------|----------|----------|----------|
| Working Memory | 当前任务上下文 | 分钟~小时 | 全量注入 prompt | 不压缩 |
| Episodic Memory | 用户事件/对话 | 天~月 | 语义 + 时序检索 | 摘要压缩 |
| Semantic Memory | 长期知识/规律 | 月~年 | 图谱遍历 + 概念检索 | 递归摘要 |
| Core Identity | 人格/偏好/价值观 | 永久 | 强权重注入 | 仅人工修改 |

**新增字段**: `Anchor.memory_layer`, `Anchor.layer_ttl`

#### 5.2.2 Event → Pattern → Identity 抽象链

**问题**: 系统记录内容，不"理解"用户

**方案**:
```
1000条事件 ──→ 100条摘要 ──→ 10条长期规律 ──→ 1个认知画像
(episodic)    (semantic)     (pattern)          (identity)
```

- **Event → Pattern**: 多次相同话题 → 推导"偏好系统级开发"
- **Pattern → Identity**: 多个关联 Pattern → "全栈工程师偏向后端"
- LLM 辅助抽象（可选），纯统计初见
- 新增 `abstraction_chain.py`，扩展现有 `abstraction.py`

#### 5.2.3 Typed Memory（多类型记忆结构）

**问题**: 代码/对话/文档/任务/工具调用本质不同，但统一存储

**方案**:

| 类型 | embedding | 压缩策略 | 检索逻辑 |
|------|-----------|----------|----------|
| `code` | 专用 code embedder | 文件级去重 | 函数/类签名索引 |
| `task` | 目标导向 embedding | 状态机压缩 | 任务状态检索 |
| `dialogue` | 通用 embedder | 对话摘要 | 时序+语义 |
| `tool_call` | 操作路径 embedding | 模式去重 | 工具链匹配 |
| `knowledge` | 概念级 embedding | 合并相似 | 概念图谱遍历 |

**改动**: `Anchor.memory_type` 字段（已有 `memory_type` 枚举，扩展即可）

#### 5.2.4 Domain-based Graph（图区域化与子图隔离）

**问题**: 图谱无限增长 → 弱连接污染 → 跨域检索噪声

**方案**:
- 按领域自动分区: 开发 / 生活 / 情感 / 项目 / 世界知识
- 每个 domain 独立 embedding subspace + 独立检索
- 跨域边自动降权（软隔离）
- Domain Router 扩展现有 `domain_router.py`
- 与现有 cortices 体系整合

### 🟡 Priority 5.3 — 检索智能化

#### 5.3.1 Context Routing Engine（上下文路由）

**问题**: "最相似" ≠ "当前任务最需要"

**方案**: 检索维度扩展，不单依赖 embedding similarity:

```
检索权重 = W1·similarity + W2·task_relevance + W3·domain_match 
          + W4·recency + W5·user_intent + W6·agent_state
```

- 当前任务目标（debugging/coding/planning）决定 W1-W6 权重分配
- 用户意图检测（问句类型、关键词信号）
- Agent 状态感知: 当前 tool 链、推理阶段
- Domain 标签: 优先检索同域记忆

#### 5.3.2 Hebbian Edge Learning（赫布边权学习）

**问题**: 图边固定，不随使用动态强化/衰减

**方案**:
- **Edge Reinforcement**: 经常一起激活的记忆对 → 边权增强
- **Edge Decay**: 长期不遍历的边 → 衰减至剪枝
- **Hebbian Rule**: `Δw = η × (激活频率) × (共现概率)`
- 与现有 `Edge` 的 `used_at` / `access_count` 字段整合
- 新增 `edge_learning.py`，扩展现有 `edges.py`

### 🟢 Priority 5.4 — Agent 自主性

#### 5.4.1 Agent State Memory

**问题**: 系统只记用户，不记 Agent 自身状态

**方案**:
- 当前任务树（goal → subgoal → action）
- Tool 调用历史 + 结果
- 推理阶段（exploration / validation / execution）
- 长任务中断恢复（checkpoint）
- 新增 `agent_state.py` 模块

#### 5.4.2 Cognitive Closure（认知闭环）

**问题**: 系统只 `存储 → 检索`，不 `学习 → 修正 → 演化`

**方案**:
```
recall → use → feedback → learn → future_recall_improved
```
- `RetrievalFeedback`: 用户是否引用了被召回的记忆
- `RecallSuccessRate`: 召回记忆被实际使用的比例
- `SelfReflection`: 定期反思——"哪些记忆应该被修订/删除/强化"
- `MemoryCorrection`: LLM 辅助修正低质量记忆
- 长期自优化: 每 N 个 sleep cycle 运行一次 reflection

#### 5.4.3 Cognitive Priority Layer（认知优先级）

**问题**: 所有记忆地位接近，但核心人格应远高于普通聊天

**方案**:

```
优先级排序:
1. 当前任务目标 (active_goals)
2. 用户长期目标 (long_term_goals)  
3. 核心人格/价值观 (core_identity)
4. 高频知识/技能 (frequent_knowledge)
5. 普通事件 (general_events)
```

- 优先级决定: 检索权重、衰减速度、压缩保护级别
- Core Identity 强制注入 prompt（不受 token budget 限制）
- 新增 `cognitive_priority.py`

### 实施优先级矩阵 (Phase 5)

```
                    高影响 ───┬─── 低影响
                   ┌──────────┼──────────┐
             高    │ 5.1.1 Memory Budget │ 5.2.2 抽象链    │
             难    │ 5.1.2 Quality Score │ 5.3.1 Context路由│
             度    │ 5.2.1 四层记忆     │ 5.2.3 Typed Mem │
                   ├──────────┼──────────┤
             低    │ 5.1.3 长期稳定性   │ 5.2.4 Domain图   │
             难    │ 5.3.2 Hebbian学习  │ 5.4.1 Agent状态  │
             度    │ 5.4.2 认知闭环     │ 5.4.3 认知优先级 │
                   └──────────┴──────────┘
```

建议实施顺序: **5.1.1 → 5.1.2 → 5.2.1 → 5.1.3 → 5.3.1 → 5.3.2 → 5.2.2 → 其他**

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
