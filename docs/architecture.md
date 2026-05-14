# NeuroWeave Cortex (NWC) — Repository Overview & Improvement Plan

> Last updated: 2026-05-15 | **v1.6.0** | 649 tests passing | 62 commits

---

## Repository Snapshot

| Metric | Value |
|--------|-------|
| **Version** | 1.6.0 |
| **Production modules** | 78 Python files in `star_graph/` |
| **Production code** | ~36,000 lines |
| **Test files** | 33 files in `tests/` |
| **Test code** | ~9,300 lines |
| **Total tests** | **649** (all passing) |
| **Total commits** | 62 |
| **License** | MIT |

### Module Inventory (78 modules)

```
star_graph/
├── __init__.py              # v1.1.0 exports — 100+ symbols
├── __main__.py              # CLI entry point
├── anchor.py                # Anchor + AnchorVector (10-dim) + memory lifecycle
├── graph.py                 # StarGraph: CRUD, edges, communities, temporal slice
├── sleep.py                 # SleepCycle: 8-phase sleep + sleep rebuild
├── runtime.py               # MemoryRuntime: DI container, auto-sleep, sharded save/load
├── manager.py               # MemoryManager: facade API, retrieval orchestration
├── retrieval_pipeline.py    # RetrievalPipeline: L0→L4 descent logic
├── config.py                # Config system: YAML defaults, schema validation, DotDict
├── defaults.yaml            # All default configuration values
│
├── ── Layer 1: Storage ──
├── storage.py               # Abstract Storage + JSONStorage
├── storage_backend.py       # StorageBackend interface
├── sqlite_storage.py        # SQLiteStorage backend
├── tiered.py                # TieredStorage: HOT/WARM/COLD disk offloading
├── tier.py                  # MemoryTier: STM/MTM/LTM/Core four-layer API + promotion pipeline
├── index.py                 # ANNIndex: approximate nearest neighbor
├── shard.py                 # MemoryShardManager: domain+time+size file sharding
│
├── ── Layer 2: Cognitive ──
├── cortex.py                # MemoryCortex: domain-specific brain regions + hierarchy
├── router.py                # CortexRouter: hierarchy-weighted sparse activation
├── retriever.py             # OSC/Vector/HybridFusion retrievers + PPR
├── dual_channel.py          # DualChannelRetriever: System-1/System-2
├── bm25.py                  # BM25Index: keyword sparse retrieval + RRF fusion
├── scheduler.py             # CognitiveMemoryScheduler: retrieval with decay
├── hub.py                   # HubLayer: cross-cortex abstraction nodes
├── brain_sphere.py          # BrainSphere: O(1) hub center lookup
├── cascade.py               # CascadeRecall: causal chain reasoning
├── timespine.py             # TimeSpine: temporal indexing
├── working_memory.py        # WorkingMemory: short-term buffer
├── hippocampus.py           # HippocampusBuffer: L1(30min)+L2(24h) transient cache
├── gate.py                  # MemoryGate: winner-take-all competition
├── competition.py           # MemoryCompetition: multi-memory scoring
├── resonance.py             # Resonator: oscillatory bridge detection
├── spreading.py             # SpreadingActivation: BFS subgraph activation with traversal weights
├── cognitive_cache.py       # Multi-level cognitive cache: query/session/topic/activation
├── compiler.py              # CognitiveCompiler: 1000→20→5→1 worldview emergence pipeline
├── reflection_loop.py       # SelfReflectionLoop: auto contradiction detection + correction
├── topology.py              # Graph topology ranking: centrality + edge-type-based scoring
├── domain_router.py          # DomainRouter: hierarchical domain→subdomain→cluster pre-filter
├── edge_budget.py            # EdgeBudgetManager: max 32 edges/node with smart retention scoring
├── write_gate.py             # MemoryWriteGate: 5-stage pre-write quality filter (noise/dup/importance)
├── four_layer.py             # FourLayerCompressor: message→event→semantic→personality compression
├── thermal_store.py          # ThermalStore: 3-tier hot/cold/archive auto promotion + demotion
├── edge_decay.py             # EdgeDecayManager: continuous time-based edge decay with adaptive rates
├── self_org.py               # SelfOrganization: auto-cluster, merge near-duplicates, emergent topic detection
├── personality.py             # PersonalityModel: Big Five traits, working style, expertise, values extraction
├── goal_tree.py               # GoalTree: hierarchical goal decomposition, progress propagation, stale archival
├── retrieval_budget.py         # RetrievalBudget: MAX_HOPS=3, MAX_NODES=24, MAX_TOKENS=6000 hard limits
├── versioned_memory.py         # CognitiveTrajectory: belief evolution chains (用户喜欢Python → 用户研究认知架构)
├── cluster_memory.py           # ClusterRouter: query→cluster centroid → scoped retrieval pre-filtering
├── causal_edges.py             # CausalEdgeClassifier: CAUSES/DEPENDS_ON/MOTIVATES/GOAL_OF/RESULT_OF/PRECEDES
├── episodic_memory.py          # EpisodicMemory: time-ordered episode streams with context + emotional arcs
│
├── ── Sleep & Consolidation ──
├── sleep.py                 # SleepCycle: 8-phase + sleep rebuild (fuse/rewire/abstract)
├── online.py                # OnlineConsolidator: real-time micro-consolidation
├── micro_sleep.py           # MicroSleepScheduler: incremental non-blocking
├── cost_estimator.py        # SleepCostEstimator: LLM cost prediction
├── compression.py           # MultiLevelCompressor: RAW→EPISODIC→STRATEGIC→META
├── atom_facts.py            # FactExtractor: LLM entity-centric fact extraction
├── abstraction.py           # AbstractionEngine + AbstractiveMemoryEngine
├── evolution.py             # MemoryEvolutionEngine: belief state transitions
│
├── ── Cognitive Enhancements ──
├── ghost.py                 # GhostSubsystem: pruned memory afterimages
├── community.py             # CommunityDetection: label propagation partitioning
├── autobiography.py         # AutobiographicalMemory: self-narrative formation
├── survival.py              # Survival functions: Ebbinghaus/PowerLaw/Exponential
├── symbolic_filter.py       # SymbolicFilter: rule-based retrieval filtering
├── multimodal.py            # MultimodalEmbedding: CLIP text+image joint embedding
│
├── ── Infrastructure ──
├── embedding.py             # EmbeddingProvider + get_embedder registry
├── math_utils.py            # cosine_sim, safe_div, clamp, sigmoid
├── logger.py                # Structured logging (get_logger/init_logging)
├── metrics.py               # CognitiveMetrics: health + performance
├── snapshot.py              # SnapshotManager: crash recovery + WAL
├── async_manager.py         # AsyncMemoryManager: async API
├── tracing.py               # MemoryTracer: OpenTelemetry spans
├── seed.py                  # Deterministic seeding
├── layers.py                # Layer boundary enforcement
│
├── ── Misc ──
├── streaming.py             # StreamingMemoryBuffer: backpressure
├── exact_cache.py           # ExactMatchCache: O(1) deterministic lookup
├── benchmark.py             # BenchmarkSuite: 5 categories, compare_systems
├── cli.py                   # CLI commands
├── mcp_server.py            # MCP server (optional)
│
└── tests/ (33 files, 649 tests)
    ├── test_v08_modules.py          # Core module smoke tests
    ├── test_sleep_consolidation.py  # Sleep + rebuild + dynamic rewiring + temporal slice + thermal + edge traversal
    ├── test_memory_tier.py          # STM/MTM/LTM/Core tier API + promotion pipeline (16 tests)
    ├── test_spreading.py            # Spreading activation with edge-type-weighted BFS (8 tests)
    ├── test_cognitive_cache.py      # Multi-level cognitive cache (query/session/topic/activation, 23 tests)
    ├── test_cognitive_compiler.py   # Cognitive compiler worldview pipeline (14 tests)
    ├── test_reflection_loop.py      # Self-reflection loop contradiction detection (9 tests)
    ├── test_topology.py             # Graph-first retrieval topology ranking (14 tests)
    ├── test_domain_router.py        # Domain routing topic hierarchy (24 tests)
    ├── test_edge_budget.py          # Edge budget smart eviction (17 tests)
    ├── test_write_gate.py           # Write gate pre-write quality filter (28 tests)
    ├── test_four_layer.py           # Four-layer memory compression (25 tests)
    ├── test_thermal_store.py        # Thermal store 3-tier auto storage (11 tests)
    ├── test_edge_decay.py           # Edge continuous time decay (17 tests)
    ├── test_self_org.py             # Self-organization, clustering, topic detection (22 tests)
    ├── test_personality.py          # Personality model, Big Five traits, expertise (28 tests)
    ├── test_goal_tree.py            # Goal tree, progress tracking, archival (36 tests)
    ├── test_retrieval_budget.py     # Retrieval budget hop/node/token limits (19 tests)
    ├── test_versioned_memory.py     # Cognitive trajectory belief chains (13 tests)
    ├── test_cluster_memory.py       # Cluster router retrieval pre-filtering (10 tests)
    ├── test_causal_edges.py         # Causal edge types inference + tracing (9 tests)
    ├── test_episodic_memory.py      # Episodic memory event streams (16 tests)
    ├── test_config_schema.py        # Config validation (15 tests)
    ├── test_abstractive_memory.py   # Cross-session pattern extraction (6 tests)
    ├── test_cortex_hierarchy.py     # Hierarchy routing + propagation (9 tests)
    ├── test_survival.py             # Survival function decay curves
    ├── test_ghost_intensity.py      # Ghost resurrection + intensity
    ├── test_oscillation_match.py    # OscillationResonanceRetriever
    ├── test_retrieval_trace.py      # RetrievalTrace + ExplainableScore
    ├── test_streaming.py            # StreamingMemoryBuffer backpressure
    ├── test_multimodal.py           # MultimodalEmbeddingProvider
    ├── test_readme_doctest.py       # README code block validation
    └── __init__.py
```

### Architecture Layers

```
Layer 3 (Behavior):  Cortex routing, memory gating, working memory,
                     dimensional reduction retrieval, adaptive replay
Layer 2 (Cognitive): Hub abstraction, cascade recall, time spine,
                     sleep consolidation, evolution, ghost revival
Layer 1 (Storage):   CRUD, persistence, indexing, ANN lookup
```

### Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0.6 | 2026-05-13 | Base: 232 tests, survival, ghost, multimodal, streaming, exact cache, micro-sleep, snapshot, async, benchmark |
| v1.0.7 | 2026-05-14 | Ghost unification, raw buffer priority, ANN incremental, manager split, cortex sleep, dual-channel auto, autobiographical, state/thermal unified, phase correction, config dot-path, sleep naming, cosine dedup, ANN contradictions, retention cache, layer3 timespine, structured logging, config schema, mypy, README updates |
| v1.0.8 | 2026-05-14 | Sleep merge ANN, BM25 hybrid, PPR sparse, EmbedderRegistry instance, AnchorVector 10-dim, tiered storage |
| v1.0.9 | 2026-05-14 | Global anchor hard cap, auto-sleep daemon, cold ghost cleanup, cortex auto-consolidation |
| **v1.1.0** | **2026-05-15** | **Hippocampus buffer, edge sparsification, file sharding, sleep rebuild (fuse/rewire/abstract), cortex hierarchy, abstractive memory engine, dynamic neural rewiring, success-rate RL, temporal slice projection — 276 tests** |
| **v1.2.0** | **2026-05-15** | **Memory tiering, decay+reinforcement loop, FROZEN thermal tier, edge traversal weights, spreading activation, cognitive cache, cognitive compiler (worldview emergence), self-reflection loop, graph-first retrieval — 373 tests** |
| **v1.3.0** | **2026-05-15** | **Domain router (hierarchical topic tree), edge budget (smart eviction, max 32), write gate (5-stage quality filter), four-layer compression (M→E→S→P) — 467 tests** |
| **v1.4.0** | **2026-05-15** | **Spreading activation primary retrieval, 3-tier thermal store (hot/cold/archive), continuous edge time decay — 496 tests** |
| **v1.5.0** | **2026-05-15** | **Renamed to NeuroWeave Cortex (NWC). Self-organization (auto-cluster/merge/topics), personality model (Big Five traits/expertise/values), goal tree (hierarchical progress tracking) — 582 tests** |
| **v1.6.0** | **2026-05-15** | **Retrieval budget (S-5: hop/node/token limits), versioned memory (A-9: belief evolution chains), cluster memory (A-10: retrieval pre-filtering), causal edge types (B-12: 6 richer types), episodic memory (B-13: event streams). All S/A/B items complete — 649 tests** |

---

> Based on comprehensive codebase audit (2026-05-13). Originally 232 tests passing, v1.0.6.

## Priority Summary

| Priority | Count | Impact |
|----------|-------|--------|
| **P0** | 6 issues | Correctness, performance, data consistency, retrieval quality |
| **P1** | 6 issues | Architecture, maintainability, retrieval quality |
| **P2** | 3 issues | Cognitive fidelity, production readiness |
| **P3** | 11 issues | Code quality, testing, developer experience |

---

## v1.0.8 — Retrieval Quality & Scale Hardening (NEW)

> Based on external code review (2026-05-13). Addresses retrieval quality gaps,
> O(n²) bottlenecks, singleton pollution, and storage tier mismatch.

| Issue | Priority | Impact |
|-------|----------|--------|
| #21 Sleep merge O(n²) → ANN | **P0** | Scale blocker — sleep() freezes on 10K+ anchors |
| #22 BM25 + embedding hybrid | **P0** | Retrieval quality — pure embedding caps at 15% has_answer |
| #23 PPR precompute/approximate | **P0** | 805ms → <200ms latency target |
| #24 EmbedderRegistry instance | **P1** | Multi-instance pollution |
| #25 AnchorVector cleanup | **P1** | Remove self-cycling dimensions |
| #26 Tiered storage (HOT→disk) | **P1** | ThermalState has no actual storage switching |

### 21. Sleep merge 用 ANN 近似 (规模化 blocker)

**现状**: `sleep.py` `_merge_similar()` 双重循环遍历所有 anchor 对，O(n²)。

**方案**: 用 HNSW/ANN 索引预筛选候选对，只对 embedding 相似的 anchor 对计算合并分数:
```python
def _merge_similar(self, anchors: list[Anchor]) -> list[MergePair]:
    candidates = self.ann.query_batch([a.embedding for a in anchors], top_k=20)
    # Only check pairs returned by ANN, O(n * k) instead of O(n²)
```

### 22. HybridFusion 引入 BM25 关键词通道 (检索质量)

**现状**: 所有检索路径最终都是 embedding cosine 变体。LoCoMo 15% has_answer 说明语义匹配不够。

**方案**: HybridFusion 增加 SparseRetrieval 通道:
```python
class HybridFusionRetriever:
    def retrieve(self, query, ...):
        dense_results = self._dense_search(query, query_emb, top_k)
        sparse_results = self._bm25_search(query, top_k)  # NEW
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
```

### 23. PPR 预计算或采样近似 (延迟优化)

**现状**: `personalized_pagerank()` 每次查询全量矩阵迭代，HybridFusion 单次查询 805ms。

**方案**:
- 预计算：sleep 时对每个社区预计算 PPR 向量
- 运行时：只取预计算向量中 query-relevant 的 top-k
- 或：用 sampling-based PPR (Monte Carlo random walks) 近似

### 24. EmbedderRegistry 改为 instance-level

**现状**: `EmbedderRegistry` 是 class-level 单例，多 MemoryManager 实例共享 embedding provider，配置互相覆盖。

**方案**: 改为 instance attribute，每个 MemoryRuntime 持有独立的 `EmbedderRegistry` 实例。

### 25. AnchorVector 维度清理

**现状**: 13 维中 `success_feedback`、`future_reusability` 没有外部输入机制，基本是随机初始值自循环。

**方案**:
- 移除无外部输入的维度（success_feedback, future_reusability）
- 合并语义重复的维度
- 保留有明确更新路径的维度

### 26. 分层存储：ThermalState → 实际介质切换

**现状**: HOT/WARM/COLD 只是属性标签，没有对应的存储切换。

**方案**:
- HOT: 内存 (Python dict)
- WARM: 内存 + 定期 flush 到磁盘
- COLD: 仅磁盘，按需加载到内存
- 在 `_on_enter_state()` 中触发介质迁移

---

## P0 — Correctness & Performance (do first)

### 1. Ghost 子系统统一 (数据一致性)

**现状**: `graph.py` 有 `add_ghost()` 创建 `GhostAnchor`，`ghost.py` 有 `GhostSubsystem` 创建 `GhostNode`。两者数据结构不同，`sleep.py` 的 `_prune_anchors()` 需要 fallback:
```python
if hasattr(self.graph, '_ghost_subsystem') and self.graph._ghost_subsystem:
    self.graph._ghost_subsystem.create(anchor, residual_edges)
else:
    self.graph.add_ghost(anchor)
```

**问题**: `GhostAnchor` 和 `GhostNode` 字段不兼容，持久化/加载可能丢失信息。

**方案**:
- 统一为 `GhostNode`，`StarGraph` 只持有 ghost ID 引用，实际数据由 `GhostSubsystem` 管理
- `add_ghost()` 委托给 subsystem
- 清理 `graph.py` 中的 `GhostAnchor` 类

### 2. Raw Buffer 优先级提升 (检索质量)

**现状**: `recall()` 中 Path 0 (Exact Cache) → Path A (Raw Buffer) → Path B (Graph)，但 Raw Buffer 结果被放在 graph 结果**之后**合并。

**问题**: Raw Buffer 存的是最近 1-2 个 session 的原始对话，对于短期事实类查询命中率远高于压缩后的 anchor。

**方案**:
```python
# 当前：exact → graph → raw
# 改为：exact → raw → graph
merged_items = exact_results + raw_results + graph_results
```

### 3. ANN 索引增量维护 (大图性能)

**现状**: `cortical_lookup()` 每次查询时如果索引不同步就全量重建:
```python
if not self._ids_in_ann_sync():
    ann.clear()
    for aid, a in self.anchors.items():
        if a.embedding: ann.add(aid, a.embedding)
    ann.rebuild()
```

**方案**:
- `add_anchor()` 时同步添加到 ANN
- `remove_anchor()` 时同步从 ANN 删除
- 取消 `_ids_in_ann_sync()` 检查
- 只在 sleep 的 Index Rebuild 阶段做全量 rebuild

---

## P1 — Architecture & Maintainability

### 4. MemoryManager 拆分 (可维护性)

**现状**: `manager.py` 已有 1700+ 行，导入 40+ 模块，20+ lazy property。同时承担:
- Facade API
- 依赖注入容器（管理所有子系统生命周期）
- 检索编排器（recall / retrieve_with_descent）
- 持久化协调器（save/load/snapshot）

**方案**: 拆分为三个角色:
```python
class MemoryRuntime:      # 依赖容器 + 生命周期管理
class RetrievalPipeline:  # 检索编排（L0→L4 降级逻辑）
class MemoryManager:      # 仅保留 facade API，委托给上面两个
```

### 5. Cortex 独立睡眠 (架构一致性)

**现状**: `add_cortex()` 创建 `MemoryCortex`，但 `sleep()` 仍对 `self.graph` 做统一睡眠，各 cortex 不独立休眠。

**方案**:
- `MemoryManager.sleep()` 遍历 `router.cortices`，对每个 cortex 调用 `cortex.sleep()`
- 全局 sleep 只做跨 cortex 的事情（Hub 桥接、全局 schema 提取）
- 各 cortex 的 `ANNIndex` 独立，不共用 `self.graph._ann_index`

### 6. Dual-Channel 自动触发 (检索质量)

**现状**: `dual_recall()` 存在，但 `recall()` 内部没有调用它。用户必须显式选择 `dual_recall()`。

**方案**: 在 `recall()` 中增加自动路由:
```python
def recall(self, query, context, max_items=10):
    result = self._system1_recall(query, context, max_items)
    system2_keywords = {"all", "which", "before", "last", "list", "every"}
    needs_system2 = (
        any(kw in query.lower() for kw in system2_keywords)
        or result.confidence < 0.35
    )
    if needs_system2:
        s2_result = self._system2_recall(query, context, max_items)
        return self._merge_channels(result, s2_result)
    return result
```

---

## P2 — Cognitive Architecture Fidelity

### 7. 自传体记忆层 ("我"的形成)

**现状**: `reflection.py` 的 reflection 是"对用户的反思"（"用户喜欢什么"），而非"对自我的反思"（"我如何理解用户"）。

**方案**: 增加 `AutobiographicalMemory` 层:
```python
@dataclass
class SelfNarrative:
    episode_summary: str      # "我和用户讨论过 Redis 超时"
    self_belief: str          # "我认为用户偏好简洁代码"
    emotional_tone: float     # 这次互动中"我"的情绪
    formed_at: float
    stability: float
```
区别于 reflection node 的"客观分析"，而是"主观体验"。

### 8. State / Thermal State 统一 (认知架构清晰度)

**现状**: 两套状态系统并存，映射关系散落在多个文件:
- `MemoryState`: ACTIVE, REHEARSING, CONSOLIDATING, DORMANT, GHOST, REACTIVATED
- `ThermalState`: HOT, WARM, COLD, DEAD

**方案**: 明确定义状态转换矩阵，在状态转换时同步更新 thermal state:
```python
_STATE_THERMAL_MAP = {
    MemoryState.ACTIVE: ThermalState.HOT,
    MemoryState.REHEARSING: ThermalState.HOT,
    MemoryState.CONSOLIDATING: ThermalState.WARM,
    MemoryState.DORMANT: ThermalState.WARM,
    MemoryState.GHOST: ThermalState.COLD,
    MemoryState.REACTIVATED: ThermalState.HOT,
}
```

### 9. 振荡共振 phase 来源修正

**现状**: `Oscillator.derive_phase()` 和 `derive_frequency()` 从 embedding 统计量推导，缺乏生物学依据。

**方案**: phase 应来自:
- 时间上下文（一天中的时段、会话序号）
- 情绪节奏（对话中的情绪波动周期）
- 而非 embedding 的统计特征

---

## P3 — Code Quality & Production Readiness

### 10. 配置访问脆弱性

**现状**: 大量 `getattr(self.cfg, 'something', None)` 链式调用，配置结构变化时静默失败。

**方案**: `_DotDict` 增加 `get_path()` 支持:
```python
self.cfg.get('exact_cache.auto_harvest', True)
```

### 11. Sleep Phase 编号统一

**现状**: `run_phased()` 中有 N1, N2, N3, REM, Wake-prep, Phase 5b, Phase 5c, Phase 6, Phase 7, Phase 8 — 命名不统一。

**方案**: 统一命名:
```
N1_Replay, N2_Merge, N3_Compression, N3b_AtomFacts,
REM_Emotion, N4_Prune, N5_HubConnect, N6_IndexRebuild
```

### 12. 余弦相似度去重

**现状**: `graph.py`, `sleep.py`, `scheduler.py`, `manager.py`, `streaming.py` 各有自己的 `_cosine_sim` 实现。

**方案**: 提取为 `star_graph/math_utils.py` 中的统一函数。

### 13. find_contradictions() O(n²) 优化

**现状**: 遍历所有 anchor 对计算 embedding 相似度。

**方案**: 使用 ANN 索引先找出高相似度候选对，再检查情绪对立:
```python
# 复杂度从 O(n²) 降到 O(n * k)
```

### 14. retrieve_with_descent() Layer 3 全量扫描

**现状**: Layer 3 (2D Plane) 遍历所有 cortex 的所有 anchors，可能 O(50K)。

**方案**: 利用 TimeSpine 索引，只扫描最近 N 天的桶。

### 15. retention_score 缓存

**现状**: `retention_score` 是 `@property`，每次访问实时计算。检索路径中每个 anchor 被访问多次。

**方案**: 状态变化时缓存，设脏标记。

### 16. 测试覆盖率提升

**现状**: 232 个测试，但很多模块覆盖不足（benchmark, community, atom_facts 等）。

**方案**:
- 每个核心模块至少单元测试
- 集成测试：模拟 100 轮对话 → sleep → 验证 ghost 复活
- 基准测试自动化

### 17. 静态类型检查

**现状**: 大量 `| None` 和 `Any`，但很多地方没有类型注解。

**方案**: 引入 mypy 或 pyright。

### 18. 配置验证增强

**现状**: `_check_ranges()` 只检查数值范围，不检查配置段存在性和兼容性。

**方案**: 增加配置 schema 验证。

### 19. 日志系统标准化

**现状**: `sleep.py` 用 `self.log: list[str]`，`manager.py` 用 `print()`。

**方案**: 引入标准 `logging`，支持结构化日志。

### 20. 文档与代码同步

**现状**: README 中的 Quick Start 示例和实际 API 可能不完全一致。

**方案**: CI 中增加 doctest，确保 README 代码示例可实际运行。

---

## Implementation Order

```
Phase 1 (P0, v1.0.7): Ghost 统一 → Raw Buffer 优先级 → ANN 增量
Phase 2 (P1, v1.0.7): Manager 拆分 → Cortex 独立睡眠 → Dual-Channel 自动触发
Phase 3 (P2, v1.0.7): 自传体记忆 → 状态机统一 → Phase 来源修正
Phase 4 (P3, v1.0.7): 代码质量逐项修复（10-15）
Phase 5 (v1.0.8):    Sleep merge ANN → BM25 hybrid → PPR approx → EmbedderRegistry → AnchorVector → Tiered storage
Phase 6 (v1.0.9):    Global hard cap → Auto-sleep daemon → Cold ghost cleanup → Cortex hard rejection
Phase 7 (v1.1.0):    Hippocampus buffer → Edge sparsification → File sharding (Memory OS)
Phase 8 (v1.1.0):    Sleep rebuild → Cortex hierarchy → Abstractive memory → Dynamic rewiring → Success-rate RL → Temporal slice projection
```

---

## v1.0.9 — Resource Bounding & Anti-Bloat

> Memory growth analysis (2026-05-14). Existing sleep pruning/merging/compression
> provides downward pressure, but there is no hard ceiling.

| Issue | Priority |
|-------|----------|
| #27 Global anchor hard cap + eviction policy | **P0** |
| #28 Auto-sleep daemon (background consolidation scheduler) | **P1** |
| #29 Cold ghost disk cleanup (TieredStorage.compact) | **P1** |
| #30 Cortex soft trigger → hard rejection on overflow | **P1** |

### 27. Global anchor hard cap + eviction policy
- Add `graph.max_total_anchors` (50K) and `graph.eviction_policy` (lru/fifo/lowest_retention)
- `remember()` triggers eviction before insert if at capacity

### 28. Auto-sleep daemon
- `AutoSleepScheduler` that triggers micro-sleep on anchor count threshold, full-sleep on time interval

### 29. Cold ghost disk cleanup
- `TieredStorage.delete(anchor_id)` for purged ghosts
- `tiered.compact()` to rewrite the JSON file without dead entries

### 30. Cortex hard rejection
- When `cortex.needs_consolidation()`, either auto-consolidate or reject new anchors

---

## v1.1.0 — Memory Operating System (Architecture Review 2026-05-14)

> **Core insight: the problem is not "too many memories" — it's "too many connections"
> and "all memories participate in every computation."

### Phase 7 — Foundation (SSS)

#### #31 Hippocampus Buffer (highest priority)

User input should NOT go directly to long-term memory. Add a hippocampus cache layer:

```
Input → Working Memory → Hippocampus Buffer → [sleep decides] → Long-term Memory
```

- **L1 (instant)**: ~30min, no vectorization, no graph, text-only cache for active conversation/task chain
- **L2 (short-term)**: ~24h, lightweight vectorization, local graph, sleep-processable
- Sleep decides: promote / summarize / merge / discard

Prevents long-term graph pollution.

#### #32 Edge Sparsification (SSS)

Current `cosine > threshold → connect()` leads to O(n²) edge explosion — everything becomes vaguely related, recall fails.

- **Only explicable relations** get edges: CAUSES, FIXES, DEPENDS_ON, CONTRADICTS, UPGRADES, SUMMARIZES, RELATED_WORKFLOW, SAME_PROJECT, SAME_USER_GOAL
- Every edge must carry: `type`, `weight`, `ttl` (auto-disconnect after inactivity)
- Ban pure cosine-similarity edges

#### #33 File Sharding: Domain + Time + Size

Don't cut files by size alone. Three-layer sharding:

```
memory/
├── procedural/       # how-to, workflows, solutions (high compression, long life)
│   ├── python/
│   │   ├── 2026_Q2_01.mem
│   │   └── ...
│   └── java/
├── episodic/         # conversations, events (high volume, fast decay)
│   ├── 2026_05_week2.mem
│   └── ...
├── semantic/         # user preferences, abstract knowledge (long-term stable)
│   ├── user_preferences.mem
│   └── world_knowledge.mem
├── reflection/       # AI self-summary, error patterns, strategy (smallest, highest weight)
│   └── strategy.mem
└── hippocampus/      # active buffer, short-term cache
    ├── active_buffer.mem
    └── short_term.mem
```

Single file: 10-50MB recommended, 100MB max before sleep/recall costs rise sharply.

### Phase 8 — Cognitive (SS)

#### #34 Sleep Rebuild (not just compress)

Sleep must **restructure the entire graph**, not just prune/merge:

- **Node fusion**: `try-except` + `python异常处理` + `错误捕获` → `Python Error Handling`
- **Graph rewiring**: drop weak/stale/low-success edges, strengthen high-frequency success paths
- **Abstractive memory**: concrete events → pattern memory (e.g., "chromedriver fix failed" → "Browser Driver Version Conflict")

#### #35 Cortex Hierarchy (not flat)

Current memories are equal-priority. Correct structure:

```
Reflection Cortex   (smallest volume, highest weight) — AI self-summary, error patterns, strategy
    ↓
Semantic Cortex     (long-term stable) — user preferences, concepts
    ↓
Procedural Cortex   (high compression, low forgetting) — workflows, solutions
    ↓
Episodic Cortex     (highest volume, fastest decay, biggest pollution source) — conversations, events
    ↓
Hippocampus Buffer  (transient cache) — active context
```

#### #36 Abstractive Memory

Don't remember concrete events forever. Form **pattern memory**:
- Extract recurring patterns across episodes
- Abstract into generalized knowledge
- Concrete source events decay faster than their abstractions

### Phase 9 — Self-Evolving (S)

#### #37 Dynamic Neural Rewiring

Graph structure is not static — it evolves:
- High-frequency co-activation → auto-form cluster
- Long-term inactivity → auto-disconnect edges
- Successful reasoning chains → strengthen weights
- Failed reasoning chains → weaken weights

#### #38 Success-Rate RL for Memory Valuation

Memory score should include `success_rate`, not just similarity + recency:

```python
memory_score = relevance * recency * reuse_frequency * success_rate * emotional_weight * novelty
```

The system must know which memories actually helped complete tasks.

#### #39 Temporal Slice Projection

Don't let ALL memories participate in recall. Only a limited-width active surface:

```
2026-05-14
├── Core memories (max 7)
├── Active memories (max 20)
├── Background summary
└── Noise layer (excluded from recall)
```

Reduces context pollution, token waste, and agent confusion.

---

## v1.1.0 (completed — all phases)

### Phase 7 — Foundation (SSS)
- [x] #31 Hippocampus Buffer: L1(30min text) + L2(24h vectorized), sleep_decide promotion
- [x] #32 Edge Sparsification: EXPLICABLE_RELATIONS gate, strong/legacy classification, edge TTL
- [x] #33 File Sharding: domain+time+size three-layer partitioning (MemoryShardManager)

### Phase 8 — Cognitive (SS)
- [x] #34 Sleep Rebuild: multi-node fusion (union-find clustering), graph rewiring (drop/strengthen/transitive), abstractive memory (concrete→pattern)
- [x] #35 Cortex Hierarchy: CORTEX_HIERARCHY levels (reflection>semantic>procedural>episodic), HIERARCHY_WEIGHTS routing boosts, factory methods, propagate_down()
- [x] #36 Abstractive Memory: AbstractiveMemoryEngine with cross-session pattern extraction, PatternMemory tracking, auto-promotion to AbstractNode, concrete decay

### Phase 9 — Self-Evolving (S)
- [x] #37 Dynamic Neural Rewiring: edge success/failure tracking, RL-based _dynamic_rewire (boost/weaken/cluster), graph.record_chain_success/failure
- [x] #38 Success-Rate RL: Edge.success_rate property, graph-level chain success/failure tracking, success_feedback integrated into retention scoring
- [x] #39 Temporal Slice Projection: graph.temporal_slice() with core(7)/active(20)/background/noise tiers

### Previously completed
- [x] v1.0.7: Ghost unification, Raw Buffer priority, ANN incremental, Manager split, Cortex sleep, Dual-Channel auto, Autobiographical, State/Thermal unified, Phase correction, Config dot-path, Sleep naming, Cosine dedup, ANN contradictions, retention cache, Layer3 TimeSpine, Structured logging, Config schema, mypy, README updates, Test coverage (250→276 tests)
- [x] v1.0.8: Sleep merge ANN, BM25 hybrid, PPR sparse, EmbedderRegistry instance, AnchorVector 10-dim, Tiered storage
- [x] v1.0.9: Global hard cap, Auto-sleep daemon, Cold ghost cleanup, Cortex hard rejection

---

## Recently Completed (v1.0.7-dev)

### P0 (correctness & performance)
- [x] Ghost subsystem unification (GhostAnchor removed, GhostNode/GhostSubsystem single source)
- [x] Raw Buffer priority elevation (exact → raw → graph merge order)
- [x] ANN index incremental maintenance (add/remove sync, no full rebuild on query)

### P1 (architecture & maintainability)
- [x] MemoryManager split (MemoryRuntime + RetrievalPipeline + MemoryManager facade)
- [x] Cortex independent sleep (per-cortex consolidate() before global cross-cortex pass)
- [x] Dual-Channel auto-trigger (structural keyword detection + low-confidence fallback)

### P2 (cognitive architecture fidelity)
- [x] Autobiographical memory layer (SelfNarrative + AutobiographicalMemory)
- [x] State/ThermalState unification (_STATE_THERMAL_MAP + synchronized transitions)
- [x] Oscillation phase source correction (temporal/emotional context, not embedding stats)

### P3 (code quality — partial)
- [x] Config dot-path access: _DotDict.get('a.b.c', default) + Config.get_path()
- [x] Sleep phase naming standardized (N1_Replay → N7_IndexRebuild)
- [x] Cosine similarity dedup: math_utils.cosine_sim (runtime + retrieval_pipeline migrated)
- [x] find_contradictions() ANN-accelerated: O(n²) → O(n*k) with near-neighbor pre-filter
- [x] retention_score caching: 0.5s TTL cache saves recomputation in hot retrieval loops
- [x] Layer 3 TimeSpine-indexed scan: O(days*buckets) replaces full O(n) cortex scan; remember()/forget() auto-populate spine
- [x] Structured logging: logger.py with get_logger/init_logging; sleep.py uses _log_event (logging + report list)
- [x] Config schema validation: CONFIG_SCHEMA with type/range/allowed_values checks; _validate_schema() for section existence + key types + cross-section compat
- [x] mypy config: pyproject.toml with basic strictness settings
- [x] README.md + README_CN.md: updated to reflect v1.0.7-dev architecture
- [x] Test coverage: 18 new tests (config schema, eviction policies, README doctest imports)
- [x] mypy config: pyproject.toml with strictness settings; real bugs fixed (FilterResult, exact_cache any→Any)
- [x] README doctest: test_readme_doctest.py validates code blocks import cleanly
- [x] All P3 items complete. Test suite: 250 tests passing.

### v1.0.9 (resource bounding — anti-bloat)
- [x] #27 Global anchor hard cap + eviction policy (lru/fifo/lowest_retention)
- [x] #28 Auto-sleep daemon: _check_auto_sleep() triggered on anchor count + time intervals
- [x] #29 Cold ghost cleanup: GhostSubsystem.decay_all() returns IDs → TieredStorage.remove() + compact()
- [x] #30 Cortex hard rejection: MemoryCortex.ensure_capacity() auto-consolidates overfull cortices

### v1.0.8 (emergency hardening — external code review)
- [x] #21 Sleep merge O(n²) → ANN-accelerated: candidate pairs pre-filtered via ANN query, O(n*k)
- [x] #22 BM25 + embedding hybrid: BM25Index with incremental add/remove, RRF fusion in System-1 search
- [x] #23 PPR sparse rewrite: seed-scoped dict-based iteration, no dense n×n matrix
- [x] #24 EmbedderRegistry instance-level: per-runtime registry prevents multi-Manager embedder pollution
- [x] #25 AnchorVector 13→10 dims: remove future_reusability, merge novelty→surprise, task_relevance→importance
- [x] #26 Tiered storage: COLD anchors offload to disk (TieredStorage), transparent thaw on access, wired into sleep

### Previous (v1.0.6)
- [x] Survival functions (Ebbinghaus / Power-law / Exponential / Custom)
- [x] Ghost intensity + NegativeGhost contradiction tracking
- [x] Multimodal memory (CLIP joint embedding text+image)
- [x] Streaming memory buffer with backpressure
- [x] Exact match cache (KV deterministic O(1) lookup)
- [x] Micro-sleep scheduler (incremental non-blocking consolidation)
- [x] Snapshot + WAL (crash recovery)
- [x] Async manager + tracing (OpenTelemetry spans)
- [x] Benchmark suite (5 categories)
- [x] Dependency manifest (requirements.txt)
- [x] Version unification (1.0.6) + orphan module exports
- [x] 232 tests passing

---

## v1.2.0 — Cognitive OS: Memory Tiering, Compiler, Activation (Architecture Review 2026-05-15)

> **核心诊断：图正在退化成"高维链表"。所有东西跟所有东西有点像，边数量指数膨胀，热点节点形成"记忆黑洞"。**
>
> 真正需要的是：**记忆分层 + 衰减 + 抽象 + 边类型 + 局部激活 + 认知编译**。
>
> 这个项目的竞争点不是 vector DB / RAG，而是 **"关系型认知记忆"** — memory is not storage, memory is structure.

### Gap Analysis: 已有 vs 缺失

| # | 方向 | 已有基础 | 关键缺失 |
|---|------|---------|---------|
| 1 | 记忆分层 STM/MTM/LTM/Core | HippocampusBuffer(L1/L2), cortex hierarchy, ThermalState | 无显式四层 API，无 MTM topic-cluster，无 Core 层 |
| 2 | 边类型系统 | EXPLICABLE_RELATIONS(25种), STRONG_RELATIONS(12种), edge sparsification | causal/temporal/preference/task_flow 区分度不够 |
| 3 | 记忆衰减 | AnchorVector decay_rate, survival functions, Ebbinghaus/PowerLaw, retention_score | 缺强化激活反馈闭环，衰减与 thermal 联动不够紧密 |
| 4 | Concept 抽象 | AbstractionEngine, AbstractiveMemoryEngine, PatternMemory, sleep rebuild | 缺 cognitive compression pipeline (1000→20→5→1) |
| 5 | 激活扩散 | CortexRouter sparse activation (1-3 cortices), temporal_slice (core/active/bg/noise) | 无图内局部扩散 (spreading activation on subgraph) |
| 6 | Memory Shard + Temperature | MemoryShardManager, TieredStorage(HOT/WARM/COLD), ThermalState | shard 与 temperature 未联动，无 frozen 层 |
| 7 | 强化记忆 | Edge.co_activation_count, Anchor.record_success/failure, success_rate | success_rate 未纳入主要检索路径权重 |
| 8 | 认知压缩 | MultiLevelCompressor(RAW→EPISODIC→STRATEGIC→META), sleep compression | 无 worldview 层，无 user profile 层，无 1000→1 全链路 |
| 9 | 向量污染防护 | Edge sparsification gate (拒绝纯 cosine 边), explicable relations only | temporal+task_relevance 多维边权未实现 |
| 10 | 认知缓存 | ExactMatchCache, retention_score cache (0.5s TTL) | 缺 query cache, session cache, topic cache |
| 11 | 自反思循环 | AutobiographicalMemory, SelfNarrative, sleep rebuild | 无自动错误记忆修正闭环 |
| 12 | Graph Cognition First | StarGraph + RichEdge + HubLayer + BrainSphere + Cortex | Embedding 仍是主要入口，需弱化 embedding-first 改为 graph-first |

---

### Phase 10 — Memory Tiering (P0, v1.2.0) ✅ COMPLETE

- [x] #40 Memory Tiering: STM/MTM/LTM/Core four-layer API + promotion pipeline
- [x] #41 Cognitive Decay + Reinforcement: FROZEN thermal tier, reinforcement-adjusted decay, five-level downgrade
- [x] #42 Edge Type Deepening: EDGE_TRAVERSAL_WEIGHTS wired into RichEdge, neighbors(), spread_activation, cascade

#### #40 Explicit STM/MTM/LTM/Core Four-Layer API

Current state: HippocampusBuffer is a transient cache, but there's no explicit tiering API. Every `remember()` call goes through the same path.

Target architecture:
```
Input → STM (deque, no graph, high churn)
     ↘ MTM (StarGraph, topic clusters, medium stability)
     ↘ LTM (summary nodes, high compression, low write)
     ↘ Core (user profile, capability model, worldview — almost never changes)
```

Implementation:
- `MemoryTier` enum: STM, MTM, LTM, CORE
- STM: `collections.deque` + embedding, max 100 items, TTL 2 hours, no graph overhead
- MTM: existing StarGraph + cortex, topic-cluster granularity (not message-level)
- LTM: summary-only nodes, write-on-consolidation, high stability (decay 365+ days)
- Core: key-value profile store, manual + auto-extracted, near-immutable
- `remember()` auto-routes by tier: new input → STM, sleep promotes STM→MTM→LTM→Core
- `recall()` searches tiers in order: Core → LTM → MTM → STM

#### #41 Cognitive Decay + Reinforcement Feedback Loop

Current state: AnchorVector has decay_rate and survival functions, but reinforcement doesn't feed back into decay curve adjustment.

Implementation:
- Decay formula: `weight_t = weight_0 × e^(-λt × (1 - reinforcement × 0.3))`
- Each successful recall → `reinforcement += 0.05`, slows future decay
- thermal downgrade path: HOT(7d) → WARM(30d) → COLD(90d) → FROZEN(archived)
- `_apply_reinforcement_decay()` in sleep: scan all anchors, adjust decay_rate by success_rate
- Frozen tier: write to disk-only shard, exclude from ANN index

#### #42 Edge Type Deepening: Causal + Temporal + Preference + TaskFlow

Current state: 25 explicable relations exist, but the retrieval system doesn't differentiate them during traversal.

Implementation:
- Edge type → traversal weight mapping:
  - causal: ×1.5 (most important for reasoning)
  - temporal(before/after): ×1.2 (ordering matters)
  - preference: ×1.3 (user intent signal)
  - task_flow: ×1.4 (workflow continuity)
  - semantic: ×1.0 (baseline)
  - contradiction: ×0.5 (useful but downweighted)
- `CascadeRecall` uses edge types to trace causal chains
- `add_edge()` auto-infers type from context when possible

---

### Phase 11 — Spreading Activation + Cognitive Cache (P1, v1.2.0)

#### #43 Local Subgraph Activation (Spreading Activation)

Current state: Retrieval searches entire cortex subgraphs (even with sparse activation, each cortex can have 5000+ anchors).

Implementation:
- Given query embedding, find seed node(s) via ANN
- From seed, spread activation to neighbors (BFS, max depth 3)
- Each hop: `activation *= edge_weight × decay(0.6)`
- Collect activated nodes, rank by accumulated activation
- Return top-k from activated subgraph only (not full graph)
- `SpreadingActivation` class in `retriever.py` or new module
- Hybrid: spreading activation for graph structure + embedding for seed finding

#### #44 Multi-Level Cognitive Cache

Current state: ExactMatchCache exists but query/session/topic caches don't.

Implementation:
- `QueryCache`: LRU cache of recent query→result pairs, TTL 5 min
- `SessionCache`: per-session working set of frequently accessed anchors
- `TopicCache`: pre-computed topic→top_anchors mapping, rebuilt on sleep
- `ActivationCache`: cached spreading activation results for hot seeds
- All caches in `cognitive_cache.py`, wired into `recall()` before full retrieval

---

### Phase 12 — Cognitive Compiler + Worldview (P2, v1.2.0) ✅ COMPLETE

- [x] #45 Cognitive Compression Pipeline (1000→20→5→1)
- [x] #46 Self-Reflection Loop (Auto Error Correction)
- [x] #47 Graph-First Retrieval (Weaken Embedding Primacy)

#### #45 Cognitive Compression Pipeline (1000→20→5→1)

Implementation:
```
1000 messages → Sleep compression → 200 episodic summaries
200 episodic → AbstractiveMemoryEngine → 20 concept nodes
20 concepts → Cross-session pattern merge → 5 worldview nodes
5 worldviews → Core profile extraction → 1 user profile
```
- `CognitiveCompiler` class: orchestrates the full pipeline
- Each level has its own compression ratio and stability threshold
- Worldview nodes: long-term stable beliefs about the user (e.g., "prefers Python", "works on memory systems", "values code simplicity")
- User profile: auto-extracted from worldview consensus

#### #46 Self-Reflection Loop (Auto Error Correction)

Current state: AutobiographicalMemory tracks self-narratives, but doesn't actively correct errors.

Implementation:
- Contradiction detection on sleep: if A contradicts B, check which has higher confidence+stability
- Lower-confidence belief gets `invalidated_by` edge, reduced weight
- `SelfCorrectionReport`: log of what was corrected and why
- Agent can query "what did I get wrong about X?" → returns corrected beliefs
- Ghost revival: if corrected belief was wrong, original ghost can be reactivated

#### #47 Graph-First Retrieval (Weaken Embedding Primacy)

Current state: Retrieval entry points are embedding-based (ANN → cosine). Graph structure is secondary.

Implementation:
- New `recall()` path: graph traversal first, embedding refinement second
- Seed selection by embedding (cheap), then graph walk for context (rich)
- `topology_rank()`: score nodes by graph centrality + edge type richness, not just embedding similarity
- Config flag: `retrieval.graph_first_weight` (default 0.6) to balance graph vs embedding signals

---

### Priority Order (v1.2.0)

| Phase | Priority | Items |
|-------|----------|-------|
| **10** | **P0** | #40 Memory Tiering, #41 Decay+Reinforcement, #42 Edge Type Deepening |
| **11** | **P1** | #43 Spreading Activation, #44 Cognitive Cache |
| **12** | **P2** | #45 Cognitive Compiler, #46 Self-Reflection, #47 Graph-First Retrieval |
| **13** | **S** | #48 Domain Router, #49 Edge Budget, #50 Write Gate, #51 Four-Layer Compression |
| **14** | **A** | #52 Spreading Activation Retrieval, #53 Thermal Store, #54 Edge Time Decay |

---

### Phase 13 — Graph Quality & Anti-Degeneration (S-Level, v1.3.0) ✅ COMPLETE

- [x] #48 Domain Router (hierarchical topic tree pre-filtering)
- [x] #49 Edge Budget Hardening (max 32 edges/node, smart eviction)
- [x] #50 Memory Write Gate (5-stage pre-write quality filter)
- [x] #51 Four-Layer Compression (message→event→semantic→personality)

#### #48 Domain Router

Implementation:
- `DomainRouter` class with `DEFAULT_DOMAIN_TREE`: 6 root domains (开发, AI, 运维, 数据库, 金融, 工具效率)
- Subdomain→Cluster tree with keyword-based routing
- `index_anchor()` assigns each anchor to best-matching domain
- `route(query)` → `{matched_domains, anchor_ids, depth, path}` for retrieval pre-filtering
- `get_candidate_scope()` narrows ANN search to domain subtree
- Wired into `retrieval_pipeline.py` recall() for domain-based score boosting

#### #49 Edge Budget Hardening

Implementation:
- `EdgeBudgetManager(max_edges=32)`: enforces per-node edge cap
- `EDGE_TYPE_RETENTION_PRIORITY`: causes(10) > fixes(9) > depends_on(8) > ... > contradicts(1)
- Retention formula: `type_score × 0.4 + weight × 0.3 + recency × 0.15 + activity × 0.15`
- `enforce(graph, node_id)`: scores and evicts weakest edges when over budget
- `enforce_all(graph)`: budget enforcement across all nodes (called during sleep)
- Wired into `runtime.connect()` and `runtime.remember()` auto-connect
- Config section: `edge_budget.max_edges` (default 32)

#### #50 Memory Write Gate

Implementation:
- `MemoryWriteGate` with 5-stage pipeline: empty check → noise patterns → emotional noise → importance → duplicate check → debounce
- `GateDecision` enum: ACCEPT, REJECT, MERGE, DEFER
- Noise patterns: regex for reactions (ok/嗯/ha+), emoji-only, greetings, bot commands
- Emotional noise: high emotion + short text → likely venting
- Importance: length bonus, tag signals, substantive content signals
- Duplicate check: ANN → cosine similarity → duplicate/merge threshold
- Debounce: MD5 hash cache prevents rapid re-writes
- Wired into `runtime.remember()` as pre-write stage (opt-in via config)
- Config section: `write_gate.enabled` (default false)

#### #51 Four-Layer Compression

Implementation:
- `FourLayerCompressor` with 4 layers: MESSAGE (TTL 2h) → EVENT (TTL 7d) → SEMANTIC (TTL 90d) → PERSONALITY (∞)
- `LayerConfig`: max_items, ttl_hours, compression_ratio, promote_stability
- `CompressedMemory` dataclass: id, text, layer, embedding, source_ids, stability, importance, tags
- `ingest_message()`: ingest into Layer 0, auto-compress if full
- `compress_layer0()` → `compress_layer1()` → `compress_layer2()`: BFS cluster synthesis at each level
- `decay_all()`: TTL-based decay across all layers
- `get_for_retrieval()`: searches all layers, personality-first priority
- Wired into `runtime.remember()` (ingest) and `runtime.sleep()` (compress + decay)
- Config section: `four_layer.enabled` (default false)

---

### Phase 14 — Spreading Activation + Thermal Store + Edge Decay (A-Level, v1.4.0) ✅ COMPLETE

- [x] #52 Spreading activation as primary retrieval path
- [x] #53 Hot/Cold/Archive three-tier auto storage
- [x] #54 Continuous edge time decay with adaptive rates

#### #52 Spreading Activation Retrieval

Implementation:
- `_spreading_recall()` method added to `RetrievalPipeline`: uses `SpreadingActivation.activate()` as Path C
- `recall_with_spreading()` method: spreading-first recall with configurable `spreading_weight`
- Spreading results merged into main `recall()` path alongside dimensional descent
- When spreading agrees with other paths, existing items get a +0.05 score boost
- `recall_with_spreading()` combines topology score × `spreading_weight` + embedding score × (1 − weight)

#### #53 Thermal Store (Hot/Cold/Archive)

Implementation:
- `ThermalStore` class with 3-tier management
- `touch(anchor_id)`: records access → triggers promotion (archive→cold, cold→hot)
- `demote_scan(graph)`: scans for idle anchors → hot→cold (72h), cold→archive (720h)
- `thaw_anchor(anchor_id, graph)`: full cold/archive→hot reconstruction
- `load_cold()` / `load_archive()`: transparent data access with auto-touch
- Wired into `runtime.sleep()` as step 6g (demotion scan)

#### #54 Edge Time Decay

Implementation:
- `EdgeDecayManager` class with per-edge decay rate by type
- `apply_decay(edge)`: lazy decay on access — `weight *= exp(-rate × hours_idle)`
- `reinforce(edge)`: strengthens weight + extends `valid_until` by type-specific hours
- `is_viable(edge)`: checks expiration + min weight
- `decay_all_edges(graph)`: bulk decay during sleep, evicts edges below `min_edge_weight`
- Decay rate adapts to success rate: high-success edges decay at half speed
- Wired into `runtime.sleep()` as step 6h (decay all edges)

---

### Phase 15 — B-Level Modules (v1.5.0) ✅ COMPLETE

- [x] #55 Self-Organization: auto-cluster, merge near-duplicates, emergent topic detection
- [x] #56 Personality Model: Big Five traits, working style, expertise, values extraction
- [x] #57 Goal Tree: hierarchical goal decomposition, progress propagation, stale archival

#### #55 Self-Organization

Implementation:
- `SelfOrganization` class: `organize(graph)` orchestrates 3 mechanisms
- Community detection: BFS label propagation, auto-assigns `community_id` to connected anchors
- Emergent topic detection: greedy embedding clustering + TF-IDF keyword labeling → `EmergentTopic`
- Near-duplicate merge: cosine similarity + tag overlap check, rewires edges on merge
- `get_topics()`, `get_topic_anchors()` query API
- Wired into `runtime.sleep()` as step 6i

#### #56 Personality Model

Implementation:
- `PersonalityProfile` dataclass: Big Five (OCEAN) traits, working style, communication preferences, expertise areas, values
- `PersonalityModel` class: `ingest_anchor()` (incremental) + `extract_from_graph()` (full scan)
- Trait extraction via keyword signals in English + Chinese
- Learning style detection: reading/doing/asking
- Value extraction: efficiency, simplicity, reliability, learning, autonomy
- Expertise inference from anchor tags and domain keywords
- Formality computation from formal/informal marker ratio
- Wired into `runtime.remember()` (incremental ingest) and `runtime.sleep()` (full extraction)

#### #57 Goal Tree

Implementation:
- `GoalNode` dataclass: hierarchical parent/child, progress 0-1, priority, confidence, staleness tracking
- `GoalStatus`: ACTIVE → ACHIEVED/ABANDONED/BLOCKED → ARCHIVED
- `GoalTree` class: regex-based goal detection from memory text (EN + ZH patterns)
- Auto-tagging: bugfix, development, learning, deployment, testing, refactoring, setup
- Priority inference from urgency signals (urgent/critical/maybe/someday)
- `mark_progress()` → auto-mark achieved at 1.0, propagates to parent
- `propagate_progress()`: bottom-up sub-goal → parent progress averaging
- `archive_stale()` (168h) + `archive_achieved()` (720h) for cleanup
- Query API: active/blocked/recently_achieved/root/stale goal retrieval
- Wired into `runtime.sleep()` as step 6k (detect + propagate + archive)

---

## v1.6.0 — Next Priorities (User-Specified)

> Based on user's priority list (2026-05-15). Graph-first architecture with retrieval budget control,
> cognitive trajectory tracking, richer edge types, and episodic memory.

### S-Level (immediate)

| # | Item | Status | Description |
|---|------|--------|-------------|
| S-1 | Graph Sparsification | ✅ Done (#49) | Edge budget: max 32 edges/node, smart eviction. Tighten to 16 |
| S-2 | Domain Routing | ✅ Done (#48) | query → domain → cluster → node, no global retrieval |
| S-3 | Memory Write Gate | ✅ Done (#50) | Importance/duplicate/temporal/personality/goal scoring before storage |
| S-4 | Layered Memory Abstraction | ✅ Done (#51) | message → event → semantic → personality → long-term cognition |
| **S-5** | **Retrieval Budget** | **✅ Done** | MAX_HOPS=3, MAX_NODES=24, MAX_TOKENS=6000 to prevent spreading activation runaway |

#### S-5 Retrieval Budget

Prevent spreading activation from exploding:

```
MAX_HOPS = 3       # max BFS depth from seed
MAX_NODES = 24     # max activated nodes per query
MAX_TOKENS = 6000  # max token budget for retrieved content
```

Implementation plan:
- `RetrievalBudget` class enforcing all three limits
- Integrate into `spreading.activate()` to halt BFS at hop limit
- Integrate into `recall()` to truncate results at node + token limits
- Token counting: use tiktoken or character-based estimation
- Config section: `retrieval_budget` with max_hops, max_nodes, max_tokens

### A-Level (next)

| # | Item | Status | Description |
|---|------|--------|-------------|
| A-6 | Spreading Activation | ✅ Done (#52) | Node activation → neighbor diffusion → weight decay → path recall |
| A-7 | Time Decay System | ✅ Done (#54) | weight = semantic × recency × frequency × importance |
| A-8 | Hot/Cold/Archive | ✅ Done (#53) | Three-tier auto-promotion/demotion storage |
| **A-9** | **Versioned Memory** | **✅ Done** | Cognitive trajectory: user_likes_python → user_prefers_ai_dev → user_studies_cognitive_arch |
| **A-10** | **Cluster Memory** | **✅ Done** | Auto-clustering: Python/AI/creative clusters. Community detection + self_org exist, integrated retrieval pre-filtering |

#### A-9 Versioned Memory (Cognitive Trajectory)

Don't create duplicate isolated cognitive nodes. Track evolution:

```
用户喜欢Python → 用户偏向AI开发 → 用户研究认知架构
```

Implementation plan:
- `CognitiveTrajectory` class: linked list of belief states with timestamps
- Auto-detect when a new memory supersedes/refines an existing belief
- `evolve_belief(old_id, new_text)`: creates new version, links to old, marks old as "superseded"
- `get_trajectory(topic)`: returns the full evolution chain
- `get_current_belief(topic)`: returns the latest version only
- Sleep integration: detect belief drift during consolidation

#### A-10 Cluster Memory

Auto-form memory clusters to reduce full-graph search:

Implementation plan (partial — community.py + self_org.py exist):
- Integrate community centroids into retrieval pre-filtering
- `ClusterRouter`: maps query → nearest cluster centroid → search within cluster
- Cluster health monitoring: fragmentation detection, auto-rebalance
- Wired into `retrieval_pipeline.py` as a pre-filter before ANN search

### B-Level (later)

| # | Item | Status | Description |
|---|------|--------|-------------|
| B-11 | Self-Organizing Memory | ✅ Done (#55) | Auto-form preferences, goals, habits, thinking patterns, behavioral trends |
| **B-12** | **Causal Edge Types** | **✅ Done** | leads_to, depends_on, motivates, goal, result — richer than "related" |
| **B-13** | **Episodic Memory** | **✅ Done** | Time + environment + context + state → memory event streams |
| B-14 | Memory Compressor | ✅ Done (#51) | 100 messages → 5 cognition → 1 long-term personality |
| B-15 | Graph Snapshot | ✅ Done | Time-sliced cognitive state (snapshot.py) |

#### B-12 Richer Causal Edge Types

Current edge types are good but need causal depth:

```
Not just "related" → CAUSES, DEPENDS_ON, MOTIVATES, GOAL_OF, RESULT_OF, PRECEDES, CONTRADICTS
```

Implementation plan:
- Extend `EXPLICABLE_RELATIONS` with causal subtypes
- `CausalChain.infer_type()`: heuristic type inference from text patterns
- Causal graph traversal: only follow causal edges for reasoning queries
- Visualization: causal chain export for debugging

#### B-13 Episodic Memory

Add temporal + contextual richness to memory:

```
Memory event stream: {time, environment, context, state, action, outcome}
```

Implementation plan:
- `EpisodeNode` dataclass: timestamp, session_id, context_snapshot, emotional_arc, participants
- `EpisodeStream`: time-ordered linked list of episodes within a session
- `contextual_recall(query, time_range, context_filter)`: time-scoped + context-filtered retrieval
- Auto-summarization: N episodes → 1 session summary
- `timespine.py` already has time-indexing; extend with context dimensions

### Architecture Direction

**Graph-first, Vector-assisted** (not vector-first):

- Vector: semantic supplement, fuzzy recall, initial filtering only
- Graph: the real core — structure, relationships, trajectories, causality
- Avoid: full-graph diffusion (use budget), permanent memory (must forget), message-level long-term storage (must abstract), super-nodes (limit connections)
