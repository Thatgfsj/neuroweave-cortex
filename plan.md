# Star Graph Memory — Architecture Plan v0.6

## Current State (v0.5)

已完成：
- 6-state memory lifecycle (ACTIVE → GHOST → REACTIVATED)
- Multiplicative retention (geometric mean of 5 factors)
- OscillationResonance + HybridFusion retrieval (C-R@3 = 0.967)
- 5-phase sleep consolidation (N1→Wake-prep)
- Memory evolution (decay, boost, conflict, interference)
- Ghost subsystem (latent traces, fuzzy recall)
- Abstraction engine (emergent categories)
- Working memory (9-item buffer, 30min TTL, auto-promotion)
- Reflection nodes (failure_analysis, success_pattern, etc.)
- RichEdge with temporal/causal/state-transition fields
- Personalized PageRank + multi-hop traversal

结构性问题（待解决）：
- **平铺图**：所有 anchor 在单一 StarGraph 中，O(n²) 连接
- **全图暴力搜索**：retrieval 扫描全部 anchor，无分区
- **无时间脊柱**：时间只是 anchor 字段，不是独立索引结构
- **无因果链回忆**：有 causal edge 但未用于链式遍历
- **无 Cortex 分区**：所有记忆在一个空间，无法按领域隔离

## Architecture Vision (v0.6)

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryManager (facade)                    │
├─────────────────────────────────────────────────────────────┤
│  CortexRouter    │  routes query → 1-3 cortices              │
├─────────────────────────────────────────────────────────────┤
│  MemoryCortex    │  independent graph + index per domain     │
│  ├─ DevCortex    │  code, architecture, debugging            │
│  ├─ FinanceCortex│  money, costs, budgets                    │
│  └─ PersonalCtx  │  preferences, identity, relationships     │
├─────────────────────────────────────────────────────────────┤
│  MemoryGate      │  winner-take-all from candidates          │
├─────────────────────────────────────────────────────────────┤
│  TimeSpine       │  temporal index: day → cluster[]          │
├─────────────────────────────────────────────────────────────┤
│  CascadeRecall   │  causal chain traversal                   │
├─────────────────────────────────────────────────────────────┤
│  HubLayer        │  cross-cortex abstraction bridges         │
└─────────────────────────────────────────────────────────────┘
```

## Design Principle: Dimensional Reduction Retrieval (降维寻找)

当高维匹配失败时，逐级降维兜底，而不是直接返回空：

```
Level 1: 3D Semantic Space (embedding full-vector cosine)
    ↓ 匹配不足 k 条？
Level 2: 2D Plane (time × importance projection)
    ↓ 仍然不足？
Level 3: Pseudo-2D Timeline (TimeSpine 时间脊柱扫描)
    → 保证总能返回结果
```

### Level 1: 3D Semantic Space
- 在当前 Cortex 的 ANN 索引中搜索
- 使用 OscillationResonance 或 HybridFusion
- 分数阈值: min_similarity = 0.5
- 目标: 找到 top-k 语义相关记忆

### Level 2: 2D Plane (Time × Importance)
- 将记忆投影到 (时间, 重要度) 二维平面
- 不再依赖语义匹配，只看 "最近的 + 重要的"
- 查询逻辑: "最近一周内，重要度 > 0.5 的记忆"
- 按 (recency DESC, importance DESC) 排序
- 这是从 "语义相关" 到 "时间+重要" 的降级

### Level 3: Pseudo-2D Timeline
- 使用 TimeSpine 结构按时间倒序扫描
- 每个时间点取 top-N 个最重要的 memory cluster
- 伪二维: Y轴=时间(倒序), X轴=重要度(降序)
- "右上到左下" 滑动窗口扫描
- 保证即使语义和时间都匹配不到，也能回溯出最近的记忆

## Pseudo-2D "右上→左下" 扫描

```
重要度 ↑
       │  ┌─ 今天 (high importance)
       │  │  ┌─ 昨天 (high)
       │  │  │  ┌─ 前天 (high)
       │  │  │  │
       │  │  │  │     ┌─ 上周 (medium)
       │  │  │  │     │
       │  │  │  │     │     ┌─ 上月 (low)
       └──┴──┴──┴─────┴─────┴──────────→ 时间 (倒序)
    
    扫描方向: 右上角 → 左下角
    = 最近且重要的 → 最近但不太重要的 → 久远但重要的 → 久远且不重要的
```

每个时间单位（天）内按重要度降序排列，横向宽度受 `max_clusters_per_day` 限制。

## 4D Hub Nodes (四维中枢节点)

中枢节点是跨 Cortex 的桥梁，不是超级节点：

```
                    Global Self Hub
                    /              \
            Dev Identity Hub    Personal Identity Hub
            /          \           /           \
    Python Hub    Docker Hub   Preferences   Relationships
       /  \         /  \         /  \          /  \
    [memories in DevCortex]    [memories in PersonalCortex]
```

### 中枢节点属性
- 不存储原始记忆，只存储摘要 + 指针
- 属于特定的抽象层 (leaf / domain / global)
- 有独立的 embedding (摘要的 embedding，不是原始记忆的)
- 可以跨 Cortex 引用（这是唯一允许跨 Cortex 的边类型）
- 稳定性极高 (stability > 0.9)，几乎不衰减

### 四维空间定位
- (X, Y, Z) = 语义 + 时间 + 重要度（三维星群）
- 第四维 = 所属脑区 (Cortex Layer)
- 中枢节点在第四维上连接不同脑区的三维星群

---

## Implementation Tasks (ordered by priority)

### P0: Cortex 分区系统 + 路由

**#46** `cortex.py` — `MemoryCortex` 类
- 独立 StarGraph + ANNIndex + embedding provider
- 独立遗忘曲线参数 (decay_half_life, retention_threshold)
- 独立 token budget
- `route(query_emb) -> float`: 判断 query 是否属于此脑区
- `recall(query, context) -> MemoryContext`: 局部召回
- `consolidate() -> SleepReport`: 独立睡眠

**#47** `router.py` — `CortexRouter` 类
- 管理所有 Cortex 实例
- `route(query) -> list[tuple[MemoryCortex, float]]`: 返回 top-3 脑区
- 路由逻辑: 语义匹配 + 标签匹配 + 最近使用
- 默认 cortex (fallback): 当所有 cortex 都不匹配时使用
- 动态创建 cortex: 当检测到新领域时自动创建

### P0: 激活门控

**#48** `gate.py` — `MemoryGate` 类
- `gate(candidates, context, k) -> list[Anchor]`: winner-take-all
- 多维竞争分数: importance, recency, emotional_valence, semantic_match, causal_relevance, user_focus
- 侧向抑制: 相似记忆互相压制
- 保证输出数量固定 (不随总记忆量增长)

### P1: 降维检索 + 伪二维扫描

**#49** 在 `scheduler.py` 中实现三级降维检索
- `_retrieve_3d_semantic()`: 现有语义搜索 (Level 1)
- `_retrieve_2d_plane()`: 时间×重要度 投影 (Level 2)
- `_retrieve_timeline()`: TimeSpine 扫描 (Level 3)
- `_dimensional_reduction_retrieve()`: 串联三级，自动降级

### P1: 时间脊柱

**#50** `timespine.py` — `TimeSpine` 类
- 按天/小时分桶的层级时间结构
- 每个时间桶挂载 Memory Cluster
- `max_clusters_per_unit`: 限制横向宽度
- `scan(start_time, end_time, direction)`: 时间窗口扫描
- "右上→左下" 扫描器: `scan_priority(recent_first=True, importance_desc=True)`
- 自动聚合: 同主题同日记忆合并为一个 cluster

### P1: 因果链回忆

**#51** `cascade.py` — `CascadeRecall` 类
- 从 query 种子节点沿因果边 (caused_by, derived_from) 遍历
- 不是随机散步，而是有方向的因果链追溯
- `cascade(query, max_depth=5) -> list[CausalChain]`
- 每条 CausalChain 是 (cause → effect → effect → ...) 的序列
- 支持向前追溯 (what caused this?) 和向后推理 (what did this cause?)

### P1: 中枢抽象层

**#52** `hub.py` — `HubLayer` + `HubNode` 类
- `HubNode`: 摘要节点，包含指向 source anchors 的指针
- `HubLayer`: 管理多级中枢 (leaf → domain → global)
- `abstract_to_hub(cluster) -> HubNode`: 从 cluster 提取中枢节点
- 跨 Cortex 边只能通过中枢节点
- 中枢节点不参与遗忘 (stability = 0.95+)

### P2: 热管理生命周期

**#53** 在 `anchor.py` 中扩展 MemoryState
- HOT (高频激活): 对应 ACTIVE + REHEARSING
- WARM (低频但重要): 对应 CONSOLIDATING + DORMANT
- COLD (冷冻仅索引): 对应 DORMANT + 低 retention
- DEAD (仅 metadata/hash): 对应 GHOST 的最终状态
- 自动降级: HOT → WARM → COLD → DEAD
- 复苏机制: COLD 可以被 GHOST revival 机制重新激活

### P2: 动态人格记忆

**#54** 自适应遗忘曲线 + 元认知层
- 每个 Cortex 独立学习最优遗忘参数
- 元认知: AI 知道自己记得什么、忘记了什么
- `know_what_i_know()`: 返回知识覆盖图
- `confidence_calibration()`: 校准记忆置信度

---

## Existing Modules to Modify

| Module | Changes |
|--------|---------|
| `manager.py` | 初始化 CortexRouter, TimeSpine, MemoryGate; 新 API: `add_cortex()`, `route_to_cortex()` |
| `scheduler.py` | 三阶段路由+门控 pipeline; 三级降维检索; 废除全图扫描 |
| `graph.py` | 添加 HubNode; 导出 ReflectionNode |
| `anchor.py` | 扩展 MemoryState 支持 HOT/WARM/COLD/DEAD; gate_score 属性 |
| `__init__.py` | 导出所有新模块 |
| `defaults.yaml` | cortex, routing, gate, timespine, cascade, hub 配置段 |

---

## File Layout (new modules only)

```
star_graph/
├── cortex.py          # NEW — MemoryCortex
├── router.py          # NEW — CortexRouter
├── gate.py            # NEW — MemoryGate
├── timespine.py       # NEW — TimeSpine, TimeBucket, MemoryCluster
├── cascade.py         # NEW — CascadeRecall, CausalChain
├── hub.py             # NEW — HubLayer, HubNode
├── ... (existing 28 files unchanged)
```

---

# v0.7: LoCoMo Benchmark Optimization Plan

## Problem Diagnosis

LoCoMo scores are low NOT due to design philosophy failure, but because:
- High-level cognitive capabilities aren't triggered in pure retrieval eval
- Basic retrieval mechanisms expose critical gaps

| Category | Current Score | Root Cause |
|----------|--------------|------------|
| Cat 1 (临时) | ~4.3% | No raw short-term channel — 1:1 compression drops immediate facts |
| Cat 2 (短时) | ~1.9% | Same as above — no uncompressed recent-session buffer |
| Cat 3-5 (综合) | Low | Pure embedding search, no goal-directed structural traversal |

---

## P0: Multi-Level Memory Architecture (Raw Chunk + Working Memory)

**Problem**: 1:1 compression irreversibly discards short-term facts. No uncompressed short-term channel.

**Solution**: Three-tier memory hierarchy:

| Tier | Retention | Content | Retrieval Priority |
|------|-----------|---------|-------------------|
| Raw Chunk Buffer | Last 1-2 sessions | Uncompressed dialogue segments | Highest |
| Mid-level Anchors | Compressed + rehearsed | Structured memory anchors | Medium |
| Long-term Graph | Persistent + decayed | Stable abstract knowledge | Normal |

**Implementation**:
- `raw_buffer.py` — Raw Chunk Buffer storing original dialogue segments per session
- No compression applied; TTL = 2 sessions max
- `MemoryManager` queries raw buffer first before graph retrieval
- Inspired by MemGPT's multi-level OS-style memory design

**Expected**: Cat 1 + 2 from 2-4% → 50-60%+

---

## P0: Dual-Write + Multi-Path Retrieval

**Problem**: 1:1 compression is irreversible and lossy. Once a fact is abstracted away, it's permanently lost.

**Solution**: Parallel write + concurrent recall:

```
Each dialogue turn:
  ├─ Write Path A: Raw chunk → RawBuffer (uncompressed, full text)
  └─ Write Path B: Anchor → Graph (compressed, structured)

Retrieval:
  ├─ Path A: BM25 + vector similarity on raw chunks (50%)
  └─ Path B: Star graph resonance mechanism (50%)
  └─ Merge/Re-rank → Final Top-K
```

**Reference**: Zep (78.94%), EverMemOS (92.3%) both use graph + chunk dual-path.

**Expected**: has_answer doubles, short-term +20-30 percentage points

---

## P1: System-1 + System-2 Dual-Channel Retrieval

**Problem**: Pure similarity-based retrieval can't handle "goal-directed structural traversal" — can only find "semantically similar", can't do "all/which/before/last" structured queries.

**Solution**: Inspired by Mnemis (93.9% on LoCoMo):

| Channel | When | Mechanism |
|---------|------|-----------|
| System-1 (Fast Association) | Default, high-confidence queries | Existing Star graph similarity search |
| System-2 (Goal-Directed) | Low confidence (< threshold) OR structural/exhaustive intent | Hierarchical graph top-down traversal |

**Implementation**:
- `dual_channel.py` — `DualChannelRetriever` combining both channels
- System-2 builds hierarchical semantic summary graph (summary → detail index)
- Triggers when: query contains "all/which/before/last", or System-1 confidence < 0.35

**Expected**: Multi-hop/composite question scores +15-20 percentage points

---

## P2: LLM-Assisted Memory Abstraction

**Problem**: Pure algorithmic abstraction (1:1 anchors) can't extract deep semantic abstractions like user preferences or cross-session relationships from noisy dialogue.

**Solution**: Lightweight LLM post-processing in sleep consolidation:

1. Pre-filter: Existing Star algorithm does initial anchor screening
2. LLM post-process: Small model (GPT-4o-mini / Qwen-2.5 3B) converts anchors into entity/event-centric Atom Facts
3. Hierarchical merge: Atom facts stored in new graph layer, hyperlinked to source anchors

**Reference**: Synthius-Mem (94.37%) philosophy — "Don't retrieve what was said, extract what is known about the user/world"

**Expected**: Adversarial QA (Cat 5) + composite fact scenarios → 50-70%

---

## Implementation Priority

| Priority | Item | Expected Gain | Complexity |
|----------|------|--------------|------------|
| P0 (now) | Raw Chunk Buffer + Multi-Path Retrieval | Cat 1+2 2-4%→50%+, has_answer 2x | Low-Med |
| P1 (next) | System-1 + System-2 Dual Channel | Cross-session reasoning foundation | High |
| P2 (later) | LLM-Assisted Abstraction | Long-term reasoning enhancement | High |

---

## Verification Flow

1. Implement raw buffer + raw chunks + multi-path recall baseline
2. Re-run LoCoMo eval on Cat 1 + 2 (has_answer metric)
3. If short-term improvement significant → push System-2 + LLM abstraction forward

---

# v0.8 — Short-Term Precision + Production Readiness

## Current Weakness

Single-token retrieval accuracy ~2% (near random). Sleep consolidation
heavy on LLM API calls. No persistence safety. No standard benchmark.

---

## 1. 修复短时精确记忆 (P0)

### 1a. KV 精确匹配旁路

对于强关联实体对（人名-生日、地名-坐标），使用键值缓存表做确定性查找，
不走模糊联想路径。

```
检索流程:
  query → 提取实体 key → KV 缓存精确查找
    ├── 命中 → 直接返回 (确定性，O(1))
    └── 未命中 → System-1/2 模糊检索 → 降级结果
```

实施:
- `exact_cache.py` — 实体对 KV 缓存表
- 每个 anchor 增加 `exact_match_keys: list[str]` 字段
- `remember()` 时自动提取实体对写入 KV 缓存
- `recall()` 时优先精确匹配，再降级到混合检索

### 1b. Salience 字段

为每个记忆节点增加显著性标记:
- `Anchor.exact_match_key: str = ""` — 精确匹配键（如 "Alice-birthday"）
- `Anchor.salience: float = 0.0` — 显著性评分（高 = 更容易被精确回忆）
- 检索时先尝试精确匹配 `exact_match_key`，命中则跳过模糊检索

### 1c. 扩展工作记忆栈

现有 WorkingMemory (9 条, 30min TTL) 扩展为:
- 容量 10-20 条
- 专门缓存最近/最频繁被精确回忆的内容
- 独立的 exact match 索引
- 与 raw_buffer 互补：WM 存最热数据，raw_buffer 存最近 2 会话

---

## 2. 降低推理调用开销 (P0)

### 2a. 增量式微睡眠

现状: `sleep()` 一次性运行完整 8 阶段，阻塞严重。

改为:
```
micro_sleep(steps: int = 2):
  每次 Agent 空闲时执行 1-2 个阶段
  记录已完成的阶段，下次从断点继续
  8 个阶段分 4-8 次微睡眠完成一轮完整周期
```

新增:
- `micro_sleep.py` — `MicroSleepScheduler` 管理阶段进度
- `MemoryManager.micro_sleep()` — 增量入口
- `SleepCycle` 增加 `resume_from(phase_idx)` 恢复能力

### 2b. 成本估算器

在运行 sleep 前输出预估:
- 当前锚点数量 → 预估 SWR 重放耗时
- 摘要节点数量 → 预估 LLM 调用次数和 token 消耗
- 总体预估成本（美元）和耗时
- 支持 dry-run 模式（只估算不执行）

新增:
- `cost_estimator.py` — `SleepCostEstimator`
- `MemoryManager.estimate_sleep_cost()` → 返回估算报告

### 2c. 轻量化后端（已完成）

- `atom_facts.py` 已支持 `provider: template` 零 API 模式
- `compression.py` 已用模板引擎替代 LLM 压缩（零 API）
- 默认所有组件使用离线模式，LLM 为可选增强

---

## 3. 补齐生产环境组件 (P1)

### 3a. 快照 + WAL

- `snapshot.py` — 定时快照 (JSON/SQLite dump)
- WAL 集成: 利用 SQLite 已有的 WAL 模式，增加应用层 snapshot 检查点
- 中断恢复: `MemoryManager.load()` 自动检测未完成的操作并回滚
- 状态回滚: 保留最近 N 个快照，支持手动回退

### 3b. 异步接口

- `async_manager.py` — `AsyncMemoryManager` (asyncio 封装)
- 连接池: 多 Agent 共享同一个 MemoryManager 实例
- 并发安全: 图操作加读写锁 `threading.RWLock`

### 3c. OpenTelemetry 追踪

- `tracing.py` — OpenTelemetry span 包装
- 记录每次检索:
  - 查询文本、响应锚点 ID
  - 相位值、共振强度、PageRank 得分
  - 经过的检索层 (raw_buffer / community / cortex / hub / timeline)
  - 耗时细分
- 便于调试"为什么这条记忆被召回"

---

## 4. 标准评估基准 (P1)

### 基准套件

| 测试类别 | 内容 | 指标 |
|---------|------|------|
| 精确事实回忆 | LoRA 风格评估集 | exact_match, has_answer |
| 联想迁移 | A→B→C 路径推理 | path_recall@k |
| 时间混淆 | 新旧记忆干扰测试 | temporal_precision |
| 抗噪能力 | 大量幽灵痕迹下的检索精度 | precision@k under noise |
| 压缩保真度 | 压缩前后 recall 对比 | recall_drop_ratio |

### 对比实验

| 系统 | 对比维度 |
|------|---------|
| FAISS + Neo4j | 纯向量 + 图 vs Star 的认知图 |
| MemGPT | OS 风格多级内存 vs Star 的认知架构 |
| Zep | 图+块双路径 vs Star 的多路召回 |
| Mnemis | 分层图检索 vs Star 的双系统路由 |

---

# v1.0 — 长期演进

## 5. 可配置的模糊回忆策略

- 开放记忆生存函数接口 `SurvivalFunction` 协议
- 内置: 艾宾浩斯曲线、幂律衰减、指数衰减、自定义 lambda
- 幽灵强度作为检索排序维度
- 负向幽灵: 降低被矛盾/证伪记忆的置信度

## 6. 多模态记忆

- CLIP 风格多模态嵌入 (文本/图像/音频 → 统一向量空间)
- 跨模态幽灵复活: 图像残余信号触发文本回忆

## 7. 多 Agent 共享记忆

- 记忆联邦层: 多 Agent 共享部分记忆子图
- 全局幽灵共鸣: 跨 Agent 推荐/警告/协作
- 记忆权限: 读/写/广播 ACL

## 8. WASM 嵌入式版本

- 核心图操作 + 相位共振编译为 Rust/WebAssembly
- micro-star: 仅保留动态遗忘 + 混合检索，适配 IoT/边缘设备

---

## v0.8 Implementation Priority

| Priority | Item | Impact | Complexity |
|----------|------|--------|------------|
| P0-1a | KV 精确匹配缓存 | 单 token 2%→60%+ | 低 |
| P0-1b | Salience + exact_match_key | 实体对确定性查找 | 低 |
| P0-1c | 扩展工作记忆栈 | 热数据命中率提升 | 低 |
| P0-2a | 增量微睡眠 | 消除长时间阻塞 | 中 |
| P0-2b | 成本估算器 | 运行前可预测 | 低 |
| P0-2c | 轻量化后端（已完成） | 零 API 成本 | ✓ |
| P1-3a | 快照 + WAL | 数据安全 | 中 |
| P1-3b | 异步接口 | 高并发支持 | 中 |
| P1-3c | OpenTelemetry | 可观测性 | 低 |
| P1-4 | 标准基准套件 | 量化对比 | 中 |
| v1.0-5 | 可配置遗忘曲线 | 学术创新 | 中 |
| v1.0-6 | 多模态 | 能力扩展 | 高 |
| v1.0-7 | 多 Agent 联邦 | 架构创新 | 高 |
| v1.0-8 | WASM 嵌入 | 新平台 | 高 |
