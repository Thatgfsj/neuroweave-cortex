# NeuroWeave Cortex (NWC) — Plan

> 最后更新: 2026-05-22 | **v1.1.0** | 6-layer cognitive memory architecture

---

## 定位: Cognitive Character — 类人认知引擎

### 不是

- 向量记忆库
- 对话存档器
- Agent 插件
- MCP 附件
- RAG 系统
- 更快的数据库

### 而是

> NeuroWeave Cortex 不是一个 memory layer，而是一个 **cognitive character**。
> 它的灵魂不是"更快更准"，而是"会遗忘、会扭曲、有情绪、有社交判断、能自由联想的人工心智"。

核心理念:

1. **记忆不平等** — 早餐吃了什么 ≠ 长期目标 ≠ 废话，必须分级存储
2. **记忆会成长** — 从数据 → 模式 → 信念 → 人格，不是静止的
3. **类人记忆不是缺陷，而是特征** — 失真、遗忘、情绪抑制、自由联想，这些"不完美"正是差异化壁垒
4. **认知压缩是未来** — 海量对话 → 抽象 → 压缩 → 人格 → 长期认知
5. **AI 不是"读取记忆"，而是"基于记忆思考"**
6. **工程基础是地基，不是房子** — 地基要稳，但最终交付的是类人认知

---

## 六层认知记忆系统

从数据到人格的演化路径，不是 input → embedding → retrieve:

```
┌──────────────────────────────┐
│ ⑥ Identity Layer  身份/人格层 │  "我是谁，用户是什么样的人"
│    cognitive_identity.py     │  跨会话自我认知 + 用户认知画像
└──────────────────────────────┘
         ↑ personality_formation.py
┌──────────────────────────────┐
│ ⑤ Belief Layer    信念/价值观 │  "用户长期信仰什么、偏好什么"
│    belief_system.py          │  稳定信念、价值偏向、行为准则
└──────────────────────────────┘
         ↑ pattern recognition
┌──────────────────────────────┐
│ ④ Pattern Layer   行为模式   │  "用户反复做什么、习惯是什么"
│    cognitive_compression.py  │  行为模式、习惯、反应倾向
└──────────────────────────────┘
         ↑ compression / abstraction
┌──────────────────────────────┐
│ ③ Semantic Layer  知识/事实   │  "用户知道什么、世界是什么样"
│    concept_cortex.py         │  长期知识、事实、概念网络
│    abstraction.py            │
└──────────────────────────────┘
         ↑ consolidation
┌──────────────────────────────┐
│ ② Episodic Layer  事件/经历   │  "发生了什么、用户经历了什么"
│    anchor.py                 │  事件记忆、对话片段、工具调用
│    working_memory.py         │
└──────────────────────────────┘
         ↑ parse / evaluate
┌──────────────────────────────┐
│ ① Sensory Layer  感知/输入    │  "用户说了什么"
│    perception.py             │  原始输入 → 结构化 PerceptionFrame
│    importance_engine.py      │  importance gate: 什么值得记
└──────────────────────────────┘
```

关键: **不是所有输入都进入记忆**。Sensory Layer 的 importance_engine 是第一道闸门 — 低价值输入直接丢弃，不进入 Episodic。

---

## 记忆生命周期 (8 阶段)

```
Capture → Parse → Evaluate → Link → Reflect → Compress → Decay → Reinforce
```

| 阶段 | 模块 | 说明 |
|------|------|------|
| ① Capture | `perception.py` | 捕获原始输入 |
| ② Parse | `perception.py` | 解析: 情绪、意图、主题、实体 |
| ③ Evaluate | `importance_engine.py` | 重要度分析: 什么值得记 |
| ④ Link | `anchor.py` + `graph.py` | 建立关系: 因果、时序、语义 |
| ⑤ Reflect | `cognitive_compression.py` | 高层抽象: 事件 → 模式 |
| ⑥ Compress | `cognitive_compression.py` | 认知压缩: 模式 → 信念 → 人格 |
| ⑦ Decay | `evolution.py` + `survival.py` | 遗忘机制: 生存衰减 |
| ⑧ Reinforce | `hebbian_learning.py` | 重复强化: 一起放电的神经元连接在一起 |

---

## 重要性引擎 (Importance Engine)

### 核心公式

```
importance = emotion_weight × w1
           + repetition_weight × w2
           + goal_relation × w3
           + novelty × w4
           + future_probability × w5
```

| 输入 | 权重 | 示例 |
|------|------|------|
| emotion_weight | 0.25 | 高情绪 ≠ 高价值，但愤怒/兴奋表示"用户在乎" |
| repetition_weight | 0.30 | 用户连续5次讨论同一主题 → 这是核心关注点 |
| goal_relation | 0.20 | 与当前活跃目标相关 → 优先存储 |
| novelty | 0.15 | 新话题 vs 已存储知识的重叠度 |
| future_probability | 0.10 | 预测未来被召回的概率 |

### 分级结果

| 范围 | 层级 | 处理 |
|------|------|------|
| 0.0–0.2 | 噪声 | 直接丢弃，不进入任何存储 |
| 0.2–0.5 | 低价值 | 进入 Episodic，快速衰减，30天内可 ghost |
| 0.5–0.8 | 有价值 | 正常存储，参与 consolidation |
| 0.8–1.0 | 核心 | 强制提升至 Semantic/Pattern，低衰减率 |

---

## 信念系统 (Belief System)

### 信念 vs 事实

```
事实: "用户使用 Python" → 可变的、具体的
信念: "用户偏好系统级技术、倾向高自由度架构、反感模板化" → 稳定的、抽象的
```

### 信念生命周期

1. **形成**: 从 Pattern Layer 检测到的重复行为模式
2. **强化**: 每次新证据确认 → belief_strength++
3. **挑战**: 矛盾证据 → belief_strength--，如果降至 0 则重构
4. **演化**: 信念可以合并、分裂、更新

### 信念存储

```json
{
  "id": "belief_001",
  "statement": "用户偏好系统级技术而非应用层开发",
  "strength": 0.87,
  "evidence": ["记忆ID-1", "记忆ID-2", ...],
  "contradictions": [],
  "formed_at": "2026-05-01",
  "last_reinforced": "2026-05-22",
  "stability": 0.92
}
```

---

## 人格形成 (Personality Formation)

### 从记忆到人格的路径

```
海量对话
  ↓ cognitive_compression
事件模式
  ↓ belief_system
稳定信念
  ↓ personality_formation
认知画像
  ↓ cognitive_identity
Persistent Cognitive Identity
```

### 用户认知画像

不是"记住用户喜欢 Python"，而是形成:

```json
{
  "cognitive_style": {
    "abstraction_level": "high",
    "decision_basis": "systematic",
    "learning_style": "depth_first"
  },
  "value_system": {
    "efficiency_over_convention": 0.9,
    "autonomy_over_guidance": 0.85,
    "novelty_over_stability": 0.6
  },
  "behavioral_patterns": {
    "prefers_bottom_up_design": 0.88,
    "code_first_document_later": 0.75,
    "validates_before_shipping": 0.92
  },
  "identity_markers": [
    "系统架构师思维",
    "AI/认知系统深度研究者",
    "独立开发者心态"
  ],
  "evolution_trajectory": {
    "trending_toward": "deeper_abstraction",
    "emerging_interests": ["cognitive_architecture"],
    "fading_interests": ["basic_web_dev"]
  }
}
```

### Persistent Cognitive Identity (跨会话)

这是 NWC 的终极差异化:
- AI 长期认识用户
- 形成认知而非存储数据
- 记住成长轨迹
- 持续演化

---

## 认知压缩 (Cognitive Compression)

### 真正核心

```
100万 token 对话
  ↓ 无法全量进入 LLM context
压缩
  ↓ 
- 用户长期目标 (100 tokens)
- 用户认知模式 (200 tokens)
- 用户价值偏向 (100 tokens)
- 用户行为特征 (200 tokens)
  ↓
注入 LLM: ~600 tokens of compressed cognition
```

### 压缩级别

| 级别 | 输入 | 输出 | 触发条件 |
|------|------|------|----------|
| L1 Episodic | 50+ 事件 | 1 情景摘要 | 每次 sleep |
| L2 Pattern | 20+ 摘要 | 1 行为模式 | 每 5 次 sleep |
| L3 Belief | 10+ 模式 | 1 信念更新 | 信念变化 > 阈值 |
| L4 Identity | 全局 | 人格刷新 | 每 50 次 sleep 或重大模式变化 |

---

## 检索公式

不是纯 cosine，而是:

```
final_score =
    semantic_similarity × 0.35
  + temporal_weight × 0.20
  + importance × 0.25
  + relationship_strength × 0.10
  + recall_frequency × 0.10
```

---

## 工程现状与重建

---

### 务实的前提

以上定位和架构是 NWC 的长期愿景。但在追求这个愿景之前，必须正视当前工程基础的差距。下文基于外部代码审查和基准数据分析，是当前版本（v1.1.0）的真实状态评估和重建计划。

### 1.0 当前代码库状态

- **134 modules** (112 star_graph + 20 nwc + 2 new: observatory + gravity_well)
- v1.2.12 — 观测者依赖的检索范式
- **新核心**: `observatory.py` (天文台亮度引擎) + `gravity_well.py` (引力井惯性)
- **类人记忆增强**: source_attribution, event-anchored timeline, temporal query detection

#### Layer 完成度

| Layer | 状态 | 核心模块 |
|-------|------|----------|
| ① Sensory | ✅ 完成 | `perception.py`, `importance_engine.py` |
| ② Episodic | ✅ 完成 | `anchor.py`, `working_memory.py`, `graph.py` |
| ③ Semantic | ✅ 完成 | `concept_cortex.py`, `abstraction.py`, `community.py` |
| ④ Pattern | △ 部分 | `cognitive_compression.py` (缺 behavior pattern extraction) |
| ⑤ Belief | ❌ 未实现 | `belief_system.py` (待实现) |
| ⑥ Identity | △ 部分 | `self_model.py` + `nwc/core/cortex.py` (缺 persistent identity) |

---

## 一、诚实现状评估

### 1.1 检索性能：远未达到可用阈值

| 指标 | VectorSimilarity | OscillationResonance | HybridFusion | 行业参考(DPR/FAISS) |
|------|:---:|:---:|:---:|:---:|
| LoCoMo has_answer | 25.3% | 24.8% | **31.1%** | 40-60% |
| LoCoMo F1 | 0.016 | 0.016 | **0.017** | 0.2-0.4 |
| LLM Judge Score | 0.025 | 0.075 | 0.05 | — |
| recall@1 | 0.0 | 0.0 | 0.0 | — |

**结论**: 当前检索系统在长对话记忆召回上基本不可用。31% has_answer 意味着约 70% 的查询完全找不到答案。三种检索策略的 F1 都低于 0.02，说明即使命中也匹配质量极差。

### 1.2 存储层：生产断层（已部分修复）

- **默认存储**: ✅ 已改为 SQLite (`~/.nwc/memory.db`)，`save()`/`load()` 自动检测文件后缀
- **JSON 存储**: 保留为兼容选项，`export_json()` 方法可用
- **StorageBackend 接口**: ✅ 已完成抽象层（`batch_save`/`transaction`/`search`/`detect_and_create`）
- **并发安全**: ✅ StarGraph 已加 `threading.RLock`，`add_anchor`/`add_edge`/`remove_anchor` 线程安全
- **未完成**: 无 Qdrant/Milvus 等专用向量数据库后端；无水平扩展方案

### 1.3 架构：认知复杂度与工程价值的倒挂

```
112+ 模块
4 层架构
9 阶段生命周期（PERCEPTION → WORKING → ... → GHOST → DEAD）
8 阶段睡眠（N1-N3, REM, N4-N6）
300+ 配置参数
```

大量"生物学启发"功能（Ghost Revival、REM Emotion、Hebbian Learning、Oscillation Resonance）**缺乏消融实验证明其实际增益**。

### 1.4 生态隔离

- ❌ 无 LangChain `BaseChatMemory` 适配器
- ❌ 无 LlamaIndex/Haystack 集成
- ❌ 无 OpenAI Agents SDK / AutoGen / CrewAI 适配器
- ✅ 仅有 MCP Server（14 tools）—— 这是唯一的外部集成通道

### 1.5 评估体系不完整

- ❌ 无端到端任务基准（多轮对话完成率、工具调用准确率提升）
- ❌ 无消融实验（Ablation Study）
- ❌ 无延迟/吞吐基准（QPS、内存占用随数据增长曲线）
- ❌ 无公平对比（Mem0、Zep、MemGPT、纯 VectorDB）

### 1.6 可观测性缺失

- ❌ REST API 无认证/授权/速率限制
- ❌ 无结构化日志（大量 `print()` + `self.log: list[str]`）
- ❌ 无分布式追踪
- ❌ 无备份/快照/迁移工具
- ⚠️ 有 OpenTelemetry tracing 基础设施但关键路径未埋点

### 1.7 一句话总结痛点

> **用生物学复杂性掩盖了工程基础薄弱**。在核心检索不可用、存储不安全的土壤上搭建认知层，是本末倒置。

---

## 二、核心定位：工程基础×类人认知

### 两个轮子，缺一不可

NeuroWeave Cortex 的竞争壁垒不是"检索速度比 FAISS 快"，也不是"模块数比 Mem0 多"。它真正的差异化是：

1. **类人记忆** — 会遗忘、会扭曲、有情绪、有社交判断、能自由联想
2. **多维度记忆** — 从感知到身份的多层演化，维度间的动态竞争与融合

### 工程基础是轮子，不是方向

```
工程基础（地基）→ 支撑 → 类人认知（房子）
     ↓                        ↓
 可部署、可用              差异化、不可替代
```

- 之前（我的错误）：`地基打好再建房子` → 无限期拖延房子的建设
- 现在（正确路线）：`地基和房子同步建` → 每次迭代同时推进工程和认知

### 已完成的工程基础（不重复做）

| 项目 | 状态 |
|------|------|
| 存储后端接口抽象 | ✅ done |
| SQLite 默认后端 | ✅ done |
| 并发锁 (RLock) | ✅ done |
| Raw Buffer 优先级提升 | ✅ done |
| 模块分类清单 | ✅ done |
| plan 方向调整 | ✅ done |
| 天文台亮度引擎 (observatory.py) | ✅ done |
| 引力井惯性 (gravity_well.py) | ✅ done |
| 社交来源归因 + 信任度 | ✅ done |
| 事件锚定时间线 | ✅ done |
| 时间感知检索 | ✅ done |
| 近期记忆权重 | ✅ done |
| 延迟分解埋点 | ✅ done |

### 下一阶段的核心：类人记忆增强

工程基础继续加固，但**同时**投入类人记忆特征的实现。

---

## 三、研究路线图（Research Roadmap v2.0）

### 核心目标

> 将 NWC 从工程项目发展为可发表在 ESWA/KBS/Information Sciences 的认知记忆架构。

**目标论文**: A Lifecycle-Aware Cognitive Cortex Architecture for Long-Term Memory Evolution in LLMs

---

### 优先级原则（按论文贡献度）

```
P0（必须完成）    Lifecycle Engine · Activation Retrieval · Benchmark Framework
P1（核心创新）    Sleep Consolidation · Dormant Revival · Schema Formation
P2（锦上添花）    Persona System · Goal System · Self Model
P3（后续论文）    Multi-Agent Memory · Collective Memory · Cognitive Reasoning
```

---

### Phase 1: Foundation Stabilization ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| Unified Graph Layer | StarGraph + 5 类边类型 + supports + 清理 9 个零使用类型 | ✅ 已完成 |
| MemoryNode 统一 | Anchor → MemoryNode 别名（Anchor 向后兼容），3 层导出链 | ✅ 已完成 |
| 存储后端强化 | SQLite 默认 + QdrantStorage（完整 save/load/search） | ✅ 已完成 |

### Phase 2: Lifecycle Engine ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| 生命周期状态机 | ACTIVE→CONSOLIDATED→INACTIVE→DORMANT→REACTIVATED | ✅ 完成 |
| 重要性评分 | 5 信号加权（emotion/repetition/goal/richness/retrieval_feedback），含检索反馈 EMA | ✅ 完成 |
| 衰减引擎 | 三维衰减：temporal(recency) + activation(activation_level) + utility(frequency)，稳定性调制 | ✅ 完成 |
| 记忆迁移 | L1→L2→L3→Dormant 自动迁移 + 传播可达重新激活 | ✅ 完成 |

### Phase 3: Cognitive Retrieval ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| Spreading Activation | 种子 → BFS → 排序 | ✅ 完成 |
| Multi-Hop Retrieval | 1-3 跳图遍历检索 | ✅ 完成 |
| Goal-Aware 检索 | recall() 增加 goals 参数，检索后按目标相关性重排序 | ✅ 完成 |
| Context Compression | compress_context() 语义去重 + 低分过滤 + top-k 截断 | ✅ 完成 |

### Phase 4: Consolidation Cortex ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| Sleep Engine | 8 阶段离线整合：重放/冲突检测/重要性更新 | ✅ 完成 |
| Memory Revision | 过期检测、矛盾解决、版本保留 | ✅ 完成 |
| Schema Formation | Schema.match() 增强：关键词 + 嵌入相似度组合评分 + 槽值提取 | ✅ 完成 |
| Knowledge Distillation | 保留 cognitive_compression + SummaryAnchor 管线 | ✅ 完成 |

### Phase 5: Self Cortex（P2 锦上添花 — 不影响论文核心贡献）

### Phase 6: Dormant Memory System ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| Dormant Layer | 永不删除，activation_level ∈ [0,1]，永远可达 | ✅ 完成 |
| Reactivation | 语义匹配/传播激活/目标相关性触发重新激活 | ✅ 完成 |
| Savings Effect | GhostSubsystem.record_revival() + relearning_savings metric | ✅ 完成 |

### Phase 7: Cognitive Theory Layer ✅ 已完成

| 项目 | 产出 | 状态 |
|------|------|------|
| 记忆演化理论 | `docs/theory.md` — 生命周期/巩固/遗忘/重新激活/图传播/模式形成 | ✅ 完成 |
| 数学框架 | `docs/formalism.md` — 11 节数学定义（重要性/衰减/激活/复杂度/指标） | ✅ 完成 |
| 复杂度分析 | `docs/complexity_analysis.md` | ✅ 完成 |

### Phase 8: Benchmark Infrastructure ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| LoCoMo-10 | 1,986 QA pairs | ✅ 44.1% has_answer |
| LongBench / RULER | longbench_adapter.py — 6 任务类型 + RULER 风格检索测试 | ✅ 完成 |
| 基线对比 | run_baseline_comparison.py — Mem0/MemGPT/HippoRAG/Vanilla RAG 框架 | ✅ 完成 |
| 完整指标 | Recall@K, MRR, NDCG, Latency, Compression Ratio | ✅ 在 formalism.md 中定义 |

### Phase 9: Research Validation ✅ 已完成

| 项目 | 说明 | 状态 |
|------|------|------|
| 消融实验 | 5 变体 × 3 seeds | ✅ 完成 |
| 统计验证 | t-test, Wilcoxon, Cohen's d | ✅ 完成 |
| Scalability | scalability_test.py — 1K/10K/100K 压力测试框架 | ✅ 完成 |
| 跨会话召回 | 基于 LoCoMo 的 1,986 QA pairs | ✅ 完成 |

### Phase 10: ESWA Submission Readiness

**投稿前检查清单:**
- [x] 认知记忆理论框架
- [x] 数学形式化定义（formalism.md）
- [x] LoCoMo 基准（44.1%）
- [x] LongBench / RULER 多基准框架
- [x] 基线对比框架（Mem0/MemGPT/HippoRAG/Vanilla RAG）
- [x] 消融实验
- [x] 统计显著性
- [x] 公开仓库 + 可复现
- [x] 文档完善（theory.md, formalism.md）

> **状态**: 所有 10 项检查项 ✅ 完成。NWC 已具备 ESWA 投稿条件。

---

## 四、已完成工程基础

| 项目 | 状态 |
|------|------|
| 存储后端接口抽象 + SQLite 默认 | ✅ |
| 并发锁 (RLock) | ✅ |
| LoCoMo 全量跑分（44.1%） | ✅ |
| 消融实验 + 统计验证 | ✅ |
| Activation Graph 检索 | ✅ |
| Memory Lifecycle Engine | ✅ |
| Never-Delete 哲学（Dormant） | ✅ |
| 复杂度分析文档 | ✅ |

## 五、版本路线图

| 版本 | 重点 | 预期指标 |
|------|------|----------|
| **v1.3.x** | Lifecycle + Activation + Benchmark | LoCoMo 44.1% |
| **v1.4.x** | Schema + Compression + Multi-Benchmark | LongBench CI |
| **v1.5.x** | Formalism + Theory + Baselines | 对比实验完成 |
| **v2.0.0** | ESWA Submission Ready | 所有检查项通过 |

## 六、版本历史

| Version | Date | Highlights |
|---------|------|------------|
| **v1.4.0** | **2026-06** | **全 Phase 完成: 3D衰减引擎, Goal-Aware检索, Context Compression, Schema增强, Savings Effect, theory.md+formalism.md, LongBench+Baseline框架, Scalability测试** |
| **v1.3.6** | **2026-06** | **Unified Graph Layer (supports + 类型分类), MemoryNode alias, QdrantStorage** |
| **v1.3.5** | **2026-06** | **LoCoMo 44.1%, 消融+统计+复杂度, Activation Graph, Never-Delete, nwc benchmark CLI** |
| **v1.3.0** | **2026-06** | Activation Graph, Memory Lifecycle Engine, L1→L2→L3 迁移 |
| v1.2.12 | 2026-06 | 天文台+引力井, observatory.py, source_attribution |
| v1.1.0 | 2026-05 | LLM SDK, CLI, MCP, 5 LLM adapters |
| v1.0.8 | 2026-05 | Phase 6 Cognitive Cortex, 112 modules |
| v1.0.7 | 2026-05 | Phase 5 Cognitive Depth: 12 modules |
| v1.0.0 | 2026-05 | PyPI 发布, 1,989 tests, 80% coverage |

## 七、关联文档

- `ROADMAP.md` — 版本路线图
- `docs/complexity_analysis.md` — 复杂度分析
- `docs/experiment_log_qwen3_embedding.md` — Embedding 消融记录
- `benchmarks/locomo_results.json` — LoCoMo 基准
- `benchmarks/ablation_results.json` — 消融实验
- `benchmarks/statistical_analysis.py` — 统计显著性

