# NeuroWeave Cortex (NWC) — Plan

> Last updated: 2026-05-22 | **v1.1.0** | 6-layer cognitive memory architecture

---

## 定位: Cognitive Memory Engine（认知记忆引擎）

### 不是

- 向量记忆库
- 对话存档器
- Agent 插件
- MCP 附件
- RAG 系统

### 而是

> NeuroWeave Cortex is a cognitive long-term memory engine for AI systems.
> 不是"让 AI 记住"，而是"让 AI 形成认知"。

核心哲学:

1. **记忆不平等** — 早餐吃了什么 ≠ 长期目标 ≠ 废话，必须分级存储
2. **记忆会成长** — 从数据 → 模式 → 信念 → 人格，不是静止的
3. **认知压缩是未来** — 海量对话 → 抽象 → 压缩 → 人格 → 长期认知
4. **LLM 在认知层之下** — NWC 先形成认知状态，LLM 只负责语言表达
5. **AI 不是"读取记忆"，而是"基于记忆思考"**

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

## 当前状态

- **132 modules** (112 star_graph + 20 nwc)
- Phase 1-6 complete + LLM SDK (v1.1.0)
- nwc/ package: Cortex SDK, CLI (Typer, 13 commands), MCP server (14 tools), OpenAI-compatible API, 5 LLM adapters

### Layer 完成度

| Layer | 状态 | 核心模块 |
|-------|------|----------|
| ① Sensory | ✅ 完成 | `perception.py`, `importance_engine.py` |
| ② Episodic | ✅ 完成 | `anchor.py`, `working_memory.py`, `graph.py` |
| ③ Semantic | ✅ 完成 | `concept_cortex.py`, `abstraction.py`, `community.py` |
| ④ Pattern | △ 部分 | `cognitive_compression.py` (缺 behavior pattern extraction) |
| ⑤ Belief | ❌ 未实现 | `belief_system.py` (待实现) |
| ⑥ Identity | △ 部分 | `self_model.py` + `nwc/core/cortex.py` (缺 persistent identity) |

---

## Phase 7 — 记忆演化 (当前实施)

### 7.1 Importance Engine（重要性引擎）

**新建模块**: `star_graph/importance_engine.py`
- 5 维评分: emotion, repetition, goal_relation, novelty, future_probability
- 噪声过滤: importance < 0.2 直接丢弃
- 与 `perception.py` 集成: 感知后立即评分

### 7.2 Belief System（信念系统）

**新建模块**: `star_graph/belief_system.py`
- 信念形成: 从 Pattern Layer 重复行为模式中提取
- 信念强化/挑战: 新证据确认或削弱信念
- 信念演化: 合并、分裂、更新
- 信念→检索权重: 与信念相关的记忆检索时加权

### 7.3 Personality Formation（人格形成引擎）

**新建模块**: `star_graph/personality_formation.py`
- 从 Belief System + Pattern Layer → Cognitive Profile
- 认知风格、价值体系、行为模式、身份标记
- 演化轨迹: 趋势方向、新兴兴趣、消退兴趣

### 7.4 Cognitive Identity（持久认知身份）

**新建模块**: `star_graph/cognitive_identity.py`
- 跨会话 Persistent Cognitive Identity
- 用户认知画像 + 自我认知
- 成长轨迹追踪
- ~/.nwc/identity/ 持久化

### 7.5 Six-Layer Pipeline Integration

**更新**: `nwc/core/cortex.py`
- 新 `ctx.identity()` API — 获取 Persistent Cognitive Identity
- 新 `ctx.beliefs()` API — 获取长期信念
- 更新 `ctx.context()` — 注入压缩后的认知画像而非仅记忆
- `SensoryParser` → `ImportanceEngine` 集成到输入管道

---

## Phase 8+ (未来)

### 8.1 混合存储后端
- SQLite/Postgres + Qdrant + 轻量 Graph
- 不是 Neo4j 全图化

### 8.2 多用户共享概念皮层
- 跨 Agent 共享 ConceptCortex
- 集体认知压缩

### 8.3 认知可视化仪表盘
- WebSocket 实时认知状态
- 信念/人格演化时间线

---

## 版本历史

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.7 | 2026-05 | Phase 5 Cognitive Depth: 12 modules (memory budget, quality score, stability control, 4-layer pyramid, typed memory, domain graph, Hebbian learning, etc.) |
| 1.0.8 | 2026-05 | Phase 6 Cognitive Cortex: 11 modules (ThoughtObject, PerceptionLayer, CognitiveWorkspace, ConceptCortex, ActivationEngine, GoalSystem, SalienceEngine, SelfModel, AutonomousReasoning, MemoryLifecycle). 4-layer architecture. 112 modules. |
| 1.1.0 | 2026-05 | LLM SDK: nwc/ package (20 files), CLI (Typer, 13 commands), MCP server (14 tools), OpenAI-compatible API, 5 LLM adapters, docs/llm_integration.md |
| 1.2.0 | 2026-05 | Phase 7 Memory Evolution: ImportanceEngine, BeliefSystem, PersonalityFormation, CognitiveIdentity. 6-layer memory architecture. |

---

## 关联文档

- `ROADMAP.md` — 版本路线图
- `docs/llm_integration.md` — LLM 接入完整指南
- `README.md` — 项目 README (英文)
- `README_CN.md` — 项目 README (中文)
