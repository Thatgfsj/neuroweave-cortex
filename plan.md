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
- v1.2.11 — 观测者依赖的检索范式
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

工程基继续加固，但**同时**投入类人记忆特征的实现。

---

## 三、行动计划

### 第一组：检索性能重建（直接影响用户价值）

#### A. 混合检索升级

**目标**: LoCoMo has_answer 从 31% → 50%+

| 措施 | 说明 | 预期提升 |
|------|------|----------|
| BM25+Embedding 真混合 | 当前 HybridFusion 只是加权组合，需要 RRF 融合 | +10-15% |
| HNSW 预筛选替代全量扫描 | 睡眠合并从 O(n²) 降为 O(n log n)，检索从全量变向量索引 | 延迟 300ms→100ms |
| 查询扩展（Query Expansion） | LLM 将用户查询扩展为 3-5 个相关表述后并行检索 | +5-10% |
| 跨编码器重排序 | 候选集 top-20 → Cross-Encoder 精排 top-5 | +5% |
| 权重回归调优 | 当前公式 0.35/0.20/0.25/0.10/0.10 是手调，改为网格搜索或贝叶斯优化 | +5% |

#### B. 原始缓冲区（Raw Buffer）优先级提升

**现状**: `recall()` 路径中 Raw Buffer 结果排在 Graph 结果之后。但 Raw Buffer 存的是最近 1-2 个 session 的原始对话，对短期事实类查询命中率远高于压缩后的 anchor。

**变更**: `exact → raw → graph`（Raw Buffer 提前）

#### C. 时间感知检索（针对 LoCoMo Category 1 Temporal）

**现状**: 主 `recall()` 路径中完全无时间权重。Category 1（时间类查询：日期/时刻/顺序）has_answer 仅 11.0%。纯 embedding 对"上周三"这样的时间表述完全不敏感。

**方案**:
1. 在 `retrieval_core.py` 的 `recall()` 中增加 `_detect_temporal_query()` 函数，识别"时间类"查询关键词（yesterday/last week/on May 7th 等）
2. 当检测到时间类查询时，调用 `timespine.query_window()` 做时间窗预过滤，只检索时间窗内的 anchor
3. 对命中的结果增加时间匹配分数加成（+0.15）

**预期提升**: Category 1 has_answer 从 11% → 25%+

#### D. 近期记忆权重提升（针对 LoCoMo Category 2-3 Short/Long Memory）

**现状**: RRF 融合路径中无时间衰减。Category 2（Short Memory）has_answer 仅 3.1%，Category 3（Long Memory）仅 7.3%。早期记忆几乎被完全遗忘。

**方案**:
1. 在 `recall()` 的 RRF 融合后、sort 之前，增加一个时间衰减权重步骤
2. 对最近 24 小时内的 anchor 加 +0.10 分数加成
3. 对最近 7 天内的 anchor 加 +0.05 分数加成
4. 权重在 `defaults.yaml` 中可配置为 `recall.recency_boost_hours` / `recall.recency_boost_weight`

**预期提升**: Category 2-3 has_answer 从 3-7% → 15%+

---

### 第二组：存储层工程化（生产就绪的基础）

#### A. 存储后端抽象层

```
StorageBackend (接口)
  ├── JSONStorage         (现有，保留为开发/测试用)
  ├── SQLiteStorage       (现有但需升级为默认后端)
  ├── QdrantStorage       (新增，开源+轻量+混合检索+HNSW)
  └── MilvusStorage       (新增，大规模生产场景)
```

**接口定义**:
```python
class StorageBackend(ABC):
    @abstractmethod
    def save(self, key: str, data: dict) -> None: ...
    @abstractmethod
    def load(self, key: str) -> dict | None: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict]: ...
    @abstractmethod
    def batch_save(self, items: list[tuple[str, dict]]) -> None: ...
    @abstractmethod
    def transaction(self) -> ContextManager: ...
```

#### B. 并发安全

- `MemoryManager` 和 `graph.py` 中的共享状态加 `threading.RLock`
- JSON 持久化改为原子写入: `write_to_temp()` + `os.rename()`
- 多线程环境下读写分离

#### C. 迁移与备份工具

- `nwc export` / `nwc import` — 完整数据导出导入
- `nwc snapshot create` / `nwc snapshot restore` — 快照管理
- `nwc migrate --from json --to sqlite` — 后端迁移

---

### 第三组：生态集成（降低 90% 潜在用户的试用门槛）

#### A. LangChain 适配器

```python
from langchain.memory import BaseChatMemory
from neuroweave import NeuroWeaveMemory

memory = NeuroWeaveMemory(
    cortex_id="my-agent",
    api_key="nwc-xxx",           # 可选：连接 NWC Server
    storage_backend="sqlite",    # 或 "qdrant"
)

# 标准 LangChain 接口
memory.save_context({"input": "Hi"}, {"output": "Hello!"})
context = memory.load_memory_variables({})
```

#### B. LlamaIndex 适配器

```python
from llama_index.core.memory import BaseMemory
from neuroweave.integrations.llama_index import NeuroWeaveLlamaMemory
```

#### C. OpenAI Agents SDK / AutoGen / CrewAI

- `openai_agents_toolkit.py` — 作为 OpenAI Agents 的 memory tool
- `autogen_memory.py` — AutoGen 的 memory 插件
- `crewai_memory_adapter.py` — CrewAI 长期记忆适配

---

### 第四组：公平评测体系（用数据证明差异化价值）

#### A. 标准对比基准

在 `benchmarks/` 目录下新增:

| 对比对象 | 测什么 | 当前状态 |
|----------|--------|----------|
| Mem0 (mem0.ai) | 相同的对话序列 → 检索 Q&A | ❌ 缺失 |
| Zep (getzep.com) | 同上 | ❌ 缺失 |
| 纯 Qdrant/FAISS | 纯向量检索基线 | ❌ 缺失 |
| MemGPT | 长期记忆管理能力 | ❌ 缺失 |

**对比规则**:
- 相同的输入对话、相同的 embedding 模型
- 相同的查询集与评判标准
- 输出: has_answer, F1, latency, memory_size

#### B. 端到端任务基准

| 任务 | 衡量指标 | 说明 |
|------|----------|------|
| 多轮对话记忆问答 | Task Success Rate | "上周三讨论的数据库连接池配置是什么？" |
| 跨会话工具调用 | Tool Selection Accuracy | 基于历史偏好选择正确工具 |
| 个性化回复 | Preference Adherence | 回复风格/内容是否符合用户历史偏好 |
| 遗忘曲线 | Forgetting Curves | 记忆质量 vs 时间的变化 |

#### C. 消融实验

| 实验 | 变体 | 验证目标 |
|------|------|----------|
| 睡眠消融 | no_sleep / 3_phase / 8_phase | 8 阶段睡眠是否过度设计 |
| 检索消融 | pure_vector / graph_only / full | 图检索是否带来实际收益 |
| 生物启发消融 | with_ghost / without_ghost | Ghost Revival 是否有量化效果 |
| 写门禁消融 | with_write_gate / without | 写门禁是否改善检索质量 |

---

### 第五组：模块化瘦身（降低认知负担）

#### 拆分方案: core vs cognitive

```
nwc/
├── core/                    # pip install NWcortex 默认安装（~20 模块）
│   ├── memory_core/
│   │   ├── graph.py         # 核心图结构
│   │   ├── anchor.py        # 记忆单元
│   │   ├── storage.py       # 存储接口
│   │   ├── storage_backend.py
│   │   ├── sqlite_storage.py
│   │   ├── tier.py
│   │   └── index.py
│   ├── retrieval_engine/
│   │   ├── retriever.py     # 检索核心
│   │   ├── bm25.py
│   │   ├── dual_channel.py
│   │   └── cognitive_cache.py
│   ├── sleep/
│   │   ├── sleep.py         # 基础睡眠
│   │   └── micro_sleep.py
│   ├── working_memory.py
│   ├── perception.py
│   └── cortex.py            # Cortex API facade
│
└── cognitive/               # pip install NWcortex[full]（高级认知模块）
    ├── hebbian_learning.py
    ├── ghost.py
    ├── personality.py
    ├── belief_system.py
    ├── cognitive_identity.py
    ├── autonomous_reasoning.py
    ├── goal_system.py
    ├── goal_tree.py
    ├── spreading.py
    ├── compiler.py
    ├── reflection_loop.py
    ├── tracing.py
    └── ... (其他认知层模块)
```

**安装方式**:
```bash
pip install nwcortex          # core 仅约 20 模块
pip install nwcortex[full]    # 全部 112+ 模块
pip install nwcortex[qdrant]  # 附带 Qdrant 后端
```

---

### 第六组：可观测性与运维

#### A. OpenTelemetry 关键路径埋点

| 路径 | 埋点指标 | 现有状态 |
|------|----------|----------|
| `recall()` | latency, result_count, cache_hit_rate | ❌ 无 |
| `sleep()` | duration, merged_count, pruned_count | ❌ 无 |
| `consolidate()` | compression_ratio, tokens_before/after | ❌ 无 |
| `remember()` | write_latency, importance_score | ❌ 无 |

#### B. Prometheus 指标暴露

```
nwc_memory_anchors_total          # 当前锚点数
nwc_memory_retrieval_latency_ms   # 检索延迟直方图
nwc_memory_cache_hit_ratio        # 缓存命中率
nwc_memory_sleep_duration_seconds # 睡眠耗时
nwc_memory_compression_ratio      # 压缩率
nwc_memory_retrieval_has_answer   # 检索命中率（自评）
```

#### C. REST API 生产级加固

- JWT 认证（`Authorization: Bearer <token>`）
- 速率限制（`nwc_server.max_rpm=60`）
- 健康检查 endpoint（`GET /health` → `{status, version, anchor_count, uptime}`）
- 优雅关闭

#### D. Docker 部署

```yaml
# docker-compose.yml
services:
  nwc:
    build: .
    ports: ["8090:8090"]
    volumes: ["./data:/data"]
    environment:
      - NWC_STORAGE_BACKEND=qdrant
      - NWC_AUTH_ENABLED=true
  qdrant:
    image: qdrant/qdrant
    volumes: ["./qdrant_data:/qdrant/storage"]
```

---

### 第八组：类人记忆增强（核心差异化）

#### P0 — 社交来源归因 + 信任度

**文件**: `anchor.py` + `sleep.py` N2b_Conflict

在 `AnchorVector` 中增加 `source_attribution` 字段，区分记忆来源：

| 来源类型 | 含义 | 初始信任度 |
|----------|------|-----------|
| `self_reported` | 用户自己说的 | 0.9 |
| `user_told_me` | Agent 被告知的 | 0.7 |
| `inferred` | 系统推断的 | 0.4 |
| `tool_output` | 工具返回的 | 0.8 |
| `observation` | 系统观察到的 | 0.6 |

**冲突解决增强**: 在睡眠 N2b_Conflict 阶段，当两个记忆矛盾时，不是单纯按时间或激活度覆盖，而是按信任度加权——高信任度来源的记忆覆盖低信任度的。

#### P0 — 事件锚定时间线（Event-Anchored Timeline）

**文件**: `timespine.py` / 新增 `event_anchor.py`

当前 Category 1（Temporal）只有 11% has_answer，因为人类不说"2024年6月7日"，而是说"就在那次 Redis 调试之后"。

方案：
1. 在 `TimeSpine` 中增加**重要事件标记**——当一条记忆的 importance > 0.8 时，自动标记为"时间地标"
2. 新增 `event_anchor.py` 模块，提供 `resolve_temporal_query(query) → time_window` 函数
3. 查询"上周三"时，先解析为"最近的项目里程碑前后"，再映射到具体日期范围

#### P1 — 情感多维化

**文件**: `salience.py` + `anchor.py`

引入轻量情感向量（效价、唤醒度、支配度 + 情绪标签），增加:
- **抑制性回忆**: 高尴尬度的记忆降低主动召回概率
- **怀旧提升**: 高怀旧度的记忆在低任务负载时提升 Ghost Revival 概率

#### P1 — 自由联想模式

**文件**: `activation_engine.py`

新增 `free_associate()` 接口，不接收查询，从当前工作记忆出发做随机漫步（random walk with temperature），返回 `association_type: "direct" / "distant" / "analogical"`。

#### P2 — 叙事连贯性检索

**文件**: `retriever.py`

新增 `narrative_weave` 模式——用图的路径搜索替代点的相似度搜索，返回"能连成一段合理叙事的记忆链"。

#### P2 — 记忆失真/重构

**文件**: `sleep.py` N2c_Revision

引入轻度重构——当两个记忆被多次同时激活时，边界模糊化产生合成记忆，打上 `reconstructed` 标签。

#### P3 — 情境信封

**文件**: 新增 `contextual_envelope.py`

给记忆增加轻量的情境包裹（地点、身体状态、前序任务），作为回忆时的氛围匹配。

---

### 第九组：工程基础继续加固

以下项目继续推进，与第八组并行：

| 项目 | 状态 |
|------|------|
| Qdrant 存储后端 | 🔜 待做 |
| LangChain 适配器 | 🔜 待做 |
| Mem0/FAISS 对比基准 | 🔜 待做 |
| 延迟分解（latency breakdown） | 🔜 待做 |
| 模块化拆分 | 🔜 待做 |

**原则**: 对于所有未被消融实验证明价值的"生物学启发"模块，标记为 **EXPERIMENTAL** 并在日志中告警。如果 3 个月内仍无消融证据，在 v2.0 中移除。

| 模块 | 当前状态 | 要求 | 截止 |
|------|----------|------|------|
| Ghost Revival | ✅ 已实现 | 证明在 has_answer 上有 >5% 增益 | 2026-Q3 |
| REM Emotion | ✅ 已实现 | 证明情感分析在检索质量上 >3% 增益 | 2026-Q3 |
| Hebbian Learning | ✅ 已实现 | 证明学习到的权重优于静态权重 | 2026-Q3 |
| Oscillation Resonance | ✅ 已实现 | 证明比纯向量检索有显著提升 | 2026-Q3 |
| Spreading Activation | ✅ 已实现 | 证明在图上传播优于纯 embedding 检索 | 2026-Q3 |

---

## 四、实施路线

### Track 1: 核心检索管道（最高优先级）

```
1.1 Raw Buffer 优先级提升          ← done, 移除 0.7 惩罚
1.2 时间感知检索（Category 1）      ← 当前：时间实体检测 + TimeSpine 预过滤
1.3 近期记忆权重（Category 2-3）    ← 当前：recall() 增加时间衰减权重
1.4 BM25+Embedding 真 RRF 融合     ← 待做：量化提升
1.5 检索权重回归调优                ← 待做：LoCoMo 数据驱动
1.6 并发安全锁 + 原子写入           ← done
```

### Track 2: 存储层工程化

```
2.1 StorageBackend 接口设计         ← done
2.2 SQLiteStorage 升级为默认后端    ← done
2.3 QdrantStorage 实现              ← 待做
2.4 导出/导入/迁移工具              ← 待做
```

### Track 3: 生态集成

```
3.1 LangChain NeuroWeaveMemory      ← 待做
3.2 基准对比脚本 (Mem0/Zep/FAISS)   ← 待做
3.3 消融实验框架                     ← 待做
3.4 OpenAI Agents SDK adapter       ← 待做
```

### Track 4: 可观测性

```
4.1 关键路径延迟分解（recall latency breakdown）  ← 当前
4.2 Prometheus 指标暴露               ← 待做
4.3 REST API 认证 + 速率限制          ← 待做
4.4 Docker Compose 部署方案           ← 待做
```

### Track 5: 模块化拆分

```
5.1 模块清单审查：core vs cognitive 划分  ← done (docs/module-classification.md)
5.2 core 包重组，依赖清理                  ← 待做
5.3 cognitive 包独立，惰性导入              ← 待做
5.4 配置简化：从 300+ 参数到 ~50 核心参数   ← 待做
```

### Track 6: 端到端评测

```
6.1 多轮对话记忆基准                     ← 当前：创建 e2e 任务基准
6.2 消融实验自动运行 + 报告生成           ← 待做
6.3 延迟/吞吐压力测试                     ← 待做
6.4 长期稳定性测试（10万+记忆）           ← 待做
```

---

## 五、质量门禁（Definition of Done）

每个合并前必须满足:

| 门禁 | 说明 |
|------|------|
| ✅ LoCoMo has_answer ≥ 45% | 当前 31%，目标 45%+ |
| ✅ 无 O(n²) 操作 | 全量扫描需有 ANN 预筛选 |
| ✅ 并发写入测试通过 | 使用 10 线程同时写入 1000 条 |
| ✅ 与 Mem0 的公平对比 | 至少在 1 个数据集上不弱于 Mem0 |
| ✅ 数据库迁移工具 | 支持 JSON ↔ SQLite ↔ Qdrant |
| ✅ 单元测试覆盖率不降 | 新代码覆盖率 ≥ 80% |

---

## 六、版本路线图

| 版本 | 重点 | 预期指标 |
|------|------|----------|
| **v1.2.0** | 检索管道重建 + 并发安全 | LoCoMo has_answer ≥ 45% |
| **v1.3.0** | 存储后端抽象 + LangChain 适配 | 存储可切换，生态可接入 |
| **v1.4.0** | 模块化拆分 + 公平评测 | 安装体积减少 60% |
| **v1.5.0** | 可观测性 + Docker 部署 | 生产可部署 |
| **v2.0.0** | 消融实验完成，移除未验证模块 | 模块从 112 精简到 50-60 |

---

## 七、版本历史

| Version | Date | Highlights |
|---------|------|------------|
| **1.2.11** | **2026-06** | **类人记忆引擎：天文台+引力井。范式迁移：从向量检索到星图观测。新增 observatory.py, gravity_well.py, source_attribution, event-anchored timeline** |
| 1.0.7 | 2026-05 | Phase 5 Cognitive Depth: 12 modules (memory budget, quality score, stability control, 4-layer pyramid, typed memory, domain graph, Hebbian learning, etc.) |
| 1.0.8 | 2026-05 | Phase 6 Cognitive Cortex: 11 modules (ThoughtObject, PerceptionLayer, CognitiveWorkspace, ConceptCortex, ActivationEngine, GoalSystem, SalienceEngine, SelfModel, AutonomousReasoning, MemoryLifecycle). 4-layer architecture. 112 modules. |
| 1.1.0 | 2026-05 | LLM SDK: nwc/ package (20 files), CLI (Typer, 13 commands), MCP server (14 tools), OpenAI-compatible API, 5 LLM adapters, docs/llm_integration.md |
| 1.2.0 | 2026-05 | Phase 7 Memory Evolution: ImportanceEngine, BeliefSystem, PersonalityFormation, CognitiveIdentity. 6-layer memory architecture. |

## 八、关联文档

- `ROADMAP.md` — 版本路线图
- `docs/architecture.md` — 架构文档（待更新）
- `docs/llm_integration.md` — LLM 接入指南
- `README.md` / `README_CN.md` — 项目说明
- `benchmarks/run_benchmarks.py` — 基准测试
