# NeuroWeave Cortex (NWC) · 神经编织皮层

不是向量数据库。不是图数据库。不是 RAG。一个**外部认知皮层**——感知、激活概念、维护工作记忆、形成推理链、追踪目标、管理注意力、演化自我模型的认知运行时。它像生物认知一样：记忆、遗忘、强化、连接、抽象和推理。

```
v1.0.8 | 112 模块 | 阶段 1-6 完成 | 认知皮层架构
```

## 与向量数据库/图数据库的区别

向量数据库检索。图数据库遍历。NeuroWeave Cortex 运行**完整的认知生命周期**：

| 能力 | 向量数据库 | 图数据库 | NW Cortex |
|---|---|---|---|
| 语义检索 | ✓ | ✗ | ✓ |
| 图遍历 | ✗ | ✓ | ✓ |
| 自动遗忘（生存衰减） | ✗ | ✗ | ✓ |
| 记忆强化（复述） | ✗ | ✗ | ✓ |
| 冲突检测与解决 | ✗ | ✗ | ✓ |
| 多策略 RRF 检索融合 | ✗ | ✗ | ✓ |
| Cross-Encoder 重排序 | ✗ | ✗ | ✓ |
| 可解释推理路径 | ✗ | ✗ | ✓ |
| 记忆修订引擎（睡眠） | ✗ | ✗ | ✓ |
| Hot/Warm/Cold 记忆分层 | ✗ | ✗ | ✓ |
| 多模态（文本+图像+音频） | ✗ | ✗ | ✓ |
| 模糊回忆（"我好像记得..."） | ✗ | ✗ | ✓ |
| 涌现抽象（模式发现） | ✗ | ✗ | ✓ |
| 时间上下文（TimeSpine 索引） | ✗ | ✗ | ✓ |
| 8阶段睡眠巩固 | ✗ | ✗ | ✓ |
| 幽灵复活（节省效应） | ✗ | ✗ | **世界首创** |
| Hebbian 边学习 | ✗ | ✗ | ✓ |
| 4层记忆金字塔 | ✗ | ✗ | ✓ |
| 目标驱动推理 | ✗ | ✗ | ✓ |
| **工作记忆工作区** | ✗ | ✗ | **✓** |
| **概念皮层** | ✗ | ✗ | **✓** |
| **扩散激活引擎** | ✗ | ✗ | **✓** |
| **显著性注意力** | ✗ | ✗ | **✓** |
| **自我模型 → LLM 提示注入** | ✗ | ✗ | **✓** |
| **自主推理循环** | ✗ | ✗ | **✓** |
| **完整记忆生命周期（9阶段）** | ✗ | ✗ | **✓** |

## 快速开始

### 经典 API（阶段 1-5）

```python
from star_graph import MemoryManager
from star_graph.scheduler import AgentContext

mgr = MemoryManager()

# 记忆
mgr.remember("用户在排查 Redis 连接超时——连接池从10调到20后修复",
             tags=["redis", "debug", "timeout"])
mgr.remember("用户偏好类型标注和简洁代码",
             tags=["preference", "style"])

# 工作记忆
mgr.remember_working("正在调试 auth 中间件超时问题",
                     tags=["debug", "auth"])

# 上下文感知检索
ctx = AgentContext(task_type="debugging", active_goals=["fix Redis connection"])
memories = mgr.recall("Redis 连接池配置", context=ctx)
print(memories.memory_summary)

# 睡眠巩固
report = mgr.sleep()

# 持久化
mgr.save("agent_memory.json")
```

### 认知皮层 API（阶段 6）

```python
from star_graph import (
    PerceptionLayer, CognitiveWorkspace, ConceptCortex,
    ActivationEngine, GoalSystem, SalienceEngine, SelfModel
)

# ── 完整认知流水线 ──
pl = PerceptionLayer()
ws = CognitiveWorkspace(max_items=20)
cc = ConceptCortex()
ae = ActivationEngine(graph=graph)
gs = GoalSystem()
se = SalienceEngine()

# 1. 感知用户输入
frame = pl.perceive("我需要紧急调试 Redis 内存泄漏！")
# → PerceptionFrame(intent="command", valence=-0.4, concepts=["redis","debug","leak"])

# 2. 纳入工作记忆工作区
ws.on_perception(frame)

# 3. 激活概念
for concept in frame.extracted_concepts:
    cc.get_or_create_concept(concept)

# 4. 在记忆图中扩散激活
result = ae.activate_from_perception(frame)

# 5. 追踪目标
goal = gs.add_goal("修复 Redis 管道内存泄漏", priority=0.9)
gs.activate_goal(goal.id)

# 6. 计算显著性——决定什么进入注意力
signals = [se.compute_salience(item.id, "thought",
    context={"task_relevance": item.relevance_to_current_task})
    for item in ws.get_items()]
winners = se.compete(signals, max_winners=5)

# 7. 构建自我模型 → LLM 提示注入
sm = SelfModel(workspace=ws, goal_system=gs, concept_cortex=cc, salience_engine=se)
prompt = sm.get_prompt_injection()
# → "# Cognitive State Summary\n## Current Focus: ...\n## Active Goals: ..."

# 这就是 LLM 看到的内容——不是原始记忆转储！
```

## 架构

四层认知设计。阶段 6 在行为层之上增加了皮层层。

```
第四层：皮层层    │  感知、工作记忆工作区、概念皮层、
    (阶段 6)      │  扩散激活、目标系统、显著性引擎、
    perception.py,│  自我模型、自主推理、认知压缩、
    cognitive_workspace.py,│  记忆生命周期管理器
    concept_cortex.py,│  "我现在在想什么？我应该关注什么？"
    activation_engine.py,│
    goal_system.py,│
    salience.py,   │
    self_model.py, │
    autonomous_reasoning.py,│
    cognitive_compression.py,│
    memory_lifecycle.py)│
                   │
第三层：行为层    │  皮层路由、记忆门控、工作记忆、
    (cortex.py,   │  双通道检索、调度器、质量评分、
     router.py,   │  记忆预算、稳定性控制、认知优先级
     gate.py,     │  "此刻应该回忆什么，以什么细节层次？"
     working_memory.py,│
     scheduler.py,│
     quality_score.py,│
     memory_budget.py,│
     stability_control.py,│
     cognitive_priority.py)│
                   │
第二层：认知层    │  Hub抽象、级联回忆、TimeSpine时间索引、
    (retriever.py,│  睡眠巩固、演化、幽灵复活、
     sleep.py,    │  抽象、社区检测、竞争、
     evolution.py,│  冲突检测、记忆修订、Cross-Encoder、
     ghost.py,    │  Hebbian学习、领域路由、上下文路由
     abstraction.py,│  "记忆如何连接、强化和消退？"
     community.py,│
     competition.py,│
     hebbian_learning.py,│
     domain_graph.py,│
     context_routing.py)│
                   │
第一层：存储层    │  CRUD、持久化、ANN索引、分层存储、
    (graph.py,    │  BM25关键词索引、多级缓存、类型化记忆、
     anchor.py,   │  4层记忆金字塔、抽象链
     storage.py,  │  "这个记忆存在哪里？"
     index.py,    │
     bm25.py,     │
     typed_memory.py,│
     memory_layers.py,│
     abstraction_chain.py)│
```

### 核心模块（阶段 1-4 基础）

| 模块 | 功能 |
|---|---|
| `manager.py` | 高层门面 API — `remember()`, `recall()`, `sleep()`, `save()` |
| `runtime.py` | 依赖容器 — 管理所有子系统生命周期 |
| `scheduler.py` | 上下文感知检索，记忆类型选择 |
| `working_memory.py` | 短期缓冲区 (15条, 1h TTL)，自动晋升至长期记忆 |
| `sleep.py` | 8阶段睡眠：N1_Replay → N2_Merge → N2b_Conflict → N2c_Revision → N3_Compression → N3d_Rebuild → REM_Emotion → N4_Prune |
| `evolution.py` | 基于生存函数的衰减 (艾宾浩斯/幂律/指数)、信念变迁 |
| `retriever.py` | HybridFusion + OscillationResonance + VectorSimilarity + 个性化PageRank |
| `bm25.py` | 稀疏关键词检索 (BM25)，与稠密向量做倒数秩融合 |
| `cross_encoder.py` | Cross-Encoder 重排序，提升检索精度 |
| `ghost.py` | 幽灵痕迹，模糊回忆，节省效应复活 |
| `abstraction.py` | 从锚点簇中涌现类别发现 |
| `community.py` | Louvain 社区检测，质心路由 |
| `anchor.py` | 记忆单元：6状态生命周期，10维 AnchorVector |
| `graph.py` | 星图：RichEdge、Schema、ReflectionNode |
| `compression.py` | 多层次会话压缩（episodic/strategic/meta） |
| `forget_certificate.py` | Ed25519 签名 JWS 删除证书 — GDPR 第17条 |
| `multimodal.py` | 跨模态文本/图像/音频记忆 |
| `zero_llm_pipeline.py` | 7阶段零LLM摄入流水线 |
| `batch_vectorizer.py` | 延迟批量嵌入写入，SQLite WAL |
| `markdown_export.py` | GBrain 对齐的 Markdown 记忆导出 |

### 阶段 5 — 认知深度

| 模块 | 功能 |
|---|---|
| `memory_budget.py` | Token + 锚点预算管理，驱逐策略 |
| `quality_score.py` | 7维记忆质量评分（使用、推理、反馈等） |
| `stability_control.py` | 长期稳定性，指数/线性衰减，漂移监控 |
| `memory_layers.py` | 4层金字塔：工作 → 情景 → 语义 → 核心身份 |
| `typed_memory.py` | 7种记忆类型（code/task/dialogue/tool_call/knowledge/event/preference） |
| `abstraction_chain.py` | 事件 → 摘要 → 模式 → 身份抽象流水线 |
| `domain_graph.py` | 基于领域的图分区，软隔离 |
| `context_routing.py` | 6维度上下文感知检索路由 |
| `hebbian_learning.py` | Hebbian 边学习——"一起放电的神经元连接在一起" |
| `agent_state.py` | 代理状态记忆：目标树、工具调用、检查点 |
| `cognitive_closure.py` | 闭环反馈：回忆 → 使用 → 学习 → 改进 |
| `cognitive_priority.py` | 5级优先级分配，核心身份强制注入 |

### 阶段 6 — 认知皮层

| 模块 | 功能 |
|---|---|
| `thought_object.py` | 统一认知基类——激活节点，非静态记忆 |
| `perception.py` | 原始文本 → 结构化 PerceptionFrame（意图、情绪、目标、概念） |
| `cognitive_workspace.py` | **核心**：工作记忆工作区，推理链、注意力、TTL衰减 |
| `concept_cortex.py` | 概念网络：激活、融合、竞争，10个内置核心概念 |
| `activation_engine.py` | 多源扩散激活（查询/目标/概念/情绪种子） |
| `goal_system.py` | 目标层级、冲突检测、目标驱动推理 |
| `salience.py` | 10组件注意力竞争，侧向抑制 |
| `cognitive_compression.py` | 事件 → 概念 → 身份 → 世界模型压缩流水线 |
| `self_model.py` | 系统自我模型 → 压缩认知状态 → LLM提示注入 |
| `autonomous_reasoning.py` | 矛盾 → 激活 → 解决循环（无需LLM） |
| `memory_lifecycle.py` | 统一9阶段生命周期：感知 → 工作 → ... → 幽灵 → 死亡 |

## 记忆生命周期（9阶段）

每个记忆经历完整的9阶段认知生命周期（阶段6.10 `memory_lifecycle.py`）：

```
PERCEPTION → WORKING → SHORT_TERM → LONG_TERM → CONSOLIDATED → ARCHIVED → DORMANT → GHOST → DEAD
```

| 阶段 | 描述 | 持续时间 |
|------|------|----------|
| **Perception（感知）** | 初始摄入，尚未存储 | 秒级 |
| **Working（工作）** | 在认知工作区中，高激活 | 分钟~小时 |
| **Short-term（短期）** | 已存储，尚未巩固 | 小时~天 |
| **Long-term（长期）** | 已巩固，可主动检索 | 天~月 |
| **Consolidated（已巩固）** | 深度整合，高稳定性 | 月~年 |
| **Archived（已归档）** | 冷存储，低访问频率 | 无限 |
| **Dormant（休眠）** | 极低访问，接近幽灵 | 无限 |
| **Ghost（幽灵）** | 已遗忘但可能复活 | 90天 |
| **Dead（死亡）** | 永久移除 | — |

配套 **ThermalState**（HOT → WARM → COLD → DEAD）用于存储介质切换，以及**4层金字塔**（工作 → 情景 → 语义 → 核心身份）。

## 睡眠巩固（8阶段）

睡眠不是清理，而是**改变图结构**：

1. **N1_Replay** — SWR评分优先重放高惊喜度、高情绪记忆
2. **N2_Merge** — ANN加速近重复融合 + 星座桥接
3. **N2b_Conflict** — 语义矛盾检测（覆盖/共存/废弃）
4. **N2c_Revision** — 惊喜优先的记忆修订引擎
5. **N3_Compression** — 海马体→皮层转移，图式形成
6. **N3d_Rebuild** — 多节点融合，边重连，抽象模式发现
7. **REM_Emotion** — 剥离已巩固记忆的情绪负载
8. **N4_Prune** — 弱锚点/边移除，幽灵痕迹创建（节省效应）

## 双通道检索

System-1（快速，embedding + BM25 混合）与 System-2（深度，层次遍历）自动触发：

- 低置信度 System-1 结果 (<0.35) 自动触发 System-2
- 结构性关键词（"所有"、"每个"、"列出"、"哪些"、"之前"、"最后"）触发穷举搜索
- 结果通过加权倒数秩融合合并

## 配置系统

```python
from star_graph.config import config, override, load_config

# 点路径访问
threshold = config.sleep.merge.default_threshold  # 0.85

# 编程覆盖
override('sleep.merge.default_threshold', 0.75)
override('gate.k', 30)

# Schema 验证
warnings = config.validate()  # 类型、范围、跨段兼容性检查

# 加载自定义 YAML
cfg = load_config("my_params.yaml")
```

所有 300+ 可调参数见 `star_graph/defaults.yaml`。

## 安装

```bash
# 从 PyPI 安装（包名：NWcortex，导入名：star_graph）
pip install NWcortex

# 包含语义嵌入支持
pip install "NWcortex[embeddings]"

# 包含 MCP 服务支持（AI 代理集成）
pip install "NWcortex[mcp]"

# 全部功能
pip install "NWcortex[all]"

# 运行演示
python examples/emergence_demo.py
```

**注意：** 不带 `[embeddings]` 时，系统使用轻量级 TF-IDF 回退进行文本编码。仅在需要语义级嵌入时才安装 `sentence-transformers`。

## MCP 服务 — AI 代理集成

NeuroWeave Cortex 内置 [Model Context Protocol](https://modelcontextprotocol.io) 服务器。连接任何 MCP 兼容的 AI 代理（Claude Desktop、OpenClaw、Cursor 等）以获得跨对话的持久认知记忆。

### 快速开始

```bash
# 安装 MCP 支持
pip install "NWcortex[mcp]"

# 在 stdio 上启动 MCP 服务
nwc-mcp

# 使用持久化存储
nwc-mcp --storage agent_memory.json

# 加载之前保存的记忆图
nwc-mcp --load my_memories.json
```

### 工具列表（12个）

| 工具 | 说明 |
|------|------|
| `remember` | 存储长期记忆（标签、重要性、情感效价） |
| `remember_working` | 存储到快速工作记忆缓冲区（当前任务上下文） |
| `recall` | 上下文感知语义检索，任务类型路由 |
| `forget` | 遗忘记忆，创建幽灵痕迹以支持潜在模糊回忆 |
| `sleep` | 5阶段睡眠巩固——合并、修剪、模式形成 |
| `consolidate` | 微巩固——增量式、非阻塞 |
| `stats` | 记忆系统统计（锚点、边、幽灵、图式、认知健康） |
| `fuzzy_recall` | 幽灵痕迹低置信度召回（"我好像记得..."） |
| `get_profile` | 从累积记忆中推断用户画像 |
| `evolve` | 记忆进化周期（衰减、增强、冲突解决） |
| `save` | 持久化记忆图到磁盘（JSON） |
| `load` | 从磁盘加载记忆图（JSON） |

### Claude Desktop 配置

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc-mcp",
      "args": ["--storage", "/path/to/agent_memory.json"]
    }
  }
}
```

### OpenClaw 配置

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc-mcp",
      "args": ["--storage", "/path/to/agent_memory.json"]
    }
  }
}
```

## REST API 服务

```bash
# 启动 REST 服务
nwc-server --port 8420

# 或通过模块
python -m star_graph.server --port 8420
```

接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，含锚点/边数量 |
| GET | `/metrics` | Prometheus 格式指标 |
| GET | `/stats` | 完整记忆系统统计 |
| POST | `/remember` | 存储记忆 `{"text": "...", "tags": [...]}` |
| POST | `/recall` | 检索记忆 `{"query": "...", "max_items": 10}` |
| POST | `/sleep` | 运行睡眠巩固 |
| POST | `/consolidate` | 运行微巩固 |

## CLI

```bash
sg-add "讨论了微服务部署模式" --tags 架构 --emotional 0.6
sg-query "数据库连接池最佳实践"
sg-query --trace "Alice 什么时候去的夏威夷？"
sg-stats --schemas --ghosts
sg-sleep --retention 0.15 --edge-prune 0.1
nwc-export --output memories/ --organize both
nwc-forget <anchor_id> --certificate --reason gdpr_art17
nwc-verify certificates/forget-<id>.jws
```

## 运行测试

```bash
pip install pytest pytest-cov
pytest tests/ -v

# 含覆盖率报告
pytest tests/ --cov=star_graph --cov-report=term
```

**状态：** 阶段 1-6 完成，112 模块，v1.0.8。

## 路线图

详见 [ROADMAP.md](ROADMAP.md)。

## 许可证

MIT
