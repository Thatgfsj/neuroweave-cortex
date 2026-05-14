# NeuroWeave Cortex (NWC) · 神经编织皮层

海马体启发的 AI 认知记忆运行时系统。不是向量数据库，不是图数据库，而是一个**记忆运行时**——它像生物记忆一样：记忆、遗忘、强化、连接、抽象和演化。

## 与向量数据库/图数据库的区别

| 能力 | 向量数据库 | 图数据库 | NeuroWeave Cortex |
|---|---|---|---|
| 语义检索 | ✓ | ✗ | ✓ |
| 图遍历 | ✗ | ✓ | ✓ |
| 自动遗忘（生存衰减） | ✗ | ✗ | ✓ |
| 记忆强化（复述） | ✗ | ✗ | ✓ |
| 冲突检测（矛盾边） | ✗ | ✗ | ✓ |
| 模糊回忆（"我好像记得..."） | ✗ | ✗ | ✓ |
| 涌现抽象（模式发现） | ✗ | ✗ | ✓ |
| 时间上下文（TimeSpine 索引） | ✗ | ✗ | ✓ |
| 8阶段睡眠巩固 | ✗ | ✗ | ✓ |
| 幽灵复活（节省效应） | ✗ | ✗ | ✓ |
| 自传体自我模型 | ✗ | ✗ | ✓ |

## 快速开始

```python
from star_graph import MemoryManager
from star_graph.scheduler import AgentContext

# 一行初始化
mgr = MemoryManager()

# 记忆
mgr.remember("用户在排查 Redis 连接超时——连接池从10调到20后修复",
             tags=["redis", "debug", "timeout"])
mgr.remember("用户熟悉 Python asyncio 异步编程",
             tags=["python", "knowledge"])
mgr.remember("用户偏好类型标注和简洁代码",
             tags=["preference", "style"])

# 工作记忆——快速、临时的活跃上下文缓冲区
mgr.remember_working("正在调试 auth 中间件超时问题",
                     tags=["debug", "auth"])

# 上下文感知检索
ctx = AgentContext(task_type="debugging", active_goals=["fix Redis connection"])
memories = mgr.recall("Redis 连接池配置", context=ctx)
print(memories.memory_summary)

# System-2 深度检索——用于穷举或低置信度查询
memories = mgr.dual_recall("列出所有 Redis 相关的问题", context=ctx)

# 微巩固——增量、非阻塞
mgr.micro_consolidate()

# 睡眠——8阶段巩固
report = mgr.sleep()
print(report)

# 持久化
mgr.save("agent_memory.json")
mgr.load("agent_memory.json")
```

## 架构

三层设计，每层仅依赖下一层：

```
第三层：行为层    │  皮层路由、记忆门控、工作记忆、
    (cortex.py,   │  双通道检索、自适应复述、自传体记忆
     router.py,   │  "此刻应该回忆什么，以什么细节层次？"
     gate.py,     │
     working_memory.py,│
     scheduler.py,│
     autobiography.py)│
                  │
第二层：认知层    │  Hub抽象、级联回忆、TimeSpine时间索引、
    (retriever.py,│  睡眠巩固、演化、幽灵复活、
     sleep.py,    │  抽象、社区检测、竞争
     evolution.py,│  "记忆如何连接、强化和消退？"
     ghost.py,    │
     abstraction.py,│
     community.py,│
     competition.py,│
     timespine.py,│
     cascade.py,  │
     hub.py)      │
                  │
第一层：存储层    │  CRUD、持久化、ANN索引、分层存储、
    (graph.py,    │  BM25关键词索引、精确匹配缓存
     anchor.py,   │  "这个记忆存在哪里？"
     storage.py,  │
     sqlite_storage.py,│
     index.py,    │
     bm25.py,     │
     exact_cache.py,│
     tiered.py)   │
```

### 核心模块

| 模块 | 功能 |
|---|---|
| `manager.py` | 高层门面 API — `remember()`, `recall()`, `sleep()`, `save()` |
| `runtime.py` | 依赖容器 — 管理所有子系统生命周期 |
| `retrieval_pipeline.py` | 5层维度降级检索 (L0→L4)，自动降级回退 |
| `scheduler.py` | 上下文感知检索，记忆类型选择 |
| `working_memory.py` | 短期缓冲区 (15条, 1h TTL)，自动晋升至长期记忆 |
| `sleep.py` | 8阶段睡眠：N1_Replay → N2_Merge → N3_Compression → N3b_AtomFacts → REM_Emotion → N4_Prune → N5_HubConnect → N6_IndexRebuild |
| `evolution.py` | 基于生存函数的衰减 (艾宾浩斯/幂律/指数)、信念变迁、干扰 |
| `retriever.py` | HybridFusion + OscillationResonance + VectorSimilarity + 个性化PageRank + 可解释评分 |
| `dual_channel.py` | System-1 (快速) + System-2 (深度) 双通道检索，自动触发 |
| `bm25.py` | 稀疏关键词检索 (BM25)，与稠密向量做倒数秩融合 |
| `ghost.py` | 潜在记忆痕迹，支持模糊回忆和矛盾追踪 (NegativeGhost) |
| `abstraction.py` | 从锚点簇中涌现类别发现 |
| `community.py` | Louvain 社区检测，质心路由 |
| `anchor.py` | 记忆单元：6状态生命周期，10维 AnchorVector，乘法留存 |
| `graph.py` | 星图：RichEdge（时序、因果、状态转换）、Schema、ReflectionNode |
| `timespine.py` | 时间索引：O(days×buckets) 时间范围检索 |
| `cascade.py` | 因果链遍历，跨连接记忆序列 |
| `hub.py` | 分层 hub-and-spoke 抽象 (leaf→domain→global) |
| `cortex.py` | 分区记忆皮层，独立睡眠和检索 |
| `exact_cache.py` | 确定性 O(1) 实体键查找 |
| `tiered.py` | HOT/WARM/COLD 分层存储，透明磁盘卸载 |
| `autobiography.py` | 自我叙事形成与自传体记忆 |
| `atom_facts.py` | LLM 驱动的原子事实提取 |
| `survival.py` | 可插拔生存函数（艾宾浩斯、幂律、指数、自定义） |
| `compression.py` | 多层次会话压缩（episodic/strategic/meta） |
| `resonance.py` | 相位锁定振荡谐振，用于时序一致检索 |
| `streaming.py` | 流式记忆缓冲区（带背压控制） |
| `benchmark.py` | 内置基准测试套件（5个类别） |
| `config.py` | 集中 YAML 配置，schema 验证，点路径访问/覆盖 |

## 记忆生命周期

每个锚点经历6个状态：

```
ACTIVE → REHEARSING → CONSOLIDATING → DORMANT → GHOST → REACTIVATED
```

- **Active（活跃）**：刚刚创建或最近被访问——完全可塑，易于更新
- **Rehearsing（复述中）**：睡眠期间被重放——临时提升重要性
- **Consolidating（巩固中）**：从海马体向皮层转移——稳定性增加
- **Dormant（休眠）**：稳定、低活动——只读，皮层检索
- **Ghost（幽灵）**：被剪枝但保留压缩痕迹——可部分回忆或完全复活
- **Reactivated（再激活）**：幽灵被新的相关经历复活——稳定性降低，可塑性高

配套 **ThermalState**（HOT → WARM → COLD → DEAD）用于存储介质切换：
- HOT：内存中，完全可访问
- WARM：内存中，定期刷新到磁盘
- COLD：仅磁盘，访问时透明解冻

## 睡眠巩固

睡眠不是清理，而是**改变图结构**：

1. **N1_Replay** — 通过 SWR 评分优先重放高惊喜度、高情绪记忆
2. **N2_Merge** — 融合近重复锚点（ANN 加速，O(n×k)），桥接星座
3. **N3_Compression** — 海马体→皮层转移，形成图式
4. **N3b_AtomFacts** — LLM 从压缩簇中提取原子事实
5. **REM_Emotion** — 剥离已巩固记忆的情绪负载
6. **N4_Prune** — 移除弱锚点/边，创建幽灵痕迹（节省效应）
7. **N5_HubConnect** — 跨皮层 Hub 桥接
8. **N6_IndexRebuild** — 刷新 ANN、BM25 和社区索引

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
git clone https://github.com/Thatgfsj/neuroweave-cortex.git
cd neuroweave-cortex

# 可编辑模式安装
pip install -e .

# 可选：SQLite 存储后端
pip install aiosqlite

# 运行演示
python examples/emergence_demo.py
```

**注意：** NeuroWeave Cortex 未发布到 PyPI，请从源码安装。

## CLI

```bash
sg-add "讨论了微服务部署模式" --tags 架构 --emotional 0.6
sg-query "数据库连接池最佳实践"
sg-query --trace "用户住在哪里？"
sg-stats --schemas --ghosts
sg-sleep --retention 0.15 --edge-prune 0.1
```

## 运行测试

```bash
pip install pytest
pytest tests/ -v
```

## 许可证

MIT
