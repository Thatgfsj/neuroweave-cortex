# 星图记忆 · Star Graph Memory

海马体启发的 AI 长期记忆系统。

对话被压缩为**锚点**（建议 ≤200 字），连接成可导航的**星图**。检索使用**振荡相位锁定谐振**而非关键词搜索。记忆巩固通过**9阶段睡眠周期**完成。

## 理论

系统基于七大神经科学原理：

| 原理 | 机制 |
|------|------|
| **尖波涟漪 (SWR)** | 睡眠期间对高优先级近期记忆进行压缩重放（~15×） |
| **记忆再巩固** | 预测误差门控：确认 / 更新 / 新建 |
| **相位进动** | θ-γ 振荡谐振将序列编码为相位偏移 |
| **图式形成** | 跨事件的共同模式抽象（类比内侧前额叶皮层） |
| **预测编码** | 自由能最小化统一编码、检索与睡眠 |
| **情绪调节** | 双因子标记（去甲肾上腺素+皮质醇）增强编码；REM 剥离情绪 |
| **自适应遗忘** | 精度加权衰减；幽灵锚点实现再学习的节省效应 |

## 架构

```
                  ┌─────────────────────────┐
                  │     新对话 / 查询         │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │   谐振引擎                │
                  │   · 海马通路：振荡锁相    │
                  │   · 皮层通路：直接查表    │
                  │   · 预测误差最小化        │
                  └───────────┬─────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
   ┌────────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
   │  确认          │  │  更新         │  │  新记忆       │
   │  强化锚点      │  │  再巩固       │  │  创建锚点     │
   └────────┬──────┘  └──────┬──────┘  └───────┬──────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │   星图                    │
                  │   · 锚点 + 振荡器         │
                  │   · 幽灵 (节省效应)       │
                  │   · 图式                 │
                  │   · 皮层索引             │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │   9阶段睡眠引擎           │
                  │   1. SWR 重放            │
                  │   2. 系统巩固            │
                  │   3. 情绪剥离            │
                  │   4. 图式提取            │
                  │   5. 合并相似            │
                  │   6. 自适应剪枝+幽灵     │
                  │   7. 桥接星座            │
                  │   8. 赫布更新            │
                  │   9. 突触稳态            │
                  └─────────────────────────┘
```

## 安装

```bash
pip install star-graph-memory
# 或从源码安装：
git clone https://github.com/Thatgfsj/star-graph-memory.git
cd star-graph-memory
pip install -e .
```

## 快速开始

```python
from star_graph import StarGraph, Anchor
from star_graph.sleep import SleepCycle
from star_graph.retriever import OscillationResonanceRetriever
from star_graph.online import OnlineConsolidator
from star_graph.storage import Storage

graph = StarGraph()
online = OnlineConsolidator(graph, interval=5)

# 添加记忆
memories = [
    ("用户偏好 Python 做爬虫，Rust 做大型项目", ["技术栈"], 0.5),
    ("用户住在北京朝阳区，通勤约40分钟", ["个人信息", "位置"], 0.3),
    ("用户正在学习日语，每天练习30分钟", ["学习"], 0.4),
    ("用户喜欢意大利菜，尤其是碳水面", ["食物"], 0.6),
]
for text, tags, emotion in memories:
    anchor = Anchor.create(text, tags=tags, emotional_valence=emotion)
    graph.add_anchor(anchor)
    online.record_interaction(anchor)

# 谐振检索
osc = OscillationResonanceRetriever(graph)
result = osc.retrieve("用户住在哪里？")
for c in result.constellations:
    for a in c.anchors:
        print(f"[{a.retention_score:.2f}] {a.text}")

# 夜间深度睡眠
cycle = SleepCycle(graph)
result = cycle.run()
print(f"合并: {result['merged']}, 剪枝: {result['pruned_anchors']}")
print(f"幽灵: {result['ghosts_created']}, 图式: {result['schemas_formed']}")

# 持久化
store = Storage()
store.save(graph)
```

## 双通路检索

```python
from star_graph.retriever import (
    OscillationResonanceRetriever,  # 创新：相位锁定谐振
    VectorSimilarityRetriever,      # 基线：余弦相似度
    compare_retrievers,             # 对比工具
)

osc_ret = OscillationResonanceRetriever(graph)
vec_ret = VectorSimilarityRetriever(graph)

# 对比两种检索器
comparisons = compare_retrievers(graph, [
    "用户喜欢什么食物？",
    "用户住在哪里？",
])
for c in comparisons:
    print(f"向量相似度: {c['vector_similarity']['top_score']:.3f}")
    print(f"振荡谐振:   {c['oscillation_resonance']['top_score']:.3f}")
```

## 在线微巩固

无需等到凌晨2点。每 N 次交互后触发轻量睡眠（<50ms）：

```python
online = OnlineConsolidator(graph, interval=5)  # 每5次交互后微巩固
online.record_interaction(anchor)  # 自动触发
```

三种模式：
- `online` — 仅微巩固（SWR重放 + 赫布更新）
- `nightly` — 完整9阶段深度睡眠
- `hybrid` — 在线微巩固 + 夜间深度睡眠

## CLI

```bash
sg-add "用户偏好 Python 做爬虫" --tags 技术栈 --emotional 0.5
sg-query "用户喜欢什么编程语言？"
sg-stats --schemas --ghosts
sg-sleep --retention 0.15 --edge-prune 0.1
```

## 运行示例

```bash
python examples/memory_basic.py
```

预期输出：添加10条记忆 → 查询"住在哪里？" → 双检索器对比 → 睡眠周期 → 再次查询 → 保存

## 运行测试

```bash
pip install pytest
pytest tests/ -v
```

## 睡眠守护进程

```powershell
# Windows：安装每日凌晨2点自动运行
powershell -ExecutionPolicy Bypass -File scripts/install_sleep_task.ps1
```

```bash
# 空闲检测模式（用户离开15分钟后触发）
python scripts/sleep_daemon.py --mode idle --idle-threshold 15
```

## 路线图

| 季度 | 目标 |
|------|------|
| 2026 Q2 | v0.3.0 可运行原型 + 对比基准 |
| 2026 Q3 | 完整9阶段睡眠 + 预印本 |
| 2026 Q4 | LangChain/LlamaIndex 集成 |
| 2027 Q1 | v1.0.0 生产候选版 |

## 更深入的阅读

- `docs/research.md` — 完整理论框架及文献引用
- `docs/neuro_mapping.md` — 每条神经科学原理到算法的详细映射
- `docs/benchmark.md` — 与 Mem0/HippoRAG 的对比基准

## 开放协作

欢迎神经科学、认知科学、AI 研究者参与讨论。如有更合理的生物-算法映射建议，请在 Issue 中提出。

## 许可证

MIT
