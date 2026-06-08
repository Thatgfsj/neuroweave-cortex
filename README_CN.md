# NeuroWeave Cortex (NWC) · 星图记忆引擎

不是向量数据库。不是图数据库。不是 RAG。甚至不是"LLM 驱动的记忆系统"。

一个**自组织的认知记忆**——记忆的结构由**使用决定，不由提示词决定**。
LLM 可以读取记忆。LLM 可以维护记忆（摘要、合并、归档）。
但 LLM **不决定**检索路径。

```
v1.3.2 | 138 模块 | 永不遗忘 | 认知记忆引擎
```

---

## 核心设计原则

> **记忆结构由记忆系统决定，不由 LLM 决定。**

亮度 = f(历史访问频率, 近因性, 边强度)
**不是** f(LLM 在查询时生成的向量)

这是"学习型记忆系统"和"每次重新计算型系统"的根本区别。

---

## 与向量数据库/图数据库的区别

| 能力 | 向量数据库 | 图数据库 | NWC |
|------|:---------:|:--------:|:---:|
| 语义检索 | ✅ | ❌ | ✅ |
| 图遍历 | ❌ | ✅ | ✅ |
| **激活图检索** | ❌ | ❌ | ✅ |
| **边衰减（真正的遗忘）** | ❌ | ❌ | ✅ |
| **记忆生命周期 L1→L2→L3** | ❌ | ❌ | ✅ |
| **无 LLM 的检索路由** | ✅ | ✅ | ✅ |
| **LLM 仅负责维护** | ❌ | ❌ | ✅ |
| 睡眠整合 | ❌ | ❌ | ✅ |
| 来源信任评分 | ❌ | ❌ | ✅ |

---

## 记忆分层（L0–L3）

```
L0: 输入层    ─── 仅存在于会话中（分钟级）
    ↓ 重复访问 >3次 或 存在 >30天 → 晋升
L1: 工作记忆层 ─── 最近活跃记忆（小时-天）
    │              LLM 维护（新增/合并/摘要）
    │              系统路由（检索不依赖 LLM）
    ↓ 巩固
L2: 长期记忆层 ─── 稳定的事实、经验、关系（月-年）
    │              只允许：强化、补充、降权
    │              不允许：完全重建
    ↓ 90天未访问或重要度低 → 存档
L3: 存档层    ─── 压缩摘要，不参与实时检索
    ↑ 查询命中摘要 → 重新激活
```

---

## 快速开始

```bash
pip install NWcortex
```

```python
from star_graph import MemoryManager, AgentContext

mgr = MemoryManager()
mgr.remember("用户偏好类型标注和简洁代码", tags=["preference"])
mgr.remember("Redis 连接池从10调到20后修复", tags=["redis","debug"])

# 无 LLM 的激活扩散检索
ctx = AgentContext(task_type="debugging")
memories = mgr.recall("Redis 连接池配置", context=ctx)

# 睡眠巩固：合并、层迁移、衰减弱边
report = mgr.sleep()
mgr.save("agent_memory.db")
```

---

## 安装选项

| 命令 | 包含 |
|------|------|
| `pip install NWcortex` | 核心引擎（138 模块） |
| `pip install "NWcortex[embeddings]"` | + sentence-transformers |
| `pip install "NWcortex[mcp]"` | + MCP 服务 |
| `pip install "NWcortex[all]"` | 全部 |

---

## 许可证

MIT
