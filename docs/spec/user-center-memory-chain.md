# 用户中心记忆链（User-Center Memory Chain）规范

> **状态**：v1.4.3 参考实现（Issue #11）。本版交付**规范 + 独立参考实现 + 测试**，
> **不接入**现有 Activation / UM 默认检索路径。
>
> **用户确认（2026-11-04）**：四维基础权重不变；记忆每次任务只调用一次，因此第 4 维
> 内部按 **本次任务/回复中心 85% + 长期技术中心 15%** 融合（见 §2.4、§3.1）；本版仍不接入
> 默认检索（见 §8 决策记录）。
>
> **权威顺序**：本 spec 属 [DOC_POLICY.md](../engineering/DOC_POLICY.md) 第 2 层权威，
> 位于 [README.md](../../README.md) 之下。本 spec 与 README 冲突时**以 README 为准**，
> 并在对应 Issue 记录待修一方；其它文档与本 spec 冲突时以本 spec 为准。

## 1. 背景与范围

用户给出的权威产品规范（原文）：

> 记忆文本的四个向量：
> 1. 相近文本接近度
> 2. 相关技术接近度
> 3. 相关事件接近度
> 4. 与用户中心接近度（技术中心、本次回复中心等等）
>
> 用户中心是核心的，会挂载最常用的一些记忆，比如常用 Python 啥的，数量可调。两个记忆文本
> 接近度高了，就继续添加下一个记忆，也就是记忆链，直到与用户要做的事情（4. 用户核心）
> 关联性 < x 的时候不再继续。

本版落地内容：

- 四维接近度的定义、公式、归一化方式与默认权重（可配置）；
- 用户中心（`UserCenter`）：容量可调、挂载权重、淘汰策略、中心向量；
- 记忆链（`build_memory_chain`）：以种子/用户中心起步，逐跳扩展并给出三类停止原因；
- 参考实现 [`star_graph/user_center.py`](../../star_graph/user_center.py) 与
  [`tests/test_user_center.py`](../../tests/test_user_center.py)。

**不在本版范围**：修改 `retrieval_core` / `runtime` / `Activation` 的默认行为；把记忆链
接进检索流水线；持久化 `UserCenter`；UI / CLI 参数。

设计约束（实现与评审共同遵守）：

1. 低耦合：只读锚点的 `id / text / embedding / tags / cortex_path / created_at /
   source_session`，不写回、不删除任何记忆（与 Never-Delete 一致）；
2. 纯标准库 + 仓库既有依赖，可独立测试；
3. 确定性：不使用随机数与时钟（`UserCenter` 的时钟可注入，便于测试淘汰）；
4. 优雅降级：空候选、空中心、无 embedding 都不抛错。

## 2. 四维接近度

实现中四个维度都是**函数**（也可用 `UserCenter` 上的同名方法），入参为
`star_graph.memory_core.anchor.Anchor` 或任意 duck-typed 对象（只要含所需字段）。
所有维度输出统一在 **[0, 1]**，越大越接近。

### 2.1 第 1 维：文本接近度 `text_proximity(a, b)`

$$
s_{text}(a,b)=
\begin{cases}
\max(0,\ \cos(\vec a,\vec b)), & \text{双方都有同维 embedding}\\
\dfrac{|T(a)\cap T(b)|}{|T(a)\cup T(b)|}, & \text{否则}
\end{cases}
$$

- `T(·)`：文本 token 集合。ASCII 按词（`[a-z0-9_+#]+`，小写），中日韩字符**按单字**切分。
- 余弦截断到 `[0, 1]`：负相关视为不相似（0），与 NWC 现有相似度阈值语义一致。
- 维度不一致、零向量、空文本：退化为 Jaccard；任一方 token 为空则返回 `0.0`。

### 2.2 第 2 维：技术接近度 `tech_proximity(a, b)`

先构造技术签名（token → 权重）：

| token 形式 | 来源 | 权重（默认，可配置） |
|---|---|---|
| `tag:<tag>` | `tags` | `tech_tag_weight = 1.0` |
| `path:<seg>` | `cortex_path` 按 `. / > : 空白` 切段 | `tech_path_weight = 0.80` |
| `kw:<keyword>` | 技术关键词表在 tags / 路径 / 文本中命中 | `tech_keyword_weight = 0.60` |

相似度为加权 Jaccard：

$$
s_{tech}(a,b)=\frac{\sum_{t}\min(w_a(t),w_b(t))}{\sum_{t}\max(w_a(t),w_b(t))}
$$

- 关键词表 `tech_keywords` 可配置（默认含 `python`、`pytest`、`docker`、`向量` 等，见
  [`star_graph/defaults.yaml`](../../star_graph/defaults.yaml) 的 `user_center.tech_keywords`）。
- 单字关键词要求 token 精确命中，避免 `go` 命中 `good`；含空格或非 ASCII 的关键词按子串匹配。
- 任一方签名为空 → `0.0`。

### 2.3 第 3 维：事件接近度 `event_proximity(a, b)`

$$
s_{event}=\frac{w_t\cdot 2^{-\Delta d / H} + w_s\cdot [session_a=session_b\neq\varnothing] + w_g\cdot J_{event}}{w_t+w_s+w_g}
$$

| 子项 | 默认权重 | 说明 |
|---|---|---|
| 时间衰减 `w_t` | 0.60 | `Δd` 为 `created_at` 相差天数，`H = event_half_life_days = 7` |
| 同会话 `w_s` | 0.25 | `source_session` 相同且非空记 1，否则 0 |
| 事件标签 `w_g` | 0.15 | tags 中以 `event:` / `event/` / `evt:` / `事件:` 开头的标签做 Jaccard |

- 任一方缺少 `created_at`（<=0）时时间项为 0，其余子项照常。
- 分母为三个子权重之和，保证输出仍在 `[0, 1]`。

### 2.4 第 4 维：与用户中心接近度 `user_center_proximity(a, query, center, task_anchors)`

用户中心是核心，但**记忆每次任务只调用一次**，所以第 4 维由两个分量融合（用户确认
2026-11-04）：

$$
s_{task}(a)=\max\Big(\text{text/tech similarity}(a,\ \text{query}),\
\max_{t\in task\_anchors}\text{text/tech similarity}(a,t)\Big)
$$

$$
s_{tech}(a)= \max_{m\in mounts}\ \Big(\underbrace{\frac{w_m}{\max_j w_j}}_{\text{相对权重}}\cdot
\frac{w_{text}\,s_{text}(a,m)+w_{tech}\,s_{tech}(a,m)}{w_{text}+w_{tech}}\Big)
$$

$$
s_{center}(a)=
\begin{cases}
\dfrac{w_{task}\,s_{task}+w_{tech\_c}\,s_{tech}}{w_{task}+w_{tech\_c}}, & \text{两个分量都可用}\\[4pt]
s_{task}, & \text{只有本次任务中心可用}\\[2pt]
s_{tech}, & \text{只有长期技术中心可用}\\[2pt]
0, & \text{都不可用}
\end{cases}
\qquad w_{task}:w_{tech\_c}=0.85:0.15
$$

- **本次任务中心为主（默认 85%）**：`query`（文本/向量/锚点）与可选的 `task_anchors`
  （本次任务内一次性挂载的记忆）代表"用户现在要做的事"。
- **长期技术中心为辅（默认 15%）**：`center` 挂载的常用记忆（如常用 Python）提供跨会话
  偏好；事件维不参与，避免时间衰减系统性惩罚长期偏好。
- **单分量归一**：只有一侧可用时直接用该侧分数，不乘 0.85/0.15；两侧都可用才按 85:15
  融合；都不可用返回 `0.0`。比例可在 `user_center.center_split.*` 调整。
- **只取分量内最大值**：任务分量取 query / task_anchors 的最大相似度，技术分量取挂载记忆
  中"相对权重 × 相似度"的最大值——一条记忆只要贴近任一核心参照即算贴近。

### 2.5 综合接近度 `combined_proximity(a, b, center, query)`

$$
s_{combined}=\frac{w_{text}s_{text}+w_{tech}s_{tech}+w_{event}s_{event}+w_{center}s_{center}}
{w_{text}+w_{tech}+w_{event}+w_{center}}
$$

默认 `w_text / w_tech / w_event / w_center = 0.40 / 0.25 / 0.15 / 0.20`。**按权重和归一化**：
用户可任意改权重（不必凑成 1），输出仍落在 `[0, 1]`；把某维权重设为 0 即关闭该维。

维度分工：第 1–3 维刻画"两条记忆之间"的邻近，第 4 维刻画"记忆与用户当前任务/中心"的关联，
后者是记忆链的**停止依据**（见 §4）。

## 3. 用户中心

### 3.1 核心地位

用户中心是长期、跨会话的"核心记忆集合"，挂载最常用的记忆（常用 Python、常用工具链、
表达偏好等）。它同时承担两个职责：

1. 作为第 4 维的参照系——判断候选记忆是否仍与"用户要做的事"相关；
2. 作为记忆链的起点/种子来源之一（§4 允许以 `center.top(1)` 起步）。

**与"本次回复中心 / 技术中心"的关系（已确认）**：第 4 维分两层并融合——**本次任务/回复
中心 85%**（`query` + 可选的 `task_anchors`，不持久化，一次任务一次调用）与**长期技术中心
15%**（持久 `UserCenter` 挂载的常用记忆）。这对应"记忆只调用一次、本次任务为主、技术为辅"
的用户语义；实现见 §2.4。

### 3.2 挂载

| API | 语义 |
|---|---|
| `mount(anchor, weight=None) -> bool` | 挂载；已挂载则更新权重与 `last_used`；容量为 0 时返回 `False` |
| `unmount(anchor) -> bool` | 卸载（接受 id 或锚点），不存在返回 `False` |
| `is_mounted(anchor) -> bool`、`anchor in center` | 是否挂载 |
| `mounted_ids -> tuple[str, ...]` | 挂载顺序的 id 列表 |
| `weight_of(anchor) -> float \| None` | 挂载权重 |
| `top(n) -> list[Anchor]` | 按挂载打分排序的前 `n` 条 |
| `center_embedding(normalize=True) -> list[float] \| None` | 挂载向量按权重加权平均，默认归一化 |
| `capacity`（属性） | 读/写容量；**写入即生效**，缩小容量立即淘汰弱者 |

### 3.3 淘汰策略

挂载打分（也是 `top` 的排序分）：

$$
rank(m)=w_m\cdot 2^{-\,\text{idle}(m)/T_{1/2}},\qquad T_{1/2}=\texttt{eviction\_half\_life\_hours}=168
$$

- 超容量（含运行时缩小容量）时，淘汰 `rank` 最小者；同分先淘汰 `last_used` 更早者，
  再按 `id` 字典序——保证确定性。
- 新挂载先腾位再插入，因此**刚挂载的记忆不会被自己挤掉**。
- 事件/文本相似度不参与淘汰；如需"按访问频率淘汰"见 §8。

## 4. 记忆链算法

### 4.1 伪代码

```text
输入: seed, candidates, center, query, config
chain ← seed（可空）
current ← seed（为空则用 query 视图）
remaining ← 候选去重（保持传入顺序，去掉 seed）

loop:
    if len(chain 含 seed) >= max_chain_length: 停止(max_chain_length)
    if remaining 为空: 停止(candidates_exhausted)

    对每个候选 c 计算:
        s_text  = text_proximity(c, current)
        s_tech  = tech_proximity(c, current)
        s_event = event_proximity(c, current)
        s_center= user_center_proximity(c, query, center, task_anchors)
        combined= Σ w_i·s_i / Σ w_i
    best ← (combined, text) 最大者；同分取先出现者

    link ← (chain_gate == "text" ? s_text : combined)
    if link < chain_similarity_threshold: 停止(similarity_below_threshold)
    if 中心可用 且 s_center < stop_center_threshold: 停止(center_below_threshold)

    chain.append(best); current ← best; remaining.remove(best)
```

### 4.2 停止条件与优先级

| 原因常量 | 含义 | 触发点 |
|---|---|---|
| `similarity_below_threshold` | 相邻记忆相似度不足，链不该继续 | 选出的最佳候选 `link < chain_similarity_threshold` |
| `center_below_threshold` | 与用户当前任务（第 4 维）关联度 `< x` | `s_center < stop_center_threshold` |
| `max_chain_length` | 达到链最大长度（含种子） | 循环开头 |
| `candidates_exhausted` | 候选耗尽（正常收敛） | `remaining` 为空 |

- `chain_gate`：`"combined"`（默认，四维综合）或 `"text"`（仅文本接近度），用于无向量/
  短文本场景下调门槛语义。
- "中心可用"= `w_center > 0` 且（中心非空 **或** 给了 `query`/`task_anchors`）。都没有时
  第 4 维恒为 0，跳过该门槛，否则链会在第一步就停，与"空中心优雅降级"冲突。
- 记忆链在**一次任务调用内**构建完成；第 4 维使用 §2.4 的 85/15 融合分，因此停止阈值 x
  衡量的是"与本次任务中心的关联度"。
- 每次只追加**一个**最佳候选，然后以它为新的 `current` 重新打分——即"文本接近度高就继续
  加下一个记忆"的链式扩展。

### 4.3 确定性

- 候选按传入顺序遍历，同分取先出现者；输出与输入一一对应（换顺序可能换结果，但同输入必同输出）。
- 不使用随机数；链扩展不读时钟（`created_at` 来自锚点自身字段）。
- 浮点求和处对 token key 排序，避免 `set` 迭代顺序带来跨进程差异。

### 4.4 降级矩阵

| 缺失条件 | 行为 |
|---|---|
| 无 embedding（一侧或两侧） | 第 1 维退化为 token Jaccard |
| 无 tags / cortex_path / 关键词命中 | 第 2 维为 0 |
| 无 `created_at` / 无 session / 无事件标签 | 对应子项为 0，第 3 维仍可计算 |
| 无 query 且无 `task_anchors`、中心有挂载 | 第 4 维只用长期技术中心分量（归一，不乘 0.15） |
| 中心为空、有 `query` 或 `task_anchors` | 第 4 维只用本次任务中心分量（归一，不乘 0.85） |
| 中心为空、无 `query`/`task_anchors` | 第 4 维为 0，跳过第 4 维门槛 |
| 候选为空 / 种子为空 | 返回仅含种子的链并给出停止原因，不抛错 |

## 5. 配置

配置段位于 [`star_graph/defaults.yaml`](../../star_graph/defaults.yaml) 的 `user_center`，
由 `UserCenterConfig.from_config()` 读取（缺省键回退 dataclass 默认值）；也支持
`from_config({"user_center": {...}})` 与构造参数覆盖。

| 键（`user_center.` 下） | 默认 | 含义 |
|---|---|---|
| `capacity` | 32 | 用户中心挂载容量 |
| `mount_weight_default` | 1.0 | 未显式给权重时的挂载权重 |
| `weights.text / tech / event / center` | 0.40 / 0.25 / 0.15 / 0.20 | 四维权重（按和归一化） |
| `center_split.task_weight / tech_weight` | 0.85 / 0.15 | 第 4 维内部：本次任务中心 / 长期技术中心 |
| `thresholds.chain_similarity` | 0.70 | 继续成链的门槛 |
| `thresholds.stop_center` | 0.35 | x：第 4 维低于此值停止 |
| `max_chain_length` | 12 | 链最大节点数（含种子） |
| `chain_gate` | `"combined"` | 门槛用综合分还是文本分（`"text"`） |
| `eviction_half_life_hours` | 168.0 | 淘汰打分的时间衰减半衰期 |
| `event.time_weight / session_weight / tag_weight` | 0.60 / 0.25 / 0.15 | 第 3 维子权重 |
| `event.half_life_days` | 7.0 | 事件时间衰减半衰期（天） |
| `tech.tag_weight / path_weight / keyword_weight` | 1.0 / 0.80 / 0.60 | 第 2 维命中权重 |
| `tech_keywords` | 中英混合技术词表（见 defaults.yaml） | 技术关键词表 |

越界值在构造时夹紧：容量/长度取非负整数，权重非负，两个阈值夹到 `[0, 1]`，非法
`chain_gate` 回退 `"combined"`。

自定义示例（`nwc.yaml`）：

```yaml
user_center:
  capacity: 64
  weights: {text: 0.50, tech: 0.20, event: 0.10, center: 0.20}
  center_split: {task_weight: 0.85, tech_weight: 0.15}   # 本次任务中心 : 技术中心
  thresholds: {chain_similarity: 0.62, stop_center: 0.30}
  max_chain_length: 20
```

### 5.1 调用示例

```python
from star_graph import Anchor, UserCenter, UserCenterConfig, build_memory_chain

center = UserCenter(UserCenterConfig(capacity=16))
center.mount(Anchor.create("用户常用 Python + pytest", tags=["python", "pytest"]), weight=2.0)
center.mount(Anchor.create("偏好函数式写法", tags=["style"]), weight=1.0)
assert center.is_mounted(center.mounted_ids[0])
print([a.text for a in center.top(2)])
print(center.center_embedding())            # 无向量时为 None

task_memory = Anchor.create("本次任务：修复 pytest 用例", tags=["pytest", "debug"])
chain = build_memory_chain(seed, candidates, center=center,
                           query="帮我改 pytest 用例", task_anchors=[task_memory])
for step in chain.steps:
    print(step.anchor_id, step.scores())    # text/tech/event/center/combined
print(chain.stop_reason, chain.node_ids())
print(chain.to_dict())                      # JSON 友好，便于日志/回放
```

## 6. 边界与不变量

- **默认检索路径零改动**：`retrieval_core`、`runtime`、`Activation` 未引用本模块；星图
  `defaults.yaml` 只新增 `user_center` 段（版本已由 Issue #5 收敛为 `_version.py` 单一源）。
- **只读锚点**：不写回重要性/激活度，不增删边，不删除记忆。
- **复杂度**：每步 `O(候选数 × 挂载数)`，受 `max_chain_length` 与 `capacity` 约束；
  纯 Python，无 ANN 依赖。
- **持久化不在本版范围**：`MemoryChain.to_dict()` 可用于日志/回放；`UserCenter` 未提供
  `to_dict/from_dict`，避免在语义未定时固化存储格式。
- **文档关系**：本 spec 与 [README.md](../../README.md) 同为第 2 层权威（见
  [ENGINEERING.md](../../ENGINEERING.md) R9 与 [DOC_POLICY.md](../engineering/DOC_POLICY.md) §1）。

## 7. 测试

[`tests/test_user_center.py`](../../tests/test_user_center.py) 覆盖：

- 容量默认值与运行时调整、容量 0 拒绝挂载；
- 按权重 + 最近使用淘汰、`top` 排序、`center_embedding` 加权平均与空/无向量；
- 配置读取（`defaults.yaml`）、覆盖、`as_dict` 往返、越界夹紧、权重可配置；
- 四维打分的边界（0 / 1）、单调性、Jaccard 回退、路径/关键词/会话/标签子项；
- 第 4 维 85/15 融合、单分量归一、可配置比例、`task_anchors` 扩展本次任务中心；
- 链增长、三类停止条件、第 4 维使用融合分的停止判断、`chain_gate="text"`、空候选/空种子/
  无 embedding 降级、确定性；
- `MemoryChain.to_dict()` 的 JSON 兼容性；
- 子进程验证 `import star_graph` 不会 eager 加载 `star_graph.user_center`。

```bash
python -m pytest tests/test_user_center.py -q
python -m pytest tests/ -q
```

## 8. 用户决策记录与开放问题

### 8.1 用户决策（2026-11-04）

| 问题 | 决策 |
|---|---|
| 四维基础权重是否调整 | **不调整**：维持 `0.40 / 0.25 / 0.15 / 0.20`。 |
| 用户中心分层 | **本次任务中心 85% + 长期技术中心 15%**（记忆每次任务只调用一次，故本次任务为主）。 |
| 是否接入默认检索 | **本版不接入**，保持参考实现；后续如需再开 Issue。 |
| 版本号规则 | 确认"每次迭代 PATCH +0.0.1"（1.4.2 → 1.4.3）。 |

### 8.2 仍开放的问题


1. ~~权重默认值~~（已确认不动；如需调整改 `user_center.weights.*` 即可，无需改代码）。
2. **x = 0.35 的语义**：当前是 `[0, 1]` 绝对分，且挂载权重做了相对归一。若希望 x 表示
   "相对本次任务中心的比例"，需要引入任务中心向量并改为相对阈值。
3. **`chain_similarity_threshold = 0.70` 偏严**：无 embedding 时 token Jaccard 常低于该值，
   链会很快停止。可下调阈值，或改用 `chain_gate: "text"` 并配更低阈值；是否按"有无向量"
   分档默认值待确认。
4. ~~中心分层~~（已确认：拆为本次任务中心 85% + 长期技术中心 15%，见 §2.4）。
5. **淘汰策略**：是否把访问频率（激活次数）纳入 `rank`，而不只是权重 + 最近使用。
6. **多模态/维度不一致**：`center_embedding` 目前跳过维度不一致的向量；是否需要投影或
   分维度聚合。
7. ~~接入方式~~（用户确认本版不接入默认检索；若后续接入，再决定预过滤/重排与 Runtime 流程）。
8. **持久化**：`UserCenter` 的存储格式、快照位置与迁移策略。
