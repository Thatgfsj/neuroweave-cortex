# A/B 实验：重要性权重 & 检索公式

> 目的：用实测对比决定 (1) 重要性权重用「代码默认」还是「plan.md 旧设计」、
> (2) 检索公式用「HybridFusion」还是「plan.md 五因子」，保留更优者。
>
> **约束**：本目录的 A/B 基建只新增文件，不修改任何已存在的源码
>（`star_graph/`、`nwc/` 下零改动）。所有变量注入均通过公开的子类 / 构造参数完成。

## 文件

| 文件 | 说明 |
|---|---|
| `ab_experiment.py` | 实验核心：配置预设、novelty 注入引擎、五因子检索器、评测与出表 |
| `run_ab_importance.py` | 实验一入口（重要性权重 W-code / W-plan / W-code+nov） |
| `run_ab_retrieval.py` | 实验二入口（检索公式 R-hybrid / R-plan） |
| `ab_results/*.json` | 每次运行的完整指标 + markdown 表格（自动生成） |

## 两组实验

### 实验一：重要性权重（作用于写入门控）

每条记忆先过被测配置的 `ImportanceEngine.evaluate()`：`should_store` 决定是否写入，
`score` 写入 `anchor.vector.importance`。评测阶段三组臂统一使用**固定不变的**
默认 HybridFusionRetriever，因此指标差异完全归因于写入侧配置。

| 配置 | 权重 |
|---|---|
| `W-code`（现行默认） | emotion=.18 repetition=.22 goal=.28 retrieval_feedback=.12 richness=.12 future=.08 |
| `W-plan`（plan.md 旧设计） | emotion=.25 repetition=.30 goal=.20 **novelty=.15** future=.10（无 retrieval_feedback / richness 项） |
| `W-code+nov` | 代码默认六项 ×0.90 + novelty=.10：emotion=.162 repetition=.198 goal=.252 retrieval_feedback=.108 richness=.108 future=.072 novelty=.100（**合计=1.00**） |

`W-code+nov` 的归一化方案：novelty 权重取 `n` 时，六个基础权重同乘 `(1-n)`，
保持权重向量为凸组合（n=0.10 时即上表数值）。改 `ab_experiment._w_code_plus_nov(n)` 可换 n。

### 实验二：检索公式（作用于 recall 排序）

**写入阶段所有臂完全共享**：一份 clean 环境、一张图、代码默认重要性引擎写入一次。
两个检索器对图都是只读（已核对 `HybridFusionRetriever.retrieve`、
`personalized_pagerank`、`graph.spread_activation` 均无锚点突变），因此在同一张图上
先后评测消除了写入 nondeterminism。

| 配置 | 公式 |
|---|---|
| `R-hybrid`（现行默认） | `HybridFusionRetriever` 原类原参：α=.50 β=.12 γ=.18 δ=.08 ε=.12 |
| `R-plan`（plan.md 五因子） | `final = semantic×.35 + temporal×.20 + importance×.25 + relationship_strength×.10 + recall_frequency×.10` |

`R-plan` 因子取值（全部来自 anchor/vector 现有字段，实现见 `PlanFiveFactorRetriever`）：

- `semantic` = cosine(query_emb, `anchor.embedding`)（无向量时退化为文本 bigram 重叠）
- `temporal` = `exp(-hours_since(anchor.last_activated_at)/168)`（与 HybridFusion 相同的 1 周半衰常数）
- `importance` = `anchor.vector.importance`
- `relationship_strength` = `graph.neighbors(aid)` 边 traversal_weight 均值（无边=0）
- `recall_frequency` = `anchor.vector.frequency`

## novelty 信号注入方式（不改 importance_engine.py）

按 plan.md 定义「新话题 vs 已存储知识的重叠度」：`novelty = 1 − 重叠率`。

- **重叠率**：当前文本关键词集合 K 与每个已存储锚点关键词集合 A 的最大覆盖比例
  `max_anchor |K ∩ A| / |K|`。取 max 而非并集：语义是"该话题已被某条记忆覆盖"，
  且避免大语料下并集饱和。
- **关键词集合**（`extract_keywords`，中英通用）：拉丁词元（len>2，去停用词）+ 中文字符 bigram。
- **锚点侧字段**：`anchor.tags` + `anchor.exact_match_keys`（写入时
  `extract_entity_keys` 收割的实体键）+ `anchor.text` 词元，三者并集。
- **注入实现**（`NoveltyImportanceEngine` 等，全部是子类）：
  1. `NoveltyImportanceWeights(ImportanceWeights)` 增加 novelty 权重字段；
  2. `NoveltyImportanceSignal(ImportanceSignal)` 增加 `novelty_signal` 字段并重写
     `compute()` = 基础六信号加权和 + `novelty_signal × weights.novelty`
     （`ImportanceSignal` 是无 novelty 字段的 dataclass，故按此扩展）；
  3. `NoveltyImportanceEngine(ImportanceEngine)` 重写 `evaluate()`：先
     `super().evaluate()`（情绪/复现/目标/richness 信号、topic tracker 更新、噪声
     强制丢弃逻辑全部原样保留），再用 Novelty 信号与被测权重**重算** score/level/
     should_store/decay_rate/consolidation_priority。基础噪声短路（composite<0.08
     且无实体/术语 → 0.05/DISCARD）刻意保留，保证各臂噪声门一致、不引入混杂变量。
  4. 引擎持有 `graph` 引用，`evaluate` 时动态读取 `graph.anchors` 作为"已知知识"
     （evaluate 先于 remember，当前文本天然不在集合内）。

## 评测协议

流程（每组配置）：clean 环境（`tempfile.mkdtemp` 私有目录作 storage path，全新
`MemoryManager`，无持久化泄漏）→ 批量写入（实验一的 W 配置在此生效）→ 逐查询
retrieve（实验二的 R 配置在此生效）→ 统计指标。所有随机性来自固定种子
`random.Random(seed)`；合成语料的 anchor id 由 `blake2b(text+source_session)`
决定，可复现。臂间 embedder 状态隔离：tfidf/hash 后端每臂重置
（词表状态不跨臂泄漏）；sentence-transformers 后端无状态、保留实例避免重复加载。

注意：`MemoryManager.remember` 内置 write gate 默认关闭
（`defaults.yaml: write_gate.enabled: false`），被测 ImportanceEngine 是唯一门控，
无第二道门的混杂。

### 指标

- `hit@1 / hit@3 / hit@5`：答案锚点 id 落在 top-k 内的查询比例
- `MRR`：首个命中排名倒数的均值（未命中记 0，深度=--top-k）
- `has_answer`：检索结果（top-`--top-k` 窗口）∩ 相关 ≠ ∅ 的查询比例
  （单答案查询下数值上等于 hit@top-k；LoCoMo 多相关锚点时二者分离）
- 另记录：存储量/总量、平均重要度、平均 novelty（实验一）、平均召回耗时

### 数据模式

- **LoCoMo 模式**：优先探测 `C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json`
  （`run_locomo_standard.py` 的默认路径）及 `benchmarks/data/` 候选路径。每轮对话
  turn 存为一条记忆；QA 的相关锚点按"答案文本被记忆文本包含"做近似匹配（README
  已声明为 pragmatic approximation；无 evidence span 精确对齐）。只取前 3 会话子集。
- **合成模式**（LoCoMo 不可得时自动降级，`--mode synthetic` 强制）：固定词表
  （人物/物品/地点/主题）模板生成 N 条事实型记忆 + M 条噪声闲聊，查询为事实的
  改写（共享区分性词元），标准答案 = 源事实锚点 id。seed 固定可复现。

## 运行命令

```bash
# Smoke test（管线验证，不出结论）：30 条记忆 10 查询
python benchmarks/run_ab_retrieval.py --smoke
python benchmarks/run_ab_importance.py --smoke

# 正式跑分（合成 200 事实 + 100 噪声，50 查询，seed=42）
python benchmarks/run_ab_importance.py --mode synthetic --facts 200 --noise 100 --queries 50 --seed 42
python benchmarks/run_ab_retrieval.py --mode synthetic --facts 200 --noise 100 --queries 50 --seed 42

# LoCoMo 本地可得时（自动探测；--mode auto 为默认行为）
python benchmarks/run_ab_importance.py --mode locomo --locomo-sessions 3 --queries 50
python benchmarks/run_ab_retrieval.py --mode locomo --locomo-sessions 3 --queries 50

# 只跑部分臂 / 更大检索深度 / 换种子做稳健性
python benchmarks/run_ab_importance.py --arms W-code,W-code+nov --top-k 10
python benchmarks/run_ab_retrieval.py --top-k 20
python benchmarks/run_ab_importance.py --mode synthetic --seed 7
```

结果同时打印 markdown 表格并写入 `benchmarks/ab_results/*.json`（含每臂完整指标）。

### 嵌入模型说明

默认 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`（本模块自动设置）：有 HF 缓存则用
all-MiniLM-L6-v2，无缓存则快速降级 TF-IDF/hash，避免在线解析挂起。
要强制在线拉取模型：`AB_ONLINE_EMBEDDINGS=1 python benchmarks/run_ab_…`。
正式对比要求两臂使用同一后端（脚本已在同进程内保证）。

### 结果解读建议（供正式跑分后使用）

- 实验一：对比 hit@k / MRR / has_answer（检索质量）与存储量、平均重要度
  （门控行为）。若某配置靠多存噪声换指标，需同时看噪声进入率。
- 实验二：两臂存储完全一致，直接对比 hit@k / MRR / has_answer 与召回耗时。
- 建议至少跑 3 个 seed 取均值再下结论；本次 smoke 只验证管线，数据不作数。

## 已知接口事实（踩坑记录）

1. `MemoryManager.recall` 是多路 RRF 融合重型管线，不适合做单一公式 A/B；
   A/B 直接对 `manager.graph` 使用 `Retriever` 子类（与生产 HybridFusion 同一接口：
   `retrieve(query, embedding, top_k) -> RetrievalResult`）。
2. `RetrievalResult.constellations` 是 `Constellation` 列表（每个含 `.anchors`），
   不是扁平的锚点/结果列表；评测评测时需展开两层级取 `anchor.id`。
   `retrieval_trace` 字段 `HybridFusionRetriever.retrieve` 并不填充（只有部分旧检索器填）。
3. `Anchor` 没有 `entities` 字段；实体信息在 `anchor.tags` 与
   `anchor.exact_match_keys`（复合键如 `alice-birthday`，直接当词元用会匹配不上，
   novelty 的锚点侧集合对其做了再分词）。
4. anchor id = `blake2b(text + source_session)`：同文本必同 id，但 source_session
   不同则 id 不同——各臂自行解析事实→锚点映射，不跨臂假设 id 一致。
5. `star_graph.embedding.get_embedder()` 是模块级单例且经 `EmbedderRegistry` 二次
   缓存；TF-IDF 后端带语料状态，臂间必须重置（`reset_embedder()` + 清 registry），
   否则先跑的臂会污染后跑臂的向量空间。
6. `HybridFusionRetriever` 构造签名
   `(graph, alpha, beta, gamma, delta, epsilon, spread_steps, bm25_index)`，
   默认值即 R-hybrid 配置；不传 bm25_index 时 ε 项退化为词面重叠——与生产默认一致。
7. `ImportanceEngine.evaluate` 会更新内部 topic tracker / 检索反馈缓存（有状态）；
   每臂新建引擎实例，跨臂无共享状态。
8. Windows 下实验结束后调用 `manager.shutdown_batch_vectorizer()`
   （脚本已处理），否则后台刷写线程可能拖延进程退出。

---

## 正式跑分结果（2026-09-11，synthetic 200 事实 + 100 噪声 / 50 查询 × 3 seeds）

原始数据：`benchmarks/ab_results/importance_ab_*.json`、`retrieval_ab_*.json`。

### 实验一 · 重要性权重 → 采纳 **W-code+nov**

| seed | W-code hit@5 / MRR | W-plan hit@5 / MRR | W-code+nov hit@5 / MRR |
|---|---|---|---|
| 42 | 0.020 / 0.007 | 0.020 / 0.007 | 0.020 / 0.007 |
| 7 | 0.060 / 0.036 | 0.060 / 0.036 | 0.060 / 0.036 |
| 13 | 0.000 / 0.003 | 0.000 / 0.003 | 0.000 / 0.003 |

三组在全部质量指标上 3/3 seeds 完全一致（存储量 200 vs 202/202，差异仅来自
门控边界）。**无证据任何权重组合更优** → 按设计要求（novelty 必须保留）采纳
**W-code+nov**（v2 六信号 ×0.90 + novelty 0.100，合计 1.000），已落地为
`ImportanceWeights` 默认值与 defaults.yaml `importance_engine.weights`。

### 实验二 · 检索公式 → 采纳 **R-hybrid（HybridFusion）**

| seed | R-hybrid hit@3/hit@5/MRR/has_answer | R-plan 同左 |
|---|---|---|
| 42 | 0.000 / 0.020 / **0.007** / 0.040 | 0.020 / 0.020 / 0.009 / 0.040 |
| 7 | 0.040 / 0.060 / **0.036** / **0.140** | 0.020 / 0.020 / 0.020 / 0.100 |
| 13 | 0.000 / 0.000 / 0.003 / 0.020 | 0.000 / 0.020 / 0.004 / 0.020 |
| **均值** | hit@5 **0.027** / MRR **0.0153** | hit@5 0.027 / MRR 0.0110 |

均值 MRR R-hybrid 更高（seed 7 上优势显著），且 R-hybrid 是 LoCoMo-10 44.2%
的生产实测管线；R-plan 仅速度占优（无 PPR）。**保留 HybridFusion 为生产检索**，
plan.md 五因子公式已标注为未采纳的设计草案。

### 已知局限
合成评测的绝对指标偏低（has_answer 2–14%，接近随机水平），说明该自建 harness
对质量差异的分辨力有限——上表"打平"结论在 3 seeds 上稳定成立，但**绝对性能
请以 LoCoMo-10 基准为准**。后续可在 LoCoMo 子集上复跑本实验做二次确认。
