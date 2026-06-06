# NeuroWeave Cortex (NWC) — Roadmap

> **方向调整**: 从"添加更多认知层"转向"夯实工程基础、验证现有认知层价值"。
> 详见 [plan.md](plan.md) 的完整分析。

---

## 当前状态 (v1.1.0 — 2026-05)

112 模块、4 层架构、8 阶段睡眠、300+ 配置参数。

### 已完成的功能

#### 存储层
- [x] 核心图结构 (StarGraph): 锚点 CRUD、边管理、社区
- [x] JSON 持久化 (默认)
- [x] SQLite 存储后端 (已实现但未默认)
- [x] 分层存储标签 (HOT/WARM/COLD)
- [x] 快照 + WAL 机制
- [x] 文件分片存储

#### 检索层
- [x] HybridFusion 检索器 (向量+图+时序融合)
- [x] OscillationResonance 检索器
- [x] VectorSimilarity 检索器
- [x] BM25 稀疏检索 (基础版)
- [x] System-1/System-2 双通道
- [x] 精确匹配缓存 (ExactMatchCache)
- [x] 认知缓存 (CognitiveCache)
- [x] 跨编码器重排序
- [x] 原始缓冲区 (Raw Buffer)
- [x] 检索预算控制 (MAX_HOPS=3, MAX_NODES=24, MAX_TOKENS=6000)

#### 睡眠与整合
- [x] 8 阶段睡眠 (N1-N3, REM, N4-N6)
- [x] 微睡眠调度器 (增量非阻塞)
- [x] 在线整合器
- [x] 认知压缩 (事件→模式→信念→人格)
- [x] 四层压缩 (消息→事件→语义→人格)
- [x] 记忆演化引擎 (衰减/增强/冲突/干扰)
- [x] 写门禁 (5 阶段质量过滤器)
- [x] 边预算 (最多 32 边/节点)

#### 认知层 (高级)
- [x] 认知工作空间 (Working Memory)
- [x] 概念皮层 (Concept Cortex)
- [x] 激活引擎 (Spreading Activation)
- [x] 感知层 (PerceptionLayer)
- [x] 目标系统 (GoalSystem + GoalTree)
- [x] 显著性引擎 (SalienceEngine)
- [x] 自我模型 (SelfModel)
- [x] 自主推理 (AutonomousReasoning)
- [x] 人格模型 (PersonalityModel)
- [x] 信念系统 (BeliefSystem)
- [x] 认知身份 (CognitiveIdentity)
- [x] 9 阶段生命周期 (Perception → ... → Ghost → Dead)
- [x] Ghost 子系统
- [x] Hebbian 学习
- [x] 认知编译器 (世界观涌现)
- [x] 自我反思循环

#### 基础设施
- [x] MCP Server (14 tools)
- [x] REST API 服务器
- [x] CLI (sg-sleep, sg-add, sg-query, sg-stats, nwc-mcp, nwc-server)
- [x] 300+ YAML 配置参数 + schema 验证
- [x] OpenTelemetry tracing 基础设施
- [x] 嵌入式/LLM 提供者 (OpenAI, Anthropic, Gemini, Ollama, DeepSeek)
- [x] 多模态 (CLIP 图像+文本)
- [x] 流式缓冲区 (带背压)

---

## 已知差距（需立即解决）

以下基于外部代码审查和基准数据分析:

| 领域 | 差距 | 严重程度 | 数据佐证 |
|------|------|----------|----------|
| **检索质量** | LoCoMo has_answer 仅 15-31%，F1 < 0.02 | 🔴 致命 | locomo_llm_results.json |
| **公平对比** | 无 Mem0/Zep/MemGPT 对比 | 🔴 致命 | benchmarks/ 缺失 |
| **存储后端** | JSON 默认，SQLite 未集成 | 🔴 致命 | 生产不可用 |
| **生态集成** | 无 LangChain/LlamaIndex/Haystack 适配器 | 🔴 致命 | 用户无法接入 |
| **并发安全** | 无锁、非原子写入 | 🟡 高危 | 多线程数据损坏 |
| **可观测性** | 无 auth、无 latency 埋点、print() 日志 | 🟡 高危 | 运维不可用 |
| **消融验证** | 无模块级消融实验 | 🟡 高危 | 无法证明价值 |
| **模块膨胀** | 112 模块耦合，新人 30 分钟无法理解 | 🟡 高危 | 维护成本高 |

---

## 路线图

### v1.2.0 — 检索重建 + 存储工程化 (当前重点工作)

**目标**: LoCoMo has_answer ≥ 45%，存储层生产就绪

- [ ] BM25+Embedding 真 RRF 融合检索 (当前 HybridFusion 效果不足)
- [ ] Raw Buffer 优先级提升 (短期事实召回先行)
- [ ] 检索权重回归调优 (数据驱动，非手调)
- [ ] ANN 索引增量维护 (消除每次查询全量重建)
- [ ] 并发安全: threading.RLock + 原子写入
- [ ] StorageBackend 抽象接口
- [ ] SQLiteStorage 升级为默认后端
- [ ] QdrantStorage 实现 (HNSW + BM25 混合)

### v1.3.0 — 生态集成 + 公平评测

**目标**: 用户可 5 分钟接入现有 Agent 工作流

- [ ] LangChain BaseChatMemory 适配器
- [ ] LlamaIndex memory 适配器
- [ ] OpenAI Agents SDK tool adapter
- [ ] Mem0/Zep/FAISS 公平对比基准
- [ ] 端到端任务基准 (多轮对话完成率)
- [ ] 消融实验自动化框架 (sleep/ghost/hebbian/spreading)

### v1.4.0 — 模块化 + 可观测性

**目标**: 安装体积缩减 60%，生产可部署

- [ ] core vs cognitive 模块拆分
- [ ] pip install nwcortex / nwcortex[full] 双模式
- [ ] 配置项从 300+ 精简到 ~50 核心项
- [ ] OpenTelemetry 关键路径埋点 (recall/sleep/remember)
- [ ] Prometheus 指标暴露 (latency/hit_rate/anchors)
- [ ] REST API JWT 认证 + 速率限制
- [ ] 结构化日志 (替换 print() + self.log)

### v1.5.0 — 部署与验证

**目标**: Docker 一站式部署，消融证据驱动

- [ ] Docker + docker-compose (含 Qdrant 侧车)
- [ ] 导出/导入/备份/迁移 CLI 工具
- [ ] 生物启发模块消融验证报告
- [ ] 延迟/吞吐基准 (QPS/内存占用曲线)
- [ ] 长期稳定性测试 (10万+ 记忆)

### v2.0.0 — 认知层重评

**目标**: 从 112 模块精简到 50-60 经过验证的模块

- [ ] 移除或降级所有未通过消融实验验证的模块
- [ ] 核心模块检索 has_answer ≥ 60%
- [ ] 与 Mem0/Zep 的公平对比中至少持平
- [ ] 文档、API、安装体验全面重构

---

## 版本历史

| 版本 | 日期 | 亮点 |
|------|------|------|
| 0.1.0 | 2026-05 | 核心图、锚点、边、星座、基础睡眠 |
| 0.2.0 | 2026-05 | 振荡器、ghost、schema、再巩固、共振引擎 |
| 0.3.0 | 2026-05 | 可插拔检索器、在线整合、13 测试套件、中文文档 |
| 0.4.0 | 2026-05 | 演化引擎、调度器、混合融合、边版本化、基准测试 |
| 1.0.5 | 2026-05 | 生存函数 (4 曲线)、ghost 强度、NegativeGhost 矛盾追踪 |
| 1.0.6 | 2026-05 | 多模态 CLIP、流式缓冲区、依赖清单、232 测试 |
| 1.0.7 | 2026-05 | Phase 5 认知深度: 12 模块 (memory_budget, quality_score, stability_control, 4-layer pyramid 等) |
| 1.0.8 | 2026-05 | Phase 6 认知皮层: 11 模块 (ThoughtObject, PerceptionLayer, CognitiveWorkspace 等) |
| 1.0.9 | 2026-05 | 全局硬上限、自动睡眠守护进程、冷 ghost 清理 |
| 1.1.0 | 2026-05 | Hippocampus 缓冲区、边稀疏化、文件分片、睡眠重建 |
| 1.2.0 | 2026-05 | 内存分层、衰减+增强、边遍历权重、传播激活、认知缓存 |
| 1.3.0 | 2026-05 | 域路由器、边预算、写门禁、四层压缩 — 467 测试 |
| 1.4.0 | 2026-05 | 传播激活检索、3 层热存储器、连续边时间衰减 — 496 测试 |
| 1.5.0 | 2026-05 | 更名为 NeuroWeave Cortex。自组织、人格模型、目标树 — 582 测试 |
| 1.0.0 | 2026-05 | 正式发布 — PyPI 发布、1,989 测试、80% 覆盖率 |

---

## 关联文档

- [plan.md](plan.md) — 完整工程重建计划与架构决策
- `docs/architecture.md` — 架构概述（待更新）
- `README.md` / `README_CN.md` — 项目说明（待更新基准数据）
