# 模块分类清单 — core vs cognitive

本文档定义哪些模块应纳入 `nwc/core/`（默认安装）和哪些进入 `nwc/cognitive/`（`[full]` 可选安装）。

## core — 必需模块 (~20 个)

实现长期记忆引擎的核心功能：存储、检索、基础记忆。

### 存储层 (6)
| 模块 | 文件 | 说明 |
|------|------|------|
| StarGraph | `memory_core/graph.py` | 核心图结构 |
| Anchor | `memory_core/anchor.py` | 记忆单元 |
| StorageBackend | `memory_core/storage_backend.py` | 存储接口 |
| JSONStorage | `memory_core/storage.py` | JSON 存储 |
| SQLiteStorage | `memory_core/sqlite_storage.py` | SQLite 存储 |
| ANNIndex | `memory_core/index.py` | 向量索引 |

### 检索层 (4)
| 模块 | 文件 | 说明 |
|------|------|------|
| RetrievalCore | `retrieval_engine/retrieval_core.py` | 检索核心管道 + RRF |
| BM25 | `retrieval_engine/bm25.py` | 关键词检索 |
| DualChannel | `retrieval_engine/dual_channel.py` | 双通道检索 |
| RetrievalBudget | `retrieval_engine/retrieval_budget.py` | 检索预算控制 |

### 基础设施 (10)
| 模块 | 文件 | 说明 |
|------|------|------|
| MemoryRuntime | `cortex_api/runtime.py` | 运行时容器 |
| RuntimeCore | `cortex_api/runtime_core.py` | CRUD 操作 |
| RuntimeLifecycle | `cortex_api/runtime_lifecycle.py` | 生命周期管理 |
| MemoryManager | `cortex_api/manager.py` | Facade API |
| RetrievalPipeline | `retrieval_engine/retrieval_pipeline.py` | 检索编排 |
| WorkingMemory | `cortex_api/working_memory.py` | 工作记忆 |
| RawBuffer | `raw_buffer.py` | 原始缓冲区 |
| Config | `config.py` | 配置系统 |
| EmbeddingProvider | `embedding.py` / `embedding_provider/` | 嵌入提供者 |
| Config defaults | `defaults.yaml` | 默认配置 |

## cognitive — 扩展模块 (~50+ 个)

### 认知加工 (5)
| 模块 | 文件 | 说明 |
|------|------|------|
| Perception | `perception.py` | 感知层 |
| CognitiveWorkspace | `cognitive_workspace.py` | 认知工作空间 |
| ConceptCortex | `concept_cortex.py` | 概念皮层 |
| ActivationEngine | `activation_engine.py` | 激活引擎 |
| Salience | `salience.py` | 显著性引擎 |

### 睡眠与整合 (6)
| 模块 | 文件 | 说明 |
|------|------|------|
| Sleep | `consolidation/sleep.py` | 8 阶段睡眠 |
| SleepNREM | `consolidation/sleep_nrem.py` | NREM 睡眠 |
| SleepREM | `consolidation/sleep_rem.py` | REM 睡眠 |
| SleepConsolidate | `consolidation/sleep_consolidate.py` | 睡眠整合 |
| MicroSleep | `consolidation/micro_sleep.py` | 微睡眠 |
| OnlineConsolidator | `online.py` | 在线整合 |

### 高级认知 (7)
| 模块 | 文件 | 说明 |
|------|------|------|
| BeliefSystem | `belief_system.py` | 信念系统 |
| PersonalityFormation | `personality_formation.py` | 人格形成 |
| CognitiveIdentity | `cognitive_identity.py` | 认知身份 |
| CognitiveCompression | `cognitive_compression.py` | 认知压缩 |
| SelfModel | `self_model.py` | 自我模型 |
| AutonomousReasoning | `autonomous_reasoning.py` | 自主推理 |
| GoalSystem | `goal_system.py` | 目标系统 |

### 图增强 (6)
| 模块 | 文件 | 说明 |
|------|------|------|
| SpreadingActivation | `spreading.py` | 传播激活 |
| HebbianLearning | `hebbian_learning.py` | Hebbian 学习 |
| GhostSubsystem | `ghost.py` / `consolidation/ghost.py` | Ghost 子系统 |
| CommunityDetection | `consolidation/community.py` | 社区检测 |
| AbstrationEngine | `abstraction.py` / `consolidation/abstraction.py` | 抽象引擎 |
| CognitiveCompiler | `compiler.py` | 认知编译器 |

### 遗忘与演化 (5)
| 模块 | 文件 | 说明 |
|------|------|------|
| EvolutionEngine | `evolution.py` / `consolidation/evolution.py` | 演化引擎 |
| SurvivalFunctions | `survival.py` / `consolidation/survival.py` | 生存函数 |
| EdgeDecay | `edge_decay.py` | 边衰减 |
| ThermalStore | `thermal_store.py` | 热存储 |
| ForgotCertificate | `forget_certificate.py` | 遗忘证书 |

### 辅助工具 (~15)
timespine, shard, hippocampus, cascade, topology, tracer, snapshot, 
hippocampus, context_routing, batch_vectorizer, markdown_export, etc.

### 评估 (3)
| 模块 | 文件 | 说明 |
|------|------|------|
| BenchmarkSuite | `contrib/benchmark.py` | 基准测试 |
| Ablation | `examples/ablation_benchmark.py` | 消融实验 |
| LoCoMoEval | `examples/locomo_eval.py` | LoCoMo 评估 |

### 集成 (3)
| 模块 | 文件 | 说明 |
|------|------|------|
| MCPServer | `mcp/server.py` | MCP 协议服务器 |
| REST API | `api/server.py` | REST API |
| integrations/ | 新增目录 | LangChain/LlamaIndex/Haystack 适配器 |

## 依赖关系检查

core 模块不允许导入 cognitive 模块。在 `__init__.py` 导入时若发现违反此规则，应报错或发出严重警告。

## 使用方法

```python
# 基础安装 — 20 核心模块
pip install nwcortex

# 全量安装 — 全部 112+ 模块
pip install nwcortex[full]

# 特定后端
pip install nwcortex[qdrant]
pip install nwcortex[sqlite]  # 已内置，无需额外依赖
```
