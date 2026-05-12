# Star Graph Memory — Refactoring & Improvement Plan

> Based on comprehensive codebase audit (2026-05-13). 232 tests passing, v1.0.6.

## Priority Summary

| Priority | Count | Impact |
|----------|-------|--------|
| **P0** | 3 issues | Correctness, performance, data consistency |
| **P1** | 3 issues | Architecture, maintainability, retrieval quality |
| **P2** | 3 issues | Cognitive fidelity, production readiness |
| **P3** | 11 issues | Code quality, testing, developer experience |

---

## P0 — Correctness & Performance (do first)

### 1. Ghost 子系统统一 (数据一致性)

**现状**: `graph.py` 有 `add_ghost()` 创建 `GhostAnchor`，`ghost.py` 有 `GhostSubsystem` 创建 `GhostNode`。两者数据结构不同，`sleep.py` 的 `_prune_anchors()` 需要 fallback:
```python
if hasattr(self.graph, '_ghost_subsystem') and self.graph._ghost_subsystem:
    self.graph._ghost_subsystem.create(anchor, residual_edges)
else:
    self.graph.add_ghost(anchor)
```

**问题**: `GhostAnchor` 和 `GhostNode` 字段不兼容，持久化/加载可能丢失信息。

**方案**:
- 统一为 `GhostNode`，`StarGraph` 只持有 ghost ID 引用，实际数据由 `GhostSubsystem` 管理
- `add_ghost()` 委托给 subsystem
- 清理 `graph.py` 中的 `GhostAnchor` 类

### 2. Raw Buffer 优先级提升 (检索质量)

**现状**: `recall()` 中 Path 0 (Exact Cache) → Path A (Raw Buffer) → Path B (Graph)，但 Raw Buffer 结果被放在 graph 结果**之后**合并。

**问题**: Raw Buffer 存的是最近 1-2 个 session 的原始对话，对于短期事实类查询命中率远高于压缩后的 anchor。

**方案**:
```python
# 当前：exact → graph → raw
# 改为：exact → raw → graph
merged_items = exact_results + raw_results + graph_results
```

### 3. ANN 索引增量维护 (大图性能)

**现状**: `cortical_lookup()` 每次查询时如果索引不同步就全量重建:
```python
if not self._ids_in_ann_sync():
    ann.clear()
    for aid, a in self.anchors.items():
        if a.embedding: ann.add(aid, a.embedding)
    ann.rebuild()
```

**方案**:
- `add_anchor()` 时同步添加到 ANN
- `remove_anchor()` 时同步从 ANN 删除
- 取消 `_ids_in_ann_sync()` 检查
- 只在 sleep 的 Index Rebuild 阶段做全量 rebuild

---

## P1 — Architecture & Maintainability

### 4. MemoryManager 拆分 (可维护性)

**现状**: `manager.py` 已有 1700+ 行，导入 40+ 模块，20+ lazy property。同时承担:
- Facade API
- 依赖注入容器（管理所有子系统生命周期）
- 检索编排器（recall / retrieve_with_descent）
- 持久化协调器（save/load/snapshot）

**方案**: 拆分为三个角色:
```python
class MemoryRuntime:      # 依赖容器 + 生命周期管理
class RetrievalPipeline:  # 检索编排（L0→L4 降级逻辑）
class MemoryManager:      # 仅保留 facade API，委托给上面两个
```

### 5. Cortex 独立睡眠 (架构一致性)

**现状**: `add_cortex()` 创建 `MemoryCortex`，但 `sleep()` 仍对 `self.graph` 做统一睡眠，各 cortex 不独立休眠。

**方案**:
- `MemoryManager.sleep()` 遍历 `router.cortices`，对每个 cortex 调用 `cortex.sleep()`
- 全局 sleep 只做跨 cortex 的事情（Hub 桥接、全局 schema 提取）
- 各 cortex 的 `ANNIndex` 独立，不共用 `self.graph._ann_index`

### 6. Dual-Channel 自动触发 (检索质量)

**现状**: `dual_recall()` 存在，但 `recall()` 内部没有调用它。用户必须显式选择 `dual_recall()`。

**方案**: 在 `recall()` 中增加自动路由:
```python
def recall(self, query, context, max_items=10):
    result = self._system1_recall(query, context, max_items)
    system2_keywords = {"all", "which", "before", "last", "list", "every"}
    needs_system2 = (
        any(kw in query.lower() for kw in system2_keywords)
        or result.confidence < 0.35
    )
    if needs_system2:
        s2_result = self._system2_recall(query, context, max_items)
        return self._merge_channels(result, s2_result)
    return result
```

---

## P2 — Cognitive Architecture Fidelity

### 7. 自传体记忆层 ("我"的形成)

**现状**: `reflection.py` 的 reflection 是"对用户的反思"（"用户喜欢什么"），而非"对自我的反思"（"我如何理解用户"）。

**方案**: 增加 `AutobiographicalMemory` 层:
```python
@dataclass
class SelfNarrative:
    episode_summary: str      # "我和用户讨论过 Redis 超时"
    self_belief: str          # "我认为用户偏好简洁代码"
    emotional_tone: float     # 这次互动中"我"的情绪
    formed_at: float
    stability: float
```
区别于 reflection node 的"客观分析"，而是"主观体验"。

### 8. State / Thermal State 统一 (认知架构清晰度)

**现状**: 两套状态系统并存，映射关系散落在多个文件:
- `MemoryState`: ACTIVE, REHEARSING, CONSOLIDATING, DORMANT, GHOST, REACTIVATED
- `ThermalState`: HOT, WARM, COLD, DEAD

**方案**: 明确定义状态转换矩阵，在状态转换时同步更新 thermal state:
```python
_STATE_THERMAL_MAP = {
    MemoryState.ACTIVE: ThermalState.HOT,
    MemoryState.REHEARSING: ThermalState.HOT,
    MemoryState.CONSOLIDATING: ThermalState.WARM,
    MemoryState.DORMANT: ThermalState.WARM,
    MemoryState.GHOST: ThermalState.COLD,
    MemoryState.REACTIVATED: ThermalState.HOT,
}
```

### 9. 振荡共振 phase 来源修正

**现状**: `Oscillator.derive_phase()` 和 `derive_frequency()` 从 embedding 统计量推导，缺乏生物学依据。

**方案**: phase 应来自:
- 时间上下文（一天中的时段、会话序号）
- 情绪节奏（对话中的情绪波动周期）
- 而非 embedding 的统计特征

---

## P3 — Code Quality & Production Readiness

### 10. 配置访问脆弱性

**现状**: 大量 `getattr(self.cfg, 'something', None)` 链式调用，配置结构变化时静默失败。

**方案**: `_DotDict` 增加 `get_path()` 支持:
```python
self.cfg.get('exact_cache.auto_harvest', True)
```

### 11. Sleep Phase 编号统一

**现状**: `run_phased()` 中有 N1, N2, N3, REM, Wake-prep, Phase 5b, Phase 5c, Phase 6, Phase 7, Phase 8 — 命名不统一。

**方案**: 统一命名:
```
N1_Replay, N2_Merge, N3_Compression, N3b_AtomFacts,
REM_Emotion, N4_Prune, N5_HubConnect, N6_IndexRebuild
```

### 12. 余弦相似度去重

**现状**: `graph.py`, `sleep.py`, `scheduler.py`, `manager.py`, `streaming.py` 各有自己的 `_cosine_sim` 实现。

**方案**: 提取为 `star_graph/math_utils.py` 中的统一函数。

### 13. find_contradictions() O(n²) 优化

**现状**: 遍历所有 anchor 对计算 embedding 相似度。

**方案**: 使用 ANN 索引先找出高相似度候选对，再检查情绪对立:
```python
# 复杂度从 O(n²) 降到 O(n * k)
```

### 14. retrieve_with_descent() Layer 3 全量扫描

**现状**: Layer 3 (2D Plane) 遍历所有 cortex 的所有 anchors，可能 O(50K)。

**方案**: 利用 TimeSpine 索引，只扫描最近 N 天的桶。

### 15. retention_score 缓存

**现状**: `retention_score` 是 `@property`，每次访问实时计算。检索路径中每个 anchor 被访问多次。

**方案**: 状态变化时缓存，设脏标记。

### 16. 测试覆盖率提升

**现状**: 232 个测试，但很多模块覆盖不足（benchmark, community, atom_facts 等）。

**方案**:
- 每个核心模块至少单元测试
- 集成测试：模拟 100 轮对话 → sleep → 验证 ghost 复活
- 基准测试自动化

### 17. 静态类型检查

**现状**: 大量 `| None` 和 `Any`，但很多地方没有类型注解。

**方案**: 引入 mypy 或 pyright。

### 18. 配置验证增强

**现状**: `_check_ranges()` 只检查数值范围，不检查配置段存在性和兼容性。

**方案**: 增加配置 schema 验证。

### 19. 日志系统标准化

**现状**: `sleep.py` 用 `self.log: list[str]`，`manager.py` 用 `print()`。

**方案**: 引入标准 `logging`，支持结构化日志。

### 20. 文档与代码同步

**现状**: README 中的 Quick Start 示例和实际 API 可能不完全一致。

**方案**: CI 中增加 doctest，确保 README 代码示例可实际运行。

---

## Implementation Order

```
Phase 1 (P0): Ghost 统一 → Raw Buffer 优先级 → ANN 增量
Phase 2 (P1): Manager 拆分 → Cortex 独立睡眠 → Dual-Channel 自动触发
Phase 3 (P2): 自传体记忆 → 状态机统一 → Phase 来源修正
Phase 4 (P3): 代码质量逐项修复（10-20）
```

---

## Recently Completed (v1.0.6)

- [x] Survival functions (Ebbinghaus / Power-law / Exponential / Custom)
- [x] Ghost intensity + NegativeGhost contradiction tracking
- [x] Multimodal memory (CLIP joint embedding text+image)
- [x] Streaming memory buffer with backpressure
- [x] Exact match cache (KV deterministic O(1) lookup)
- [x] Micro-sleep scheduler (incremental non-blocking consolidation)
- [x] Snapshot + WAL (crash recovery)
- [x] Async manager + tracing (OpenTelemetry spans)
- [x] Benchmark suite (5 categories)
- [x] Dependency manifest (requirements.txt)
- [x] Version unification (1.0.6) + orphan module exports
- [x] 232 tests passing
