# CI 策略（CI_POLICY.md）

## 1. 分层原则

| 阶段 | 时机 | 做什么 | 不做什么 |
|------|------|--------|----------|
| **PR CI（静态）** | 每次 push / PR 更新 | 语法、版本一致性、文件行数、PR 规范 | 不构建、不安装依赖、不跑测试套件 |
| **最终验证** | 合并到 `main` 后 / 手动触发 | 构建 wheel + 简单 smoke 测试 | 不替代 PR 评审 |
| **本地自检** | 提交前 | 与 PR CI 等价的静态命令 + 按需 smoke | 不作为唯一证据省略 |

**为什么 PR CI 不构建**：本机与 CI runner 资源有限，依赖安装慢且易受网络影响；构建失败会掩盖静态问题，让评审失去焦点；PR 阶段也不应接触发布凭据。构建与冒烟是"发布前一次"的动作，放在最终验证。

## 2. PR CI 检查项与脚本对应关系

工作流 `.github/workflows/ci.yml` 是 PR 入口，复用 `.github/workflows/pr-check.yml`；每一项检查只有一个对应脚本，规则与实现一一对应：

| 检查项 | 脚本 | 判定 | 失败时 |
|--------|------|------|--------|
| Python 语法（不 import 第三方依赖） | `scripts/ci/check_syntax.py` | 全部 `.py` 可解析 | 修语法错误 |
| 版本号一致性 | `scripts/check_version.py` | 以单一版本源为准，各副本一致 | 修版本字符串，见 [VERSIONING.md](VERSIONING.md) |
| 单文件行数预算 | `scripts/check_line_budget.py` | 每个代码文件 ≤ 1000 行 | 拆分文件 |
| PR 规范与代码量 | `scripts/ci/check_pr.py` | 有 `Closes #N`、代码量 < 1000、超限须 `size/exception` | 改 PR 描述或拆 PR，见 [PR_POLICY.md](PR_POLICY.md) |

本地等价命令（提交前跑一遍，能省一轮 CI）：

```bash
python scripts/ci/check_syntax.py
python scripts/check_version.py
python scripts/check_line_budget.py
python scripts/ci/check_pr.py
```

## 3. 最终验证（build + smoke）

- 由 `.github/workflows/verify.yml` 在合并到 `main` 后或手动触发时执行：安装依赖 → 构建 wheel → 运行 `scripts/smoke_test.sh`。
- 本地等价：

```bash
python -m build --wheel            # 或 pip wheel . -w dist
bash scripts/smoke_test.sh
```

- smoke 只验证"装得上、导得进、核心链路能跑通"（例如 `remember` / `recall` / `sleep` 的最小调用），**不追求覆盖率**；完整测试套件按需在本地执行。
- 构建产物（`dist/`、`build/`、`*.egg-info`）不入库、不进 PR。
- PR 若涉及打包元数据（`pyproject.toml`、`MANIFEST.in`、包内数据文件），作者必须本地跑一次构建 + smoke，并把结果写进 PR 的"验证证据"。

## 4. 行数预算与 ratchet allowlist

- 硬上限：每个代码文件 ≤ 1000 行。
- 存量超限文件在拆分完成前进入 allowlist（由 `scripts/check_line_budget.py` 维护），当前 4 个：
  - `star_graph/retrieval_engine/retrieval_core.py`
  - `star_graph/memory_core/graph.py`
  - `star_graph/cortex_api/runtime.py`
  - `examples/memory_benchmark.py`
- allowlist 规则：**只减不增**。新增超限文件一律不允许；拆分任一文件后必须在同一个 PR 中把它从 allowlist 移除，否则评审 B 拒绝。

## 5. 迁移状态与禁止事项

- 当前 `.github/workflows/ci.yml` 仍是旧版（安装依赖 + `pytest`），由工具链 Issue（#4）替换为本文结构；替换前以本文为目标态，不得再强化"PR 阶段构建/测试"。
- 禁止：在 PR 工作流中新增 `pip install`、`python -m build`、全量 `pytest`；用 `continue-on-error` 掩盖失败；把 secrets 暴露给 `pull_request` 工作流。
- 新增检查项必须：先有脚本 → 再进工作流 → 同步更新本文档表格（同一个 PR 内完成）；CI 因基础设施失败时新建 Issue 记录，不要把失败常态化。
