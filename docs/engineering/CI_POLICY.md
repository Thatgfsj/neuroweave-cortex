# CI 策略（CI_POLICY.md）

## 1. 分层原则

| 阶段 | 触发时机 | 做什么 | 不做什么 |
|------|----------|--------|----------|
| **PR CI（静态）** | `pull_request` 事件（`opened` / `synchronize` / `reopened` / `edited` / `labeled` / `unlabeled`） | 语法、版本一致性、文件行数、PR 规范（Issue 关联 + 代码量） | 不构建、不安装依赖、不跑测试套件 |
| **最终验证** | 手动 `workflow_dispatch`，或推送 `v*` tag | 构建 wheel + smoke 测试 | 不替代 PR 评审；**不会**因合并到 `main` 自动触发 |
| **本地自检** | 提交前 | 与 PR CI 等价的静态命令 + 按需 smoke | 不作为唯一证据省略 |

**为什么 PR CI 不构建**：本机与 CI runner 资源有限，依赖安装慢且易受网络影响；构建失败会掩盖静态问题，让评审失去焦点；PR 阶段也不应接触发布凭据。构建与冒烟是"发布前一次"的动作，放在最终验证。

## 2. 工作流拓扑（最终态）

| 文件 | 角色 | 触发 |
|------|------|------|
| `.github/workflows/ci.yml` | **PR 薄入口**：只声明 `pull_request` 事件、权限与并发，`uses: ./.github/workflows/pr-check.yml` | `pull_request` → `main` |
| `.github/workflows/pr-check.yml` | **可复用静态检查**（`workflow_call`）：元数据/代码量、版本、行数预算、语法；无 secrets、不安装、不构建 | 被 `ci.yml` 调用 |
| `.github/workflows/verify.yml` | **最终验证**：校验 tag 与版本一致 → 构建 wheel → smoke | `workflow_dispatch`；`push` tags `v*` |

`verify.yml` 的实际触发配置只有下面两项，**没有** `push: branches: [main]`，也**没有** `pull_request`：

```yaml
on:
  workflow_dispatch:
  push:
    tags: ["v*"]
```

- 因此"合并到 `main` 后自动验证"不成立：发布者必须在打 tag 或手动 Run workflow 后确认 `verify` 为绿色。
- tag 触发时先校验 tag 名与 `star_graph/_version.py` 的 `__version__` 一致（`v1.4.3` ↔ `1.4.3`），不一致直接失败；该版本从 `_version.py` 用 `ast` 静态读取，**不读** `pyproject.toml`（版本已改为 dynamic，没有 `[project].version`）。
- 分支保护的必需检查是 `PR checks / PR checks (no build)`（即 `pr-check.yml` 的 job），不是 `verify`；配置见 [BRANCH_POLICY.md](BRANCH_POLICY.md)。

## 3. PR CI 检查项与脚本对应关系

每一项检查只有一个对应脚本，规则与实现一一对应：

| 检查项 | 脚本 | 判定 | 失败时 |
|--------|------|------|--------|
| PR 规范与代码量 | `scripts/ci/check_pr.py` | body 有 `Closes #N` 且 Issue 存在、早于 PR、有类型标签；代码量 < 1000 行，超限须 `size/exception` | 改 PR 描述/Issue 关联或拆 PR，见 [PR_POLICY.md](PR_POLICY.md) |
| 版本号一致性 | `scripts/check_version.py` | 单一源 `star_graph/_version.py` 与其余 15 项（pyproject dynamic、两个 `__init__.py`、5 处文档镜像、defaults、4 个 server、verify.yml tag 读取）一致，共 16 项（见 [VERSIONING.md](VERSIONING.md)） | 同步镜像/导入，不改唯一源 |
| 单文件行数预算 | `scripts/check_line_budget.py` | 每个被 git 跟踪的 `.py` 文件 ≤ 1000 行（allowlist 现为空，见 §5） | 拆分文件 |
| Python 语法（不 import 第三方依赖） | `scripts/ci/check_syntax.py` | 全部 `.py` 可 `ast.parse`；改动的 `*.toml` / `*.json` 可解析 | 修语法错误 |

**代码量口径**（与 `scripts/ci/check_pr.py` 的 `CODE_EXTS` 完全一致，完整清单也见 [PR_POLICY.md](PR_POLICY.md) §2）：

- 计入（源码扩展名，共 23 种）：`.py` `.pyi` `.ts` `.tsx` `.js` `.jsx` `.mjs` `.cjs` `.sh` `.bash` `.ps1` `.sql` `.rs` `.go` `.java` `.kt` `.c` `.h` `.cpp` `.hpp` `.cs` `.rb` `.lua`。
- **不计入**：文档与数据/配置，包括 `.md` `.json` `.jsonl` `.csv` `.yml` `.yaml` `.toml` `.lock` 等。
- 判定方式：只看文件后缀（`Path(name).suffix.lower()` 是否在 `CODE_EXTS`），不做目录白名单；测试代码若为 `.py` 同样计入。

本地等价命令（提交前跑一遍，能省一轮 CI）：

```bash
python scripts/ci/check_syntax.py
python scripts/check_version.py
python scripts/check_line_budget.py
python scripts/ci/check_pr.py
```

`check_pr.py` 需要 `GITHUB_TOKEN` / `GITHUB_REPOSITORY` / `PR_NUMBER`；离线自检用 `python scripts/ci/check_pr.py --self-test`。

## 4. 最终验证（build + smoke）

- 由 `.github/workflows/verify.yml` 在**手动触发（Actions → Verify → Run workflow）或推送 `v*` tag** 时执行。
- 构建与 smoke 合并为一步：`bash scripts/smoke_test.sh`（脚本内部构建 wheel、装进仓库外的临时 venv，再跑 smoke 检查）；不再拆成"先 `python -m build` 再测试"的两段工作流步骤。
- 本地等价：

```bash
bash scripts/smoke_test.sh
```

- smoke 只验证"装得上、导得进、核心链路能跑通"（例如 `remember` / `recall` / `sleep` 的最小调用），**不追求覆盖率**；完整测试套件按需在本地执行。
- 构建产物（`dist/`、`build/`、`*.egg-info`）不入库、不进 PR。
- PR 若涉及打包元数据（`pyproject.toml`、`MANIFEST.in`、包内数据文件），作者必须本地跑一次构建 + smoke，并把结果写进 PR 的"验证证据"。
- 发布时"CI 通过"指**最近一次 `verify` 运行（tag 或手动触发）为绿色**，不存在"合并到 main 后自动跑绿"这一环节。

## 5. 行数预算与 ratchet allowlist

- 硬上限：每个被 git 跟踪的 `.py` 文件 ≤ 1000 行（`scripts/check_line_budget.py`）。
- allowlist（`scripts/ci/line_budget_allowlist.txt`）当前为**空**（**Issue #24** 已清零）：**全部代码文件 ≤ 1000 行，不存在例外**。
- allowlist 规则：**只减不增**，现为 0 例外。**新增超限必须先拆分**才能合并，不得往 allowlist 里加条目；拆分 PR 必须在同一个 PR 内保证文件 ≤ 1000 行，不能靠条目过渡。
- 历史：拆分 PR（#20/#21/#25/#26）合入后，`star_graph/retrieval_engine/retrieval_core.py`（#7）、`star_graph/memory_core/graph.py`（#8）、`star_graph/cortex_api/runtime.py`（#9）、`examples/memory_benchmark.py`（#10）四个文件均 ≤ 1000 行；#24 已移除这 4 条 stale 条目，allowlist 文件只保留头部说明。

## 6. 迁移状态、清理与禁止事项

- 周度分支清理工作流由 **Issue #12** 承接（脚本 `scripts/cleanup_branches.sh` + 定时工作流）；本文不描述尚未落地的清理时间表，以 Issue #12 的验收为准。
- 禁止：在 PR 工作流中新增 `pip install`、`python -m build`、全量 `pytest`；用 `continue-on-error` 掩盖失败；把 secrets 暴露给 `pull_request` 工作流。
- 新增检查项必须：先有脚本 → 再进工作流 → 同步更新本文档表格与分支保护的必需检查（同一个 PR 内完成）；CI 因基础设施失败时新建 Issue 记录，不要把失败常态化。
