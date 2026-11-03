# NWC 工程宪法（ENGINEERING.md）

> 本文件是 **neuroweave-cortex 唯一的工程规则入口**，适用于所有改动；细则见 [docs/engineering/](docs/engineering/)。细则与本文件冲突时**以本文件为准**，并请开 Issue 修正细则。

## 一句话

**Issue first → 分支 → PR → 静态 CI → 双评审 → squash 合并 → 删分支**；版本每次迭代 +0.0.1。

## 硬规则（全部强制）

### R1 没有 Issue 就没有改动

- 任何改动先建 Issue，并打类型标签：`bug` / `feat` / `docs` / `ci` / `refactor` / `chore` / `governance` / `spec`。
- 每个 PR 必须关联一个**为此 PR 新建**的 Issue；PR body 必须包含 `Closes #N`。
- 发现真实缺陷时新建 Issue 并打 `bug`，修复 PR 用 `Fixes #N`。

### R2 禁止直推 main

- `main` 只接受经过评审的 PR（squash 合并）。禁止 `git push origin main`、禁止本地合并后直推。
- 分支一律从 `origin/main` 新建，命名 `type/short-slug`（如 `docs/governance`、`fix/retrieval-deadlock`）。

### R3 PR 代码量 <1000 行

- 口径：**代码文件**的 additions + deletions 合计 <1000 行；文档、数据文件不计入。计入的源码扩展名清单与 `scripts/ci/check_pr.py` 的 `CODE_EXTS` 完全一致（见 [PR_POLICY.md](docs/engineering/PR_POLICY.md)、[DOC_CONFLICTS.md](docs/engineering/DOC_CONFLICTS.md) §2）。
- 超出必须由 Lead 审核并添加 `size/exception` 标签，且在 PR 描述中写明拆分计划与理由。
- 单文件同样不得超过 1000 行；4 个存量超限文件在拆分完成前进入 ratchet allowlist，拆分完成后由 #24 统一清理条目（见 [PR_POLICY.md](docs/engineering/PR_POLICY.md)、[CI_POLICY.md](docs/engineering/CI_POLICY.md)）。

### R4 PR CI 只做静态检查

- PR 阶段 CI **不构建、不安装依赖、不跑测试套件**；只做语法/版本/行数/PR 规范等静态检查（见 [CI_POLICY.md](docs/engineering/CI_POLICY.md)）。
- 最终验证（构建 wheel + 简单 smoke 测试）由 `verify` 工作流（手动 `workflow_dispatch` 或推送 `v*` tag 触发）或本地 `scripts/smoke_test.sh` 完成，结果作为 PR 的验证证据。

### R5 双评审

- 两个独立评审（A：正确性/行为/API/测试；B：工程合规/文档/CI/分支/版本）都通过后才能合并。
- 清单见 [REVIEW_CHECKLIST.md](docs/engineering/REVIEW_CHECKLIST.md)，结论以 PR 评论留痕。

### R6 squash 合并 + 删分支

- 合并方式固定为 squash；合并后立即删除分支（本地与远程）。
- 同一 PR 不夹带无关格式化、无关重构。

### R7 分支总数 ≤30 且及时清理

- 活跃分支（本地 + 远程，同名算 1）总数 ≤30；合并后自动删除；每周清理一次。
- 见 [BRANCH_POLICY.md](docs/engineering/BRANCH_POLICY.md)。

### R8 版本号每次迭代 +0.0.1

- SemVer；每次迭代 PATCH +1（例如 `1.4.2` → `1.4.3`，当前值以 `star_graph/_version.py` 为准）。
- **唯一版本源：`star_graph/_version.py`** 的 `__version__`（零 import）；`pyproject.toml` 使用 `dynamic = ["version"]` + `[tool.setuptools.dynamic] version = {attr = "star_graph._version.__version__"}`，不再有静态 `[project].version`。
- `scripts/check_version.py` 静态校验 **16 项**：源文件本身与包内 `__version__` 唯一性、pyproject dynamic 接线、`star_graph/__init__.py` 与 `nwc/__init__.py` 的导入、5 处文档镜像（README 徽章 / README_CN 首行 / plan.md 顶部 / ROADMAP 现状标题 / ROADMAP 当前版本）、`defaults.yaml` 无 `version:`、4 个 server 模块从 `_version` 导入、`verify.yml` 的 tag 校验读 `_version.py`。
- 功能 PR 不得顺手改版本；版本只能通过 `python scripts/bump_version.py` 变更，发布单独走一个 Issue/PR。见 [VERSIONING.md](docs/engineering/VERSIONING.md)。

### R9 文档冲突以权威文档为准

- 以 `README.md` 与 [docs/spec/user-center-memory-chain.md](docs/spec/user-center-memory-chain.md)（用户中心记忆链路规范）为权威；该 spec 已合入仓库，与 README 同为权威（第 2 层），二者或其它文档冲突时**以 README 为准**。
- 禁止硬编码易过期计数（模块数、测试数、行数）；必须给出统计口径与脚本/命令来源。全仓统一口径为 `python scripts/repo_stats.py`（见 [DOC_POLICY.md](docs/engineering/DOC_POLICY.md)）。

### R10 文档与代码同 PR 更新

- 行为、API、配置、流程变更必须同步更新受影响文档，不允许"下个 PR 补文档"。

## 标准流程

```
新建 Issue（打类型标签）
  → 从 origin/main 建分支 type/short-slug
  → 提交（area: 变更描述）
  → PR（body 首行 Closes #N，填模板全部字段）
  → PR CI 静态检查通过
  → 双评审（A/B）通过
  → squash 合并 + 删除分支
  → 发布迭代时统一 +0.0.1
```

命令示例与各环节细节见 [WORKFLOW.md](docs/engineering/WORKFLOW.md)。

## 文档索引

| 文档 | 内容 |
|------|------|
| [WORKFLOW.md](docs/engineering/WORKFLOW.md) | Issue-first 全流程与命令 |
| [PR_POLICY.md](docs/engineering/PR_POLICY.md) | PR 规则、代码量口径、`size/exception` |
| [BRANCH_POLICY.md](docs/engineering/BRANCH_POLICY.md) | 分支命名、上限、清理 |
| [CI_POLICY.md](docs/engineering/CI_POLICY.md) | 静态 CI / 最终验证分工与脚本映射 |
| [VERSIONING.md](docs/engineering/VERSIONING.md) | SemVer、版本源、校验 |
| [DOC_POLICY.md](docs/engineering/DOC_POLICY.md) | 文档权威顺序与防过期规则 |
| [docs/spec/user-center-memory-chain.md](docs/spec/user-center-memory-chain.md) | 用户中心记忆链规范（与 README 同为权威） |
| [scripts/repo_stats.py](scripts/repo_stats.py) | 全仓统计口径：模块/测试/行数/版本 |
| [REVIEW_CHECKLIST.md](docs/engineering/REVIEW_CHECKLIST.md) | 双评审清单 |
| [RELEASE_CHECKLIST.md](docs/engineering/RELEASE_CHECKLIST.md) | 发布前检查 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献者入口（最短路径） |

## 违反规则时

CI 或评审可直接拒绝；已合并的违规改动需新建 `governance` Issue 记录并回滚/修正，不静默忽略。
