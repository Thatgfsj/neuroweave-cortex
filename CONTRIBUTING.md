# 参与贡献（CONTRIBUTING.md）

本文是最短入口。**完整规则以 [ENGINEERING.md](ENGINEERING.md) 为唯一权威**，细则见 [docs/engineering/](docs/engineering/)。

## 五分钟上手

1. **建 Issue**：说明背景、目标、验收标准，打类型标签（`bug` / `feat` / `docs` / `ci` / `refactor` / `chore` / `governance` / `spec`）。没有 Issue 不接收 PR。
2. **建分支**：从 `origin/main` 新建 `type/short-slug`，例如 `docs/governance`。
3. **改代码/文档**：保持一个 PR 只做一件事，代码改动 <1000 行。
4. **本地自检**：跑静态检查（下面命令）；涉及打包或核心链路时再跑构建与 smoke。
5. **开 PR**：用 [PR 模板](.github/PULL_REQUEST_TEMPLATE.md)，首行写 `Closes #N`，通过 CI 与双评审后由维护者 squash 合并并删除分支。

```bash
git fetch origin
git worktree add ../nwc-wt/<slug> -b docs/governance origin/main

# 静态检查（与 PR CI 等价）
python scripts/ci/check_syntax.py
python scripts/check_version.py
python scripts/check_line_budget.py
python scripts/ci/check_pr.py
```

## 规则速查

| 主题 | 规则 | 详情 |
|------|------|------|
| 工作流 | Issue first → 分支 → PR → 静态 CI → 双评审 → squash 合并 → 删分支 | [WORKFLOW.md](docs/engineering/WORKFLOW.md) |
| PR | 一个 PR 一件事；`Closes #N`；代码量 <1000 行 | [PR_POLICY.md](docs/engineering/PR_POLICY.md) |
| 分支 | `type/short-slug`；活跃分支 ≤30；禁止直推 `main` | [BRANCH_POLICY.md](docs/engineering/BRANCH_POLICY.md) |
| CI | PR 只做静态检查，不构建、不装依赖；最终验证另跑 | [CI_POLICY.md](docs/engineering/CI_POLICY.md) |
| 版本 | SemVer，每次迭代 +0.0.1（当前 1.4.2 → 1.4.3） | [VERSIONING.md](docs/engineering/VERSIONING.md) |
| 文档 | 冲突以 README 为准；不写易过期计数 | [DOC_POLICY.md](docs/engineering/DOC_POLICY.md) |
| 评审 | 双评审 A（正确性）/ B（工程合规） | [REVIEW_CHECKLIST.md](docs/engineering/REVIEW_CHECKLIST.md) |
| 发布 | 版本、构建、smoke、清理逐项确认 | [RELEASE_CHECKLIST.md](docs/engineering/RELEASE_CHECKLIST.md) |

## 提交信息

```
area: 变更描述

正文说明为什么（而不是复述做了什么）。
```

示例：`sleep: 修复合并相似记忆时的 O(n^2) 退化`。

## 代码约定

- Python 3.11+，公开 API 必须有类型注解，优先 `X | None` 写法。
- **单个代码文件 ≤ 1000 行**；超限必须拆分（4 个存量超限文件在拆分前受 ratchet allowlist 约束，见 [CI_POLICY.md](docs/engineering/CI_POLICY.md)）。
- 注释解释**为什么**，而不是复述代码在做什么。
- 核心只做记忆 CRUD、图操作、检索、睡眠巩固；更高层认知特性放 `star_graph/contrib/` 或独立包。
- `star_graph/__init__.py` 使用 PEP 562 惰性导入：新增符号请登记到 `_LAZY`，不要在文件顶部直接 `from .new_module import X`，并验证 `from star_graph import YourSymbol` 可用。
- 安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开开 Issue。
