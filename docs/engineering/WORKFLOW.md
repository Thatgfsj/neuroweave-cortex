# 工作流：Issue-first（WORKFLOW.md）

凡是要落实到仓库的改动，一律走本流程。规则总览见 [ENGINEERING.md](../../ENGINEERING.md)。

## 1. 新建 Issue

每个 PR 必须有**一个为此 PR 新建**的 Issue（不允许"借用"老 Issue 或 epic 顶替）。

```bash
gh issue create --repo Thatgfsj/neuroweave-cortex \
  --title "docs(governance): 工程化维护文档与 Issue/PR 模板" \
  --label docs --label governance \
  --body "## 背景
...
## 目标
...
## 验收
..."
```

类型标签（必选其一）：`bug` / `feat` / `docs` / `ci` / `refactor` / `chore` / `governance` / `spec`；可叠加 `qa`（`size/exception` 仅 Lead 使用）。
正文写清背景、目标、验收标准、不做什么（边界），模板见 [.github/ISSUE_TEMPLATE/](../../.github/ISSUE_TEMPLATE/)。

## 2. 建分支

从 `origin/main` 新建，命名 `type/short-slug`（见 [BRANCH_POLICY.md](BRANCH_POLICY.md)）：

```bash
git fetch origin
git worktree add ../nwc-wt/<slug> -b docs/governance origin/main
```

多任务并行时使用独立 worktree，避免互相污染工作区；**不要**在同一目录切换分支做不同 Issue。

## 3. 提交

- 提交信息 `area: 变更描述`，正文写**为什么**；一个提交表达一个意图。
- 不提交构建产物、`.venv`、数据库、临时日志；提交前先跑第 5 步的静态检查。

```bash
git add -A
git commit -m "docs(governance): 工程化维护文档与 Issue/PR 模板 (#3)"
```

## 4. 建 PR

- body 首行必须是 `Closes #N`（缺陷修复用 `Fixes #N`）。
- 必须填满 [PR 模板](../../.github/PULL_REQUEST_TEMPLATE.md)：变更摘要、动机、验证证据、检查清单。
- 未完成的草稿用 Draft PR，但仍需先有 Issue。

```bash
git push -u origin docs/governance
gh pr create --repo Thatgfsj/neuroweave-cortex --base main --head docs/governance \
  --title "docs(governance): 工程化维护文档与 Issue/PR 模板 (#3)" \
  --label docs --label governance \
  --body "Closes #3"
```

## 5. CI：PR 阶段只跑静态检查

PR CI **不构建、不安装依赖**，只做静态检查：语法、版本一致性、文件行数预算、PR 规范。
明细与脚本对应关系见 [CI_POLICY.md](CI_POLICY.md)，本地等价命令：

```bash
python scripts/ci/check_syntax.py
python scripts/check_version.py
python scripts/check_line_budget.py
python scripts/ci/check_pr.py
```

失败就修，不要靠重跑（retry）掩盖；确属环境/基础设施问题时新建 Issue 记录。

## 6. 双评审

两个独立评审角色，各自按 [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) 出具结论：**A** 正确性/行为/API/测试，**B** 工程合规/文档/CI/分支/版本。
结论粘贴到 PR 评论（`gh pr review <N> --comment --body "..."`），两位都通过后由 Lead 合并；评审只针对本 PR 的 diff 与受影响调用方，其它改进另开 Issue。

## 7. 合并与清理

固定 squash，合并即删分支：

```bash
gh pr merge <N> --repo Thatgfsj/neuroweave-cortex --squash --delete-branch
git worktree remove ../nwc-wt/<slug> && git branch -d docs/governance
```

合并后确认：PR 显示 merged、远程分支已删除、Issue 自动关闭。

## 8. Bug 流程

1. 新建 Issue，打 `bug`，填现象 / 复现 / 期望 / 环境。
2. 建 `fix/short-slug` 分支复现问题（先加失败用例，再改代码）。
3. PR body 用 `Fixes #N`，验证证据包含修复前后的行为对比。
4. 走同一套 CI + 双评审 + squash 合并流程。

## 9. 例外

- 紧急修复也必须先有 Issue（可事后补齐细节），不得直推 `main`；回滚以 revert PR 执行，同样关联新建 Issue。
