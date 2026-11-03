# 分支规范（BRANCH_POLICY.md）

## 1. 命名

格式：`type/short-slug`

- `type` 取 Issue 类型标签：`feat` / `docs` / `ci` / `refactor` / `chore` / `governance` / `spec`；缺陷修复用 `fix/`（对应 `bug` 标签）。
- `short-slug`：小写英文与数字，单词间用 `-`，2–5 个词，整体分支名 ≤ 40 字符；不要用中文、空格、大写、下划线、日期后缀。
- 一个分支只服务一个 Issue；分支名里的 slug 应能让人猜到 Issue 内容。

| 示例 | 说明 |
|------|------|
| `docs/governance` | 工程治理文档（#3） |
| `fix/retrieval-deadlock` | 缺陷修复 |
| `ci/static-pr-check` | PR CI 静态检查（#4） |
| `feat/memory-decay-curve` | 新功能 |
| `chore/bump-1.4.3` | 版本发布 |

反例：`patch-1`、`test2`、`feat/new-feature-final`、`docs/2026-11-04`、`ZhangSan-fix`。

## 2. 上限与统计口径

- **活跃分支（本地 + 远程）总数 ≤ 30**，同名分支只算 1 个，不含 `main`。
- 统计命令：

```bash
{ git branch --format='%(refname:short)'; git branch -r --format='%(refname:short)' | sed 's#^origin/##'; } \
  | grep -v -E '^(main|HEAD)$' | sort -u | wc -l
```

- 超过 30 时先删已合并分支、再关停无进展 PR；仍超限则暂停新分支，由 Lead 取舍。长期分支（存活 > 2 周）视为流程异常。

## 3. 删除与清理

- **合并后自动删除**：仓库已开启 `delete_branch_on_merge=true`，PR 合并时 GitHub 自动删除 head 分支（本地再 `git branch -d <name>` + `git fetch --prune`）。合并命令 `gh pr merge <N> --squash` 即可，`--delete-branch` 可省略（重复指定无害）。
- **周度清理**（建议每周一次，由维护者执行；定时工作流由 Issue #12 承接）：
  1. 删除已合并分支（远程 + 本地）；
  2. 关闭 14 天无活动的 PR/分支，或催办；
  3. `git worktree remove` 回收临时 worktree，`git worktree prune` 清理失效登记；
  4. 用 `scripts/cleanup_branches.sh` 辅助（由 Issue #12 提供，参数以 `--help` 为准），执行前确认输出，不要盲删。
- 清理只删"已合并 / 已关闭 PR / 明确废弃"的分支；不确定的先问作者。

## 4. `main` 分支保护（已启用，见 Issue #17）

`main` 的保护规则以仓库实际设置为准（`gh api` 于 2026-11-04 核对）：

| 设置 | 实际值 | 含义 |
|------|--------|------|
| 必须通过 PR 合并 | 是（`required_pull_request_reviews` 已启用，必需批准数 0） | 禁止直接 push `main`；批准数 0 表示不强制额外 approve，双评审仍按 R5 执行 |
| 必需状态检查 | `PR checks / PR checks (no build)` | 该检查未通过不得合并；`strict=false` 表示不强制先同步 main 才可合并 |
| `enforce_admins` | `true` | 管理员同样受上述约束 |
| `required_linear_history` | `true` | 只接受 squash/rebase，禁止 merge commit |
| `allow_force_pushes` | `false` | 禁止 force push |
| `allow_deletions` | `false` | 禁止删除 `main` |
| `delete_branch_on_merge` | `true` | PR 合并后自动删除 head 分支（仓库级设置） |

核对命令：

```bash
gh api repos/Thatgfsj/neuroweave-cortex/branches/main/protection \
  --jq '{checks: .required_status_checks.contexts, strict: .required_status_checks.strict,
         reviews: .required_pull_request_reviews, admins: .enforce_admins.enabled,
         linear: .required_linear_history.enabled,
         force: .allow_force_pushes.enabled, delete: .allow_deletions.enabled}'
gh api repos/Thatgfsj/neuroweave-cortex --jq .delete_branch_on_merge
```

- 必需检查名对应 `.github/workflows/ci.yml` → `pr-check.yml` 的 `PR checks (no build)` job（见 [CI_POLICY.md](CI_POLICY.md)）；新增或重命名该检查必须同步更新保护规则与本表。
- 保护规则是兜底，不替代流程：Issue 关联（R1）、双评审（R5）、squash 合并（R6）仍须执行。
- 分支总数 ≤30 与周度清理由 Issue #12 承接；本节只约束 `main` 的合并入口。

## 5. 禁止事项

- 禁止直接向 `main` 推送（含 `git push -f origin main`）。
- 禁止在 `main` 上直接提交后"再开分支补 PR"。
- 禁止一个分支跨多个 Issue、禁止多人共用同一分支。
- 禁止删除他人分支或他人 worktree。
- 禁止把 `main` 的更新用 merge 提交带进分支；统一 `git rebase origin/main`（冲突较多时先与 Lead 确认）。

## 6. 与 worktree 的关系

并行任务使用独立 worktree 隔离（`git worktree add <path> -b <branch> origin/main`），任务结束即移除；worktree 路径不属于仓库内容，不提交、不引用绝对路径。
