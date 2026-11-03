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

- **活跃分支（本地 + 远程）总数 ≤ 30**，同名分支只算 1 个，不含 `main`（或 `--base` 指定的基准分支）。
- 统计命令（远程分支以 `git ls-remote` 为准，避免本地过期的远程跟踪引用）：

```bash
{ git branch --format='%(refname:short)';
  git ls-remote --heads origin | awk '{print $2}' | sed 's#^refs/heads/##'; } \
  | grep -v -x main | sort -u | wc -l
```

- 超过 30 时先删已合并分支、再关停无进展 PR；仍超限则暂停新分支，由 Lead 取舍。长期分支（存活 > 2 周）视为流程异常。
- 统计与清理统一走 `scripts/cleanup_branches.sh`（见 §3），不要各自手写命令，避免口径漂移。

## 3. 删除与清理

- **合并后自动删除**：仓库已开启 `delete_branch_on_merge=true`，PR 合并时 GitHub 自动删除 head 分支（本地再 `git branch -d <name>` + `git fetch --prune`）。合并命令 `gh pr merge <N> --squash` 即可，`--delete-branch` 可省略（重复指定无害）。
- **周度清理**：`.github/workflows/branch-cleanup.yml` 每周一 03:00 UTC 定时运行，也可在 Actions → Branch cleanup → Run workflow 手动触发；执行 `bash scripts/cleanup_branches.sh --execute`，并把统计与清理结果写入 Step Summary。清理后活跃分支仍 > 30 时 job 失败，需人工关闭/合并长期 PR、删除分支或回收 worktree。该工作流不在 `pull_request` 上触发，不构建、不安装依赖。
- **脚本用法**（以 `bash scripts/cleanup_branches.sh --help` 为准）：

```bash
bash scripts/cleanup_branches.sh --dry-run          # 默认：只统计 + 打印计划，不改动任何东西
bash scripts/cleanup_branches.sh --execute          # 执行删除
bash scripts/cleanup_branches.sh --execute --force  # 清理后仍超限时不返回非零（只改退出码）
```

  行为约定：
  1. 统计本地 + 远程活跃分支（不含 `main`），输出清理前后计数；远程清单用 `git ls-remote --heads origin`。
  2. 本地：删除已合并进 `main`、不是当前分支、未被其它 worktree 占用的分支（`git branch --merged` 口径）。
  3. 远程：用 `gh pr list --state merged` 找出已合并 PR 的 head 分支并 `git push origin --delete`；跳过 `main`、仍有 open PR 的分支、受保护分支与 fork（跨仓库）分支。
  4. worktree：指向已合并分支、工作区干净、未 locked、不是当前/主 worktree 的登记会被 `git worktree remove --force` 移除，随后 `git worktree prune`。
  5. 幂等：重复执行不会重复删除；默认 dry-run，只有 `--execute` 才修改；删除失败时以非零状态退出。
- 清理只删"已合并 / 已关闭 PR / 明确废弃"的分支；不确定的先问作者。worktree 里有未提交改动时脚本跳过并告警（不丢改动），需人工确认后再删。
- 本地用 squash 合并的分支不在 `--merged` 结果中（提交不是 `main` 的祖先），由"已合并 PR"路径清理远程分支，本地同名分支人工确认后 `git branch -D`。
- 远程删除依据"已合并 PR 的 head 分支名"：若分支名在合并后被复用、且尚未开 PR，可能被误删；复用旧分支名时请先开 PR，或直接改用新名字。

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
- `branch-cleanup.yml` 只删除已合并 PR 的非保护 head 分支，不修改本节任何保护设置，也不会直接向 `main` 推送。
- 分支总数 ≤30 与周度清理见 §2–§3（Issue #12，已落地）；本节只约束 `main` 的合并入口。

## 5. 禁止事项

- 禁止直接向 `main` 推送（含 `git push -f origin main`）。
- 禁止在 `main` 上直接提交后"再开分支补 PR"。
- 禁止一个分支跨多个 Issue、禁止多人共用同一分支。
- 禁止手工删除他人**未合并**分支或他人正在使用的 worktree；已合并分支的批量清理由 §3 的脚本/工作流按规则执行。
- 禁止把 `main` 的更新用 merge 提交带进分支；统一 `git rebase origin/main`（冲突较多时先与 Lead 确认）。

## 6. 与 worktree 的关系

并行任务使用独立 worktree 隔离（`git worktree add <path> -b <branch> origin/main`），任务结束即移除；worktree 路径不属于仓库内容，不提交、不引用绝对路径。
