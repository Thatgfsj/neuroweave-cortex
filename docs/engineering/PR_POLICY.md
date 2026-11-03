# PR 规范（PR_POLICY.md）

## 1. 基本要求

- 一个 PR 只做一件事，且必须关联**为其新建**的 Issue；body 首行 `Closes #N`（缺陷用 `Fixes #N`）。
- 标题格式：`type(scope): 摘要 (#N)`，`type` 与 Issue 标签一致，例如 `docs(governance): ... (#3)`。
- 必须走 [PR 模板](../../.github/PULL_REQUEST_TEMPLATE.md) 的全部字段；验证证据要可复现（命令 + 输出摘要）。
- 基线固定为 `main`，分支从 `origin/main` 新建；不得把多个 Issue 塞进一个 PR。
- 合并方式：squash；合并后删除分支。PR 作者不自行合并（由 Lead 执行）。

## 2. PR 代码量口径

**门槛：代码文件 additions + deletions < 1000 行。**

| 类别 | 是否计入 | 示例（与 `scripts/ci/check_pr.py` 的 `CODE_EXTS` 同源） |
|------|:--------:|------|
| 源码（23 种后缀） | ✅ | `.py` `.pyi` `.ts` `.tsx` `.js` `.jsx` `.mjs` `.cjs` `.sh` `.bash` `.ps1` `.sql` `.rs` `.go` `.java` `.kt` `.c` `.h` `.cpp` `.hpp` `.cs` `.rb` `.lua` |
| 测试代码 | ✅ | `tests/**/*.py`（后缀是 `.py`，同样计入） |
| 文档 | ❌ | `.md` `.rst` `.txt`、无后缀的 `LICENSE` |
| 数据 / 配置 / 生成物描述 | ❌ | `.json` `.jsonl` `.csv` `.yml` `.yaml` `.toml` `.lock` |

口径以脚本为准：只看 `Path(filename).suffix.lower()` 是否在 `CODE_EXTS`，没有目录白名单。脚本与本文同源；若发现不一致，按脚本执行并提 Issue 修正本文。

统计命令（PR 视角，与 CI 同源：分页拉取 PR 文件列表，再按同一份扩展名清单聚合；`<N>` 换成 PR 号）：

```bash
EXT='py|pyi|ts|tsx|js|jsx|mjs|cjs|sh|bash|ps1|sql|rs|go|java|kt|c|h|cpp|hpp|cs|rb|lua'
gh api --paginate repos/Thatgfsj/neuroweave-cortex/pulls/<N>/files \
  --jq '.[] | "\(.additions)\t\(.deletions)\t\(.filename)"' \
  | awk -F'\t' -v ext="$EXT" '
      BEGIN { re = "^\\.(" ext ")$" }
      { if (tolower($3) ~ re) { a += $1; d += $2 } }
      END { printf "代码改动：%d+ / %d- = %d 行（门槛 <1000）\n", a, d, a + d }'
```

本地分支自检（同一扩展名清单；结果可能与 GitHub 文件列表略有差异，以 PR 视角为准）：

```bash
EXT='py|pyi|ts|tsx|js|jsx|mjs|cjs|sh|bash|ps1|sql|rs|go|java|kt|c|h|cpp|hpp|cs|rb|lua'
git fetch origin main
git diff --numstat origin/main...HEAD \
  | awk -F'\t' -v ext="$EXT" '
      BEGIN { re = "^\\.(" ext ")$" }
      { if (tolower($3) ~ re) { a += $1; d += $2 } }
      END { printf "代码改动：%d+ / %d- = %d 行（门槛 <1000）\n", a, d, a + d }'
```

口径要点：

- 以 **diff 行数**（additions + deletions）为准，不看最终文件大小；重命名/移动按实际增删行计。
- 自动生成的文件（锁文件、导出数据）不进 PR；文档 PR 不受 1000 行限制，但仍受"一 PR 一事"与评审约束。
- 超过门槛必须走 §3 的 `size/exception` 特批，不能通过改名/换后缀规避。

## 3. `size/exception` 特批流程

1. **先拆**：优先把改动拆成多个 PR（每个 <1000 行），这是默认答案。
2. 确实无法拆分时（如单文件大规模重写、机械式批量替换），在 PR 描述中写明：
   - 为什么不能拆（耦合点、原子性要求）；
   - 预计代码行数与文件分布；
   - 风险与回滚方式。
3. **Lead 审核**后给 PR 添加 `size/exception` 标签；无该标签时 CI 的 `check_pr.py` 判定失败。
4. 特批只对本次 PR 生效，不形成先例；连续特批需在 Issue 中复盘。

## 4. PR 描述要求

必须包含（模板已内置）：

- `Closes #N`，以及**验收标准对照**：用模板中的表格逐条对应 Issue 的验收项（Issue 原文不必重复粘贴）。
- **变更摘要**：改了哪些文件/模块，行为差异是什么。
- **动机**：为什么现在做，不改会怎样。
- **验证证据**：执行过的命令与结果摘要（静态检查、本地 smoke、构建、手工验证），以及未覆盖的部分。
- **检查清单**：Issue 关联、代码量、文档同步、无构建型 CI、本地/CI 结果。
- 风险与回滚：涉及数据格式、公开 API、配置默认值时必填。

## 5. 禁止事项

- 直接推送 `main`，或合并未经双评审的 PR。
- PR 中夹带无关格式化、无关重命名、无关依赖升级。
- 提交构建产物（`dist/`、`build/`、`*.whl`、`*.egg-info`）、虚拟环境、数据库与日志文件。
- 在 PR CI 中加入构建、依赖安装或全量测试（见 [CI_POLICY.md](CI_POLICY.md)）。
- 修改他人 worktree/分支，或强推（force push）已被评审的提交（需重新评审）。
- 用 `--no-verify` 绕过本地钩子、用"CI 会过"代替本地自检。
- 在同一个 PR 里顺手升级版本号（发布 PR 除外，见 [VERSIONING.md](VERSIONING.md)）。
- 单文件超过 1000 行且未进入 ratchet allowlist（见 [CI_POLICY.md](CI_POLICY.md)）。
