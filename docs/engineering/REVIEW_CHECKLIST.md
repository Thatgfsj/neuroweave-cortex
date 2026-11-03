# 双评审清单（REVIEW_CHECKLIST.md）

每个 PR 由两个**独立**评审角色检查，二者互不替代。评审只针对本 PR 的 diff 与受影响调用方；范围外的改进另开 Issue。

## 评审 A：正确性

- [ ] 改动与 Issue 的验收标准逐条对应，没有夹带无关修改
- [ ] 行为变更向后兼容；公开 API 的签名、默认值、返回类型未被意外改变（若有变更，已在 PR 描述中声明）
- [ ] 错误路径完整：异常被捕获后有明确处理，没有 `except: pass`、没有静默吞错
- [ ] 边界条件：空输入、超长输入、重复调用、并发/重入、资源释放（文件/连接/锁）
- [ ] 持久化兼容：旧数据库/旧快照仍可读取，字段缺失时有默认值
- [ ] 算法与复杂度：检索、扩散激活、睡眠合并等热路径没有引入 O(n²) 或全量扫描
- [ ] 测试：新增或修改的测试覆盖 golden path + 至少一个边界；测试可在无网络、无 GPU 环境下运行
- [ ] 无调试残留（`print`、断点、临时开关）、TODO 已关联 Issue
- [ ] 代码可读性：命名表达意图，注释解释"为什么"而非"做什么"

## 评审 B：工程合规

- [ ] 存在**为此 PR 新建**的 Issue，类型标签正确（`bug`/`feat`/`docs`/`ci`/`refactor`/`chore`/`governance`/`spec`）
- [ ] PR body 首行含 `Closes #N`（缺陷为 `Fixes #N`），模板字段全部填写
- [ ] 代码量口径正确：代码文件 additions+deletions < 1000；超出时有 Lead 添加的 `size/exception` 且写明理由
- [ ] 单文件 ≤ 1000 行；新增文件未进 allowlist；存量条目由 #24 统一清理，拆分 PR 不强制在同一 PR 移除（见 [CI_POLICY.md](CI_POLICY.md) §5）
- [ ] 分支命名符合 `type/short-slug`，从 `origin/main` 新建，一个分支只对应一个 Issue
- [ ] 未直推 `main`；未强推已评审的提交（若已强推，历史被重写，需重新评审）
- [ ] 版本号：功能/修复 PR 未改版本；发布 PR 为 +0.0.1 且 `scripts/check_version.py` 16 项全绿（唯一源 `star_graph/_version.py`，[VERSIONING.md](VERSIONING.md)）
- [ ] PR CI 仅含静态检查，未新增构建、依赖安装或全量测试
- [ ] 文档同步：行为/API/配置变更已更新 README 或对应文档；新增脚本/工作流已更新 [CI_POLICY.md](CI_POLICY.md) 表格
- [ ] 文档链接为相对路径且目标存在；无硬编码易过期计数（[DOC_POLICY.md](DOC_POLICY.md)）
- [ ] 无构建产物、虚拟环境、数据库、日志、大文件；无无关格式化/重命名
- [ ] 提交信息符合 `area: 变更描述`，正文说明原因
- [ ] 合并计划明确：squash + 合并后删除分支，本地 worktree 清理

## 评审结论格式

每条结论必须可执行（通过 / 需修改 + 具体位置 + 建议）。

```markdown
### 评审 A（正确性）：需修改
- [x] 行为兼容
- [ ] 边界：`recall(query="")` 未覆盖空查询，会走全量扫描 → 建议在 `retrieval_core` 入口短路返回空列表
- [x] 测试覆盖 golden path

### 评审 B（工程合规）：通过
- 代码量 412 行；分支 `docs/governance`；版本未变；CI 静态检查通过
```

结论用 PR 评论留痕：

```bash
gh pr review <N> --repo Thatgfsj/neuroweave-cortex --comment --body "<结论>"
```

## 合并门槛

- 两份评审结论均为"通过"且 PR CI 全绿，才可由 Lead 执行 squash 合并。
- 评审提出修改后作者逐条响应（已改 / 说明为何不改），不得静默忽略；评审者不得批准自己的 PR、不得直接改作者代码。

## 评审者禁止事项

- 只看 PR 描述就下结论、未读 diff；以"风格偏好"否决（除非仓库已有明确规范）。
- 把安全/数据丢失风险当作"后续优化"放过。
- 在同一个 PR 里追加新需求（应新建 Issue）。
