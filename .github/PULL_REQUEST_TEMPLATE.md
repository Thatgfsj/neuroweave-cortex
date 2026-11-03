Closes #<issue-number>

<!-- 一个 PR 只做一件事；首行必须是 Closes #N（缺陷修复用 Fixes #N）。规则见 docs/engineering/PR_POLICY.md -->

## 变更摘要

<!-- 改了哪些文件/模块，行为差异是什么。避免逐行复述 diff。 -->

-

## 动机

<!-- 为什么现在做？不改会怎样？对应 Issue 的哪条验收标准？ -->

-

## 验收标准对照

<!-- 逐条列出 Issue 的验收项，并给出本 PR 中的证据位置（文件/命令输出/CI 结果）；未完成或不适用要写原因。 -->

| # | Issue 验收项 | 状态 | 证据 |
|---|--------------|:----:|------|
| 1 |  |  |  |

## 验证证据

<!-- 贴可复现的命令与结果摘要；未覆盖的部分要写出来。 -->

```bash
# 静态检查（与 PR CI 等价）
python scripts/ci/check_syntax.py
python scripts/check_version.py
python scripts/check_line_budget.py
python scripts/ci/check_pr.py

# 需要构建/运行时（涉及打包、依赖、核心链路改动时必填）
# python -m build --wheel
# bash scripts/smoke_test.sh
```

结果：

-

## 检查清单

- [ ] **Issue 关联**：本 PR 关联的是为此 PR 新建的 Issue，类型标签正确，body 含 `Closes #N`
- [ ] **代码量**：代码文件 additions + deletions < 1000 行（文档/数据不计入）；超出已获 Lead 添加 `size/exception` 并写明理由
- [ ] **验收标准对照**：已按上表逐条对应 Issue 验收项，未完成项已注明原因
- [ ] **文档同步**：行为/API/配置/流程变更已更新 README 或 `docs/`；文档链接为相对路径且目标存在
- [ ] **无构建型 CI**：PR 工作流只做静态检查，未新增构建、依赖安装或全量测试
- [ ] **本地或 CI 结果**：上述静态检查通过；如涉及构建/运行，已给出本地 smoke 或 verify 工作流的结果
- [ ] **单文件行数** ≤ 1000；新文件未加入 ratchet allowlist；拆分完成时已从 allowlist 移除
- [ ] **分支规范**：分支名 `type/short-slug`，从 `origin/main` 新建，未直推 `main`
- [ ] **版本号**：功能/修复 PR 未改版本（发布 PR 除外，且 +0.0.1）
- [ ] **无残留**：没有构建产物、虚拟环境、数据库、日志、调试代码与无关格式化

## 风险与回滚

<!-- 涉及数据格式、公开 API、配置默认值时必填 -->

-
