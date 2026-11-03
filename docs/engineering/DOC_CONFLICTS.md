# 文档冲突处理（DOC_CONFLICTS.md）

> 本文件是 [DOC_POLICY.md](DOC_POLICY.md) §1 的落地索引，Issue #6 的验收物之一。

## 1. 权威顺序（高 → 低）

1. **`README.md`** —— 项目对外唯一事实源：版本、安装、Quick Start、能力总览。
2. **`docs/spec/*.md`** —— 产品/算法规范（目前为 `user-center-memory-chain.md`）。与 README 冲突时以 README 为准，并在对应 Issue 记录待修一方。
3. **`ENGINEERING.md` + `docs/engineering/*`** —— 工程流程与 CI/分支/版本规则；不得与 README 的产品事实冲突。
4. **`docs/architecture.md`、`plan.md`、`ROADMAP.md`** —— 说明性/历史性文档。冲突时以 1–3 为准；历史数值必须带"历史快照/统计口径"标注，不得作为现状引用。

## 2. 冲突判定规则

- **事实冲突**（版本号、模块/测试计数、安装命令、API 名称）：以代码 + `README` + 统计脚本实测为准；文档必须改，不能保留两套说法。
- **口径冲突**（同一个数字不同统计范围）：以 `python scripts/repo_stats.py` 与 CI 脚本（`check_version.py`、`check_line_budget.py`）的实现为准，并在文档里写清口径与命令。
- **历史叙述冲突**：允许保留历史快照，但必须明确标注"历史（版本/日期）"并指向现状文档。
- 同一事实只允许有一个"当前值"；其它位置引用它，不复制数字。

## 3. 处理流程

1. 新建议题（`docs` 标签），列出冲突双方、文件:行、事实依据（命令输出/SHA）。
2. 一个 PR 只解决一类冲突；改完运行：
   - `python scripts/check_version.py`
   - `python scripts/repo_stats.py`
   - `python -m pytest tests/test_readme_doctest.py -q`
   - 相对链接扫描（改动文件内链接目标必须存在）。
3. PR 描述附"旧值 → 新表述"清单；双评审通过后 squash 合并。

## 4. 已知的权威文件

| 文件 | 层级 | 备注 |
|------|:----:|------|
| `README.md` / `README_CN.md` | 1 | 版本、安装、Quick Start、能力总览 |
| `docs/spec/user-center-memory-chain.md` | 2 | 用户中心记忆链（四维接近度）规范 |
| `ENGINEERING.md` / `docs/engineering/*` | 3 | 流程、CI、分支、版本、评审 |
| `docs/architecture.md` / `plan.md` / `ROADMAP.md` | 4 | 历史/规划，标注快照 |
