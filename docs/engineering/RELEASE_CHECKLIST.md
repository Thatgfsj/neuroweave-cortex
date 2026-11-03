# 发布检查清单（RELEASE_CHECKLIST.md）

发布 = 一个版本 +0.0.1 的完整动作（见 [VERSIONING.md](VERSIONING.md)）。以下每项都要有可复现的证据，写进发布 PR 或发布 Issue。

## 1. 版本与变更

- [ ] 发布 Issue 已建（`chore` / `governance` 标签），写明本迭代范围与不包含项
- [ ] 版本号已 +0.0.1（本次 `1.4.2` → `1.4.3`），且 `star_graph/_version.py` 为唯一事实源
- [ ] 所有副本已同步（`pyproject.toml`、`star_graph/__init__.py`、`star_graph/defaults.yaml`、`star_graph/contrib/mcp_server.py`、README 徽章）
- [ ] `python scripts/check_version.py` 通过
- [ ] 发布说明已整理：按 `feat` / `fix` / `docs` / `ci` / `chore` 分组，且与 `git log` 对得上
- [ ] 破坏性变更已显式标注（若有，说明迁移方式）

## 2. 构建

- [ ] 干净环境下构建 wheel：`python -m build --wheel`（或 `pip wheel . -w dist`）
- [ ] 构建日志无警告级以上的打包问题（缺失文件、重复文件、`MANIFEST.in` 漏项）
- [ ] 产物未入库：`git status` 干净，`dist/`、`build/`、`*.egg-info` 均被忽略
- [ ] wheel 内容抽查：包内模块与 `defaults.yaml` 等数据文件齐全

## 3. smoke 测试

- [ ] `bash scripts/smoke_test.sh` 通过
- [ ] 全新虚拟环境安装产物的冒烟：

```bash
python -m venv /tmp/nwc-smoke && /tmp/nwc-smoke/bin/pip install dist/*.whl
/tmp/nwc-smoke/bin/python -c "import star_graph; print(star_graph.__version__)"
```

- [ ] 核心链路最小验证：`MemoryManager` 的 `remember` → `recall` → `sleep` 可跑通（无需外部 LLM、无需 GPU）
- [ ] 结果摘要（版本号、耗时、失败项）记录到发布 PR
- [ ] 测试用临时目录/数据库在结束后删除

## 4. 分支与仓库清理

- [ ] 活跃分支（本地 + 远程，同名算 1，不含 `main`）总数 ≤ 30
- [ ] 已合并分支全部删除：远程 `git fetch --prune`、本地 `git branch -d`
- [ ] 临时 worktree 已移除（`git worktree list` 只剩主克隆）
- [ ] 无未关闭的过期 PR；确需保留的已在 PR 中说明原因
- [ ] 发布相关 Issue 在合并后自动关闭，未关闭的手动确认

## 5. 发布动作

- [ ] 发布 PR 已 review + squash 合并，`main` CI（最终验证）通过
- [ ] 打 annotated tag 并推送：

```bash
git tag -a v1.4.3 -m "v1.4.3"
git push origin v1.4.3
```

- [ ] GitHub Release 创建，说明取自发布 PR 的变更摘要
- [ ] 确认 PyPI 等分发渠道（如启用）使用同一版本号，未重复发布

## 6. 发布后

- [ ] 检查 `main` 上 `verify` 工作流为绿色；24 小时内无回归 Issue（若有按 [WORKFLOW.md](WORKFLOW.md) 走 `bug` 流程）
- [ ] 更新下一迭代的 ROADMAP/plan 状态（另开 Issue），并记录本次发布卡点

## 发布禁止事项

- 未跑构建与 smoke 就打 tag，或用本地 editable 包冒充"产物可用"。
- 发布 PR 里夹带功能改动。
- 发布后修改已推送的 tag 或复用同一版本号。
