# 版本规范（VERSIONING.md）

## 1. 版本号规则

- 采用 **SemVer**：`MAJOR.MINOR.PATCH`。
- **每次迭代发布 +0.0.1**（PATCH +1），例如 `1.4.2` → `1.4.3`；当前版本始终以 `pyproject.toml` 的 `[project].version` 为准（见 §2）。
- `MAJOR` / `MINOR` 的跳变只在明确的不兼容变更或大版本里程碑时由 Lead 决定，不随日常迭代发生。
- 版本号属于**发布动作**，不属于功能 PR：一个迭代只在一个专门的发布 PR 里改一次。

## 2. 版本源：当前基准与目标态

**当前事实基准（`scripts/check_version.py` 的 `REFERENCE`）：`pyproject.toml` 的 `[project].version`。**

**目标态（#5 落地中）：`star_graph/_version.py` 作为单一版本源，其余位置为副本。** 该文件当前不存在；#5 合入后把本节改为最终口径，并删除 §4 的迁移说明。

`scripts/check_version.py` 实际校验下列 4 处（静态文本检查，不 import 包）：

| 位置 | 形态 | 校验 |
|------|------|:----:|
| `pyproject.toml` | `[project]` 段的 `version = "x.y.z"` | ✅ 基准 |
| `star_graph/__init__.py` | `__version__ = "x.y.z"`（对外导出） | ✅ |
| `nwc/__init__.py` | `__version__ = "x.y.z"` | ✅ |
| `README.md` | 版本徽章 `version-x.y.z`（接受徽章路径或 shields.io URL） | ✅ |
| `star_graph/_version.py` | `__version__ = "x.y.z"` | ⏳ 目标态，**#5 落地后**加入 |
| `star_graph/defaults.yaml` | `version: "x.y.z"` | ❌ 漂移中，**#14 收敛后**纳入 |
| `nwc/api/server.py` | `version="x.y.z"` | ❌ 漂移中，**#14 收敛后**纳入 |
| `star_graph/contrib/mcp_server.py` | `version="x.y.z"` | ❌ 漂移中，**#14 收敛后**纳入 |

- `check_version.py` 源码顶部以 `TODO(#14)` 明确排除 defaults.yaml 与 server 模块；因此**脚本通过 ≠ 所有副本一致**，发布前必须人工核对表中标 ❌ 的位置。
- 版本字符串的迁移期例外：目标态要求代码不硬编码版本号，需要展示版本时从 `star_graph.__version__` 读取（#5 后从 `star_graph._version` 读取）；但 `nwc/api/server.py` 与 `star_graph/contrib/mcp_server.py` 目前仍是硬编码副本，**#5/#14 收敛前允许存在**，由发布 PR 手工同步。收敛后删除本例外。

## 3. 校验

```bash
python scripts/check_version.py
```

- 该校验是静态文本检查（不 import 包），已纳入 PR CI，见 [CI_POLICY.md](CI_POLICY.md)。
- 校验失败时：**改副本，不改变基准**（当前基准是 `pyproject.toml`）；若基准本身要改，只能通过发布 PR。

## 4. 迁移状态

- 当前事实基准是 `pyproject.toml` 的 `[project].version`；`star_graph/_version.py` 是 **Issue #5** 的目标态，**不是** #4 引入的——#4（PR #15）只交付 `scripts/check_version.py`。
- `star_graph/_version.py` 尚未创建；在此之前不要在任何文档或脚本中把它当作已存在的事实源。
- 已知漂移（由 **Issue #14** 收敛）：`star_graph/defaults.yaml` 与 server 模块（`nwc/api/server.py`、`star_graph/contrib/mcp_server.py`）落后于基准，`check_version.py` 有意不检查。
- #5 合入后：把 §2 表格收敛为"`_version.py` 单一基准 + 副本"，删除本节与 §2 的迁移期例外。

## 5. 发布动作

1. 新建 Issue（`chore` 或 `governance` 标签），说明本次迭代范围。
2. 一个发布 PR：把 `pyproject.toml` 的 `[project].version` 与 4 处受校验的版本源同步 +0.0.1，并人工同步 `defaults.yaml` 与 server 模块副本 → 更新发布说明。
3. 通过静态检查与双评审后 squash 合并。
4. 打 annotated tag 并推送（tag 必须与 `pyproject.toml` 版本一致）：

```bash
git tag -a v1.4.3 -m "v1.4.3"   # 示例：实际版本以 pyproject.toml 为准
git push origin v1.4.3
```

5. 推送 tag 会触发 `verify` 工作流（构建 + smoke）；执行 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) 并确认其绿色。

## 6. 禁止事项

- 跳号（`1.4.2` → `1.4.5`）、回退版本号、复用已发布版本号。
- 在功能/修复 PR 中顺手改版本。
- 只改一处版本字符串就提 PR（其余副本不一致）。
- 移动或删除已发布的 tag。
