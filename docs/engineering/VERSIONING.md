# 版本规范（VERSIONING.md）

## 1. 版本号规则

- 采用 **SemVer**：`MAJOR.MINOR.PATCH`。
- **每次迭代发布 +0.0.1**（PATCH +1），例如 `1.4.2` → `1.4.3`；当前版本始终以 `star_graph/_version.py` 的 `__version__` 为准（见 §2）。
- `MAJOR` / `MINOR` 的跳变只在明确的不兼容变更或大版本里程碑时由 Lead 决定，不随日常迭代发生。
- 版本号属于**发布动作**，不属于功能 PR：一个迭代只在一个专门的发布 PR 里改一次。

## 2. 版本源：单一源 `star_graph/_version.py`

**唯一事实源：`star_graph/_version.py` 的 `__version__`。** 该模块**零 import**：setuptools 通过 `[tool.setuptools.dynamic]` 读 `attr`，`scripts/check_version.py` 用 `ast` 静态解析，都不会为了读一个字符串而导入整个运行时。文件同时定义 `__version_info__` 元组，由 `scripts/bump_version.py` 一并更新。

| 位置 | 形态 | 角色 |
|------|------|------|
| `star_graph/_version.py` | `__version__ = "x.y.z"` | **唯一源**（只有这里可以改版本） |
| `pyproject.toml` | `dynamic = ["version"]` + `[tool.setuptools.dynamic] version = {attr = "star_graph._version.__version__"}` | 无 `[project].version`；构建元数据从唯一源派生 |
| `star_graph/__init__.py` | `from ._version import __version__` | 导入再导出 |
| `nwc/__init__.py` | `from star_graph._version import __version__` | 导入再导出 |
| `README.md` | 版本徽章 `badge/version-x.y.z` | 文档镜像 |
| `README_CN.md` | 首行 `vx.y.z \| ...` | 文档镜像 |
| `plan.md` | 顶部 `> 最后更新: ... \| **vx.y.z**` | 文档镜像 |
| `ROADMAP.md` | `## 现状（vx.y.z — ...）` 与 `当前版本 vx.y.z` | 文档镜像 |
| `nwc/api/server.py` | `from star_graph._version import __version__` → `version=__version__` | 从唯一源导入 |
| `star_graph/contrib/mcp_server.py` | 同上 | 从唯一源导入 |
| `nwc/mcp/server.py` | 同上（`serverInfo.version`） | 从唯一源导入 |
| `star_graph/cortex_api/server.py` | `from .._version import __version__` | 从唯一源导入 |
| `star_graph/defaults.yaml` | **无** `version:` 字段 | 版本不进入运行时配置 |

`scripts/check_version.py` 的 **16 项**检查（静态文本 / `ast` / `tomllib`，不 import 包）：

| 类别 | 项数 | 内容 |
|------|:----:|------|
| 唯一源 | 2 | `_version.py` 存在且为字符串字面量 `__version__`、模块内无 import；`star_graph/` 与 `nwc/` 内没有第二个 `__version__ = ...` |
| pyproject | 1 | `dynamic == ["version"]`、`attr == star_graph._version.__version__`、无静态 `[project].version` |
| 包入口 | 2 | `star_graph/__init__.py`、`nwc/__init__.py` 从 `_version` 导入且不硬编码版本字符串 |
| 文档镜像 | 5 | README 徽章、README_CN 首行、plan.md 顶部、ROADMAP 现状标题、ROADMAP `当前版本` |
| defaults | 1 | `star_graph/defaults.yaml` 无顶层 `version:` |
| server 模块 | 4 | `nwc/api/server.py`、`star_graph/contrib/mcp_server.py`、`nwc/mcp/server.py`、`star_graph/cortex_api/server.py` 从 `_version` 导入 |
| release 工作流 | 1 | `.github/workflows/verify.yml` 的 tag 校验读 `_version.py`，而非已不存在的静态 `[project].version` |

合计 2 + 1 + 2 + 5 + 1 + 4 + 1 = **16 项**。`--verbose` 会逐项打印 `file:line` 证据。

## 3. 校验

```bash
python scripts/check_version.py --verbose   # 16 项逐项证据
python scripts/bump_version.py --check      # 复用同一套校验（只读）
```

- 两个脚本都是纯标准库、静态检查（不 import 包），已纳入 PR CI，见 [CI_POLICY.md](CI_POLICY.md)。
- 校验失败时：**改镜像/导入，不改唯一源**；确需改版本号只能走发布 PR（§5）。
- `--check` 与 `bump_version.py --check` 共用 `check_version` 的实现，不会出现两套口径。

## 4. 历史（v1.4.3 已完成迁移）

- v1.4.3（PR #22，`Closes #5`、`Fixes #14`）完成从"多副本 + `pyproject.toml` 基准"到"`_version.py` 单一源"的迁移：
  - 新增 `star_graph/_version.py`（零 import，含 `__version_info__`）；
  - `pyproject.toml` 改为 `dynamic = ["version"]` + `[tool.setuptools.dynamic]`，删除静态 `[project].version`；
  - `star_graph/__init__.py`、`nwc/__init__.py` 与 4 个 server 模块（`nwc/api/server.py`、`star_graph/contrib/mcp_server.py`、`nwc/mcp/server.py`、`star_graph/cortex_api/server.py`）改为从 `_version` 导入；
  - 删除 `star_graph/defaults.yaml` 中未被 `Config.from_yaml` 使用的 `version: "1.3.5"`；
  - `scripts/check_version.py` 从 4 项文本比对扩展为 16 项；新增 `scripts/bump_version.py`。
- 迁移前的事实基准是 `pyproject.toml` 的 `[project].version`，且 defaults.yaml 与部分 server 存在硬编码漂移（由 #14 收敛）。**这些旧口径已过期**：文档中不应再出现"4 处副本""手工同步 defaults/server""通过 ≠ 一致"之类表述。

## 5. 发布动作

1. 新建 Issue（`chore` 或 `governance` 标签），说明本次迭代范围。
2. 一个发布 PR：执行 `python scripts/bump_version.py --bump patch`。**`star_graph/_version.py` 是唯一修改点**，脚本在同一次提交里自动同步 4 个文档镜像（README 徽章、README_CN 首行、plan.md 顶部、ROADMAP 现状/当前版本）；不要手工改这些文件，历史条目（plan.md 版本历史、ROADMAP 历史 bullet）不会被改写。
3. 跑 `python scripts/bump_version.py --check`（等价 `check_version.py` 的 16 项）确认全绿，再通过静态检查与双评审后 squash 合并。
4. 打 annotated tag 并推送（tag 必须与 `star_graph/_version.py` 一致，`verify.yml` 直接读该文件比对）：

```bash
git tag -a v1.4.3 -m "v1.4.3"   # 示例：实际版本以 star_graph/_version.py 为准
git push origin v1.4.3
```

5. 推送 tag 会触发 `verify` 工作流（构建 + smoke）；执行 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) 并确认其绿色。

## 6. 禁止事项

- 跳号（`1.4.2` → `1.4.5`）、回退版本号、复用已发布版本号。
- 在功能/修复 PR 中顺手改版本。
- 绕过 `scripts/bump_version.py` 手工改 `_version.py` 或文档镜像（必然漏掉某一处）。
- 在 `pyproject.toml` 恢复静态 `[project].version`，或在 `star_graph/`、`nwc/` 内新增第二个 `__version__` 定义。
- 移动或删除已发布的 tag。
