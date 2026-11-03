# 版本规范（VERSIONING.md）

## 1. 版本号规则

- 采用 **SemVer**：`MAJOR.MINOR.PATCH`。
- **每次迭代发布 +0.0.1**（PATCH +1）：
  - 当前发布版本 `1.4.2` → 下个迭代 **`1.4.3`** → 再下个 `1.4.4` ……
  - `MAJOR` / `MINOR` 的跳变只在明确的不兼容变更或大版本里程碑时由 Lead 决定，不随日常迭代发生。
- 版本号属于**发布动作**，不属于功能 PR：一个迭代只在一个专门的发布 PR 里改一次。

## 2. 单一版本源

- 唯一事实源：`star_graph/_version.py`

```python
__version__ = "1.4.3"
```

- 其它位置**只允许是副本**，由发布 PR 同步，并由 `scripts/check_version.py` 校验一致：

| 位置 | 形态 |
|------|------|
| `pyproject.toml` | `version = "1.4.3"` |
| `star_graph/__init__.py` | `__version__ = "1.4.3"`（对外导出） |
| `star_graph/defaults.yaml` | `version: "1.4.3"` |
| `star_graph/contrib/mcp_server.py` | `version="1.4.3"` |
| `README.md` | 版本徽章 `version-1.4.3` |

- 任何代码都不得硬编码版本字符串；需要展示版本时从 `star_graph._version` / `star_graph.__version__` 读取。

## 3. 校验

```bash
python scripts/check_version.py
```

- 该校验是静态文本检查（不 import 包），已纳入 PR CI，见 [CI_POLICY.md](CI_POLICY.md)。
- 校验失败时：**改副本，不改变基准**；若基准本身要改，只能通过发布 PR。

## 4. 迁移状态（截至 v1.4.2 / `f63648a`）

- `star_graph/_version.py` 与 `scripts/check_version.py` 由工具链 Issue（#4）引入。落地前，`pyproject.toml` 的 `version` 是事实基准，其余副本必须与它一致。
- 已知漂移：`star_graph/defaults.yaml` 为 `1.3.5`、`star_graph/contrib/mcp_server.py` 为 `1.4.0`，落后于 `1.4.2`；发布 `1.4.3` 时必须一次性对齐。迁移完成后删除本节。

## 5. 发布动作

1. 新建 Issue（`chore` 或 `governance` 标签），说明本次迭代范围。
2. 一个发布 PR：`star_graph/_version.py` 改为 `1.4.3` → 同步全部副本（含 README 徽章）→ 更新发布说明。
3. 通过静态检查与双评审后 squash 合并。
4. 打 annotated tag 并推送：

```bash
git tag -a v1.4.3 -m "v1.4.3"
git push origin v1.4.3
```

5. 执行 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)，确认构建与 smoke 通过。

## 6. 禁止事项

- 跳号（`1.4.2` → `1.4.5`）、回退版本号、复用已发布版本号。
- 在功能/修复 PR 中顺手改版本。
- 只改一处版本字符串就提 PR（其余副本不一致）。
- 移动或删除已发布的 tag。
