# NWCortex OpenClaw Memory Plugin

将 NWC（NeuroWeave Cortex）接入 OpenClaw 作为记忆后端。

## 安装

```bash
# 1. 安装 NWC MCP server
pip install "NWcortex[mcp]"

# 2. 将此目录软链接到 OpenClaw 的 extensions 目录
ln -s $(pwd)/openclaw-plugin /path/to/openclaw/extensions/nwcortex

# 3. 在 openclaw 配置中启用
# plugins.slots.memory = "nwcortex"
```

## 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `storagePath` | `~/.openclaw/nwcortex/memory.json` | 记忆存储路径 |
| `pythonCommand` | `nwc-mcp` | MCP server 启动命令 |
| `autoRecall` | `false` | 自动注入相关记忆到上下文 |
| `autoCapture` | `false` | 自动捕获对话中的重要信息 |

## 功能

- **3 个工具**: `memory_store`, `memory_forget`, `memory_stats`
- **2 个生命周期钩子**: `before_prompt_build`（自动回忆）、`agent_end`（自动捕获）
- **5 个 CLI 命令**: `nwc recall|store|stats|sleep|profile`
- **记忆后端**: QMD（通过 nwc-mcp JSON-RPC over stdio）
