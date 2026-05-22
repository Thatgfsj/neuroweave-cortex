# NeuroWeave Cortex (NWC) — LLM Integration Guide

## Quick Start (3 minutes)

```bash
pip install NWcortex

nwc init          # Interactive setup: pick provider, model, API key
nwc serve         # Start API server on localhost:8765
```

That's it. NWC is now running as a cognitive memory backend for any LLM.

## Architecture

```
┌─────────────────────────────────────┐
│  LLM / Agent (Claude, GPT, etc.)    │
├─────────────────────────────────────┤
│  NWC Runtime (Cognitive Layer)      │
│  ┌───────────────────────────────┐  │
│  │  Perception → Working Memory  │  │
│  │  → Concept Cortex → Salience  │  │
│  │  → Self Model → Prompt        │  │
│  └───────────────────────────────┘  │
├─────────────────────────────────────┤
│  Memory Graph (Storage Layer)       │
│  ┌───────────────────────────────┐  │
│  │  Anchors · Edges · Ghosts     │  │
│  │  TimeSpine · Schemas · Tiers  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

NWC is NOT a dependency of your LLM — it's a cognitive runtime that sits between the LLM and memory. The LLM sees compressed cognitive context, not raw memory dumps.

## Integration Modes

### 1. Python SDK (simplest)

```python
from nwc import Cortex

ctx = Cortex()

# Store memories
ctx.remember("用户喜欢科幻小说", tags=["preference", "books"])
ctx.remember("用户在工作中使用 Rust 和 Python", tags=["skills"])
ctx.remember_working("正在实现新的认证中间件", tags=["current_task"])

# Retrieve with structured output
result = ctx.recall("用户喜欢什么类型的书")
# → RecallResult(
#     memory=[{id, content, score, tags}, ...],
#     entities=["preference", "books"],
#     relations=[],
#     summary="Retrieved 1 memories for: 用户喜欢什么类型的书"
#   )

# Get compressed cognitive context for LLM injection
frame = ctx.context("推荐一本书")
system_prompt = frame.to_system_prompt()
# → "# Cognitive State Summary
#    ## Current Focus: 推荐一本书
#    ## Active Concepts: preference, books, 文学
#    ## Relevant Memories
#    - 用户喜欢科幻小说
#    ..."

# Pass to any LLM
import openai
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "推荐一本书"}
    ]
)

# Maintain memory
ctx.reflect()   # Sleep consolidation — merges, prunes, forms schemas
ctx.evolve()    # Memory evolution — decay, boost, conflict resolution
ctx.save("memories.json")
```

### 2. MCP Server (for MCP-compatible agents)

```bash
# Start MCP server
nwc mcp

# With persistent storage
nwc mcp --storage ~/.nwc/agent_memory.json

# Load existing memories
nwc mcp --load my_memories.json
```

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc",
      "args": ["mcp", "--storage", "~/.nwc/agent_memory.json"]
    }
  }
}
```

**OpenClaw config**:

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc",
      "args": ["mcp", "--storage", "~/.nwc/agent_memory.json"]
    }
  }
}
```

**Cursor config** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "neuroweave-cortex": {
      "command": "nwc",
      "args": ["mcp", "--storage", "~/.nwc/agent_memory.json"]
    }
  }
}
```

**MCP Tools** (14 tools):

| Tool | Description |
|------|-------------|
| `remember` | Store a long-term memory |
| `remember_working` | Store in working memory buffer |
| `recall` | Context-aware semantic retrieval |
| `context` | Get compressed cognitive context for LLM injection |
| `reflect` | Run sleep consolidation |
| `evolve` | Memory evolution cycle |
| `forget` | Remove a memory with ghost trace |
| `fuzzy_recall` | Low-confidence recall from ghost traces |
| `stats` | Memory system statistics |
| `get_profile` | Inferred user profile |
| `save` | Persist memory graph to disk |
| `load` | Load memory graph from disk |
| `consolidate` | Micro-consolidation |
| `reflect` | Full cognitive reflection / summary |

### 3. OpenAI-Compatible API Server

```bash
nwc serve --port 8765
```

The API server is **OpenAI-compatible** — any tool that works with OpenAI's API can use NWC.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (with memory injection) |
| POST | `/memory/write` | Store a memory |
| POST | `/memory/query` | Retrieve memories |
| POST | `/memory/context` | Get cognitive context |
| GET | `/graph` | Memory graph overview |

**Chat with memory injection:**

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nwc-cortex",
    "messages": [
      {"role": "user", "content": "推荐一本适合我的书"}
    ],
    "inject_memory": true
  }'
```

Response includes NWC cognitive context injected as system prompt, or forwarded to your configured LLM.

**Compatible with:**
- OpenWebUI
- Cherry Studio
- NextChat / Lobechat
- OpenClaw
- Any OpenAI-compatible client

### 4. LangChain Integration

```python
from nwc import Cortex
from nwc.llm import get_llm
from nwc.config import get_config

cfg = get_config()
ctx = Cortex()
llm = get_llm(cfg.llm.provider, cfg.llm.model, cfg.llm.api_key)
from nwc.llm.base import ChatMessage

async def chat_with_memory(user_input: str):
    # Get cognitive context
    frame = ctx.context(user_input)
    
    # Build messages with memory
    messages = [
        ChatMessage(role="system", content=frame.to_system_prompt()),
        ChatMessage(role="user", content=user_input),
    ]
    
    response = await llm.chat(messages)
    
    # Store the exchange
    ctx.remember(user_input, tags=["conversation"])
    ctx.remember(response.content, tags=["conversation", "assistant"])
    
    return response.content
```

### 5. AutoGen Integration

```python
from nwc import Cortex

ctx = Cortex()

# Register NWC as a memory tool for AutoGen agents
def nwc_remember(text: str, tags: list[str] = None) -> str:
    """Remember something about the user or conversation."""
    return ctx.remember(text, tags=tags or [])

def nwc_recall(query: str) -> str:
    """Recall relevant memories."""
    result = ctx.recall(query)
    return result.summary + "\n" + "\n".join(
        m.get("content", "") for m in result.memory
    )

def nwc_context(prompt: str = "") -> str:
    """Get cognitive context for decision-making."""
    return ctx.context(prompt).to_system_prompt()

# Register with AutoGen agent
# agent.register_tool(nwc_remember)
# agent.register_tool(nwc_recall)
# agent.register_tool(nwc_context)
```

## Configuration

### Config file (`~/.nwc/config.yaml`)

```yaml
llm:
  provider: deepseek
  model: deepseek-chat
  api_key: sk-xxxx

embedding:
  provider: bge
  model: BAAI/bge-m3

memory:
  backend: cortexgraph
  max_depth: 5
  working_capacity: 20

retrieval:
  top_k: 8
  rerank: true
  fusion: hybrid

storage:
  path: ~/.nwc/data

server:
  host: 127.0.0.1
  port: 8765
```

### Environment variables (override config file)

```bash
export NWC_API_KEY=sk-xxxx
export NWC_MODEL=deepseek-chat
export NWC_PROVIDER=deepseek
export NWC_BASE_URL=https://api.deepseek.com/v1
export NWC_SERVER_PORT=8765
```

**Priority:** ENV > config.yaml > defaults

## Supported LLM Providers

| Provider | Adapter | Models |
|----------|---------|--------|
| OpenAI | `OpenAIAdapter` | gpt-4o, gpt-4-turbo, gpt-3.5-turbo |
| DeepSeek | `DeepSeekAdapter` | deepseek-chat, deepseek-reasoner |
| Anthropic | `AnthropicAdapter` | claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5 |
| Ollama | `OllamaAdapter` | llama3, mistral, qwen, any local model |
| Gemini | `GeminiAdapter` | gemini-2.0-flash, gemini-2.0-pro |

## Supported Embedding Providers

| Provider | Class | Model |
|----------|-------|-------|
| Sentence Transformers | `SentenceTransformersProvider` | BAAI/bge-m3 (default) |
| TF-IDF Fallback | `TfidfFallbackProvider` | No download needed |

## CLI Reference

```
nwc init                  Interactive setup (provider, model, API key)
nwc mcp                   Start MCP server
nwc serve                 Start API server
nwc remember <text>       Store a memory
nwc recall <query>        Retrieve memories
nwc context [prompt]      Get cognitive context
nwc reflect               Run sleep consolidation
nwc evolve                Run memory evolution
nwc stats                 Memory system statistics
nwc save <path>           Persist to disk
nwc load <path>           Load from disk
nwc config                Show current configuration
nwc config-reset          Reset to defaults
```

## Memory Modes (Cortex)

| Mode | Description | API |
|------|-------------|-----|
| **Episodic** | Event/task memory | `ctx.remember("Fixed Redis timeout...")` |
| **Semantic** | Knowledge memory | `ctx.remember("Python 3.12 adds...", tags=["knowledge"])` |
| **Working** | Short-term buffer | `ctx.remember_working("Currently debugging...")` |
| **Emotional** | Emotion-weighted | `ctx.remember("...", emotional_valence=-0.5)` |
| **Reflection** | Long-term summaries | `ctx.reflect()` — sleep consolidation |
| **Procedural** | How-to / patterns | `ctx.remember("Steps to deploy: ...", tags=["procedure"])` |

## What Makes NWC Different

NWC is NOT a vector database with a pretty API. It's a **cognitive runtime**:

- **Memory activation & competition** — not just retrieve, but activate related concepts and let them compete for attention
- **Spreading activation** — query → concept → related concepts → memories (not just KNN)
- **Salience-based attention** — 10-component salience scoring with lateral inhibition
- **Self-model prompt injection** — LLM sees compressed cognitive state, not raw memory dumps
- **Autonomous reasoning** — contradiction detection → activation → resolution (no LLM required)
- **Sleep consolidation** — 8-phase sleep that actually changes the memory graph
- **Ghost revival** — forgotten memories can be revived by related new experiences (savings effect)
- **9-stage lifecycle** — memories move through Perception → Working → ... → Ghost → Dead

This is the difference between "a library that stores vectors" and "an external cognitive cortex for AI agents."
