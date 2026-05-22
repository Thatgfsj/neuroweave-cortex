"""NeuroWeave Cortex (NWC) — Cognitive Runtime for LLM Agents.

Usage:
    from nwc import Cortex
    ctx = Cortex()
    ctx.remember("用户喜欢科幻小说")
    results = ctx.recall("用户喜欢什么")
    context = ctx.context("推荐一本书")

CLI:
    nwc init          # Interactive setup
    nwc mcp           # Start MCP server
    nwc serve         # Start API server
    nwc remember "text"
    nwc recall "query"
"""

__version__ = "1.2.4"
__all__ = ["Cortex", "RecallResult", "ContextFrame"]

from nwc.core.cortex import Cortex, RecallResult, ContextFrame
