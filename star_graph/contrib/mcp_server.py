"""MCP Server — exposes star-graph-memory as tools for AI agents.

Usage:
    python -m star_graph.mcp_server

Or from Claude Desktop / MCP client config:
    {
        "mcpServers": {
            "star-graph-memory": {
                "command": "python",
                "args": ["-m", "star_graph.mcp_server"],
                "cwd": "/path/to/star-graph-memory"
            }
        }
    }

Tools exposed:
    remember     — store a memory
    recall       — context-aware retrieval
    forget       — remove a memory (with ghost trace)
    sleep        — run sleep consolidation
    stats        — memory system statistics
    fuzzy_recall — low-confidence recall from ghost traces
    get_profile  — inferred user profile from accumulated memories
    evolve       — run memory evolution (decay, boost, conflict resolution)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server import Server
from mcp.types import Tool, TextContent


# ── Global MemoryManager instance ────────────────────────────

_mgr = None
_storage_path = os.environ.get("STAR_GRAPH_STORAGE_PATH", "")


def _get_manager():
    global _mgr
    if _mgr is None:
        from star_graph import MemoryManager
        _mgr = MemoryManager()
        if _storage_path and os.path.exists(_storage_path):
            _mgr.load(_storage_path)
            sys.stderr.write(f"[star-graph] Loaded from {_storage_path}\n")
    return _mgr


# ── Server ──────────────────────────────────────────────────

server = Server(
    "star-graph-memory",
    version="1.0.2",
    instructions="Cognitive memory runtime for AI agents. Remembers, forgets, "
                  "strengthens, connects, abstracts, and evolves memories across "
                  "conversations. Stores a persistent memory graph with sleep "
                  "consolidation, ghost traces, and emergent abstraction.",
)

TOOLS = [
    Tool(
        name="remember",
        description="Store a memory. Use for facts, preferences, events, decisions, "
                    "and anything worth recalling later.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The memory content to store",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for classification: 'preference', 'debug', "
                                   "'knowledge', 'fact', 'bug', 'style', 'python', etc.",
                },
                "importance": {
                    "type": "number",
                    "description": "0..1 how important this memory is (default 0.5)",
                },
                "emotional_valence": {
                    "type": "number",
                    "description": "-1..+1 emotional charge (negative for bugs/errors, "
                                   "positive for successes)",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session or conversation identifier",
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="recall",
        description="Retrieve memories relevant to a query with context-aware scoring. "
                    "Returns the most relevant memories ranked by semantic similarity, "
                    "recency, and graph structure.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["coding", "debugging", "planning", "reflection", "conversation"],
                    "description": "What the agent is doing — affects which memory types are prioritized",
                },
                "max_items": {
                    "type": "integer",
                    "description": "Maximum memories to return (default 5)",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="forget",
        description="Remove a memory. Creates a ghost trace for potential fuzzy recall later.",
        inputSchema={
            "type": "object",
            "properties": {
                "anchor_id": {
                    "type": "string",
                    "description": "ID of the memory to forget (from recall results)",
                },
            },
            "required": ["anchor_id"],
        },
    ),
    Tool(
        name="sleep",
        description="Run a full 5-phase sleep consolidation cycle. Merges similar memories, "
                    "forms schemas, prunes weak anchors, creates abstractions, and decouples "
                    "emotion from consolidated memories.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="stats",
        description="Get memory system statistics — anchor count, edge count, ghosts, "
                    "schemas, sleep cycles, and cognitive health metrics.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="fuzzy_recall",
        description="Low-confidence recall from ghost traces — 'I seem to remember...'. "
                    "Returns partial memories that were previously forgotten.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in ghost traces",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_profile",
        description="Get an inferred user profile from accumulated memories — preferences, "
                    "habits, technical stack, coding style, and key facts.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="evolve",
        description="Run a memory evolution cycle — applies time decay, frequency boost, "
                    "conflict detection, and interference resolution without sleep.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


@server.list_tools()
async def list_tools():
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    mgr = _get_manager()

    try:
        if name == "remember":
            anchor = mgr.remember(
                text=arguments["text"],
                tags=arguments.get("tags", []),
                importance=arguments.get("importance", 0.5),
                emotional_valence=arguments.get("emotional_valence", 0.0),
                source_session=arguments.get("session_id", ""),
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "stored",
                    "anchor_id": anchor.id,
                    "text_preview": anchor.text[:120],
                    "tags": anchor.tags,
                    "state": anchor.state.value,
                }, ensure_ascii=False, indent=2),
            )]

        elif name == "recall":
            from star_graph.scheduler import AgentContext
            task_type = arguments.get("task_type", "conversation")
            ctx = AgentContext(
                task_type=task_type,
                context_budget_tokens=4000,
            )
            result = mgr.recall(
                query=arguments["query"],
                context=ctx,
                max_items=arguments.get("max_items", 5),
            )
            items_json = []
            for item in result.items:
                items_json.append({
                    "anchor_id": item.anchor.id,
                    "text": item.compressed_text or item.anchor.text[:200],
                    "relevance": round(item.relevance_score, 3),
                    "confidence": round(item.confidence, 3),
                    "memory_type": item.memory_type.value,
                    "compression_level": item.compression_level,
                    "tags": item.anchor.tags,
                })
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": arguments["query"],
                    "summary": result.memory_summary,
                    "total_tokens": result.total_tokens,
                    "latency_ms": round(result.retrieval_latency_ms, 1),
                    "items": items_json,
                    "relevant_facts": result.relevant_facts[:5],
                    "active_patterns": result.active_patterns[:3],
                }, ensure_ascii=False, indent=2),
            )]

        elif name == "forget":
            anchor_id = arguments["anchor_id"]
            anchor = mgr.forget(anchor_id, create_ghost=True)
            if anchor:
                ghost_count = len(mgr.ghosts.ghosts) if mgr._ghosts else 0
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "forgotten",
                        "anchor_id": anchor_id,
                        "text_preview": anchor.text[:120],
                        "ghost_created": True,
                        "total_ghosts": ghost_count,
                    }, ensure_ascii=False, indent=2),
                )]
            return [TextContent(
                type="text",
                text=json.dumps({"status": "not_found", "anchor_id": anchor_id}),
            )]

        elif name == "sleep":
            t0 = time.perf_counter()
            result = mgr.sleep()
            elapsed = time.perf_counter() - t0
            report = result.get("global_report")
            summary = report.summary() if report else "Sleep complete"
            evo = result.get("evolution", {})
            cortex_reports = result.get("cortex_reports", {})
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "sleep_complete",
                    "duration_seconds": round(elapsed, 2),
                    "summary": summary,
                    "anchors_before": report.anchors_before if report else 0,
                    "anchors_after": report.anchors_after if report else 0,
                    "merged": report.memories_merged if report else 0,
                    "pruned": report.memories_pruned if report else 0,
                    "ghosts_created": report.ghosts_created if report else 0,
                    "schemas_formed": report.schemas_formed if report else 0,
                    "compression_ratio": round(report.compression_ratio, 2) if report else 1.0,
                    "evolution_events": evo.get("total_events", 0),
                    "cortex_count": len(cortex_reports),
                }, ensure_ascii=False, indent=2),
            )]

        elif name == "stats":
            s = mgr.stats
            # Add cognitive health
            try:
                health = mgr.metrics.snapshot()
            except Exception:
                health = {}
            return [TextContent(
                type="text",
                text=json.dumps({
                    "anchors": s.anchors,
                    "edges": s.edges,
                    "ghosts": s.ghosts,
                    "schemas": s.schemas,
                    "abstracts": s.abstracts,
                    "sleep_cycles": s.sleep_cycles,
                    "evolutions": s.total_evolutions,
                    "uptime_seconds": round(s.uptime_seconds, 0),
                    "cognitive_health": {
                        "memory_stability": round(health.get("memory_stability", 0), 2),
                        "recall_plasticity": round(health.get("recall_plasticity", 0), 2),
                        "compression_ratio": round(health.get("compression_ratio", 0), 2),
                        "semantic_drift_resistance": round(health.get("semantic_drift_resistance", 0), 2),
                    },
                }, ensure_ascii=False, indent=2),
            )]

        elif name == "fuzzy_recall":
            embedder = mgr._get_embedder()
            embedding = embedder.encode(arguments["query"])
            results = mgr.fuzzy_recall(embedding=embedding)
            items = []
            for desc, score in results[:5]:
                items.append({"fuzzy_text": desc, "confidence": round(score, 3)})
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": arguments["query"],
                    "results": items,
                    "note": "Fuzzy recall from ghost traces — low confidence partial memories",
                }, ensure_ascii=False, indent=2),
            )]

        elif name == "get_profile":
            ctx = mgr.scheduler.get_user_profile()
            return [TextContent(
                type="text",
                text=json.dumps({
                    "summary": ctx.memory_summary,
                    "patterns": ctx.active_patterns,
                    "facts": ctx.relevant_facts,
                    "total_tokens": ctx.total_tokens,
                }, ensure_ascii=False, indent=2),
            )]

        elif name == "evolve":
            result = mgr.evolve()
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "evolution_complete",
                    "cycle": result.get("cycle", 0),
                    "decayed": result.get("decay", {}).get("decayed", 0),
                    "boosted": result.get("boost", {}).get("boosted", 0),
                    "conflicts_resolved": result.get("conflicts", {}).get("resolved", 0),
                    "interference_applied": result.get("interference", {}),
                    "total_events": result.get("total_events", 0),
                }, ensure_ascii=False, indent=2),
            )]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )]

    except Exception as e:
        sys.stderr.write(f"[star-graph] Error in tool '{name}': {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "tool": name}),
        )]


# ── Entry point ─────────────────────────────────────────────

def main():
    """Run the MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
