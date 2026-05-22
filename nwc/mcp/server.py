"""NWC MCP Server — Model Context Protocol server for AI agent integration.

Enhanced MCP server with rich return format (memory, entities, relations, summary),
context tool, reflect tool, and Phase 6 SelfModel prompt injection.

Tools exposed:
    remember, remember_working, recall, forget, sleep, consolidate,
    stats, fuzzy_recall, get_profile, evolve, save, load,
    context (new), reflect (new)
"""

import json
import sys
from pathlib import Path
from typing import Optional


class McpServer:
    """NWC MCP Server — stdio-based, Model Context Protocol compatible.

    Connect from Claude Desktop, OpenClaw, Cursor, or any MCP-compatible agent.
    """

    def __init__(self, storage_path: str = "", load_path: str = ""):
        self.storage_path = storage_path
        self.load_path = load_path
        self._cortex = None

    @property
    def cortex(self):
        if self._cortex is None:
            from nwc.core.cortex import Cortex
            self._cortex = Cortex()
            if self.load_path:
                self._cortex.load(self.load_path)
            elif self.storage_path:
                try:
                    self._cortex.load(self.storage_path)
                except Exception:
                    pass
        return self._cortex

    def run(self):
        """Run the MCP server on stdio (JSON-RPC)."""
        # Import here to avoid hard dependency
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            self._run_mcp_lib(Server, stdio_server)
        except ImportError:
            self._run_manual()

    def _run_mcp_lib(self, Server, stdio_server):
        """Run using the mcp Python library."""
        import asyncio

        server = Server("neuroweave-cortex")

        @server.list_tools()
        async def list_tools():
            return [
                {"name": "remember", "description": "Store a long-term memory", "inputSchema": {
                    "type": "object", "properties": {
                        "text": {"type": "string", "description": "Memory content"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
                        "importance": {"type": "number", "description": "Importance (0-1)"},
                        "emotional_valence": {"type": "number", "description": "Emotional valence (-1 to 1)"},
                    }, "required": ["text"],
                }},
                {"name": "remember_working", "description": "Store in working memory buffer", "inputSchema": {
                    "type": "object", "properties": {
                        "text": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    }, "required": ["text"],
                }},
                {"name": "recall", "description": "Context-aware semantic retrieval", "inputSchema": {
                    "type": "object", "properties": {
                        "query": {"type": "string"},
                        "max_items": {"type": "integer", "default": 8},
                    }, "required": ["query"],
                }},
                {"name": "context", "description": "Get compressed cognitive context for LLM injection", "inputSchema": {
                    "type": "object", "properties": {
                        "prompt": {"type": "string", "description": "Current prompt/task for context"},
                    },
                }},
                {"name": "reflect", "description": "Run sleep consolidation — merges, prunes, forms schemas", "inputSchema": {
                    "type": "object", "properties": {},
                }},
                {"name": "evolve", "description": "Memory evolution cycle (decay, boost, conflict resolution)", "inputSchema": {
                    "type": "object", "properties": {},
                }},
                {"name": "forget", "description": "Remove a memory, creating a ghost trace", "inputSchema": {
                    "type": "object", "properties": {
                        "anchor_id": {"type": "string"},
                    }, "required": ["anchor_id"],
                }},
                {"name": "fuzzy_recall", "description": "Low-confidence recall from ghost traces", "inputSchema": {
                    "type": "object", "properties": {
                        "query": {"type": "string"},
                        "threshold": {"type": "number", "default": 0.2},
                    }, "required": ["query"],
                }},
                {"name": "stats", "description": "Memory system statistics", "inputSchema": {
                    "type": "object", "properties": {},
                }},
                {"name": "get_profile", "description": "Inferred user profile from accumulated memories", "inputSchema": {
                    "type": "object", "properties": {},
                }},
                {"name": "save", "description": "Persist memory graph to disk", "inputSchema": {
                    "type": "object", "properties": {
                        "path": {"type": "string"},
                    }, "required": ["path"],
                }},
                {"name": "load", "description": "Load memory graph from disk", "inputSchema": {
                    "type": "object", "properties": {
                        "path": {"type": "string"},
                    }, "required": ["path"],
                }},
                {"name": "consolidate", "description": "Micro-consolidation — incremental, non-blocking", "inputSchema": {
                    "type": "object", "properties": {},
                }},
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            return await asyncio.to_thread(self._handle_tool, name, arguments)

        async def main():
            async with stdio_server() as (read, write):
                await server.run(read, write)

        asyncio.run(main())

    def _handle_tool(self, name: str, args: dict) -> list:
        """Dispatch tool calls. Returns list of content blocks (MCP format)."""
        ctx = self.cortex

        if name == "remember":
            anchor_id = ctx.remember(
                args["text"],
                tags=args.get("tags", []),
                importance=args.get("importance", 0.5),
                emotional_valence=args.get("emotional_valence", 0.0),
            )
            return [{"type": "text", "text": json.dumps({"id": anchor_id, "status": "stored"}, ensure_ascii=False)}]

        elif name == "remember_working":
            # remember_working returns a dict with pseudo-id (WorkingMemoryEntry has no .id)
            result = ctx.remember_working(args["text"], tags=args.get("tags", []))
            return [{"type": "text", "text": json.dumps({**result, "status": "stored_working"}, ensure_ascii=False)}]

        elif name == "recall":
            result = ctx.recall(args["query"], max_items=args.get("max_items", 8))
            return [{"type": "text", "text": json.dumps({
                "memory": result.memory,
                "entities": result.entities,
                "relations": result.relations,
                "summary": result.summary,
            }, ensure_ascii=False)}]

        elif name == "context":
            frame = ctx.context(args.get("prompt", ""))
            return [{"type": "text", "text": json.dumps({
                "focus": frame.focus,
                "active_goals": frame.active_goals,
                "active_concepts": frame.active_concepts,
                "emotional_tone": frame.emotional_tone,
                "summary": frame.summary,
                "system_prompt": frame.to_system_prompt(),
            }, ensure_ascii=False)}]

        elif name == "reflect":
            report = ctx.reflect()
            return [{"type": "text", "text": str(report)}]

        elif name == "evolve":
            report = ctx.evolve()
            return [{"type": "text", "text": str(report)}]

        elif name == "forget":
            ok = ctx.forget(args["anchor_id"])
            return [{"type": "text", "text": json.dumps({"forgotten": ok}, ensure_ascii=False)}]

        elif name == "fuzzy_recall":
            result = ctx.fuzzy_recall(args["query"], threshold=args.get("threshold", 0.2))
            return [{"type": "text", "text": json.dumps({
                "memory": result.memory, "summary": result.summary,
            }, ensure_ascii=False)}]

        elif name == "stats":
            s = ctx.stats()
            # ManagerStats has cognitive_health that may be slow to compute;
            # only include if already loaded (not on first access)
            if hasattr(s, 'to_dict'):
                d = s.to_dict()
            else:
                d = dict(s) if isinstance(s, dict) else {"error": "stats unavailable"}
            # Remove cognitive_health to avoid slow snapshot computation on first access
            d.pop("cognitive_health", None)
            return [{"type": "text", "text": json.dumps(d, ensure_ascii=False, default=str)}]

        elif name == "get_profile":
            profile = ctx.get_profile()
            return [{"type": "text", "text": json.dumps(profile, ensure_ascii=False, default=str)}]

        elif name == "save":
            ctx.save(args["path"])
            return [{"type": "text", "text": f"Saved to {args['path']}"}]

        elif name == "load":
            ctx.load(args["path"])
            return [{"type": "text", "text": f"Loaded from {args['path']}"}]

        elif name == "consolidate":
            self._ensure_legacy()
            if hasattr(ctx._manager, 'micro_consolidate'):
                report = ctx._manager.micro_consolidate()
                return [{"type": "text", "text": str(report)}]
            return [{"type": "text", "text": "Consolidation not available (use reflect/evolve instead)"}]

        else:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]

    def _ensure_legacy(self):
        self.cortex._ensure_loaded()

    def _run_manual(self):
        """Manual JSON-RPC over stdio (fallback when mcp lib not installed)."""
        import asyncio

        async def serve():
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                try:
                    request = json.loads(line.strip())
                    response = await self._handle_rpc(request)
                    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    err = {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None}
                    sys.stdout.write(json.dumps(err) + "\n")
                    sys.stdout.flush()

        asyncio.run(serve())

    async def _handle_rpc(self, request: dict) -> dict:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "neuroweave-cortex", "version": "1.1.0"},
                    "capabilities": {"tools": {}},
                },
            }
        elif method == "tools/list":
            tools = [
                {"name": "remember", "description": "Store a long-term memory"},
                {"name": "remember_working", "description": "Store in working memory buffer"},
                {"name": "recall", "description": "Context-aware semantic retrieval"},
                {"name": "context", "description": "Get compressed cognitive context for LLM injection"},
                {"name": "reflect", "description": "Run sleep consolidation"},
                {"name": "evolve", "description": "Memory evolution cycle"},
                {"name": "forget", "description": "Remove a memory with ghost trace"},
                {"name": "fuzzy_recall", "description": "Low-confidence recall from ghost traces"},
                {"name": "stats", "description": "Memory system statistics"},
                {"name": "get_profile", "description": "Inferred user profile"},
                {"name": "save", "description": "Persist memory graph to disk"},
                {"name": "load", "description": "Load memory graph from disk"},
                {"name": "consolidate", "description": "Micro-consolidation"},
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            import asyncio
            result = await asyncio.to_thread(self._handle_tool, tool_name, arguments)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": result}}
        elif method == "notifications/initialized":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def entry():
    McpServer().run()


if __name__ == "__main__":
    entry()
