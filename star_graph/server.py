"""REST service wrapper for NeuroWeave Cortex memory runtime.

Provides /health, /metrics (Prometheus), and memory CRUD endpoints
for integration with LangChain / LlamaIndex / external agents.

Usage:
    python -m star_graph.server --port 8420
"""

from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .manager import MemoryManager

# Global references set on startup
_manager: Optional[MemoryManager] = None
_start_time: float = 0.0


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


class MemoryHTTPHandler(BaseHTTPRequestHandler):
    """Lightweight REST handler — no external web framework required."""

    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        mgr = get_manager()

        if self.path == "/health":
            self._send_json({
                "status": "healthy",
                "uptime_seconds": time.time() - _start_time,
                "version": "1.0.1",
                "anchors": len(mgr.graph.anchors),
                "edges": len(mgr.graph.edges),
            })

        elif self.path == "/metrics":
            # Prometheus text format
            metrics_lines = [
                "# HELP nwc_anchors_total Total number of memory anchors.",
                "# TYPE nwc_anchors_total gauge",
                f"nwc_anchors_total {len(mgr.graph.anchors)}",
                "# HELP nwc_edges_total Total graph edges.",
                "# TYPE nwc_edges_total gauge",
                f"nwc_edges_total {len(mgr.graph.edges)}",
                "# HELP nwc_uptime_seconds Process uptime in seconds.",
                "# TYPE nwc_uptime_seconds gauge",
                f"nwc_uptime_seconds {time.time() - _start_time:.2f}",
                "# HELP nwc_embedding_fallback_count Embedding provider fallback count.",
                "# TYPE nwc_embedding_fallback_count counter",
            ]
            # Try to get fallback count from mixed provider
            try:
                from .embedding import get_embedder
                embedder = get_embedder()
                fb = getattr(getattr(embedder, 'metrics', None), 'fallback_count', 0)
                metrics_lines.append(f"nwc_embedding_fallback_count {fb}")
            except Exception:
                metrics_lines.append("nwc_embedding_fallback_count 0")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write("\n".join(metrics_lines).encode() + b"\n")

        elif self.path == "/stats":
            self._send_json(mgr.stats.to_dict())

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        mgr = get_manager()
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        if self.path == "/remember":
            text = data.get("text", "")
            if not text:
                self._send_json({"error": "missing 'text'"}, 400)
                return
            tags = data.get("tags", [])
            anchor = mgr.remember(text, tags=tags)
            if anchor is None:
                self._send_json({"error": "memory rejected by write gate"}, 422)
                return
            self._send_json(anchor.to_dict(), 201)

        elif self.path == "/recall":
            query = data.get("query", "")
            max_items = data.get("max_items", 10)
            ctx = mgr.recall(query, max_items=max_items)
            self._send_json(ctx.to_dict())

        elif self.path == "/sleep":
            result = mgr.sleep()
            global_rpt = result.get("global_report")
            if global_rpt is not None:
                rpt_dict = global_rpt.to_dict()
                rpt_dict["duration_seconds"] = round(global_rpt.total_duration_ms / 1000, 2)
                self._send_json(rpt_dict)
            else:
                self._send_json({"status": "sleep_complete", "detail": result}, 200)

        elif self.path == "/consolidate":
            result = mgr.micro_consolidate()
            self._send_json(result)

        else:
            self._send_json({"error": "not found"}, 404)


def start_server(host: str = "0.0.0.0", port: int = 8420):
    """Start the REST service. Blocks until interrupted."""
    global _start_time, _manager
    _start_time = time.time()
    _manager = MemoryManager()

    server = HTTPServer((host, port), MemoryHTTPHandler)
    print(f"NeuroWeave Cortex REST server on http://{host}:{port}")
    print(f"  GET  /health")
    print(f"  GET  /metrics")
    print(f"  GET  /stats")
    print(f"  POST /remember  {{\"text\": \"...\", \"tags\": [...]}}")
    print(f"  POST /recall    {{\"query\": \"...\", \"max_items\": 10}}")
    print(f"  POST /sleep     {{}}")
    print(f"  POST /consolidate {{}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("Server stopped.")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="NeuroWeave Cortex REST Server")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
