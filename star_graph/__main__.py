"""Entry point: python -m star_graph [mcp|demo]

    mcp   — start MCP server on stdio (for Claude Desktop, etc.)
    demo  — run emergence demo
"""
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run emergence demo from examples/
        demo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples", "emergence_demo.py",
        )
        with open(demo_path, encoding="utf-8") as f:
            exec(f.read())
    else:
        from .mcp_server import main
        main()
