"""MCP Server — CLI entry point for NeuroWeave Cortex MCP integration.

Usage:
    nwc-mcp
    nwc-mcp --storage agent_memory.json
    nwc-mcp --load /path/to/saved/memory.json
    python -m star_graph.mcp_server
"""


def entry():
    """CLI entry point registered as ``nwc-mcp`` in pyproject.toml."""
    from ..contrib.mcp_server import main
    main()


# Legacy import support
from ..contrib.mcp_server import server as mcp_server

__all__ = ["mcp_server", "entry"]

# Support ``python -m star_graph.mcp_server``
if __name__ == "__main__":
    entry()
