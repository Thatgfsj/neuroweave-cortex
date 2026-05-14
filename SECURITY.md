# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.6.x   | yes |
| < 1.6.0 | no |

## Reporting a Vulnerability

NeuroWeave Cortex is an infrastructure-level memory runtime for AI agents. Vulnerabilities may include:

- **Injection**: malicious text content causing unexpected behavior in embedding/search paths
- **Data leakage**: unauthorized access to stored memories
- **Denial of service**: graph explosion or memory exhaustion attacks

**Do NOT open a public issue for security vulnerabilities.**

Report vulnerabilities to the repository maintainer via GitHub's private vulnerability reporting:  
https://github.com/Thatgfsj/neuroweave-cortex/security/advisories/new

Include:
- Steps to reproduce
- Affected components
- Potential impact
- Suggested fix (if available)

Response target: 48 hours for critical, 7 days for moderate.

## Security Design Notes

NWC stores user conversation memories. Defense in depth:

1. **No network ingress by default** — NWC runs in-process, not as a server
2. **No default LLM integration** — atom_facts and compiler use template providers unless explicitly configured
3. **Graph size limits** — `max_total_anchors`, `max_edges_per_node` prevent unbounded growth
4. **Write gate** — pre-storage quality filter blocks noise/junk entry
5. **MCP server is optional** — not loaded unless explicitly enabled
