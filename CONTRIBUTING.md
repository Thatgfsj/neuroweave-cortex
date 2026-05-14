# Contributing to NeuroWeave Cortex (NWC)

## Module Addition Review

NeuroWeave Cortex has 78 modules and 33 test files. Before adding a new module:

1. **Scope check** — does this belong in core, or should it be a plugin? Core handles: memory CRUD, graph operations, retrieval, sleep consolidation. Higher-level cognitive features (personality, compiler, self-org) should go in `contrib/` or a separate package.
2. **Open an issue** describing the motivation and design before writing code.
3. **Keep modules small** — target <500 lines. If a module grows past 1000 lines, split it.

## Commit Format

```
area: brief description

Body explaining why, not what.
```

Examples:
- `sleep: fix merge O(n^2) regression in _merge_similar`
- `retrieval: add RetrievalBudget enforcement to recall()`
- `tests: cover write_gate edge cases`

## Code Standards

- Python 3.11+ required — use modern syntax (`X | None`, PEP 604 unions)
- No comments stating *what* code does — names should say that. Comments explain *why*.
- Tests required for new modules — minimum coverage of golden-path + edge-case classes.
- Run `pytest tests/ -q` before pushing. All 649 tests must pass.
- Type annotations required on all public API functions.
- Don't add half-finished implementations or feature flags for hypothetical use cases.

## Pull Requests

- Keep PRs focused: one concern per PR.
- CI must pass (tests + version consistency check).
- Rebase, don't merge.

## Lazy Import Policy

`star_graph/__init__.py` uses PEP 562 lazy imports. When adding a new module:

1. Do NOT add `from .new_module import X` at the top of `__init__.py`.
2. Add entries to the `_LAZY` dict mapping each symbol name to its module and attribute.
3. Verify `from star_graph import YourSymbol` works after the change.
