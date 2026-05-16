# NeuroWeave Cortex (NWC) — Plan

> Last updated: 2026-05-16 | **v1.0.0** | 1,989 tests passing

## Current State

- **78 modules** in `star_graph/`, **33 test files**, **649 tests**
- Full S/A/B implementation complete: Retrieval Budget, Versioned Memory, Cluster Memory, Causal Edge Types, Episodic Memory
- Lazy imports (PEP 562) — all symbols loaded on first access
- CI pipeline with version consistency checks

## Active Priorities (2026-05-15)

### P1 — Engineering
- [ ] Split oversized modules: sleep.py (84KB), runtime.py (78KB), retrieval_pipeline.py (39KB)
- [ ] PyPI publishing (`pip install neuroweave-cortex`)
- [ ] Test coverage ≥ 80% (currently ~48%)
- [ ] Strict mypy (disallow_untyped_defs = true) — gradual typing

### P2 — Architecture
- [ ] Plugin extraction: non-core modules → `contrib/`
- [ ] Module merges: tier+tiered, edge_budget+edge_decay, cognitive_cache+exact_cache
- [ ] multimodal.py true optional import (no torch/transformers on import)

### P3 — Performance
- [ ] LoCoMo has_answer improvement (15.3% → target 25%)
- [ ] find_contradictions() ANN pre-filter (O(n²) → O(n×k))
- [ ] TimeSpine temporal QA debugging (4.3%)

### P4 — Interface
- [ ] `--format json` on all CLI commands (stats done, query/sleep remaining)
- [ ] Unified sync/async interface

## Architecture Detail

See [docs/architecture.md](docs/architecture.md) for full module inventory, version history, and implementation history.
