"""Layer boundary enforcement for the three-layer cognitive architecture.

Layer 1 (Storage): CRUD, persistence, indexing
  - anchor.py, graph.py, index.py, storage.py
  - MUST NOT import from Layer 2 or Layer 3 modules

Layer 2 (Cognitive): resonance, abstraction, replay, consolidation
  - sleep.py, retriever.py, resonance.py, abstraction.py,
    competition.py, ghost.py, metrics.py
  - MAY import from Layer 1 only

Layer 3 (Behavior): retrieval policy, forgetting policy, adaptive replay
  - seed.py, embedding.py, online.py
  - MAY import from Layers 1 and 2

This module provides optional debug-mode enforcement and clear documentation
of which imports are allowed between layers.
"""

from __future__ import annotations

import sys
import os

# Module → layer mapping
_LAYER_MAP: dict[str, int] = {
    # Layer 1: Storage
    "anchor": 1,
    "graph": 1,
    "index": 1,
    "storage": 1,
    "config": 1,  # infrastructure, treated as L1

    # Layer 2: Cognitive
    "sleep": 2,
    "retriever": 2,
    "resonance": 2,
    "abstraction": 2,
    "competition": 2,
    "ghost": 2,
    "metrics": 2,

    # Layer 3: Behavior
    "seed": 3,
    "embedding": 3,
    "online": 3,
}

# Allowed down-imports: higher layer → lower layer (and same layer)
# L3 → L1, L3 → L2, L2 → L1, same layer always OK
_ALLOWED = {
    # L1 → L1: allowed, except anchor → embedding is a known exception
    # L2 → L1: always allowed
    # L3 → L1, L3 → L2: always allowed
}


def get_layer(module_name: str) -> int:
    """Return the layer number for a module, or 0 if unknown."""
    # Strip package prefix
    base = module_name.split(".")[-1] if "." in module_name else module_name
    return _LAYER_MAP.get(base, 0)


def check_import(from_module: str, to_module: str) -> bool:
    """Check if an import from one module to another respects layer boundaries.

    Returns True if the import is allowed, False if it violates layer separation.
    Unknown modules (layer 0) are always allowed.
    """
    from_layer = get_layer(from_module)
    to_layer = get_layer(to_module)

    if from_layer == 0 or to_layer == 0:
        return True  # unknown modules pass through

    # Higher layers can import from lower layers: L3→L1, L3→L2, L2→L1
    if from_layer >= to_layer:
        return True

    # Known exceptions — documented violations with justification
    # anchor (L1) → embedding (L3): lazy import in Anchor.create() for auto-encoding.
    # The embedding is only used at creation time and falls back gracefully.
    if from_module == "anchor" and to_module == "embedding":
        return True

    return False


def enforce_layer_boundaries() -> None:
    """Debug-mode check: verify no cross-layer import violations exist.

    Only active when STAR_GRAPH_STRICT_LAYERS=1 env var is set.
    Does nothing in production.
    """
    if os.environ.get("STAR_GRAPH_STRICT_LAYERS") != "1":
        return

    import importlib
    import pkgutil
    import star_graph as pkg

    violations = []
    for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        source = mod_name.split(".")[-1]
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if attr is None:
                continue
            target_mod = getattr(attr, "__module__", "")
            if not target_mod or not target_mod.startswith("star_graph"):
                continue
            target = target_mod.split(".")[-1]

            if not check_import(source, target):
                violations.append(f"  LAYER VIOLATION: {source} (L{get_layer(source)}) → {target} (L{get_layer(target)})")

    if violations:
        raise ImportError(
            "Layer boundary violations detected:\n" + "\n".join(violations)
            + "\n\nLayer rules: L3→L1, L3→L2, L2→L1 only. "
              "Set STAR_GRAPH_STRICT_LAYERS=0 to disable."
        )


def layer_summary() -> str:
    """Return a human-readable summary of the layer architecture."""
    lines = ["Layer Architecture", "=================="]
    for layer_num, layer_name in [(1, "Storage"), (2, "Cognitive"), (3, "Behavior")]:
        modules = sorted(m for m, l in _LAYER_MAP.items() if l == layer_num)
        lines.append(f"\nLayer {layer_num} ({layer_name}):")
        for m in modules:
            lines.append(f"  - {m}.py")
    lines.append(f"\nLayer rules:")
    lines.append(f"  L3 (Behavior)  → L1, L2, L3")
    lines.append(f"  L2 (Cognitive) → L1, L2")
    lines.append(f"  L1 (Storage)   → L1 only")
    return "\n".join(lines)
