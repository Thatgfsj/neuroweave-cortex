"""RetrievalCore — composition of the retrieval mixins.

The original 1472-line module was split by responsibility (issue #7) — pure
move, no behaviour change. The orchestration methods now live in four mixins:

  - retrieval_core_recall.py     RecallMixin     recall, System-2 fallback,
                                                 temporal-query detection
  - retrieval_core_support.py    SupportMixin    BM25 anchor search, semantic
                                                 scan, compression, explain
  - retrieval_core_spreading.py  SpreadingMixin  spreading activation, sparse
                                                 recall, dimensional descent,
                                                 cascade recall, dual recall
  - retrieval_core_graph.py      GraphMixin      graph-first recall, topology
                                                 ranking, cross-modal search

This module keeps the composed ``RetrievalCore`` class plus the shared imports
that the historical star-import surface relies on
(``star_graph/retrieval_core.py`` does ``from .retrieval_engine.retrieval_core
import *``).

Note: every mixin is a bare mixin — they expect ``self._rt`` (a MemoryRuntime
reference) to be available, set by the inheriting class.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np

from ..scheduler import AgentContext, MemoryContext, MemoryItem, MemoryType
from ..anchor import Anchor
from ..router import RouteResult
from .dual_channel import DualChannelOutput
from ..multimodal import CrossModalResult
from ..math_utils import cosine_sim as _cosine_sim
from ..topology import graph_first_recall, topology_rank
from .retriever import RetrievalTrace, RetrievalTraceEntry

from .retrieval_core_recall import (
    RecallMixin,
    _TEMPORAL_PATTERNS,
    _detect_temporal_query,
)
from .retrieval_core_support import SupportMixin
from .retrieval_core_spreading import SpreadingMixin
from .retrieval_core_graph import GraphMixin


class RetrievalCore(RecallMixin, SupportMixin, SpreadingMixin, GraphMixin):
    """Mixin composition providing all retrieval orchestration methods.

    Expects self._rt to be a MemoryRuntime instance (set by the inheriting class).
    """
