"""Regression tests for the top-level public API surface (Issue #23).

`star_graph` is a PEP 562 lazy-export package: every top-level name must be
registered in `star_graph.__init__._LAZY`. `RetrievalCore` was exported by
`star_graph.retrieval_engine` but missing from `_LAZY`, so
`from star_graph import RetrievalCore` raised ImportError even though the
class existed. These tests pin the export and its identity.
"""

import star_graph


class TestRetrievalCoreExport:
    def test_importable_from_top_level(self):
        from star_graph import RetrievalCore

        assert RetrievalCore is not None
        assert RetrievalCore.__name__ == "RetrievalCore"

    def test_identity_with_subpackage_and_module(self):
        from star_graph import RetrievalCore
        from star_graph.retrieval_engine import RetrievalCore as SubpackageRetrievalCore
        from star_graph.retrieval_engine.retrieval_core import (
            RetrievalCore as ModuleRetrievalCore,
        )

        assert RetrievalCore is SubpackageRetrievalCore
        assert RetrievalCore is ModuleRetrievalCore

    def test_registered_in_lazy_registry_and_dir(self):
        from star_graph import _LAZY

        assert _LAZY["RetrievalCore"] == (
            "star_graph.retrieval_engine.retrieval_core",
            "RetrievalCore",
        )
        assert "RetrievalCore" in dir(star_graph)

    def test_attribute_access_after_import(self):
        from star_graph import RetrievalCore

        # __getattr__ caches the resolved object in module globals; a second
        # access must return the same object.
        assert star_graph.RetrievalCore is RetrievalCore


class TestExistingEntryPointsStillExport:
    def test_memory_runtime(self):
        from star_graph import MemoryRuntime
        from star_graph.cortex_api.runtime import MemoryRuntime as RuntimeMemoryRuntime

        assert MemoryRuntime is RuntimeMemoryRuntime

    def test_retrieval_pipeline(self):
        from star_graph import RetrievalPipeline
        from star_graph.retrieval_engine.retrieval_pipeline import (
            RetrievalPipeline as ModuleRetrievalPipeline,
        )

        assert RetrievalPipeline is ModuleRetrievalPipeline
