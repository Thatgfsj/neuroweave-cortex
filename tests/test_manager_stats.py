"""Tests for manager_stats module — ManagerStats dataclass."""

import pytest

from star_graph.manager_stats import ManagerStats


class TestManagerStats:
    def test_defaults(self):
        ms = ManagerStats()
        assert ms.anchors == 0
        assert ms.edges == 0
        assert ms.ghosts == 0
        assert ms.schemas == 0
        assert ms.abstracts == 0
        assert ms.working_memory == 0
        assert ms.cortices == 0
        assert ms.hubs == 0
        assert ms.clusters == 0
        assert ms.cold_anchors == 0
        assert ms.sleep_cycles == 0
        assert ms.total_evolutions == 0
        assert ms.auto_micro_sleeps == 0
        assert ms.auto_full_sleeps == 0
        assert ms.anchors_since_micro == 0
        assert ms.uptime_seconds == 0.0
        assert ms.cognitive_health is None

    def test_custom_values(self):
        ms = ManagerStats(
            anchors=100, edges=200, ghosts=5,
            sleep_cycles=3, uptime_seconds=3600.0,
        )
        assert ms.anchors == 100
        assert ms.edges == 200
        assert ms.ghosts == 5
        assert ms.sleep_cycles == 3
        assert ms.uptime_seconds == 3600.0

    def test_cognitive_health_dict(self):
        ms = ManagerStats(
            cognitive_health={"modularity": 0.5, "communities": 4},
        )
        assert ms.cognitive_health is not None
        assert ms.cognitive_health["modularity"] == 0.5
