"""Tests for sleep report data classes."""

from star_graph.sleep_report import PhaseMetrics, SleepReport


class TestPhaseMetrics:
    def test_defaults(self):
        pm = PhaseMetrics()
        assert pm.phase == ""
        assert pm.duration_ms == 0.0
        assert pm.items_processed == 0
        assert pm.details == {}

    def test_with_values(self):
        pm = PhaseMetrics(phase="REM", duration_ms=150.0, items_processed=42,
                          details={"gamma": 0.9})
        assert pm.phase == "REM"
        assert pm.duration_ms == 150.0
        assert pm.items_processed == 42
        assert pm.details["gamma"] == 0.9


class TestSleepReport:
    def test_defaults(self):
        sr = SleepReport()
        assert sr.cycle == 0
        assert sr.total_duration_ms == 0.0
        assert sr.phases == []
        assert sr.compression_ratio == 1.0

    def test_summary_empty_cycle(self):
        sr = SleepReport(cycle=1)
        result = sr.summary()
        assert "no significant changes" in result

    def test_summary_with_replays(self):
        sr = SleepReport(cycle=2, memories_replayed=10)
        result = sr.summary()
        assert "Replayed 10" in result

    def test_summary_with_merges(self):
        sr = SleepReport(cycle=2, memories_merged=5)
        result = sr.summary()
        assert "Merged 5" in result

    def test_summary_with_schemas(self):
        sr = SleepReport(cycle=2, schemas_formed=3)
        result = sr.summary()
        assert "Created 3 schemas" in result

    def test_summary_with_pruned(self):
        sr = SleepReport(cycle=2, memories_pruned=4, ghosts_created=2)
        result = sr.summary()
        assert "Pruned 4 (2 ghosts)" in result

    def test_summary_with_bridges(self):
        sr = SleepReport(cycle=2, bridges_created=7)
        result = sr.summary()
        assert "Bridged 7" in result

    def test_summary_multiple_parts(self):
        sr = SleepReport(cycle=3, memories_replayed=20, memories_merged=8,
                         schemas_formed=2, bridges_created=3)
        result = sr.summary()
        assert " | " in result
        assert "Replayed 20" in result
        assert "Merged 8" in result

    def test_detailed_includes_cycle(self):
        sr = SleepReport(cycle=5, total_duration_ms=1234.0)
        result = sr.detailed()
        assert "Sleep Cycle #5" in result
        assert "1234ms" in result

    def test_detailed_with_phases(self):
        phases = [
            PhaseMetrics(phase="NREM_SWR", duration_ms=100.0, items_processed=30),
            PhaseMetrics(phase="REM", duration_ms=200.0, items_processed=15),
        ]
        sr = SleepReport(cycle=1, total_duration_ms=300.0, phases=phases,
                         anchors_before=100, anchors_after=95,
                         edges_before=200, edges_after=190,
                         avg_retention_before=0.7, avg_retention_after=0.75,
                         compression_ratio=1.05)
        result = sr.detailed()
        assert "NREM_SWR" in result
        assert "REM" in result
        assert "100 → 95" in result
        assert "200 → 190" in result
        assert "0.700" in result
        assert "0.750" in result

    def test_detailed_with_merges_and_schemas(self):
        sr = SleepReport(cycle=1, memories_merged=3, schemas_formed=2)
        result = sr.detailed()
        assert "Merged:   3" in result
        assert "Schemas:  2" in result

    def test_detailed_with_emotional_and_cortical(self):
        sr = SleepReport(cycle=1, emotional_decoupled=4, cortical_transferred=2)
        result = sr.detailed()
        assert "decoupled from 4" in result
        assert "2 memories transferred" in result

    def test_detailed_with_ghosts(self):
        sr = SleepReport(cycle=1, memories_pruned=5, ghosts_created=3)
        result = sr.detailed()
        assert "Pruned:   5" in result
        assert "3 ghosts" in result
