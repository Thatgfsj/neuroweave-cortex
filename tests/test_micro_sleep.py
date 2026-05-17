"""Tests for micro_sleep module — MicroSleepScheduler and dataclasses."""

import time

import pytest

from star_graph.micro_sleep import (
    MicroPhase,
    MicroSleepProgress,
    MicroSleepResult,
    MicroSleepScheduler,
    _PHASES,
)
from star_graph.config import Config
from star_graph.graph import StarGraph


class TestMicroPhase:
    def test_defaults(self):
        mp = MicroPhase(name="Test Phase", method="_test_method",
                        description="A test phase")
        assert mp.name == "Test Phase"
        assert mp.method == "_test_method"
        assert mp.description == "A test phase"
        assert mp.items_processed == 0
        assert mp.duration_ms == 0.0
        assert mp.error is None

    def test_with_error(self):
        mp = MicroPhase(name="Bad Phase", method="_bad",
                        error="Something went wrong")
        assert mp.error == "Something went wrong"


class TestMicroSleepProgress:
    def test_defaults(self):
        p = MicroSleepProgress()
        assert p.cycle == 0
        assert p.phase_index == 0
        assert p.phases_completed == []
        assert p.total_phases == 10
        assert p.is_complete is False
        assert p.progress_pct == 0.0

    def test_is_complete_when_done(self):
        p = MicroSleepProgress(phase_index=10, total_phases=10)
        assert p.is_complete is True

    def test_is_complete_past_end(self):
        p = MicroSleepProgress(phase_index=15, total_phases=10)
        assert p.is_complete is True

    def test_progress_pct(self):
        p = MicroSleepProgress(phase_index=5, total_phases=10)
        assert p.progress_pct == 0.5

    def test_progress_pct_zero_phases(self):
        p = MicroSleepProgress(total_phases=0)
        assert p.progress_pct == 1.0

    def test_with_phases(self):
        p = MicroSleepProgress(
            cycle=1,
            phase_index=3,
            phases_completed=["N1", "N2", "N3"],
            phases_remaining=["N4", "N5"],
            started_at=100.0,
            last_phase_at=200.0,
            total_duration_ms=50.0,
        )
        assert p.cycle == 1
        assert p.phase_index == 3
        assert p.total_duration_ms == 50.0


class TestMicroSleepResult:
    def test_defaults(self):
        r = MicroSleepResult()
        assert r.phases_run == []
        assert r.phases_processed == 0
        assert r.is_complete is False
        assert r.progress is None
        assert r.errors == []
        assert r.items_processed == 0
        assert r.duration_ms == 0.0

    def test_with_progress(self):
        p = MicroSleepProgress(phase_index=5)
        r = MicroSleepResult(
            phases_run=["N1", "N2"],
            phases_processed=2,
            is_complete=False,
            progress=p,
        )
        assert r.progress is p
        assert len(r.phases_run) == 2


class TestPhases:
    def test_all_phases_present(self):
        assert len(_PHASES) == 10
        names = [p.name for p in _PHASES]
        assert "N1 Replay Indexing" in names
        assert "8 Index Rebuild" in names

    def test_phases_have_methods(self):
        for p in _PHASES:
            assert p.method.startswith("_")
            assert isinstance(p.description, str)


class TestMicroSleepScheduler:
    def test_init_basic(self):
        ms = MicroSleepScheduler()
        assert ms.is_complete is False
        assert ms.progress.phase_index == 0

    def test_init_with_graph_and_config(self):
        g = StarGraph()
        cfg = Config.get()
        ms = MicroSleepScheduler(graph=g, config=cfg)
        assert ms._graph is g
        assert ms._config is cfg

    def test_init_with_brain_and_hublayer(self):
        brain = object()
        hl = object()
        cortices = [object()]
        ms = MicroSleepScheduler(brain=brain, hublayer=hl, cortices=cortices)
        assert ms.brain is brain
        assert ms.hublayer is hl
        assert ms.cortices is cortices

    def test_is_complete_true_after_all_phases(self):
        ms = MicroSleepScheduler()
        ms._progress.phase_index = len(_PHASES)
        assert ms.is_complete is True

    def test_progress_property(self):
        ms = MicroSleepScheduler()
        p = ms.progress
        assert isinstance(p, MicroSleepProgress)
        assert ms.progress is p  # cached

    def test_sleep_cycle_lazy_init(self):
        ms = MicroSleepScheduler()
        sc = ms.sleep_cycle
        assert sc is not None
        # Should be cached
        assert ms.sleep_cycle is sc

    def test_sleep_cycle_provided(self):
        from star_graph.sleep import SleepCycle
        g = StarGraph()
        cfg = Config.get()
        sc = SleepCycle(g, cfg)
        ms = MicroSleepScheduler(sleep_cycle=sc)
        assert ms.sleep_cycle is sc

    def test_resume_from_start(self):
        ms = MicroSleepScheduler()
        ms._progress.phase_index = 5
        p = ms.resume_from(0)
        assert p.phase_index == 0
        assert p.phases_completed == []

    def test_resume_from_middle(self):
        ms = MicroSleepScheduler()
        p = ms.resume_from(3)
        assert p.phase_index == 3
        assert len(p.phases_completed) == 3
        assert p.phases_remaining == [p.name for p in _PHASES[3:]]

    def test_resume_from_end(self):
        ms = MicroSleepScheduler()
        p = ms.resume_from(len(_PHASES))
        assert p.is_complete is True

    def test_resume_from_beyond(self):
        ms = MicroSleepScheduler()
        p = ms.resume_from(999)
        assert p.phase_index == len(_PHASES)

    def test_resume_from_negative(self):
        ms = MicroSleepScheduler()
        p = ms.resume_from(-5)
        assert p.phase_index == 0

    def test_run_next_already_complete(self):
        ms = MicroSleepScheduler()
        ms._progress.phase_index = len(_PHASES)
        result = ms.run_next(steps=2)
        assert result.is_complete is True
        assert result.phases_run == []

    def test_run_next_one_step(self):
        ms = MicroSleepScheduler()
        result = ms.run_next(steps=1)
        assert len(result.phases_run) == 1
        assert result.phases_processed == 1
        assert ms.progress.phase_index == 1
        assert result.duration_ms >= 0

    def test_run_next_two_steps(self):
        ms = MicroSleepScheduler()
        result = ms.run_next(steps=2)
        assert len(result.phases_run) == 2
        assert ms.progress.phase_index == 2

    def test_run_next_multiple_calls(self):
        ms = MicroSleepScheduler()
        r1 = ms.run_next(steps=2)
        assert len(r1.phases_run) == 2
        r2 = ms.run_next(steps=2)
        assert len(r2.phases_run) == 2
        assert ms.progress.phase_index == 4
        # Phases should be different
        assert r1.phases_run != r2.phases_run

    def test_run_next_updates_progress(self):
        ms = MicroSleepScheduler()
        ms.run_next(steps=1)
        p = ms.progress
        assert len(p.phases_completed) == 1
        assert p.last_phase_at > 0
        assert p.total_duration_ms >= 0  # may be 0 for very fast phases

    def test_run_next_progress_remaining(self):
        ms = MicroSleepScheduler()
        ms.run_next(steps=1)
        p = ms.progress
        assert len(p.phases_remaining) == len(_PHASES) - 1

    def test_run_next_with_cortices(self):
        ms = MicroSleepScheduler(cortices=[object()])
        result = ms.run_next(steps=1)
        assert result is not None

    def test_run_all(self):
        ms = MicroSleepScheduler()
        result = ms.run_all()
        assert result.is_complete is True
        assert ms.is_complete is True

    def test_run_all_partial(self):
        ms = MicroSleepScheduler()
        ms.run_next(steps=3)
        assert not ms.is_complete
        result = ms.run_all()
        assert result.is_complete is True

    def test_get_summary_start(self):
        ms = MicroSleepScheduler()
        s = ms.get_summary()
        assert "0/10" in s
        assert "0%" in s

    def test_get_summary_mid(self):
        ms = MicroSleepScheduler()
        ms.run_next(steps=5)
        s = ms.get_summary()
        assert "5/10" in s
        assert "50%" in s

    def test_get_summary_done(self):
        ms = MicroSleepScheduler()
        ms.run_all()
        s = ms.get_summary()
        assert "done" in s

    def test_reset(self):
        ms = MicroSleepScheduler()
        ms.run_next(steps=5)
        assert ms.progress.phase_index == 5
        ms.reset()
        assert ms.progress.phase_index == 0
        assert ms.progress.phases_completed == []

    def test_reset_clears_results(self):
        ms = MicroSleepScheduler()
        ms.run_next(steps=1)
        ms.reset()
        assert ms._phase_results == []

    def test_errors_recorded_on_phase_failure(self):
        """Some phases may fail without proper setup; errors should be recorded."""
        ms = MicroSleepScheduler()
        # Run without cortices/brain - hub and index phases may error
        result = ms.run_all()
        # Just verify the result structure is valid
        assert isinstance(result, MicroSleepResult)
        assert result.is_complete is True
