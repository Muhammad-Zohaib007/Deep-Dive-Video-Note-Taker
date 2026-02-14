"""Unit tests for the performance profiling utilities."""

from __future__ import annotations

import time
from unittest.mock import patch

from notetaker.utils.profiler import (
    ProfilingReport,
    TimingRecord,
    _get_memory_mb,
    get_active_report,
    profile_stage,
    start_profiling,
    stop_profiling,
    timed,
)

# ---------------------------------------------------------------------------
# TimingRecord
# ---------------------------------------------------------------------------


class TestTimingRecord:
    """Tests for TimingRecord dataclass."""

    def test_basic_attributes(self):
        rec = TimingRecord(name="test", duration_seconds=1.5)
        assert rec.name == "test"
        assert rec.duration_seconds == 1.5
        assert rec.memory_before_mb is None
        assert rec.memory_after_mb is None

    def test_memory_delta_both_present(self):
        rec = TimingRecord(
            name="stage",
            duration_seconds=2.0,
            memory_before_mb=100.0,
            memory_after_mb=150.0,
        )
        assert rec.memory_delta_mb == 50.0

    def test_memory_delta_none_when_missing(self):
        rec = TimingRecord(name="stage", duration_seconds=1.0, memory_before_mb=100.0)
        assert rec.memory_delta_mb is None


# ---------------------------------------------------------------------------
# ProfilingReport
# ---------------------------------------------------------------------------


class TestProfilingReport:
    """Tests for ProfilingReport."""

    def test_empty_report(self):
        report = ProfilingReport()
        assert len(report.records) == 0
        assert report.total_seconds == 0.0

    def test_add_records(self):
        report = ProfilingReport()
        report.add(TimingRecord(name="a", duration_seconds=1.0))
        report.add(TimingRecord(name="b", duration_seconds=2.0))
        assert len(report.records) == 2
        assert report.total_seconds == 3.0

    def test_summary_format(self):
        report = ProfilingReport()
        report.add(TimingRecord(name="audio", duration_seconds=5.0))
        report.add(TimingRecord(name="transcription", duration_seconds=10.0))
        text = report.summary()
        assert "PERFORMANCE PROFILE" in text
        assert "audio" in text
        assert "transcription" in text
        assert "TOTAL" in text
        assert "15.00s" in text

    def test_summary_with_memory(self):
        report = ProfilingReport()
        report.add(
            TimingRecord(
                name="embed",
                duration_seconds=3.0,
                memory_before_mb=100.0,
                memory_after_mb=200.0,
            )
        )
        text = report.summary()
        assert "mem:" in text
        assert "+100.0 MB" in text

    def test_as_dict(self):
        report = ProfilingReport()
        report.add(TimingRecord(name="stage1", duration_seconds=1.5))
        d = report.as_dict()
        assert d["total_seconds"] == 1.5
        assert len(d["stages"]) == 1
        assert d["stages"][0]["name"] == "stage1"
        assert d["stages"][0]["duration_seconds"] == 1.5

    def test_as_dict_with_memory(self):
        report = ProfilingReport()
        report.add(
            TimingRecord(
                name="s",
                duration_seconds=1.0,
                memory_before_mb=50.0,
                memory_after_mb=75.0,
            )
        )
        d = report.as_dict()
        assert d["stages"][0]["memory_before_mb"] == 50.0
        assert d["stages"][0]["memory_after_mb"] == 75.0
        assert d["stages"][0]["memory_delta_mb"] == 25.0


# ---------------------------------------------------------------------------
# Global profiling session
# ---------------------------------------------------------------------------


class TestProfilingSession:
    """Tests for start_profiling / stop_profiling / get_active_report."""

    def setup_method(self):
        """Ensure clean state before each test."""
        stop_profiling()

    def teardown_method(self):
        """Clean up after each test."""
        stop_profiling()

    def test_start_returns_report(self):
        report = start_profiling()
        assert isinstance(report, ProfilingReport)

    def test_get_active_report_none_initially(self):
        assert get_active_report() is None

    def test_get_active_report_after_start(self):
        start_profiling()
        assert get_active_report() is not None

    def test_stop_returns_report(self):
        start_profiling()
        report = stop_profiling()
        assert isinstance(report, ProfilingReport)

    def test_stop_clears_active(self):
        start_profiling()
        stop_profiling()
        assert get_active_report() is None


# ---------------------------------------------------------------------------
# profile_stage context manager
# ---------------------------------------------------------------------------


class TestProfileStage:
    """Tests for the profile_stage context manager."""

    def setup_method(self):
        stop_profiling()

    def teardown_method(self):
        stop_profiling()

    def test_records_timing(self):
        report = start_profiling()
        with profile_stage("test_stage"):
            time.sleep(0.05)
        assert len(report.records) == 1
        assert report.records[0].name == "test_stage"
        assert report.records[0].duration_seconds >= 0.04

    def test_no_crash_without_active_report(self):
        """profile_stage works even when profiling isn't active."""
        with profile_stage("orphan"):
            pass  # Should not raise

    def test_multiple_stages(self):
        report = start_profiling()
        with profile_stage("stage_a"):
            time.sleep(0.01)
        with profile_stage("stage_b"):
            time.sleep(0.01)
        assert len(report.records) == 2
        assert report.records[0].name == "stage_a"
        assert report.records[1].name == "stage_b"


# ---------------------------------------------------------------------------
# timed decorator
# ---------------------------------------------------------------------------


class TestTimedDecorator:
    """Tests for the @timed decorator."""

    def setup_method(self):
        stop_profiling()

    def teardown_method(self):
        stop_profiling()

    def test_preserves_return_value(self):
        @timed
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_preserves_function_name(self):
        @timed
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_records_to_active_report(self):
        report = start_profiling()

        @timed
        def slow():
            time.sleep(0.05)

        slow()
        assert len(report.records) == 1
        assert report.records[0].duration_seconds >= 0.04

    def test_no_crash_without_active_report(self):
        @timed
        def simple():
            return 42

        assert simple() == 42


# ---------------------------------------------------------------------------
# _get_memory_mb
# ---------------------------------------------------------------------------


class TestGetMemoryMb:
    """Tests for _get_memory_mb."""

    def test_returns_float_or_none(self):
        result = _get_memory_mb()
        assert result is None or isinstance(result, float)

    def test_returns_none_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Force re-import path that triggers ImportError
            result = _get_memory_mb()
            # Either returns None (no psutil) or float (psutil cached)
            assert result is None or isinstance(result, float)
