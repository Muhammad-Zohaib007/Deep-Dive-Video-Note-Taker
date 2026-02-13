"""Performance profiling utilities.

Provides timing decorators and a profiling context manager for
measuring pipeline stage performance and memory usage.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from notetaker.utils.logging import get_logger

logger = get_logger("profiler")


@dataclass
class TimingRecord:
    """A single timing measurement."""
    name: str
    duration_seconds: float
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None

    @property
    def memory_delta_mb(self) -> Optional[float]:
        if self.memory_before_mb is not None and self.memory_after_mb is not None:
            return self.memory_after_mb - self.memory_before_mb
        return None


@dataclass
class ProfilingReport:
    """Aggregated profiling data from a pipeline run."""
    records: list[TimingRecord] = field(default_factory=list)
    total_seconds: float = 0.0

    def add(self, record: TimingRecord) -> None:
        self.records.append(record)
        self.total_seconds = sum(r.duration_seconds for r in self.records)

    def summary(self) -> str:
        """Generate a formatted summary string."""
        lines = ["=" * 60, "PERFORMANCE PROFILE", "=" * 60]
        for rec in self.records:
            pct = (rec.duration_seconds / self.total_seconds * 100) if self.total_seconds > 0 else 0
            mem_str = ""
            if rec.memory_delta_mb is not None:
                mem_str = f"  (mem: {rec.memory_delta_mb:+.1f} MB)"
            lines.append(f"  {rec.name:<30s} {rec.duration_seconds:>7.2f}s  ({pct:>5.1f}%){mem_str}")
        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':<30s} {self.total_seconds:>7.2f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "total_seconds": round(self.total_seconds, 3),
            "stages": [
                {
                    "name": r.name,
                    "duration_seconds": round(r.duration_seconds, 3),
                    "memory_before_mb": round(r.memory_before_mb, 1) if r.memory_before_mb is not None else None,
                    "memory_after_mb": round(r.memory_after_mb, 1) if r.memory_after_mb is not None else None,
                    "memory_delta_mb": round(r.memory_delta_mb, 1) if r.memory_delta_mb is not None else None,
                }
                for r in self.records
            ],
        }


def _get_memory_mb() -> Optional[float]:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


# Global profiling report (set when --profile is active)
_active_report: Optional[ProfilingReport] = None


def get_active_report() -> Optional[ProfilingReport]:
    """Get the active profiling report, if profiling is enabled."""
    return _active_report


def start_profiling() -> ProfilingReport:
    """Start a new profiling session."""
    global _active_report
    _active_report = ProfilingReport()
    return _active_report


def stop_profiling() -> Optional[ProfilingReport]:
    """Stop profiling and return the report."""
    global _active_report
    report = _active_report
    _active_report = None
    return report


@contextmanager
def profile_stage(name: str):
    """Context manager to time a pipeline stage.

    Usage:
        with profile_stage("transcription"):
            result = transcribe_audio(...)

    Automatically records timing (and memory if psutil is available)
    to the active profiling report.
    """
    mem_before = _get_memory_mb()
    start = time.perf_counter()

    yield

    elapsed = time.perf_counter() - start
    mem_after = _get_memory_mb()

    record = TimingRecord(
        name=name,
        duration_seconds=elapsed,
        memory_before_mb=mem_before,
        memory_after_mb=mem_after,
    )

    if _active_report is not None:
        _active_report.add(record)

    logger.debug(f"[PROFILE] {name}: {elapsed:.2f}s")


def timed(func: Callable) -> Callable:
    """Decorator to time a function and log the result.

    If profiling is active, also records to the profiling report.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__qualname__
        mem_before = _get_memory_mb()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        mem_after = _get_memory_mb()

        record = TimingRecord(
            name=name,
            duration_seconds=elapsed,
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
        )

        if _active_report is not None:
            _active_report.add(record)

        logger.debug(f"[TIMED] {name}: {elapsed:.2f}s")
        return result

    return wrapper
