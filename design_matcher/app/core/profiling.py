"""
Profiling and performance measurement utilities for the Design Matcher indexing engine.
Provides low-overhead stage timing, throughput tracking, and aggregated percentile metrics.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def measure_stage(stage_name: str, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Executes operation() while measuring wall-clock duration with high precision.

    Args:
        stage_name: Human-readable stage identifier (e.g. 'discovery', 'stage_a_metadata').
        operation: Callable to invoke.
        *args, **kwargs: Arguments passed to operation.

    Returns:
        The return value of operation().
    """
    start = time.perf_counter()
    try:
        return operation(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        logger.debug("%s: %.4fs", stage_name, elapsed)


class StageTimer:
    """
    Context manager for measuring and recording execution duration of a block.

    Usage:
        with StageTimer("render_page") as timer:
            render_pdf_page(...)
        print(f"Took {timer.elapsed:.3f}s")
    """

    def __init__(self, name: str, session: Optional["ProfilingSession"] = None) -> None:
        self.name = name
        self.session = session
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.session is not None:
            self.session.record_measurement(self.name, self.elapsed)
        logger.debug("[%s] Elapsed: %.4fs", self.name, self.elapsed)


@dataclass
class StageStats:
    """Aggregated statistics for an individual pipeline stage."""
    count: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0
    samples: List[float] = field(default_factory=list)

    def record(self, duration: float) -> None:
        self.count += 1
        self.total_seconds += duration
        if duration < self.min_seconds:
            self.min_seconds = duration
        if duration > self.max_seconds:
            self.max_seconds = duration
        if len(self.samples) < 5000:
            self.samples.append(duration)

    @property
    def avg_seconds(self) -> float:
        return (self.total_seconds / self.count) if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(round((len(s) - 1) * (p / 100.0)))
        return s[max(0, min(len(s) - 1, idx))]


class ProfilingSession:
    """
    Tracks and aggregates metrics across discovery, metadata extraction,
    rendering, vector embedding, and database writes.
    """

    def __init__(self, name: str = "indexing_run") -> None:
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None
        self.stages: Dict[str, StageStats] = {}
        self.items_processed: int = 0
        self.tiles_generated: int = 0

    def timer(self, stage_name: str) -> StageTimer:
        """Returns a StageTimer bound to this session."""
        return StageTimer(stage_name, self)

    def record_measurement(self, stage_name: str, duration: float) -> None:
        if stage_name not in self.stages:
            self.stages[stage_name] = StageStats()
        self.stages[stage_name].record(duration)

    def finish(self, items_processed: int = 0, tiles_generated: int = 0) -> None:
        self.end_time = time.perf_counter()
        self.items_processed = items_processed
        self.tiles_generated = tiles_generated

    @property
    def total_duration(self) -> float:
        end = self.end_time or time.perf_counter()
        return max(0.0001, end - self.start_time)

    @property
    def items_per_second(self) -> float:
        return self.items_processed / self.total_duration

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured dictionary of benchmark metrics."""
        stage_summary = {}
        for name, stats in self.stages.items():
            stage_summary[name] = {
                "count": stats.count,
                "total_sec": round(stats.total_seconds, 4),
                "avg_sec": round(stats.avg_seconds, 5),
                "min_sec": round(stats.min_seconds if stats.count > 0 else 0.0, 5),
                "max_sec": round(stats.max_seconds, 5),
                "p50_sec": round(stats.percentile(50), 5),
                "p95_sec": round(stats.percentile(95), 5),
            }

        return {
            "session_name": self.name,
            "total_duration_sec": round(self.total_duration, 3),
            "items_processed": self.items_processed,
            "tiles_generated": self.tiles_generated,
            "items_per_second": round(self.items_per_second, 2),
            "stages": stage_summary,
        }

    def format_report(self) -> str:
        """Renders formatted ASCII summary table of stage timings and throughput."""
        summary = self.get_summary()
        lines = [
            f"=== Indexing Benchmark Report [{self.name}] ===",
            f"Total Duration: {summary['total_duration_sec']:.2f}s | "
            f"Files: {self.items_processed} | "
            f"Throughput: {summary['items_per_second']:.1f} files/s | "
            f"Tiles: {self.tiles_generated}",
            "-" * 72,
            f"{'Stage Name':<24} {'Count':>6} {'Total(s)':>9} {'Avg(s)':>9} {'p50(s)':>8} {'p95(s)':>8}",
            "-" * 72,
        ]
        for name, s in summary["stages"].items():
            lines.append(
                f"{name:<24} {s['count']:>6} {s['total_sec']:>9.3f} {s['avg_sec']:>9.4f} "
                f"{s['p50_sec']:>8.4f} {s['p95_sec']:>8.4f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)
