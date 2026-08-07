"""
TimingCollector — named-segment accumulation on a monotonic clock (LEV-14 / 2.1).

Deliberately knows nothing about the engine, the caches or the embedding
manager: it is handed to them, they push durations into it. That direction of
dependency is what lets the production path stay untouched — when no collector
is passed, no clock is read and no branch is taken beyond one `is None` check.

Durations are milliseconds, taken from `time.perf_counter()` (monotonic, and
the highest resolution the platform offers). `time.time()` is deliberately not
used: it can step backwards under NTP correction, which would produce a
negative segment in a measurement whose whole point is a duration.
"""

import time
from contextlib import contextmanager, nullcontext
from typing import ContextManager, Dict, Iterator, List, Optional

# Segment names are constants rather than free strings so the engine, the
# benchmark and the CSV writer cannot drift apart on spelling.
SEGMENT_EMBED = "embed"
SEGMENT_INDEX_SEARCH = "index_search"
SEGMENT_EXACT_LOOKUP = "exact_lookup"
SEGMENT_TOTAL_LOOKUP = "total_lookup"

SEGMENTS = (SEGMENT_EMBED, SEGMENT_INDEX_SEARCH, SEGMENT_EXACT_LOOKUP, SEGMENT_TOTAL_LOOKUP)


class TimingCollector:
    """
    Accumulates named segment durations, in milliseconds.

    A collector is per-measurement, not per-process: the benchmark creates one,
    passes it through a single `generate()` call, reads the samples, and drops
    it. Multiple observations of the same segment within one call (e.g. two
    index searches) accumulate under `record`, and are readable individually
    via `samples()`.
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, segment: str, duration_ms: float) -> None:
        """Record one observation of `segment`. Negative durations are rejected."""
        if duration_ms < 0:
            raise ValueError(f"negative duration {duration_ms!r} for segment {segment!r}")
        self._samples.setdefault(segment, []).append(duration_ms)

    @contextmanager
    def measure(self, segment: str) -> Iterator[None]:
        """
        Time the enclosed block and record it under `segment`.

        The duration is recorded even when the block raises, so a failed call
        does not silently vanish from a measurement of the path it failed on.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(segment, (time.perf_counter() - start) * 1000.0)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def samples(self, segment: str) -> List[float]:
        """All observations recorded for `segment`, in order. Empty if never recorded."""
        return list(self._samples.get(segment, []))

    def total(self, segment: str) -> float:
        """Sum of the observations for `segment`; 0.0 if it was never recorded."""
        return float(sum(self._samples.get(segment, ())))

    def recorded_segments(self) -> List[str]:
        """Segment names that carry at least one observation, in recording order."""
        return list(self._samples.keys())

    def as_dict(self) -> Dict[str, float]:
        """Per-segment totals, for a caller that wants one number per segment."""
        return {segment: self.total(segment) for segment in self._samples}

    def reset(self) -> None:
        self._samples.clear()


# ----------------------------------------------------------------------
# Helpers for instrumented call sites
# ----------------------------------------------------------------------
#
# Both accept `Optional[TimingCollector]` so an instrumented call site reads the
# same whether or not measurement is on, without the caller growing an `if` per
# segment. With no collector, neither reads the clock.


def segment(timing: Optional[TimingCollector], name: str) -> ContextManager[None]:
    """Time the enclosed block into `timing`, or do nothing when it is None."""
    return timing.measure(name) if timing is not None else nullcontext()


def mark() -> float:
    """Monotonic start mark for a span whose end is not lexically enclosed."""
    return time.perf_counter()


def since_ms(start: float) -> float:
    """Milliseconds elapsed since a `mark()`."""
    return (time.perf_counter() - start) * 1000.0
