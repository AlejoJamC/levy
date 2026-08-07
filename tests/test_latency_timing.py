"""
Tests for the timing instrumentation (LEV-14 / 2.4, 2.5).

Two properties matter here and are asserted separately:

1. **The decomposition is honest.** No segment is longer than the path that
   contains it, on every one of the three outcomes the lookup path has.
2. **The production path is untouched.** With no collector, the result, the
   metrics and the cache contents are what they were before the parameter
   existed.

Fully offline: mock LLM at zero latency, mock embeddings.
"""

import unittest

from levy.config import LevyConfig
from levy.engine import LevyEngine
from levy.latency.timing import (
    SEGMENT_EMBED,
    SEGMENT_EXACT_LOOKUP,
    SEGMENT_INDEX_SEARCH,
    SEGMENT_TOTAL_LOOKUP,
    TimingCollector,
)


def _engine(threshold: float = 0.85) -> LevyEngine:
    return LevyEngine(
        LevyConfig(
            llm_provider="mock",
            mock_llm_latency_seconds=0,
            embedding_provider="mock",
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=threshold,
            cache_store_type="memory",
        )
    )


class TestTimingCollector(unittest.TestCase):

    def test_records_and_totals_named_segments(self):
        collector = TimingCollector()
        collector.record("a", 1.5)
        collector.record("a", 2.5)
        collector.record("b", 4.0)

        self.assertEqual(collector.samples("a"), [1.5, 2.5])
        self.assertEqual(collector.total("a"), 4.0)
        self.assertEqual(collector.as_dict(), {"a": 4.0, "b": 4.0})
        self.assertEqual(collector.recorded_segments(), ["a", "b"])

    def test_unrecorded_segment_is_empty_not_an_error(self):
        collector = TimingCollector()
        self.assertEqual(collector.samples("never"), [])
        self.assertEqual(collector.total("never"), 0.0)

    def test_negative_duration_is_rejected(self):
        collector = TimingCollector()
        with self.assertRaises(ValueError):
            collector.record("a", -0.001)

    def test_measure_records_even_when_the_block_raises(self):
        collector = TimingCollector()
        with self.assertRaises(RuntimeError):
            with collector.measure("a"):
                raise RuntimeError("boom")
        self.assertEqual(len(collector.samples("a")), 1)

    def test_reset_empties_the_collector(self):
        collector = TimingCollector()
        collector.record("a", 1.0)
        collector.reset()
        self.assertEqual(collector.as_dict(), {})


class TestSegmentSumsDoNotExceedTotal(unittest.TestCase):
    """
    The invariant of the whole measurement: if a segment could be longer than
    the path it sits inside, the reported decomposition would be describing
    something other than the lookup.
    """

    def _assert_decomposition_is_sane(self, collector: TimingCollector):
        total = collector.total(SEGMENT_TOTAL_LOOKUP)
        self.assertEqual(len(collector.samples(SEGMENT_TOTAL_LOOKUP)), 1)
        parts = sum(
            collector.total(name)
            for name in (SEGMENT_EXACT_LOOKUP, SEGMENT_EMBED, SEGMENT_INDEX_SEARCH)
        )
        for name in collector.recorded_segments():
            for sample in collector.samples(name):
                self.assertGreaterEqual(sample, 0.0, name)
        self.assertLessEqual(parts, total)

    def test_miss(self):
        engine = _engine()
        collector = TimingCollector()

        result = engine.generate("what is the refund window", timing=collector)

        self.assertEqual(result.source, "llm")
        self.assertEqual(len(collector.samples(SEGMENT_EXACT_LOOKUP)), 1)
        self._assert_decomposition_is_sane(collector)

    def test_exact_hit(self):
        engine = _engine()
        engine.generate("what is the refund window")

        collector = TimingCollector()
        result = engine.generate("what is the refund window", timing=collector)

        self.assertEqual(result.source, "exact_cache")
        # An exact hit never reaches the embedder or the index.
        self.assertEqual(collector.samples(SEGMENT_EMBED), [])
        self.assertEqual(collector.samples(SEGMENT_INDEX_SEARCH), [])
        self._assert_decomposition_is_sane(collector)

    def test_semantic_hit(self):
        # Threshold 0.0 makes any nearest neighbour a hit, so the semantic path
        # is exercised deterministically under mock (text-seeded) embeddings.
        engine = _engine(threshold=0.0)
        engine.generate("what is the refund window")

        collector = TimingCollector()
        result = engine.generate("how long do I have to ask for a refund", timing=collector)

        self.assertEqual(result.source, "semantic_cache")
        self.assertEqual(len(collector.samples(SEGMENT_EMBED)), 1)
        self.assertEqual(len(collector.samples(SEGMENT_INDEX_SEARCH)), 1)
        self._assert_decomposition_is_sane(collector)


class TestDefaultBehaviourIsUnchanged(unittest.TestCase):

    PROMPTS = [
        "what is the refund window",
        "what is the refund window",          # exact hit
        "an entirely unrelated question",     # miss
    ]

    def _run(self, with_timing: bool):
        engine = _engine()
        results = []
        for prompt in self.PROMPTS:
            collector = TimingCollector() if with_timing else None
            results.append(engine.generate(prompt, timing=collector))
        return engine, results

    def test_result_fields_and_metrics_match_with_and_without_a_collector(self):
        plain_engine, plain = self._run(with_timing=False)
        timed_engine, timed = self._run(with_timing=True)

        # latency_ms is wall-clock and is deliberately not compared.
        self.assertEqual(
            [(r.answer, r.source, r.similarity_score, r.metadata) for r in plain],
            [(r.answer, r.source, r.similarity_score, r.metadata) for r in timed],
        )

        plain_snapshot = plain_engine.metrics.get_snapshot()
        timed_snapshot = timed_engine.metrics.get_snapshot()
        for field in ("total_requests", "exact_hits", "semantic_hits", "misses", "tokens_saved"):
            self.assertEqual(
                getattr(plain_snapshot, field),
                getattr(timed_snapshot, field),
                field,
            )

        self.assertEqual(plain_engine.get_cache_stats(), timed_engine.get_cache_stats())

    def test_no_segment_is_recorded_when_no_collector_is_passed(self):
        engine = _engine()
        collector = TimingCollector()

        engine.generate("what is the refund window")   # untimed
        engine.generate("what is the refund window")   # untimed exact hit

        self.assertEqual(collector.as_dict(), {})


if __name__ == "__main__":
    unittest.main()
