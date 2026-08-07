"""
Tests for the offline lookup benchmark (LEV-14 / 3.3, 3.4).

The two claims worth pinning:

- **Cold means cold.** A cold sample reaches the underlying embedding client;
  a warm one does not. Asserted by counting client invocations, not by
  comparing durations — a duration comparison would pass on a fast machine
  even if the memo were never cleared.
- **Percentiles come from the measured repetitions.** Warm-up iterations are
  absent from the computation, and the sample count is exactly what was asked
  for.

Fully offline: mock LLM at zero latency, mock embeddings.
"""

import unittest
from collections import Counter

from levy.dataset.schema import QueryPair
from levy.embedding_manager import EmbeddingManager
from levy.embeddings import MockEmbeddingClient
from levy.experiment.config import ExperimentConfig
from levy.latency.benchmark import (
    BenchmarkError,
    ConfigurationLatency,
    DEFAULT_PROBE_SEED,
    SEGMENT_EMBED_COLD,
    SEGMENT_EMBED_WARM,
    benchmark_configuration,
    benchmark_configurations,
    percentile,
    probe_texts,
    require_samples,
)
from levy.latency.timing import SEGMENT_EXACT_LOOKUP, SEGMENT_INDEX_SEARCH, SEGMENT_TOTAL_LOOKUP


class _CountingEmbeddingClient(MockEmbeddingClient):
    """A mock embedding client that records every text it was actually asked to embed."""

    def __init__(self, dimension: int = 384):
        super().__init__(dimension=dimension)
        self.texts = []

    @property
    def calls(self) -> int:
        return len(self.texts)

    def embed(self, text):
        self.texts.append(text)
        return super().embed(text)


def _manager_with_counting_client():
    manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", provider="mock")
    client = _CountingEmbeddingClient()
    manager._clients["mock"] = client
    return manager, client


def _pairs(n: int = 6, workload: str = "faq"):
    return [
        QueryPair(
            pair_id=f"{workload}-{index:03d}",
            workload=workload,
            source_corpus="synthetic-fixture",
            source_pair_id=str(index),
            query_1=f"fixture question {index} about the account",
            query_2=f"fixture rephrasing {index} about the account",
            original_label=index % 2,
        )
        for index in range(n)
    ]


class TestPercentile(unittest.TestCase):

    def test_nearest_rank_reports_an_observed_value(self):
        values = [5.0, 1.0, 4.0, 2.0, 3.0]
        self.assertEqual(percentile(values, 50), 3.0)
        self.assertEqual(percentile(values, 95), 5.0)
        self.assertEqual(percentile(values, 100), 5.0)

    def test_single_value(self):
        self.assertEqual(percentile([2.5], 95), 2.5)

    def test_empty_sample_is_an_error_not_a_zero(self):
        with self.assertRaises(BenchmarkError):
            percentile([], 50)


class TestRequireSamples(unittest.TestCase):

    def test_a_segment_with_no_sample_is_refused_by_name(self):
        with self.assertRaises(BenchmarkError) as ctx:
            require_samples("model|faq|0.85", {"index_search": [], "embed_cold": [1.0]})
        message = str(ctx.exception)
        self.assertIn("index_search", message)
        self.assertIn("model|faq|0.85", message)

    def test_fully_populated_segments_pass(self):
        self.assertIsNone(require_samples("c", {"index_search": [1.0], "embed_cold": [2.0]}))


class TestProbeTexts(unittest.TestCase):

    def test_probes_are_unique_and_deterministic(self):
        pairs = _pairs()
        first = probe_texts(pairs, 20, seed=42)
        second = probe_texts(pairs, 20, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)

    def test_different_seeds_select_different_bases(self):
        pairs = _pairs()
        self.assertNotEqual(probe_texts(pairs, 20, seed=1), probe_texts(pairs, 20, seed=2))

    def test_no_pairs_is_an_error(self):
        with self.assertRaises(BenchmarkError):
            probe_texts([], 5)


class TestColdAndWarmSampling(unittest.TestCase):

    CONFIG = ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.85)

    def test_cold_samples_invoke_the_client_and_warm_samples_do_not(self):
        """
        Counted per probe text, not in aggregate.

        Every measured probe is seeded into the memo once. A warm probe's
        measured lookup must then find it there (one client call in total); a
        cold probe's must not, because its entry was evicted first (two).
        Comparing durations instead would pass on a fast machine even if the
        eviction never happened.
        """
        manager, client = _manager_with_counting_client()
        pairs = _pairs()
        warmup, repetitions = 2, 4

        result = benchmark_configuration(
            self.CONFIG,
            pairs,
            embedding_manager=manager,
            warmup=warmup,
            repetitions=repetitions,
        )
        self.assertIsInstance(result, ConfigurationLatency)

        probes = probe_texts(pairs, warmup + 2 * repetitions, seed=DEFAULT_PROBE_SEED)
        warm_probes = probes[warmup : warmup + repetitions]
        cold_probes = probes[warmup + repetitions :]
        counts = Counter(client.texts)

        for text in warm_probes:
            self.assertEqual(counts[text], 1, f"warm probe re-embedded: {text!r}")
        for text in cold_probes:
            self.assertEqual(counts[text], 2, f"cold probe not re-embedded: {text!r}")

        self.assertEqual(result.segments[SEGMENT_EMBED_COLD].n, repetitions)
        self.assertEqual(result.segments[SEGMENT_EMBED_WARM].n, repetitions)

    def test_forget_makes_the_next_embed_reach_the_client(self):
        """
        The mechanism the cold phase depends on, isolated from the benchmark.

        `forget` must evict exactly one entry: if it flushed the memo, every
        later sample in the run would be cold and `embed_warm` would be
        measuring the encoder too.
        """
        manager, client = _manager_with_counting_client()

        manager.embed("alpha")
        manager.embed("beta")
        self.assertEqual(client.calls, 2)

        manager.embed("alpha")                       # memoized: no client call
        self.assertEqual(client.calls, 2)

        self.assertTrue(manager.forget("alpha"))     # evicted
        manager.embed("alpha")                       # cold: reaches the client
        self.assertEqual(client.calls, 3)

        manager.embed("beta")                        # untouched by the eviction
        self.assertEqual(client.calls, 3)

        self.assertFalse(manager.forget("never embedded"))



class TestPercentilesComeFromTheMeasuredRepetitions(unittest.TestCase):

    CONFIG = ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.85)

    def test_sample_counts_equal_the_configured_repetitions(self):
        manager, _ = _manager_with_counting_client()
        repetitions, warmup = 7, 3

        result = benchmark_configuration(
            self.CONFIG,
            _pairs(),
            embedding_manager=manager,
            warmup=warmup,
            repetitions=repetitions,
        )

        # embed is split by regime: one phase each.
        self.assertEqual(result.segments[SEGMENT_EMBED_COLD].n, repetitions)
        self.assertEqual(result.segments[SEGMENT_EMBED_WARM].n, repetitions)
        self.assertEqual(result.segments[SEGMENT_TOTAL_LOOKUP].n, repetitions)
        # index search and exact lookup are pooled across both phases.
        self.assertEqual(result.segments[SEGMENT_INDEX_SEARCH].n, 2 * repetitions)
        self.assertEqual(result.segments[SEGMENT_EXACT_LOOKUP].n, 2 * repetitions)
        # warm-up contributes to none of them.
        self.assertEqual(result.n_lookups, 2 * repetitions)

    def test_warmup_count_does_not_change_the_sample_counts(self):
        manager_a, _ = _manager_with_counting_client()
        manager_b, _ = _manager_with_counting_client()

        few = benchmark_configuration(self.CONFIG, _pairs(), embedding_manager=manager_a, warmup=0, repetitions=5)
        many = benchmark_configuration(self.CONFIG, _pairs(), embedding_manager=manager_b, warmup=9, repetitions=5)

        for segment in (SEGMENT_EMBED_COLD, SEGMENT_EMBED_WARM, SEGMENT_TOTAL_LOOKUP, SEGMENT_INDEX_SEARCH):
            self.assertEqual(few.segments[segment].n, many.segments[segment].n, segment)

    def test_percentiles_are_non_negative_and_ordered(self):
        manager, _ = _manager_with_counting_client()
        result = benchmark_configuration(self.CONFIG, _pairs(), embedding_manager=manager, warmup=1, repetitions=5)
        for name, stats in result.segments.items():
            self.assertGreaterEqual(stats.p50_ms, 0.0, name)
            self.assertGreaterEqual(stats.p95_ms, stats.p50_ms, name)

    def test_invalid_protocol_parameters_are_rejected(self):
        manager, _ = _manager_with_counting_client()
        with self.assertRaises(BenchmarkError):
            benchmark_configuration(self.CONFIG, _pairs(), embedding_manager=manager, repetitions=0)
        with self.assertRaises(BenchmarkError):
            benchmark_configuration(self.CONFIG, _pairs(), embedding_manager=manager, warmup=-1)


class TestBenchmarkConfigurations(unittest.TestCase):

    def test_one_manager_per_model_is_shared_across_thresholds(self):
        configs = [
            ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.70),
            ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.90),
            ExperimentConfig(model="modernbert", workload="faq", threshold=0.70),
        ]
        results, identities = benchmark_configurations(
            configs,
            _pairs(),
            embedding_provider="mock",
            warmup=1,
            repetitions=3,
        )

        self.assertEqual([r.config.config_id for r in results], [c.config_id for c in configs])
        self.assertEqual(sorted(identities), ["all-MiniLM-L6-v2", "modernbert"])


if __name__ == "__main__":
    unittest.main()
