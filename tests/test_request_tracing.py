"""
Tests for the opt-in observation hooks and the console's trace recorder.

Fully offline: mock embeddings and a zero-latency mock LLM. The engine hooks
are checked for what they record *and* for changing nothing: the same prompt
sequence through engines with and without them must resolve identically.
"""

import unittest

from levy.config import LevyConfig
from levy.console.options import ConsoleSettings
from levy.console.tracing import (
    LiveTimingCollector,
    TraceRecorder,
    call_cost_usd,
    llm_block,
    observation_dict,
    provider_endpoint,
    token_usage,
)
from levy.engine import LevyEngine
from levy.latency.timing import (
    SEGMENT_EMBED,
    SEGMENT_EXACT_LOOKUP,
    SEGMENT_INDEX_SEARCH,
    SEGMENT_LLM_CALL,
    SEGMENT_STORE,
    SEGMENT_TOTAL_LOOKUP,
    SEGMENTS,
    TimingCollector,
)
from levy.models import LLMResponse, SemanticLookupObservation


def _engine(threshold: float = 0.85, **overrides) -> LevyEngine:
    config = LevyConfig(
        llm_provider="mock",
        mock_llm_latency_seconds=0.0,
        embedding_provider="mock",
        similarity_threshold=threshold,
        vector_index_backend="brute_force",
        **overrides,
    )
    return LevyEngine(config)


def _settings(**overrides) -> ConsoleSettings:
    values = dict(
        threshold=0.85,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="anthropic",
        llm_model="test-anthropic-model",
        exact_cache=True,
        semantic_cache=True,
        vector_index_backend="brute_force",
        cache_store="memory",
    )
    values.update(overrides)
    return ConsoleSettings(**values)


class TestSemanticLookupObservation(unittest.TestCase):

    def test_miss_records_the_nearest_neighbour(self):
        engine = _engine(threshold=0.99)
        engine.generate("what is the refund window")
        obs = SemanticLookupObservation()

        result = engine.generate("an entirely unrelated question", observation=obs)

        self.assertEqual(result.source, "llm")
        self.assertTrue(obs.reached)
        self.assertEqual(obs.index_size, 1)
        self.assertEqual(obs.threshold, 0.99)
        self.assertEqual(obs.nearest_prompt, "what is the refund window")
        self.assertAlmostEqual(obs.similarity, 1.0 / (1.0 + obs.distance))
        self.assertLess(obs.similarity, 0.99)
        self.assertFalse(obs.hit)

    def test_semantic_hit_matches_the_result(self):
        engine = _engine(threshold=0.0)
        engine.generate("what is the refund window")
        obs = SemanticLookupObservation()

        result = engine.generate("how long do I have to ask for a refund", observation=obs)

        self.assertEqual(result.source, "semantic_cache")
        self.assertTrue(obs.hit)
        self.assertEqual(obs.nearest_prompt, "what is the refund window")
        self.assertAlmostEqual(obs.similarity, result.similarity_score)

    def test_exact_hit_never_reaches_the_semantic_cache(self):
        engine = _engine()
        engine.generate("same prompt")
        obs = SemanticLookupObservation()

        result = engine.generate("same prompt", observation=obs)

        self.assertEqual(result.source, "exact_cache")
        self.assertFalse(obs.reached)

    def test_empty_index_is_reached_with_no_neighbour(self):
        engine = _engine()
        obs = SemanticLookupObservation()

        engine.generate("first prompt ever", observation=obs)

        self.assertTrue(obs.reached)
        self.assertEqual(obs.index_size, 0)
        self.assertIsNone(obs.nearest_prompt)
        self.assertIsNone(obs.similarity)

    def test_decisions_are_unchanged_by_observation_and_timing(self):
        prompts = [
            "what is the refund window",
            "what is the refund window",
            "how long do I have to ask for a refund",
            "an entirely unrelated question",
            "what is the refund window?",
        ]
        plain = _engine(threshold=0.5)
        watched = _engine(threshold=0.5)
        for prompt in prompts:
            a = plain.generate(prompt)
            b = watched.generate(prompt, timing=TimingCollector(), observation=SemanticLookupObservation())
            self.assertEqual((a.source, a.similarity_score, a.answer), (b.source, b.similarity_score, b.answer))
        self.assertEqual(plain.metrics.get_snapshot().exact_hits, watched.metrics.get_snapshot().exact_hits)
        self.assertEqual(plain.metrics.get_snapshot().semantic_hits, watched.metrics.get_snapshot().semantic_hits)
        self.assertEqual(plain.metrics.get_snapshot().misses, watched.metrics.get_snapshot().misses)


class TestProviderCallAndStoreSegments(unittest.TestCase):

    def test_miss_records_llm_call_and_store(self):
        engine = _engine()
        collector = TimingCollector()

        engine.generate("a fresh prompt", timing=collector)

        self.assertEqual(len(collector.samples(SEGMENT_LLM_CALL)), 1)
        self.assertEqual(len(collector.samples(SEGMENT_STORE)), 1)
        lookup_parts = sum(collector.total(s) for s in (SEGMENT_EXACT_LOOKUP, SEGMENT_EMBED, SEGMENT_INDEX_SEARCH))
        self.assertLessEqual(lookup_parts, collector.total(SEGMENT_TOTAL_LOOKUP))

    def test_hits_record_neither(self):
        engine = _engine(threshold=0.0)
        engine.generate("what is the refund window")
        for prompt in ("what is the refund window", "how long do I have to ask for a refund"):
            collector = TimingCollector()
            engine.generate(prompt, timing=collector)
            self.assertEqual(collector.samples(SEGMENT_LLM_CALL), [])
            self.assertEqual(collector.samples(SEGMENT_STORE), [])

    def test_new_segments_stay_out_of_the_benchmark_set(self):
        self.assertNotIn(SEGMENT_LLM_CALL, SEGMENTS)
        self.assertNotIn(SEGMENT_STORE, SEGMENTS)


class TestTraceRecorder(unittest.TestCase):

    def test_cursor_returns_only_changes(self):
        recorder = TraceRecorder()
        first = recorder.open({"prompt": "a"})
        seen = recorder.version
        second = recorder.open({"prompt": "b"})

        changed = recorder.since(seen)

        self.assertEqual([t["id"] for t in changed], [second])
        recorder.update(first, outcome="miss")
        self.assertEqual([t["id"] for t in recorder.since(recorder.version - 1)], [first])

    def test_buffer_evicts_the_oldest(self):
        recorder = TraceRecorder(max_traces=2)
        ids = [recorder.open({"prompt": str(i)}) for i in range(3)]

        self.assertEqual([t["id"] for t in recorder.since(0)], ids[1:])
        recorder.update(ids[0], outcome="miss")  # evicted while running: ignored
        recorder.span_finished(ids[0], "embed", 0.0, 1.0)
        self.assertIsNone(recorder.get(ids[0]))
        self.assertIsNone(recorder.current_stage(ids[0]))

    def test_clear_starts_a_new_epoch_and_keeps_origins(self):
        recorder = TraceRecorder()
        recorder.open({"prompt": "a"})
        recorder.remember_origin(("cfg", "key"), {"origin_id": "x"})
        epoch = recorder.epoch

        new_epoch = recorder.clear()

        self.assertNotEqual(new_epoch, epoch)
        self.assertEqual(recorder.since(0), [])
        self.assertEqual(recorder.origin(("cfg", "key")), {"origin_id": "x"})
        recorder.clear_origins()
        self.assertIsNone(recorder.origin(("cfg", "key")))

    def test_returned_traces_are_copies(self):
        recorder = TraceRecorder()
        trace_id = recorder.open({"prompt": "a"})
        recorder.since(0)[0]["prompt"] = "changed"
        self.assertEqual(recorder.get(trace_id)["prompt"], "a")

    def test_finish_marks_done(self):
        recorder = TraceRecorder()
        trace_id = recorder.open({"prompt": "a"})
        recorder.finish(trace_id, outcome="miss")
        trace = recorder.get(trace_id)
        self.assertEqual(trace["status"], "done")
        self.assertIsNone(trace["current_stage"])


class TestLiveTimingCollector(unittest.TestCase):

    def test_spans_carry_start_offsets_and_durations(self):
        recorder = TraceRecorder()
        trace_id = recorder.open({"prompt": "p"})
        obs = SemanticLookupObservation()
        engine = _engine(threshold=0.99)
        engine.generate("seed prompt")
        import time
        collector = LiveTimingCollector(recorder, trace_id, time.perf_counter(), observation=obs)

        engine.generate("another prompt", timing=collector, observation=obs)

        trace = recorder.get(trace_id)
        names = [s["name"] for s in trace["spans"]]
        self.assertEqual(names, ["exact_lookup", "embed", "index_search", "llm_call", "store"])
        starts = [s["start_ms"] for s in trace["spans"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(s["duration_ms"] >= 0 for s in trace["spans"]))
        # The lookup's decision is published before the provider call returns.
        self.assertEqual(trace["semantic"]["nearest_prompt"], "seed prompt")
        self.assertEqual(collector.samples("embed")[0], trace["spans"][1]["duration_ms"])


class TestTraceFields(unittest.TestCase):

    def test_observation_dict_derives_cosine(self):
        obs = SemanticLookupObservation(reached=True, index_size=1, threshold=0.7,
                                        nearest_prompt="p", distance=0.4, similarity=1 / 1.4, hit=True)
        data = observation_dict(obs)
        self.assertAlmostEqual(data["cosine"], 1 - 0.4 ** 2 / 2)
        self.assertIsNone(observation_dict(SemanticLookupObservation(reached=True))["cosine"])

    def test_provider_endpoint_and_location(self):
        base = LevyConfig(llm_provider="mock")
        self.assertEqual(provider_endpoint(_settings(llm_provider="ollama", llm_model="test-ollama-model"), base, object()),
                         ("http://localhost:11434", "local"))
        self.assertEqual(provider_endpoint(_settings(), base, object()), ("https://api.anthropic.com", "remote"))
        remote_ollama = LevyConfig(llm_provider="mock", ollama_base_url="http://gpu-box:11434")
        self.assertEqual(provider_endpoint(_settings(llm_provider="ollama", llm_model="test-ollama-model"), remote_ollama, object())[1],
                         "remote")

    def test_token_usage_per_provider(self):
        anthropic = LLMResponse(text="", metadata={"input_tokens": 3, "output_tokens": 4, "stop_reason": "end_turn"})
        ollama = LLMResponse(text="", metadata={"prompt_eval_count": 5, "eval_count": 6, "done_reason": "stop"})
        self.assertEqual(token_usage(anthropic), (3, 4, "end_turn"))
        self.assertEqual(token_usage(ollama), (5, 6, "stop"))
        self.assertEqual(token_usage(LLMResponse(text="", token_usage=9)), (None, None, None))

    def test_cost_uses_the_model_prices_or_reports_none(self):
        self.assertAlmostEqual(call_cost_usd((1.0, 5.0), 1_000_000, 1_000_000), 6.0)
        self.assertIsNone(call_cost_usd(None, 10, 10))
        self.assertIsNone(call_cost_usd((1.0, 5.0), None, None))

    def test_llm_block(self):
        response = LLMResponse(text="x", token_usage=30, model="test-anthropic-model",
                               metadata={"input_tokens": 10, "output_tokens": 20, "stop_reason": "end_turn"})
        base = LevyConfig(llm_provider="mock", anthropic_models=(("test-anthropic-model", 1.0, 5.0),))
        block = llm_block(_settings(), base, object(), response, 1234.5)
        self.assertEqual(block["location"], "remote")
        self.assertEqual(block["total_tokens"], 30)
        self.assertEqual(block["duration_ms"], 1234.5)
        self.assertAlmostEqual(block["cost_usd"], 10 / 1e6 * 1.0 + 20 / 1e6 * 5.0)


if __name__ == "__main__":
    unittest.main()
