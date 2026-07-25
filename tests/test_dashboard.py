"""
Tests for the results dashboard's testable core (LEV-10 / D6).

Covers bundle loading/validation, curve selection, the live query decision,
and its semantics parity with a direct `SemanticCache` query -- all offline,
headless (no Streamlit process started).
"""

import json
import tempfile
import unittest
from pathlib import Path

from analysis_fixtures import additive_cells, build_results, write_harness_dir

from levy.analysis.report import build_analysis_bundle
from levy.cache.semantic_cache import SemanticCache
from levy.dashboard.bundle import (
    BundleContractError,
    BundleNotFoundError,
    load_bundle,
)
from levy.dashboard.curves import (
    available_model_workload_pairs,
    available_models,
    available_workloads,
    select_curve,
)
from levy.dashboard.query import build_query_index, evaluate_query
from levy.dataset.schema import QueryPair
from levy.models import LLMRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sample_bundle(out_dir: Path) -> Path:
    """A full, schema-valid analysis bundle via the real LEV-8 pipeline."""
    harness_dir = out_dir / "harness"
    write_harness_dir(harness_dir, build_results(additive_cells()))
    bundle_dir = out_dir / "analysis"
    build_analysis_bundle(harness_dir=harness_dir, out_dir=bundle_dir)
    return bundle_dir


def _pair(pair_id, workload, query_1, query_2, label=1) -> QueryPair:
    return QueryPair(
        pair_id=pair_id,
        workload=workload,
        source_corpus="synthetic-fixture",
        source_pair_id=pair_id,
        query_1=query_1,
        query_2=query_2,
        original_label=label,
    )


class _ControlledEmbeddingManager:
    """Test double for `EmbeddingManager`: fixed vectors, counts every embed_with call."""

    def __init__(self, vectors: dict) -> None:
        self._vectors = vectors
        self.embed_calls = []

    def embed_with(self, model_name, text):
        self.embed_calls.append(text)
        return list(self._vectors[text])

    def get_dimension(self, model_name=None):
        return len(next(iter(self._vectors.values())))


# ---------------------------------------------------------------------------
# 1. Bundle loading
# ---------------------------------------------------------------------------

class TestLoadBundle(unittest.TestCase):

    def test_valid_bundle_loads_all_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = _build_sample_bundle(Path(tmp))
            bundle = load_bundle(bundle_dir)

            self.assertEqual(bundle.bundle_dir, bundle_dir)
            for column in ("model", "workload", "threshold", "metric", "value", "zero_div", "n"):
                self.assertIn(column, bundle.hit_rate.columns)
                self.assertIn(column, bundle.precision.columns)
            self.assertIn("hypothesis", bundle.anova.columns)
            self.assertIn("group1", bundle.tukey.columns)
            self.assertEqual(list(bundle.tukey_status.columns), ["effect", "ran", "reason"])
            self.assertIsInstance(bundle.kappa, dict)
            self.assertIsInstance(bundle.meta, dict)

    def test_missing_directory_raises_bundle_not_found(self):
        with self.assertRaises(BundleNotFoundError) as ctx:
            load_bundle("/nonexistent/bundle/dir")
        self.assertIn("scripts/reproduce.sh", str(ctx.exception))

    def test_missing_file_names_the_path_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = _build_sample_bundle(Path(tmp))
            (bundle_dir / "anova.csv").unlink()

            with self.assertRaises(BundleNotFoundError) as ctx:
                load_bundle(bundle_dir)
            message = str(ctx.exception)
            self.assertIn("anova.csv", message)
            self.assertIn("scripts/reproduce.sh", message)

    def test_missing_column_raises_bundle_contract_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = _build_sample_bundle(Path(tmp))
            # Corrupt curves_hit_rate.csv by dropping the zero_div column.
            path = bundle_dir / "curves_hit_rate.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            header = lines[0].split(",")
            drop_index = header.index("zero_div")
            rewritten = []
            for line in lines:
                fields = line.split(",")
                del fields[drop_index]
                rewritten.append(",".join(fields))
            path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

            with self.assertRaises(BundleContractError) as ctx:
                load_bundle(bundle_dir)
            message = str(ctx.exception)
            self.assertIn("zero_div", message)
            self.assertIn("curves_hit_rate.csv", message)
            self.assertIn("scripts/reproduce.sh", message)


# ---------------------------------------------------------------------------
# 2. Curve selection
# ---------------------------------------------------------------------------

class TestCurveSelection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        bundle_dir = _build_sample_bundle(Path(self._tmp.name))
        self.bundle = load_bundle(bundle_dir)

    def test_available_models_and_workloads(self):
        self.assertEqual(available_models(self.bundle.hit_rate), ["all-MiniLM-L6-v2", "modernbert"])
        self.assertEqual(available_workloads(self.bundle.hit_rate), ["chat", "code", "faq"])

    def test_available_pairs_cover_full_grid(self):
        pairs = available_model_workload_pairs(self.bundle.hit_rate)
        self.assertEqual(len(pairs), 6)  # 2 models x 3 workloads
        self.assertIn(("all-MiniLM-L6-v2", "faq"), pairs)

    def test_select_curve_preserves_threshold_order_and_flags(self):
        points = select_curve(self.bundle.hit_rate, "all-MiniLM-L6-v2", "faq")
        self.assertEqual(len(points), 5)
        thresholds = [p.threshold for p in points]
        self.assertEqual(thresholds, sorted(thresholds))
        for point in points:
            self.assertIsInstance(point.zero_div, bool)
            self.assertIsInstance(point.n, int)

    def test_select_curve_zero_div_flag_from_precision_table(self):
        # additive_cells() has no zero-division cells by default; the flag column
        # must still be present and correctly False throughout.
        points = select_curve(self.bundle.precision, "modernbert", "chat")
        self.assertTrue(all(p.zero_div is False for p in points))

    def test_select_curve_unknown_pair_returns_empty(self):
        self.assertEqual(select_curve(self.bundle.hit_rate, "no-such-model", "faq"), [])


# ---------------------------------------------------------------------------
# 3. Query decision
# ---------------------------------------------------------------------------

def _angle_vector(angle_deg: float):
    import math
    rad = math.radians(angle_deg)
    return [math.cos(rad), math.sin(rad)]


class TestQueryDecision(unittest.TestCase):

    def test_near_duplicate_reports_hit_with_similarity(self):
        vectors = {
            "q1": _angle_vector(0),
            "q2": _angle_vector(5),
            "near_dup": _angle_vector(3),
        }
        manager = _ControlledEmbeddingManager(vectors)
        pairs = [_pair("faq-0001", "faq", "q1", "q2")]
        cache = build_query_index(manager, "all-MiniLM-L6-v2", pairs)

        decision = evaluate_query(cache, "near_dup", threshold=0.5)
        self.assertTrue(decision.hit)
        self.assertIsNotNone(decision.similarity)
        self.assertIn(decision.matched_text, ("q1", "q2"))

    def test_unrelated_query_reports_miss_with_nearest_similarity(self):
        vectors = {
            "q1": _angle_vector(0),
            "q2": _angle_vector(5),
            "unrelated": _angle_vector(180),
        }
        manager = _ControlledEmbeddingManager(vectors)
        pairs = [_pair("faq-0001", "faq", "q1", "q2")]
        cache = build_query_index(manager, "all-MiniLM-L6-v2", pairs)

        decision = evaluate_query(cache, "unrelated", threshold=0.5)
        self.assertFalse(decision.hit)
        self.assertIsNotNone(decision.similarity)
        self.assertIsNotNone(decision.matched_text)

    def test_threshold_change_flips_decision_without_reembedding_dataset(self):
        vectors = {
            "q1": _angle_vector(0),
            "q2": _angle_vector(5),
            "borderline": _angle_vector(3),
        }
        manager = _ControlledEmbeddingManager(vectors)
        pairs = [_pair("faq-0001", "faq", "q1", "q2")]
        cache = build_query_index(manager, "all-MiniLM-L6-v2", pairs)
        dataset_embed_count = len(manager.embed_calls)

        loose = evaluate_query(cache, "borderline", threshold=0.5)
        strict = evaluate_query(cache, "borderline", threshold=0.999999)

        self.assertTrue(loose.hit)
        self.assertFalse(strict.hit)
        # Re-evaluating at a different threshold never re-embeds the dataset.
        self.assertEqual(manager.embed_calls.count("q1"), 1)
        self.assertEqual(manager.embed_calls.count("q2"), 1)
        self.assertEqual(len(manager.embed_calls), dataset_embed_count + 2)

    def test_build_query_index_dedups_repeated_query_text(self):
        vectors = {"same": _angle_vector(0)}
        manager = _ControlledEmbeddingManager(vectors)
        pairs = [
            _pair("faq-0001", "faq", "same", "same"),
            _pair("faq-0002", "faq", "same", "same"),
        ]
        build_query_index(manager, "all-MiniLM-L6-v2", pairs)
        self.assertEqual(manager.embed_calls.count("same"), 1)

    def test_empty_index_reports_miss_with_no_similarity(self):
        manager = _ControlledEmbeddingManager({})
        cache = build_query_index(manager, "all-MiniLM-L6-v2", pairs=[])
        decision = evaluate_query(cache, "anything", threshold=0.5)
        self.assertFalse(decision.hit)
        self.assertIsNone(decision.similarity)
        self.assertIsNone(decision.matched_text)

    def test_search_finds_no_ids_reports_miss(self):
        class _EmptySearchIndex:
            def add(self, vector, entry_id):
                pass

            def search(self, vector, k=1):
                return [], []

            def reset(self):
                pass

            def size(self):
                return 1  # non-zero, but search() yields no ids

        class _Client:
            def embed(self, text):
                return [1.0, 0.0]

        cache = SemanticCache(embedding_client=_Client(), threshold=0.0, vector_index=_EmptySearchIndex())
        decision = evaluate_query(cache, "q", threshold=0.5)
        self.assertFalse(decision.hit)
        self.assertIsNone(decision.similarity)
        self.assertIsNone(decision.matched_text)


# ---------------------------------------------------------------------------
# 4. Semantics parity with a direct SemanticCache query
# ---------------------------------------------------------------------------

class TestSemanticsParity(unittest.TestCase):

    def test_hit_matches_direct_semantic_cache_query(self):
        vectors = {
            "q1": _angle_vector(0),
            "near_dup": _angle_vector(3),
        }

        class _Client:
            def embed(self, text):
                return list(vectors[text])

        threshold = 0.5
        sc = SemanticCache(embedding_client=_Client(), threshold=threshold, backend="brute_force")
        sc.set(LLMRequest(prompt="q1"), "resp-q1", embedding=vectors["q1"])

        direct = sc.get(LLMRequest(prompt="near_dup"))
        self.assertIsNotNone(direct)

        manager = _ControlledEmbeddingManager(vectors)
        pairs = [_pair("faq-0001", "faq", "q1", "q1")]
        cache = build_query_index(manager, "all-MiniLM-L6-v2", pairs)
        decision = evaluate_query(cache, "near_dup", threshold=threshold)

        self.assertTrue(decision.hit)
        self.assertAlmostEqual(
            decision.similarity, direct.metadata["last_similarity_score"], places=6
        )

    def test_miss_matches_direct_semantic_cache_query(self):
        vectors = {
            "q1": _angle_vector(0),
            "unrelated": _angle_vector(180),
        }

        class _Client:
            def embed(self, text):
                return list(vectors[text])

        threshold = 0.5
        sc = SemanticCache(embedding_client=_Client(), threshold=threshold, backend="brute_force")
        sc.set(LLMRequest(prompt="q1"), "resp-q1", embedding=vectors["q1"])
        direct = sc.get(LLMRequest(prompt="unrelated"))
        self.assertIsNone(direct)

        manager = _ControlledEmbeddingManager(vectors)
        pairs = [_pair("faq-0001", "faq", "q1", "q1")]
        cache = build_query_index(manager, "all-MiniLM-L6-v2", pairs)
        decision = evaluate_query(cache, "unrelated", threshold=threshold)

        self.assertFalse(decision.hit)


if __name__ == "__main__":
    unittest.main()
