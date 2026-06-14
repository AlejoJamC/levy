"""
Tests for VectorIndex abstraction and SemanticCache rewrite (LEV-2).

Covers: add/search/reset/size, L2 normalization, similarity transform,
threshold decisions, id→entry mapping, per-configuration reset,
Faiss↔brute-force agreement (skipped if Faiss unavailable),
and engine end-to-end with mock embeddings.
"""

import math
import unittest

import numpy as np

from levy.cache.vector_index import (
    BruteForceVectorIndex,
    FaissHNSWVectorIndex,
    _l2_normalize,
    make_vector_index,
)
from levy.cache.semantic_cache import SemanticCache
from levy.config import LevyConfig
from levy.engine import LevyEngine
from levy.models import LLMRequest, CacheEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAISS_AVAILABLE = True
try:
    import faiss  # noqa: F401
except ImportError:
    FAISS_AVAILABLE = False


def _unit_vec(dim: int, angle_deg: float = 0.0) -> list:
    """2-D unit vector at `angle_deg` degrees, padded to `dim` with zeros."""
    rad = math.radians(angle_deg)
    v = [math.cos(rad), math.sin(rad)] + [0.0] * (dim - 2)
    arr = np.array(v, dtype=np.float32)
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()


def _known_l2(a: list, b: list) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# ---------------------------------------------------------------------------
# 6.1  VectorIndex: add / search / reset / size / normalization / zero guard
# ---------------------------------------------------------------------------

class TestBruteForceVectorIndex(unittest.TestCase):

    def _make(self):
        return BruteForceVectorIndex()

    def test_add_and_search_returns_nearest(self):
        idx = self._make()
        v0 = _unit_vec(4, 0)
        v1 = _unit_vec(4, 90)
        idx.add(v0, entry_id=0)
        idx.add(v1, entry_id=1)
        ids, dists = idx.search(v0, k=1)
        self.assertEqual(ids[0], 0)
        self.assertAlmostEqual(dists[0], 0.0, places=5)

    def test_size_increments(self):
        idx = self._make()
        self.assertEqual(idx.size(), 0)
        idx.add(_unit_vec(4, 0), entry_id=0)
        self.assertEqual(idx.size(), 1)
        idx.add(_unit_vec(4, 45), entry_id=1)
        self.assertEqual(idx.size(), 2)

    def test_empty_index_returns_empty(self):
        idx = self._make()
        ids, dists = idx.search(_unit_vec(4, 0), k=1)
        self.assertEqual(ids, [])
        self.assertEqual(dists, [])

    def test_reset_empties_index(self):
        idx = self._make()
        idx.add(_unit_vec(4, 0), entry_id=0)
        idx.reset()
        self.assertEqual(idx.size(), 0)
        ids, dists = idx.search(_unit_vec(4, 0), k=1)
        self.assertEqual(ids, [])

    def test_zero_vector_is_safe(self):
        idx = self._make()
        zero = [0.0, 0.0, 0.0, 0.0]
        idx.add(zero, entry_id=99)
        ids, dists = idx.search(zero, k=1)
        self.assertIsInstance(dists[0], float)
        self.assertFalse(math.isnan(dists[0]))

    def test_l2_normalize_zero_guard(self):
        zero = np.zeros(4, dtype=np.float32)
        result = _l2_normalize(zero)
        self.assertTrue(np.all(result == 0.0))

    def test_l2_normalize_unit_vector(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        normed = _l2_normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(normed)), 1.0, places=6)


# ---------------------------------------------------------------------------
# 6.2  Similarity transform + threshold decisions
# ---------------------------------------------------------------------------

class TestSimilarityTransform(unittest.TestCase):
    """
    Use controlled unit vectors with a KNOWN angle to get a predictable L2 distance,
    then assert the 1/(1+L2) transform and threshold logic.
    """

    def _sc(self, threshold: float) -> SemanticCache:
        from levy.embeddings import MockEmbeddingClient
        client = MockEmbeddingClient(dimension=2)
        return SemanticCache(
            embedding_client=client,
            threshold=threshold,
            backend="brute_force",
        )

    def test_hit_at_threshold(self):
        """When similarity == threshold exactly (identical vectors → sim=1.0), it is a hit."""
        class ControlledClient:
            def embed(self, text): return [1.0, 0.0]
            def get_dimension(self): return 2

        sc = SemanticCache(embedding_client=ControlledClient(), threshold=1.0, backend="brute_force")
        req = LLMRequest(prompt="hello")
        sc.set(req, "response", embedding=[1.0, 0.0])
        entry = sc.get(LLMRequest(prompt="hello"))
        # L2 distance = 0 → similarity = 1.0 = threshold → should be a hit
        self.assertIsNotNone(entry)

    def test_miss_below_threshold(self):
        """Two orthogonal unit vectors: L2 = sqrt(2) → similarity ≈ 0.414."""
        # Override embedding_client to return controlled vectors.
        class ControlledClient:
            def __init__(self, vec):
                self._vec = vec
            def embed(self, text):
                return list(self._vec)
            def get_dimension(self):
                return len(self._vec)

        store_vec = np.array([1.0, 0.0], dtype=np.float32)
        query_vec = np.array([0.0, 1.0], dtype=np.float32)
        l2 = float(np.linalg.norm(store_vec - query_vec))  # sqrt(2) ≈ 1.414
        expected_sim = 1.0 / (1.0 + l2)  # ≈ 0.414

        sc = SemanticCache(embedding_client=ControlledClient(store_vec), threshold=0.5, backend="brute_force")
        req = LLMRequest(prompt="q")
        sc.set(req, "resp", embedding=store_vec.tolist())

        sc.embedding_client = ControlledClient(query_vec)
        result = sc.get(LLMRequest(prompt="q"))
        self.assertIsNone(result, f"Expected miss (sim={expected_sim:.3f} < 0.5)")

    def test_hit_above_threshold(self):
        """Same vector: L2=0 → similarity=1.0. Should hit at any threshold ≤ 1."""
        class ControlledClient:
            def embed(self, text): return [1.0, 0.0]
            def get_dimension(self): return 2

        sc = SemanticCache(embedding_client=ControlledClient(), threshold=0.9, backend="brute_force")
        req = LLMRequest(prompt="q")
        sc.set(req, "resp", embedding=[1.0, 0.0])
        entry = sc.get(LLMRequest(prompt="q"))
        self.assertIsNotNone(entry)
        sim = entry.metadata.get("last_similarity_score", 0)
        self.assertGreaterEqual(sim, 0.9)

    def test_similarity_score_written_to_metadata(self):
        class ControlledClient:
            def embed(self, text): return [1.0, 0.0]
            def get_dimension(self): return 2

        sc = SemanticCache(embedding_client=ControlledClient(), threshold=0.0, backend="brute_force")
        req = LLMRequest(prompt="q")
        sc.set(req, "resp", embedding=[1.0, 0.0])
        entry = sc.get(LLMRequest(prompt="q"))
        self.assertIn("last_similarity_score", entry.metadata)
        self.assertIsInstance(entry.metadata["last_similarity_score"], float)


# ---------------------------------------------------------------------------
# 6.3  id→entry resolution
# ---------------------------------------------------------------------------

class TestIdEntryMapping(unittest.TestCase):

    def test_retrieved_id_resolves_to_correct_entry(self):
        class ControlledClient:
            def embed(self, text): return [1.0, 0.0]
            def get_dimension(self): return 2

        sc = SemanticCache(embedding_client=ControlledClient(), threshold=0.0, backend="brute_force")
        req = LLMRequest(prompt="my query")
        model_meta = {"canonical_name": "all-MiniLM-L6-v2", "checkpoint": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 2}
        sc.set(req, "my response", embedding=[1.0, 0.0], metadata=model_meta)

        entry = sc.get(LLMRequest(prompt="my query"))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.prompt, "my query")
        self.assertEqual(entry.response_text, "my response")
        self.assertEqual(entry.metadata.get("canonical_name"), "all-MiniLM-L6-v2")

    def test_multiple_entries_nearest_is_returned(self):
        class ControlledClient:
            def __init__(self, v): self._v = v
            def embed(self, text): return list(self._v)
            def get_dimension(self): return 2

        store_client = ControlledClient([1.0, 0.0])
        sc = SemanticCache(embedding_client=store_client, threshold=0.0, backend="brute_force")

        sc.set(LLMRequest(prompt="a"), "resp-a", embedding=[1.0, 0.0])   # id=0, angle=0°
        sc.set(LLMRequest(prompt="b"), "resp-b", embedding=[0.0, 1.0])   # id=1, angle=90°

        # Query close to "a" (angle≈5°)
        q_vec = [math.cos(math.radians(5)), math.sin(math.radians(5))]
        sc.embedding_client = ControlledClient(q_vec)
        entry = sc.get(LLMRequest(prompt="query"))
        self.assertEqual(entry.response_text, "resp-a")


# ---------------------------------------------------------------------------
# 6.4  Faiss ↔ brute-force agreement
# ---------------------------------------------------------------------------

@unittest.skipUnless(FAISS_AVAILABLE, "faiss-cpu not installed — skipping agreement test")
class TestFaissAgreement(unittest.TestCase):

    def test_same_fixture_same_result(self):
        """Faiss HNSW and brute-force must agree on hit/miss for the same fixture."""
        dim = 4
        store_vec = _unit_vec(dim, 0)     # [1, 0, 0, 0]
        other_vec = _unit_vec(dim, 90)    # [0, 1, 0, 0]
        query_vec = _unit_vec(dim, 5)     # close to store_vec

        brute = BruteForceVectorIndex()
        brute.add(store_vec, entry_id=0)
        brute.add(other_vec, entry_id=1)
        b_ids, b_dists = brute.search(query_vec, k=1)

        faiss_idx = FaissHNSWVectorIndex(m=16, ef_construction=100, ef_search=32)
        faiss_idx.add(store_vec, entry_id=0)
        faiss_idx.add(other_vec, entry_id=1)
        f_ids, f_dists = faiss_idx.search(query_vec, k=1)

        self.assertEqual(b_ids[0], f_ids[0], "Nearest id should be the same")
        self.assertAlmostEqual(b_dists[0], f_dists[0], places=4,
                               msg="L2 distances should agree within 4 decimal places")


# ---------------------------------------------------------------------------
# 6.5  Engine end-to-end with mock embeddings
# ---------------------------------------------------------------------------

class TestEngineSemanticCache(unittest.TestCase):

    def _config(self, threshold=0.0):
        return LevyConfig(
            enable_exact_cache=False,
            enable_semantic_cache=True,
            similarity_threshold=threshold,
            llm_provider="mock",
            embedding_provider="mock",
            vector_index_backend="brute_force",
        )

    def test_miss_then_hit_via_semantic_cache(self):
        """After a miss, a same-text query returns source='semantic_cache'."""
        engine = LevyEngine(self._config(threshold=0.0))
        r1 = engine.generate("hello world")
        self.assertEqual(r1.source, "llm")

        # Mock embeddings are seeded by text, so same text → same vector → similarity=1.
        r2 = engine.generate("hello world")
        self.assertEqual(r2.source, "semantic_cache")
        self.assertIsNotNone(r2.similarity_score)

    def test_similarity_score_in_result(self):
        """Semantic hit carries a float similarity_score in [0, 1]."""
        engine = LevyEngine(self._config(threshold=0.0))
        engine.generate("test query")
        result = engine.generate("test query")
        if result.source == "semantic_cache":
            self.assertIsInstance(result.similarity_score, float)
            self.assertGreaterEqual(result.similarity_score, 0.0)
            self.assertLessEqual(result.similarity_score, 1.0)

    def test_semantic_disabled_skips_index(self):
        """With semantic cache disabled, source is always 'llm' or 'exact_cache'."""
        config = LevyConfig(
            enable_exact_cache=False,
            enable_semantic_cache=False,
            llm_provider="mock",
            embedding_provider="mock",
        )
        engine = LevyEngine(config)
        r1 = engine.generate("prompt")
        r2 = engine.generate("prompt")
        self.assertNotEqual(r1.source, "semantic_cache")
        self.assertNotEqual(r2.source, "semantic_cache")

    def test_reset_clears_semantic_cache(self):
        """After reset(), a previously-stored entry is no longer retrievable."""
        engine = LevyEngine(self._config(threshold=0.0))
        engine.generate("hello")
        engine.semantic_cache.reset()
        # Next call must miss (goes to LLM)
        result = engine.generate("hello")
        self.assertEqual(result.source, "llm")

    def test_existing_exact_cache_behavior_preserved(self):
        """Existing test intent: with semantic cache at threshold=0 and mock,
        storing one entry gives one entry (in the exact store). Semantic cache
        now has its own separate index."""
        config = LevyConfig(
            enable_exact_cache=False,
            enable_semantic_cache=True,
            similarity_threshold=0.0,
            llm_provider="mock",
            embedding_provider="mock",
            vector_index_backend="brute_force",
        )
        engine = LevyEngine(config)
        engine.generate("Hello")
        # Exact store still has 1 entry
        self.assertEqual(len(engine.store.entries), 1)
        # Semantic index also has 1 entry
        self.assertEqual(engine.semantic_cache._index.size(), 1)


if __name__ == "__main__":
    unittest.main()
