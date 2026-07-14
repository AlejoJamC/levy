"""
Tests for the Algorithm 2 replay protocol (LEV-4 / 4.3).

Mock embeddings are text-hashed random vectors that don't capture semantic
similarity, so these tests inject a small scripted embedding manager that
maps specific query texts to controlled 2-D unit vectors at known angles --
the same technique `tests/test_vector_index.py` uses to make similarity
outcomes predictable.
"""

import math
import unittest

from levy.dataset.schema import QueryPair
from levy.embedding_manager import ModelIdentity
from levy.experiment.config import ExperimentConfig
from levy.experiment.replay import run_experiment


def _vec(angle_deg: float) -> list:
    rad = math.radians(angle_deg)
    return [math.cos(rad), math.sin(rad)]


class ScriptedEmbeddingManager:
    """Duck-types the engine's embedding_manager interface with fixed vectors per text."""

    def __init__(self, vectors: dict):
        self._vectors = vectors
        dim = len(next(iter(vectors.values())))
        self._identity = ModelIdentity(canonical_name="scripted", checkpoint="scripted", dimension=dim)

    def embed(self, text: str) -> list:
        return list(self._vectors[text])

    def get_model_identity(self) -> ModelIdentity:
        return self._identity


_CFG = ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.70)


def _pair(pair_id, q1, q2, label):
    return QueryPair(
        pair_id=pair_id,
        workload="faq",
        source_corpus="synthetic-fixture",
        source_pair_id=pair_id,
        query_1=q1,
        query_2=q2,
        original_label=label,
    )


class TestConfusionOutcomes(unittest.TestCase):
    """Each scenario replays a single, isolated pair (empty cache at start)."""

    def test_near_identical_pair_is_true_positive(self):
        pair = _pair("p-tp", "near query one", "near query two", label=1)
        manager = ScriptedEmbeddingManager({"near query one": _vec(0), "near query two": _vec(2)})
        result = run_experiment(_CFG, [pair], embedding_manager=manager, llm_latency_seconds=0)
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (1, 0, 0, 0))
        self.assertEqual(result.decisions[0].outcome, "TP")
        self.assertEqual(result.decisions[0].source, "semantic_cache")

    def test_near_identical_pair_with_negative_label_is_false_positive(self):
        pair = _pair("p-fp", "near query one", "near query two", label=0)
        manager = ScriptedEmbeddingManager({"near query one": _vec(0), "near query two": _vec(2)})
        result = run_experiment(_CFG, [pair], embedding_manager=manager, llm_latency_seconds=0)
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (0, 1, 0, 0))
        self.assertEqual(result.decisions[0].outcome, "FP")

    def test_unrelated_pair_is_true_negative(self):
        pair = _pair("p-tn", "far query one", "far query two", label=0)
        manager = ScriptedEmbeddingManager({"far query one": _vec(0), "far query two": _vec(90)})
        result = run_experiment(_CFG, [pair], embedding_manager=manager, llm_latency_seconds=0)
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (0, 0, 1, 0))
        self.assertEqual(result.decisions[0].outcome, "TN")

    def test_unrelated_pair_with_positive_label_is_false_negative(self):
        pair = _pair("p-fn", "far query one", "far query two", label=1)
        manager = ScriptedEmbeddingManager({"far query one": _vec(0), "far query two": _vec(90)})
        result = run_experiment(_CFG, [pair], embedding_manager=manager, llm_latency_seconds=0)
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (0, 0, 0, 1))
        self.assertEqual(result.decisions[0].outcome, "FN")

    def test_exact_duplicate_decided_by_exact_cache(self):
        """query_1 == query_2 verbatim -> exact cache hit, logged with its source."""
        pair = _pair("p-exact", "identical text", "identical text", label=1)
        manager = ScriptedEmbeddingManager({"identical text": _vec(0)})
        result = run_experiment(_CFG, [pair], embedding_manager=manager, llm_latency_seconds=0)
        self.assertEqual(result.decisions[0].outcome, "TP")
        self.assertEqual(result.decisions[0].source, "exact_cache")


class TestCacheAccumulation(unittest.TestCase):

    def test_second_pair_matches_first_pairs_stored_query(self):
        """query_2 of the second pair is compared against everything stored so far,
        not only its own pair's query_1."""
        pair1 = _pair("acc-1", "alpha entry", "beta entry", label=0)  # far apart -> TN, both stored
        pair2 = _pair("acc-2", "gamma entry", "delta entry", label=1)  # delta is close to alpha
        manager = ScriptedEmbeddingManager(
            {
                "alpha entry": _vec(0),
                "beta entry": _vec(170),
                "gamma entry": _vec(90),
                "delta entry": _vec(2),  # close to "alpha entry", not to "gamma entry"
            }
        )
        result = run_experiment(_CFG, [pair1, pair2], embedding_manager=manager, llm_latency_seconds=0)
        self.assertEqual(result.n, 2)
        second_decision = result.decisions[1]
        self.assertEqual(second_decision.pair_id, "acc-2")
        self.assertEqual(second_decision.decision, "hit")
        self.assertEqual(second_decision.source, "semantic_cache")
        self.assertGreater(second_decision.similarity, 0.9)


class TestFreshCachePerConfiguration(unittest.TestCase):

    def test_no_state_leaks_between_run_experiment_calls(self):
        """Even sharing one embedding_manager (as the sweep runner does), a second
        run_experiment call starts from an empty cache."""
        manager = ScriptedEmbeddingManager(
            {
                "z entry": _vec(2),
                "w entry": _vec(170),
                "q1 entry": _vec(90),
                "q2 entry": _vec(4),  # close to "z entry", which only exists in the first run
            }
        )

        first_pair = _pair("leak-z", "z entry", "w entry", label=0)
        run_experiment(_CFG, [first_pair], embedding_manager=manager, llm_latency_seconds=0)

        second_pair = _pair("leak-q", "q1 entry", "q2 entry", label=1)
        result = run_experiment(_CFG, [second_pair], embedding_manager=manager, llm_latency_seconds=0)

        # If "z entry" had leaked into this run's cache, q2 would spuriously hit.
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (0, 0, 0, 1))
        self.assertEqual(result.decisions[0].outcome, "FN")
        self.assertEqual(result.decisions[0].decision, "miss")


if __name__ == "__main__":
    unittest.main()
