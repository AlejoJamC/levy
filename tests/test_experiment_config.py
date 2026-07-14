"""
Tests for the frozen experimental grid (LEV-4 / 4.1).
"""

import unittest

from levy.dataset.schema import WORKLOADS
from levy.experiment.config import EMBEDDING_MODELS, THRESHOLDS, ExperimentConfig, full_grid


class TestFullGrid(unittest.TestCase):

    def test_yields_exactly_30_configs(self):
        self.assertEqual(len(full_grid()), 30)

    def test_all_configs_distinct(self):
        grid = full_grid()
        identities = {(c.model, c.workload, c.threshold) for c in grid}
        self.assertEqual(len(identities), 30)

    def test_covers_every_model_workload_threshold_combination(self):
        grid = full_grid()
        identities = {(c.model, c.workload, c.threshold) for c in grid}
        expected = {
            (model, workload, threshold)
            for model in EMBEDDING_MODELS
            for workload in WORKLOADS
            for threshold in THRESHOLDS
        }
        self.assertEqual(identities, expected)

    def test_frozen_models_and_workloads(self):
        self.assertEqual(EMBEDDING_MODELS, ("all-MiniLM-L6-v2", "modernbert"))
        self.assertEqual(set(WORKLOADS), {"faq", "code", "chat"})

    def test_thresholds_unmodified(self):
        self.assertEqual(THRESHOLDS, (0.70, 0.75, 0.80, 0.85, 0.90))
        for config in full_grid():
            self.assertIn(config.threshold, THRESHOLDS)

    def test_config_id_is_stable_and_unique(self):
        grid = full_grid()
        ids = {c.config_id for c in grid}
        self.assertEqual(len(ids), 30)
        self.assertEqual(
            ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.7).config_id,
            "all-MiniLM-L6-v2|faq|0.70",
        )


if __name__ == "__main__":
    unittest.main()
