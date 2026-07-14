"""
Tests for confusion-matrix metrics and sanity checks (LEV-4 / 4.2).
"""

import unittest

from levy.experiment.config import ExperimentConfig
from levy.experiment.metrics import (
    EvaluationResult,
    ExperimentSanityError,
    check_sanity,
    evaluate_confusion,
)

_CFG = ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.85)


class TestKnownCounts(unittest.TestCase):
    """Hand-computed case from the frozen spec: TP=8, FP=2, TN=7, FN=3."""

    def setUp(self):
        self.result = evaluate_confusion(_CFG, tp=8, fp=2, tn=7, fn=3, n=20)

    def test_precision(self):
        self.assertAlmostEqual(self.result.precision, 0.8, places=6)

    def test_recall(self):
        self.assertAlmostEqual(self.result.recall, 8 / 11, places=6)

    def test_f0_5(self):
        self.assertAlmostEqual(self.result.f0_5, 0.7843137254901961, places=6)

    def test_fpr(self):
        self.assertAlmostEqual(self.result.fpr, 2 / 9, places=6)

    def test_hit_rate(self):
        self.assertAlmostEqual(self.result.hit_rate, 0.5, places=6)

    def test_no_zero_division_flags(self):
        self.assertFalse(self.result.precision_zero_div)
        self.assertFalse(self.result.recall_zero_div)
        self.assertFalse(self.result.fpr_zero_div)


class TestZeroDivision(unittest.TestCase):

    def test_no_hits_at_all_flags_precision(self):
        """TP+FP=0 -> precision reported as 0.0 with the zero-division flag set."""
        result = evaluate_confusion(_CFG, tp=0, fp=0, tn=10, fn=5, n=15)
        self.assertEqual(result.precision, 0.0)
        self.assertTrue(result.precision_zero_div)

    def test_no_positives_flags_recall(self):
        """TP+FN=0 -> recall reported as 0.0 with the zero-division flag set."""
        result = evaluate_confusion(_CFG, tp=0, fp=3, tn=7, fn=0, n=10)
        self.assertEqual(result.recall, 0.0)
        self.assertTrue(result.recall_zero_div)

    def test_no_negatives_flags_fpr(self):
        """FP+TN=0 -> FPR reported as 0.0 with the zero-division flag set."""
        result = evaluate_confusion(_CFG, tp=5, fp=0, tn=0, fn=2, n=7)
        self.assertEqual(result.fpr, 0.0)
        self.assertTrue(result.fpr_zero_div)

    def test_never_nan(self):
        result = evaluate_confusion(_CFG, tp=0, fp=0, tn=0, fn=0, n=0)
        for value in (result.precision, result.recall, result.f0_5, result.fpr, result.hit_rate):
            self.assertEqual(value, 0.0)
            self.assertFalse(value != value)  # NaN would fail equality with itself


class TestSanityChecks(unittest.TestCase):

    def test_mismatched_n_raises(self):
        with self.assertRaises(ExperimentSanityError):
            evaluate_confusion(_CFG, tp=8, fp=2, tn=7, fn=3, n=19)

    def test_error_names_offending_configuration(self):
        with self.assertRaises(ExperimentSanityError) as ctx:
            evaluate_confusion(_CFG, tp=8, fp=2, tn=7, fn=3, n=100)
        self.assertIn(_CFG.config_id, str(ctx.exception))

    def test_out_of_range_rate_raises(self):
        bad_result = EvaluationResult(
            config=_CFG,
            n=10,
            tp=5,
            fp=5,
            tn=0,
            fn=0,
            precision=1.5,  # invalid on purpose
            recall=0.5,
            f0_5=0.5,
            fpr=0.0,
            hit_rate=1.0,
            precision_zero_div=False,
            recall_zero_div=False,
            fpr_zero_div=False,
        )
        with self.assertRaises(ExperimentSanityError):
            check_sanity(bad_result)


if __name__ == "__main__":
    unittest.main()
