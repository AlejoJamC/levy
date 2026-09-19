"""
Tests for the LEV-22 robustness re-analysis, pinned against hand-computed
fixtures.

Model permutation fixture (one workload, two thresholds, n = 3 per model):
    stratum t1: model A 2 FP, model B 0 FP  -> k = 2
    stratum t2: model A 1 FP, model B 0 FP  -> k = 1
    A-count per stratum ~ Hypergeom(6, k, 3):
        t1: P(0) = 3/15, P(1) = 9/15, P(2) = 3/15
        t2: P(0) = P(1) = 1/2
    total A in {0,1,2,3} with probs {0.1, 0.4, 0.4, 0.1}; observed A = 3,
    statistic proportional to 2A - 3, |2A - 3| >= 3 iff A in {0, 3}
    -> exact two-sided p = 0.2.

Workload permutation fixture (one stratum, workloads x/y/z with n = 2/1/1,
one event, observed in y):
    event in x (prob 1/2): rates (1/2, 0, 0), SS = 1/6
    event in y or z (prob 1/4 each): SS = 2/3
    observed SS = 2/3 -> exact p = 1/2.

Logistic: for a 2x2 table the ML log odds ratio is log(ad / bc), and the
Firth estimate of a saturated binary logistic model equals the log odds ratio
with 1/2 added to every cell.
"""

import itertools
import math
import unittest

import numpy as np
import pandas as pd

from levy.analysis.robustness import (
    _agree,
    firth_fit,
    ml_fit,
    model_cluster_signflip_exact,
    model_permutation_exact,
    model_permutation_mc,
    negatives_from_frame,
    separation_check,
    workload_permutation_exact,
)


def _rows(model, workload, threshold, outcomes, pair_prefix):
    return [
        {"model": model, "workload": workload, "threshold": threshold, "pair_id": f"{pair_prefix}-{i}", "label": 0, "outcome": "FP" if y else "TN"}
        for i, y in enumerate(outcomes)
    ]


def model_fixture():
    rows = []
    rows += _rows("A", "w", "0.70", [1, 1, 0], "p")
    rows += _rows("B", "w", "0.70", [0, 0, 0], "p")
    rows += _rows("A", "w", "0.75", [1, 0, 0], "p")
    rows += _rows("B", "w", "0.75", [0, 0, 0], "p")
    return negatives_from_frame(pd.DataFrame(rows))


def workload_fixture():
    rows = []
    rows += _rows("A", "x", "0.70", [0, 0], "x")
    rows += _rows("A", "y", "0.70", [1], "y")
    rows += _rows("A", "z", "0.70", [0], "z")
    return negatives_from_frame(pd.DataFrame(rows))


def _brute_force_model_p(neg):
    """Enumerate every relabelling within each stratum."""
    strata = []
    for _, g in neg.groupby(["workload", "threshold"]):
        y = g["is_fp"].to_numpy()
        n = len(y) // 2
        strata.append([(y[list(c)].sum() - (y.sum() - y[list(c)].sum())) / n for c in itertools.combinations(range(len(y)), n)])
    obs = model_permutation_exact(neg)["diff_mean_config_fpr"]
    stats_all = [sum(combo) / len(strata) for combo in itertools.product(*strata)]
    return sum(abs(t) >= abs(obs) - 1e-12 for t in stats_all) / len(stats_all)


class ModelPermutationTest(unittest.TestCase):
    def test_exact_p_matches_hand_computation(self):
        result = model_permutation_exact(model_fixture())
        self.assertAlmostEqual(result["p_mean_config"], 0.2, places=12)
        self.assertAlmostEqual(result["p_pooled"], 0.2, places=12)
        self.assertAlmostEqual(result["diff_mean_config_fpr"], (2 / 3 + 1 / 3) / 2, places=12)
        self.assertEqual((result["events_a"], result["events_b"]), (3, 0))

    def test_exact_p_matches_full_enumeration(self):
        neg = model_fixture()
        self.assertAlmostEqual(model_permutation_exact(neg)["p_mean_config"], _brute_force_model_p(neg), places=12)

    def test_monte_carlo_approaches_exact(self):
        mc = model_permutation_mc(model_fixture(), n_perm=20_000, seed=1)
        self.assertAlmostEqual(mc["p_value"], 0.2, delta=0.02)

    def test_unequal_n_per_model_is_rejected(self):
        neg = model_fixture().iloc[1:]
        with self.assertRaises(ValueError):
            model_permutation_exact(neg)

    def test_cluster_signflip_hand_computed(self):
        # Pairs p-0 and p-1 are discordant (A only); p-0 contributes at both
        # thresholds, p-1 at one: d = (2/3, 1/3) / 2 strata. Sign patterns:
        # |+-1/2|, |+-1/6| -> 2 of 4 reach the observed 1/2.
        result = model_cluster_signflip_exact(model_fixture())
        self.assertEqual(result["discordant_pairs"], 2)
        self.assertAlmostEqual(result["statistic"], 0.5, places=12)
        self.assertAlmostEqual(result["p_value"], 0.5, places=12)


class WorkloadPermutationTest(unittest.TestCase):
    def test_exact_p_matches_hand_computation(self):
        result = workload_permutation_exact(workload_fixture())
        self.assertAlmostEqual(result["statistic"], 2 / 3, places=12)
        self.assertAlmostEqual(result["p_value"], 0.5, places=12)


class LogisticTest(unittest.TestCase):
    X = np.array([[1.0, 0.0], [1.0, 1.0]])

    def test_firth_equals_haldane_corrected_log_odds_ratio(self):
        events, n = np.array([3.0, 7.0]), np.array([20.0, 25.0])
        fit = firth_fit(self.X, events, n)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["beta"][1], math.log((7.5 / 18.5) / (3.5 / 17.5)), places=8)
        self.assertAlmostEqual(fit["beta"][0], math.log(3.5 / 17.5), places=8)

    def test_firth_finite_under_separation(self):
        events, n = np.array([10.0, 0.0]), np.array([50.0, 40.0])
        self.assertTrue(separation_check(self.X, events, n)["separation"])
        fit = firth_fit(self.X, events, n)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["beta"][1], math.log((0.5 / 40.5) / (10.5 / 40.5)), places=8)

    def test_ml_log_odds_ratio_and_no_separation(self):
        events, n = np.array([3.0, 7.0]), np.array([20.0, 25.0])
        self.assertFalse(separation_check(self.X, events, n)["separation"])
        cells = pd.DataFrame({"events": events, "n": n})
        fit = ml_fit(pd.DataFrame(self.X, columns=["Intercept", "x"]), cells)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["result"].params.iloc[1], math.log((7 * 17) / (18 * 3)), places=8)

    def test_fixed_coefficient_is_held(self):
        events, n = np.array([3.0, 7.0]), np.array([20.0, 25.0])
        fit = firth_fit(self.X, events, n, fixed={1: 0.0})
        self.assertEqual(fit["beta"][1], 0.0)


class VerdictRuleTest(unittest.TestCase):
    def test_agree_disagree(self):
        self.assertEqual(_agree("retain", "retain"), "AGREE")
        self.assertEqual(_agree("reject", "retain"), "DISAGREE")
        self.assertEqual(_agree("no_verdict_not_converged", "retain"), "NO_VERDICT")


if __name__ == "__main__":
    unittest.main()
