"""
Tests for the two-way ANOVA and conditional Tukey HSD (LEV-8 / 4.2, 4.3).

Expected values are double-sourced: every fixture's sums of squares are
computed by hand here from the balanced 2 x 3 x 5 design formulas and
compared against what statsmodels produces. If the two disagree the test
fails -- neither source is trusted alone.

Hand formulas for a balanced two-factor design with a=2 models, b=3
workloads, n=5 replicates per cell:

    SS_A   = b*n * sum_i (mean_i.  - grand)^2
    SS_B   = a*n * sum_j (mean_.j  - grand)^2
    SS_AB  = n   * sum_ij (mean_ij - mean_i. - mean_.j + grand)^2
    SS_E   = sum over cells of the within-cell sum of squares
"""

import math
import unittest
from itertools import product

from analysis_fixtures import (
    MODELS,
    OFFSETS,
    RESIDUAL_DF,
    RESIDUAL_SS,
    WORKLOADS,
    additive_cells,
    anova_frame,
)

from levy.analysis.hypothesis import (
    ANOVA_COLUMNS,
    EFFECT_SIZE_COLUMNS,
    AnovaDesignError,
    run_tukey_hsd,
    run_two_way_anova,
)

N_REPLICATES = len(OFFSETS)


def hand_sums_of_squares(cell_fpr):
    """Balanced-design sums of squares, computed independently of statsmodels."""
    grand = sum(cell_fpr.values()) / len(cell_fpr)
    model_means = {
        model: sum(cell_fpr[(model, w)] for w in WORKLOADS) / len(WORKLOADS) for model in MODELS
    }
    workload_means = {
        workload: sum(cell_fpr[(m, workload)] for m in MODELS) / len(MODELS) for workload in WORKLOADS
    }

    ss_model = len(WORKLOADS) * N_REPLICATES * sum((mean - grand) ** 2 for mean in model_means.values())
    ss_workload = len(MODELS) * N_REPLICATES * sum((mean - grand) ** 2 for mean in workload_means.values())
    ss_interaction = N_REPLICATES * sum(
        (cell_fpr[(m, w)] - model_means[m] - workload_means[w] + grand) ** 2
        for m, w in product(MODELS, WORKLOADS)
    )
    # Every cell carries the same replicate offsets, so the within-cell sum
    # of squares is the same for all six cells.
    ss_residual = len(cell_fpr) * sum(offset**2 for offset in OFFSETS)
    return ss_model, ss_workload, ss_interaction, ss_residual


class TestHandComputationAgreesWithStatsmodels(unittest.TestCase):
    """The double-sourcing itself: hand formulas vs the reference implementation."""

    def test_sums_of_squares_match_on_a_non_additive_design(self):
        cells = {
            ("all-MiniLM-L6-v2", "faq"): 0.10,
            ("all-MiniLM-L6-v2", "code"): 0.22,
            ("all-MiniLM-L6-v2", "chat"): 0.31,
            ("modernbert", "faq"): 0.44,
            ("modernbert", "code"): 0.25,
            ("modernbert", "chat"): 0.53,
        }
        ss_model, ss_workload, ss_interaction, ss_residual = hand_sums_of_squares(cells)
        table = run_two_way_anova(anova_frame(cells)).table.set_index("effect")

        self.assertAlmostEqual(table.loc["model", "sum_sq"], ss_model, places=10)
        self.assertAlmostEqual(table.loc["workload", "sum_sq"], ss_workload, places=10)
        self.assertAlmostEqual(table.loc["model:workload", "sum_sq"], ss_interaction, places=10)
        self.assertAlmostEqual(table.loc["Residual", "sum_sq"], ss_residual, places=10)
        self.assertAlmostEqual(ss_residual, RESIDUAL_SS, places=12)


class TestModelEffectFixture(unittest.TestCase):
    """One model's FPR uniformly higher across every workload."""

    def setUp(self):
        # A: 0.10 / 0.20 / 0.30, B: 0.30 / 0.40 / 0.50 -- additive, so the
        # interaction sum of squares is exactly zero.
        self.cells = additive_cells(model_effect=0.20, workload_effect=0.10, base=0.10)
        self.anova = run_two_way_anova(anova_frame(self.cells))
        self.table = self.anova.table.set_index("hypothesis")

    def test_hand_computed_sums_of_squares(self):
        ss_model, ss_workload, ss_interaction, ss_residual = hand_sums_of_squares(self.cells)
        # Hand arithmetic, spelled out: grand mean 0.30; model means 0.20/0.40;
        # SS_model = 3*5*(0.01+0.01) = 0.30. Workload means 0.20/0.30/0.40;
        # SS_workload = 2*5*(0.01+0+0.01) = 0.20. Additive => SS_interaction = 0.
        self.assertAlmostEqual(ss_model, 0.30, places=12)
        self.assertAlmostEqual(ss_workload, 0.20, places=12)
        self.assertAlmostEqual(ss_interaction, 0.0, places=12)
        self.assertAlmostEqual(ss_residual, 0.006, places=12)

        self.assertAlmostEqual(self.table.loc["H0_1", "sum_sq"], 0.30, places=10)
        self.assertAlmostEqual(self.table.loc["H0_2", "sum_sq"], 0.20, places=10)
        self.assertAlmostEqual(self.table.loc["H0_3", "sum_sq"], 0.0, places=10)

    def test_degrees_of_freedom_and_f_statistics(self):
        self.assertEqual(self.table.loc["H0_1", "df"], 1.0)
        self.assertEqual(self.table.loc["H0_2", "df"], 2.0)
        self.assertEqual(self.table.loc["H0_3", "df"], 2.0)
        self.assertEqual(self.anova.table.set_index("effect").loc["Residual", "df"], RESIDUAL_DF)

        mean_sq_residual = RESIDUAL_SS / RESIDUAL_DF  # 0.00025
        self.assertAlmostEqual(self.table.loc["H0_1", "F"], 0.30 / mean_sq_residual, places=6)
        self.assertAlmostEqual(self.table.loc["H0_2", "F"], (0.20 / 2) / mean_sq_residual, places=6)

    def test_h0_1_is_rejected_with_supporting_statistics(self):
        self.assertEqual(self.anova.decision("H0_1"), "reject")
        self.assertLess(self.anova.p_value("H0_1"), 0.05)
        self.assertGreater(self.table.loc["H0_1", "F"], 0.0)
        self.assertIn("model", self.anova.significant_effects)

    def test_additive_design_retains_the_interaction(self):
        self.assertEqual(self.anova.decision("H0_3"), "retain")
        self.assertGreaterEqual(self.anova.p_value("H0_3"), 0.05)
        self.assertNotIn("model:workload", self.anova.significant_effects)

    def test_tukey_runs_for_the_significant_effects_only(self):
        tukey = run_tukey_hsd(anova_frame(self.cells), self.anova)

        self.assertTrue(tukey.ran)
        self.assertIn("model", tukey.statement)
        self.assertTrue(tukey.per_effect["model"].startswith("ran"))
        self.assertTrue(tukey.per_effect["workload"].startswith("ran"))
        self.assertTrue(tukey.per_effect["model:workload"].startswith("skipped"))

        model_rows = tukey.table.loc[tukey.table["effect"] == "model"]
        self.assertEqual(len(model_rows), 1)  # 2 models -> 1 comparison
        self.assertEqual(len(tukey.table.loc[tukey.table["effect"] == "workload"]), 3)
        self.assertEqual(len(tukey.table.loc[tukey.table["effect"] == "model:workload"]), 0)

        row = model_rows.iloc[0]
        self.assertAlmostEqual(abs(row["meandiff"]), 0.20, places=6)  # 0.40 - 0.20
        self.assertLess(row["p_adj"], 0.05)
        self.assertTrue(row["reject"])
        self.assertLess(row["lower"], row["upper"])

    def test_diagnostics_describe_the_design(self):
        design = self.anova.diagnostics["design"]
        self.assertTrue(design["balanced"])
        self.assertEqual(sorted(design["models"]), sorted(MODELS))
        self.assertEqual(set(design["cell_counts"].values()), {5})
        self.assertEqual(self.anova.diagnostics["n_observations"], 30)
        self.assertEqual(self.anova.diagnostics["sum_of_squares_type"], "II")
        self.assertIn("shapiro_wilk", self.anova.diagnostics)
        self.assertIn("levene", self.anova.diagnostics)
        self.assertFalse(self.anova.degenerate)


class TestInteractionFixture(unittest.TestCase):
    """A non-additive design: Tukey must compare the 6 model x workload cells."""

    def setUp(self):
        self.cells = {
            ("all-MiniLM-L6-v2", "faq"): 0.10,
            ("all-MiniLM-L6-v2", "code"): 0.10,
            ("all-MiniLM-L6-v2", "chat"): 0.10,
            ("modernbert", "faq"): 0.60,
            ("modernbert", "code"): 0.10,
            ("modernbert", "chat"): 0.10,
        }
        self.anova = run_two_way_anova(anova_frame(self.cells))

    def test_interaction_is_rejected(self):
        self.assertEqual(self.anova.decision("H0_3"), "reject")
        self.assertLess(self.anova.p_value("H0_3"), 0.05)

    def test_tukey_compares_all_six_cells(self):
        tukey = run_tukey_hsd(anova_frame(self.cells), self.anova)
        cell_rows = tukey.table.loc[tukey.table["effect"] == "model:workload"]

        self.assertEqual(len(cell_rows), 15)  # C(6, 2)
        groups = set(cell_rows["group1"]) | set(cell_rows["group2"])
        self.assertEqual(len(groups), 6)
        self.assertIn("modernbert|faq", groups)
        self.assertTrue(cell_rows["reject"].any())


class TestNullFixture(unittest.TestCase):
    """Identical FPR distributions in every cell: all three H0 retained."""

    def setUp(self):
        self.cells = additive_cells(model_effect=0.0, workload_effect=0.0, base=0.20)
        self.anova = run_two_way_anova(anova_frame(self.cells))

    def test_all_three_hypotheses_are_retained_with_p_values(self):
        for hypothesis in ("H0_1", "H0_2", "H0_3"):
            self.assertEqual(self.anova.decision(hypothesis), "retain", hypothesis)
            p_value = self.anova.p_value(hypothesis)
            self.assertFalse(math.isnan(p_value), hypothesis)
            self.assertGreaterEqual(p_value, 0.05, hypothesis)
        self.assertEqual(self.anova.significant_effects, [])
        self.assertFalse(self.anova.degenerate)

    def test_tukey_is_skipped_and_says_so(self):
        tukey = run_tukey_hsd(anova_frame(self.cells), self.anova)

        self.assertFalse(tukey.ran)
        self.assertTrue(tukey.table.empty)
        self.assertIn("not run", tukey.statement)
        self.assertIn("alpha=0.05", tukey.statement)
        for effect in ("model", "workload", "model:workload"):
            self.assertTrue(tukey.per_effect[effect].startswith("skipped"), effect)
            self.assertIn(effect, tukey.per_effect[effect])


class TestDegenerateResponse(unittest.TestCase):
    """
    Zero variance in FPR (the state of the committed synthetic fixture under
    mock embeddings): the F-tests are undefined and must be reported as such,
    never laundered into a retention of the null.
    """

    def setUp(self):
        cells = {key: 0.0 for key in additive_cells()}
        frame = anova_frame(cells)
        frame["fpr"] = 0.0  # flatten the replicate offsets too
        self.frame = frame
        self.anova = run_two_way_anova(frame)

    def test_decisions_are_undefined(self):
        self.assertTrue(self.anova.degenerate)
        for hypothesis in ("H0_1", "H0_2", "H0_3"):
            self.assertEqual(self.anova.decision(hypothesis), "undefined", hypothesis)
        self.assertEqual(self.anova.significant_effects, [])
        self.assertIn("degenerate_response", self.anova.diagnostics)

    def test_diagnostics_record_the_skipped_assumption_tests(self):
        self.assertIn("skipped", self.anova.diagnostics["shapiro_wilk"])
        self.assertIn("skipped", self.anova.diagnostics["levene"])

    def test_tukey_statement_explains_the_degeneracy(self):
        tukey = run_tukey_hsd(self.frame, self.anova)
        self.assertFalse(tukey.ran)
        self.assertIn("zero variance", tukey.statement)
        self.assertIn("undefined", tukey.per_effect["model"])


def hand_effect_sizes(ss_effect, df_effect, ss_total, ss_residual, df_residual):
    """eta^2, partial eta^2 and omega^2 from their definitions, not the module's code."""
    ms_residual = ss_residual / df_residual
    return (
        ss_effect / ss_total,
        ss_effect / (ss_effect + ss_residual),
        (ss_effect - df_effect * ms_residual) / (ss_total + ms_residual),
    )


class TestEffectSizes(unittest.TestCase):
    """LEV-20: eta^2, partial eta^2, omega^2 per term, pinned against hand arithmetic."""

    def setUp(self):
        # Same fixture as TestModelEffectFixture: SS 0.30 / 0.20 / 0.0, residual
        # 0.006 on 24 df, so SS_total = 0.506 and MS_residual = 0.00025.
        self.cells = additive_cells(model_effect=0.20, workload_effect=0.10, base=0.10)
        self.table = run_two_way_anova(anova_frame(self.cells)).table.set_index("effect")

    def test_values_match_the_hand_computed_formulas(self):
        # Spelled out for the model term:
        #   eta^2         = 0.30 / 0.506                     = 0.592885...
        #   partial eta^2 = 0.30 / (0.30 + 0.006)            = 0.980392...
        #   omega^2       = (0.30 - 1*0.00025) / (0.506 + 0.00025) = 0.592098...
        self.assertAlmostEqual(self.table.loc["model", "eta_sq"], 0.30 / 0.506, places=10)
        self.assertAlmostEqual(self.table.loc["model", "partial_eta_sq"], 0.30 / 0.306, places=10)
        self.assertAlmostEqual(self.table.loc["model", "omega_sq"], 0.29975 / 0.50625, places=10)

        ss_model, ss_workload, ss_interaction, ss_residual = hand_sums_of_squares(self.cells)
        ss_total = ss_model + ss_workload + ss_interaction + ss_residual
        for effect, ss, df in (
            ("model", ss_model, 1),
            ("workload", ss_workload, 2),
            ("model:workload", ss_interaction, 2),
        ):
            eta, partial, omega = hand_effect_sizes(ss, df, ss_total, ss_residual, RESIDUAL_DF)
            self.assertAlmostEqual(self.table.loc[effect, "eta_sq"], eta, places=10, msg=effect)
            self.assertAlmostEqual(self.table.loc[effect, "partial_eta_sq"], partial, places=10, msg=effect)
            self.assertAlmostEqual(self.table.loc[effect, "omega_sq"], omega, places=10, msg=effect)

    def test_negative_omega_squared_is_reported_not_clamped(self):
        # SS_interaction = 0 < 2 * MS_residual, so omega^2 = -0.0005 / 0.50625.
        omega = self.table.loc["model:workload", "omega_sq"]
        self.assertLess(omega, 0.0)
        self.assertAlmostEqual(omega, -0.0005 / 0.50625, places=12)

    def test_residual_row_carries_no_effect_size(self):
        for column in EFFECT_SIZE_COLUMNS:
            self.assertTrue(math.isnan(self.table.loc["Residual", column]), column)

    def test_total_sum_of_squares_is_double_sourced_on_a_non_additive_design(self):
        # eta^2 * SS_total_hand must reproduce each term's hand SS, which pins the
        # module's denominator (statsmodels' centered_tss) to the hand total.
        cells = {
            ("all-MiniLM-L6-v2", "faq"): 0.10,
            ("all-MiniLM-L6-v2", "code"): 0.22,
            ("all-MiniLM-L6-v2", "chat"): 0.31,
            ("modernbert", "faq"): 0.44,
            ("modernbert", "code"): 0.25,
            ("modernbert", "chat"): 0.53,
        }
        hand = hand_sums_of_squares(cells)
        ss_total = sum(hand)
        table = run_two_way_anova(anova_frame(cells)).table.set_index("effect")
        for effect, ss in zip(("model", "workload", "model:workload"), hand[:3]):
            self.assertAlmostEqual(table.loc[effect, "eta_sq"] * ss_total, ss, places=10, msg=effect)

    def test_degenerate_response_leaves_every_effect_size_undefined(self):
        frame = anova_frame({key: 0.0 for key in additive_cells()})
        frame["fpr"] = 0.0
        table = run_two_way_anova(frame).table
        for column in EFFECT_SIZE_COLUMNS:
            self.assertTrue(table[column].isna().all(), column)

    def test_existing_columns_keep_their_positions(self):
        table = run_two_way_anova(anova_frame(self.cells)).table
        self.assertEqual(list(table.columns), ANOVA_COLUMNS + EFFECT_SIZE_COLUMNS)
        self.assertTrue(set(ANOVA_COLUMNS).isdisjoint(EFFECT_SIZE_COLUMNS))


class TestDesignValidation(unittest.TestCase):

    def test_single_model_is_rejected(self):
        frame = anova_frame(additive_cells(0.2))
        frame = frame.loc[frame["model"] == MODELS[0]]
        with self.assertRaises(AnovaDesignError) as ctx:
            run_two_way_anova(frame)
        self.assertIn("2 embedding models", str(ctx.exception))

    def test_single_workload_is_rejected(self):
        frame = anova_frame(additive_cells(0.2))
        frame = frame.loc[frame["workload"] == "faq"]
        with self.assertRaises(AnovaDesignError) as ctx:
            run_two_way_anova(frame)
        self.assertIn("2 workloads", str(ctx.exception))

    def test_empty_cell_is_named(self):
        frame = anova_frame(additive_cells(0.2))
        frame = frame.loc[~((frame["model"] == MODELS[1]) & (frame["workload"] == "chat"))]
        with self.assertRaises(AnovaDesignError) as ctx:
            run_two_way_anova(frame)
        self.assertIn("empty cell", str(ctx.exception))
        self.assertIn("modernbert|chat", str(ctx.exception))

    def test_unreplicated_cell_is_named(self):
        frame = anova_frame(additive_cells(0.2))
        keep = ~((frame["model"] == MODELS[1]) & (frame["workload"] == "chat") & (frame["threshold"] > 0.70))
        with self.assertRaises(AnovaDesignError) as ctx:
            run_two_way_anova(frame.loc[keep])
        self.assertIn("under-replicated", str(ctx.exception))

    def test_missing_response_column_is_named(self):
        frame = anova_frame(additive_cells(0.2)).drop(columns=["fpr"])
        with self.assertRaises(AnovaDesignError) as ctx:
            run_two_way_anova(frame)
        self.assertIn("fpr", str(ctx.exception))

    def test_unknown_hypothesis_id_raises(self):
        anova = run_two_way_anova(anova_frame(additive_cells(0.2)))
        with self.assertRaises(KeyError):
            anova.p_value("H0_9")
        with self.assertRaises(KeyError):
            anova.decision("H0_9")


if __name__ == "__main__":
    unittest.main()
