"""
Tests for the threshold-selection curve tables and figures (LEV-8 / 4.4).

The tables are the machine-readable source of truth (the figures are
regenerated from them), so the assertions are on table shape, ordering,
values, and the preserved zero-division flags; figures are checked for
existence and non-emptiness, not pixels.
"""

import tempfile
import unittest
from pathlib import Path

from analysis_fixtures import MODELS, THRESHOLDS, WORKLOADS, additive_cells, build_results

from levy.analysis.curves import (
    CURVE_COLUMNS,
    HIT_RATE_VIABILITY,
    build_curve_tables,
    write_curve_figures,
)
from levy.analysis.io import load_results
from analysis_fixtures import write_harness_dir


def _loaded_results(**kwargs):
    """Round-trip through the loader so the tables see real dtypes."""
    with tempfile.TemporaryDirectory() as tmp:
        harness = write_harness_dir(Path(tmp) / "h", build_results(additive_cells(0.2, 0.1), **kwargs))
        return load_results(harness / "results.csv")


class TestCurveTables(unittest.TestCase):

    def setUp(self):
        self.results = _loaded_results()
        self.hit_rate, self.precision = build_curve_tables(self.results)

    def test_both_tables_cover_six_pairs_of_five_thresholds(self):
        for table in (self.hit_rate, self.precision):
            self.assertEqual(list(table.columns), CURVE_COLUMNS)
            self.assertEqual(len(table), 30)
            counts = table.groupby(["model", "workload"]).size()
            self.assertEqual(len(counts), 6)
            self.assertEqual(set(counts.values), {5})
            self.assertEqual(sorted(table["threshold"].unique()), sorted(THRESHOLDS))
            self.assertEqual(sorted(table["model"].unique()), sorted(MODELS))
            self.assertEqual(sorted(table["workload"].unique()), sorted(WORKLOADS))

    def test_metric_column_identifies_the_curve(self):
        self.assertEqual(set(self.hit_rate["metric"]), {"hit_rate"})
        self.assertEqual(set(self.precision["metric"]), {"precision"})

    def test_values_come_from_the_results_rows(self):
        source = self.results.set_index(["model", "workload", "threshold"])
        for _, row in self.precision.iterrows():
            expected = source.loc[(row["model"], row["workload"], row["threshold"]), "precision"]
            self.assertAlmostEqual(row["value"], float(expected), places=9)

    def test_rows_are_sorted_for_byte_stable_output(self):
        keys = list(zip(self.hit_rate["model"], self.hit_rate["workload"], self.hit_rate["threshold"]))
        self.assertEqual(keys, sorted(keys))

    def test_row_order_is_independent_of_input_order(self):
        shuffled = self.results.sample(frac=1.0, random_state=17).reset_index(drop=True)
        hit_rate, precision = build_curve_tables(shuffled)
        self.assertTrue(hit_rate.equals(self.hit_rate))
        self.assertTrue(precision.equals(self.precision))


class TestZeroDivisionFlags(unittest.TestCase):

    def test_precision_flag_is_carried_through(self):
        results = _loaded_results(precision_zero_div_cells=[("modernbert", "code")])
        _, precision = build_curve_tables(results)

        flagged = precision.loc[precision["zero_div"]]
        self.assertEqual(len(flagged), 5)
        self.assertEqual(set(flagged["model"]), {"modernbert"})
        self.assertEqual(set(flagged["workload"]), {"code"})
        # A flagged cell reads 0.0: visible as degenerate, not as a real zero.
        self.assertEqual(set(flagged["value"]), {0.0})

    def test_hit_rate_flag_is_derived_from_an_empty_replay(self):
        results = _loaded_results()
        results.loc[results.index[:5], "n"] = 0
        hit_rate, _ = build_curve_tables(results)

        self.assertEqual(int(hit_rate["zero_div"].sum()), 5)
        self.assertEqual(int((~hit_rate["zero_div"]).sum()), 25)


class TestFigures(unittest.TestCase):

    def test_figures_exist_and_are_non_empty(self):
        hit_rate, precision = build_curve_tables(_loaded_results())
        with tempfile.TemporaryDirectory() as tmp:
            figures_dir = Path(tmp) / "figures"
            written = write_curve_figures(hit_rate, precision, figures_dir)

            self.assertEqual(
                [path.name for path in written],
                [
                    "curve_hit_rate.png",
                    "curve_hit_rate.pdf",
                    "curve_precision.png",
                    "curve_precision.pdf",
                ],
            )
            for path in written:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_figures_regenerate_from_the_tables_alone(self):
        hit_rate, precision = build_curve_tables(_loaded_results())
        with tempfile.TemporaryDirectory() as tmp:
            # No results frame in scope for this call -- only the tidy tables.
            written = write_curve_figures(hit_rate.copy(), precision.copy(), Path(tmp) / "again")
            self.assertTrue(all(path.stat().st_size > 0 for path in written))

    def test_viability_reference_is_the_frozen_thirty_percent(self):
        self.assertAlmostEqual(HIT_RATE_VIABILITY, 0.30, places=12)


if __name__ == "__main__":
    unittest.main()
