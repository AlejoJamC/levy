"""
Tests for the D3 bundle assembly and the analysis CLI (LEV-8 / 4.5, 4.7).

Covers the one-command contract (every table, figure, and sidecar from a
single invocation), the kappa section's provenance labelling (sourced from
the LEV-3 dataset tooling, never recomputed here), and byte-identical tables
across re-runs on identical input.
"""

import json
import tempfile
import unittest
from pathlib import Path

from analysis_fixtures import additive_cells, build_results, write_harness_dir
from test_analysis_io import load_script

from levy.analysis.report import (
    ANOVA_FILENAME,
    CURVES_HIT_RATE_FILENAME,
    CURVES_PRECISION_FILENAME,
    DETERMINISTIC_TABLES,
    KAPPA_FILENAME,
    KAPPA_THRESHOLD,
    META_FILENAME,
    TUKEY_FILENAME,
    TUKEY_STATUS_FILENAME,
    build_analysis_bundle,
    build_kappa_section,
)
from levy.dataset.io import load_dataset, save_csv
from levy.dataset.kappa import kappa_report
from levy.dataset.schema import QueryPair

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DATASET = REPO_ROOT / "data" / "ground_truth.csv"


class TestBundleContents(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.harness = write_harness_dir(
            tmp / "harness",
            build_results(additive_cells(model_effect=0.20, workload_effect=0.10)),
            dataset_path=FIXTURE_DATASET,
        )
        self.out_dir = tmp / "analysis"
        self.bundle = build_analysis_bundle(self.harness, self.out_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_every_table_figure_and_sidecar_is_written(self):
        for name in DETERMINISTIC_TABLES + (KAPPA_FILENAME, META_FILENAME):
            path = self.out_dir / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)

        figures = sorted(p.name for p in (self.out_dir / "figures").iterdir())
        self.assertEqual(
            figures,
            ["curve_hit_rate.pdf", "curve_hit_rate.png", "curve_precision.pdf", "curve_precision.png"],
        )

    def test_anova_table_carries_the_three_hypotheses(self):
        text = (self.out_dir / ANOVA_FILENAME).read_text(encoding="utf-8")
        for hypothesis in ("H0_1", "H0_2", "H0_3"):
            self.assertIn(hypothesis, text)
        self.assertIn("Residual", text)
        self.assertEqual(self.bundle.anova.decision("H0_1"), "reject")

    def test_tukey_status_states_ran_or_skipped_per_effect(self):
        text = (self.out_dir / TUKEY_STATUS_FILENAME).read_text(encoding="utf-8")
        for effect in ("model", "workload", "model:workload", "__overall__"):
            self.assertIn(effect, text)
        self.assertIn("skipped", text)
        self.assertIn("ran", text)
        self.assertGreater(len((self.out_dir / TUKEY_FILENAME).read_text(encoding="utf-8").splitlines()), 1)

    def test_curve_tables_hold_thirty_rows_each(self):
        for name in (CURVES_HIT_RATE_FILENAME, CURVES_PRECISION_FILENAME):
            lines = (self.out_dir / name).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 31, name)  # header + 30 configurations

    def test_no_table_contains_a_timestamp_or_version(self):
        meta = json.loads((self.out_dir / META_FILENAME).read_text(encoding="utf-8"))
        stamp = meta["generated_at"]
        version = meta["library_versions"]["statsmodels"]
        for name in DETERMINISTIC_TABLES:
            text = (self.out_dir / name).read_text(encoding="utf-8")
            self.assertNotIn(stamp, text, name)
            self.assertNotIn(f"statsmodels {version}", text, name)

    def test_metadata_records_inputs_diagnostics_and_versions(self):
        meta = self.bundle.meta
        self.assertEqual(meta["inputs"]["n_configurations"], 30)
        self.assertEqual(meta["inputs"]["harness_dir"], str(self.harness))
        self.assertEqual(meta["inputs"]["dataset_path_for_kappa"], str(FIXTURE_DATASET))
        self.assertEqual(meta["anova"]["formula"], "fpr ~ C(model) * C(workload)")
        self.assertEqual(meta["anova"]["decisions"]["H0_1"], "reject")
        self.assertFalse(meta["anova"]["degenerate_response"])
        self.assertIn("design", meta["anova"]["diagnostics"])
        for library in ("pandas", "statsmodels", "matplotlib", "scipy", "python"):
            self.assertIn(library, meta["library_versions"])


class TestKappaSection(unittest.TestCase):

    def test_kappa_is_sourced_from_the_dataset_tooling(self):
        section = build_kappa_section(FIXTURE_DATASET)
        expected = kappa_report(load_dataset(FIXTURE_DATASET))

        self.assertEqual(section["status"], "computed")
        self.assertEqual(section["source"], "levy.dataset.kappa.kappa_report")
        self.assertEqual(section["overall"]["kappa"], expected.overall.kappa)
        self.assertEqual(section["overall"]["confusion"], expected.overall.confusion)
        self.assertEqual(
            sorted(section["per_workload"]), sorted(expected.per_workload)
        )
        self.assertEqual(section["threshold"], KAPPA_THRESHOLD)

    def test_fixture_data_is_labelled_fixture_only(self):
        section = build_kappa_section(FIXTURE_DATASET)

        self.assertTrue(section["provenance"]["fixture_only"])
        self.assertIn("synthetic-fixture", section["provenance"]["source_corpora"])
        self.assertIn("FIXTURE ONLY", section["provenance"]["note"])

    def test_fully_annotated_dataset_gets_a_pass_fail_verdict(self):
        pairs = [
            QueryPair(
                pair_id=f"faq-{index:04d}",
                workload="faq",
                source_corpus="quora-qqp",
                source_pair_id=str(index),
                query_1=f"question {index} a",
                query_2=f"question {index} b",
                original_label=index % 2,
                author_label=index % 2,  # perfect agreement -> kappa 1.0
            )
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "annotated.csv"
            save_csv(pairs, dataset)
            section = build_kappa_section(dataset)

        self.assertTrue(section["fully_annotated"])
        self.assertEqual(section["overall"]["kappa"], 1.0)
        self.assertIs(section["passes_threshold"], True)
        self.assertFalse(section["provenance"]["fixture_only"])
        self.assertIn("released dataset", section["provenance"]["note"])

    def test_partially_annotated_dataset_withholds_the_verdict(self):
        pairs = [
            QueryPair(
                pair_id=f"faq-{index:04d}",
                workload="faq",
                source_corpus="quora-qqp",
                source_pair_id=str(index),
                query_1=f"question {index} a",
                query_2=f"question {index} b",
                original_label=index % 2,
                author_label=(index % 2) if index < 4 else None,
            )
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "partial.csv"
            save_csv(pairs, dataset)
            section = build_kappa_section(dataset)

        self.assertFalse(section["fully_annotated"])
        self.assertIsNone(section["passes_threshold"])
        self.assertEqual(section["overall"]["n_excluded_unannotated"], 4)

    def test_missing_dataset_is_reported_not_fatal(self):
        self.assertEqual(build_kappa_section(None)["status"], "unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            section = build_kappa_section(Path(tmp) / "absent.csv")
        self.assertEqual(section["status"], "unavailable")
        self.assertIn("not found", section["reason"])

    def test_bundle_falls_back_to_the_run_meta_dataset_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(
                Path(tmp) / "harness", build_results(additive_cells(0.2)), dataset_path=FIXTURE_DATASET
            )
            bundle = build_analysis_bundle(harness, Path(tmp) / "out")

        self.assertEqual(bundle.kappa["status"], "computed")
        self.assertEqual(bundle.kappa["dataset_path"], str(FIXTURE_DATASET))

    def test_bundle_without_any_dataset_still_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2)))
            out_dir = Path(tmp) / "out"
            bundle = build_analysis_bundle(harness, out_dir)

            self.assertEqual(bundle.kappa["status"], "unavailable")
            self.assertTrue((out_dir / KAPPA_FILENAME).is_file())


class TestDeterminism(unittest.TestCase):

    def test_tables_are_byte_identical_across_runs(self):
        run_analysis = load_script("run_analysis")
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(
                Path(tmp) / "harness",
                build_results(additive_cells(model_effect=0.20, workload_effect=0.10)),
                dataset_path=FIXTURE_DATASET,
            )
            first, second = Path(tmp) / "run1", Path(tmp) / "run2"
            for out_dir in (first, second):
                code = run_analysis.main(
                    ["--results-dir", str(harness), "--out-dir", str(out_dir)],
                    output_fn=lambda _message: None,
                )
                self.assertEqual(code, 0)

            for name in DETERMINISTIC_TABLES:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    f"{name} differs between identical runs",
                )
            self.assertEqual(
                (first / KAPPA_FILENAME).read_bytes(), (second / KAPPA_FILENAME).read_bytes()
            )


class TestAnalysisCli(unittest.TestCase):

    def test_one_invocation_emits_the_whole_bundle(self):
        run_analysis = load_script("run_analysis")
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(
                Path(tmp) / "harness",
                build_results(additive_cells(model_effect=0.20)),
                dataset_path=FIXTURE_DATASET,
            )
            out_dir = Path(tmp) / "analysis"
            messages = []
            code = run_analysis.main(
                ["--results-dir", str(harness), "--out-dir", str(out_dir), "--dataset", str(FIXTURE_DATASET)],
                output_fn=messages.append,
            )

            self.assertEqual(code, 0)
            for name in DETERMINISTIC_TABLES + (KAPPA_FILENAME, META_FILENAME):
                self.assertTrue((out_dir / name).is_file(), name)

        joined = "\n".join(messages)
        self.assertIn("H0_1", joined)
        self.assertIn("Tukey HSD", joined)
        self.assertIn("FIXTURE ONLY", joined)

    def test_cli_reports_an_unavailable_kappa_section(self):
        run_analysis = load_script("run_analysis")
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2)))
            messages = []
            code = run_analysis.main(
                ["--results-dir", str(harness), "--out-dir", str(Path(tmp) / "out")],
                output_fn=messages.append,
            )

        self.assertEqual(code, 0)
        self.assertIn("kappa: unavailable", "\n".join(messages))

    def test_cli_rejects_an_unusable_design(self):
        run_analysis = load_script("run_analysis")
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2))
            frame = frame.loc[frame["model"] == "modernbert"]
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            out_dir = Path(tmp) / "out"
            code = run_analysis.main(
                ["--results-dir", str(harness), "--out-dir", str(out_dir)],
                output_fn=lambda _message: None,
            )

            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
