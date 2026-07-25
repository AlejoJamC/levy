"""
Tests for the harness-contract loader (LEV-8 / 4.1).

The analysis pipeline's only input is a harness output directory, so a
contract violation must fail loudly -- naming the offending column -- before
any statistics or output files exist.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from analysis_fixtures import additive_cells, build_results, write_harness_dir

from levy.analysis.io import (
    DECISIONS_REQUIRED_COLUMNS,
    HarnessContractError,
    RESULTS_REQUIRED_COLUMNS,
    load_decisions,
    load_harness_outputs,
    load_results,
    load_run_meta,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DATASET = REPO_ROOT / "data" / "ground_truth.csv"


def load_script(name: str):
    """Import a `scripts/*.py` CLI as a module so tests can call `main()`."""
    spec = importlib.util.spec_from_file_location(f"levy_scripts_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLoadResults(unittest.TestCase):

    def test_valid_harness_dir_loads_thirty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2, 0.1)))
            outputs = load_harness_outputs(harness)

        self.assertEqual(outputs.n_configurations, 30)
        self.assertEqual(list(outputs.results.columns), RESULTS_REQUIRED_COLUMNS)
        self.assertEqual(sorted(outputs.results["model"].unique()), ["all-MiniLM-L6-v2", "modernbert"])
        self.assertEqual(len(outputs.results["threshold"].unique()), 5)

    def test_zero_division_flags_become_real_booleans(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2), precision_zero_div_cells=[("modernbert", "chat")])
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            results = load_results(harness / "results.csv")

        flags = results["precision_zero_div"]
        self.assertTrue(all(isinstance(value, bool) for value in flags))
        self.assertEqual(int(flags.sum()), 5)  # the 5 thresholds of one cell

    def test_hand_written_csv_flags_are_coerced(self):
        """A hand-written fixture may spell the flags as text or as 0/1."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2))
            frame["precision_zero_div"] = frame["precision_zero_div"].astype(object)
            frame["recall_zero_div"] = frame["recall_zero_div"].astype(object)
            frame.loc[0, "precision_zero_div"] = "TRUE"
            frame.loc[1, "precision_zero_div"] = "false"
            frame.loc[2, "recall_zero_div"] = 1
            frame.loc[3, "recall_zero_div"] = ""
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            results = load_results(harness / "results.csv")

        self.assertEqual(results["precision_zero_div"].dtype, bool)
        self.assertTrue(results.loc[0, "precision_zero_div"])  # "TRUE"
        self.assertFalse(results.loc[1, "precision_zero_div"])  # "false"
        self.assertTrue(results.loc[2, "recall_zero_div"])  # 1
        self.assertFalse(results.loc[3, "recall_zero_div"])  # blank cell

    def test_missing_column_names_the_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2)).drop(columns=["fpr"])
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            with self.assertRaises(HarnessContractError) as ctx:
                load_harness_outputs(harness)

        self.assertIn("fpr", str(ctx.exception))
        self.assertIn("missing required column", str(ctx.exception))

    def test_non_numeric_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2))
            frame.loc[3, "fpr"] = "not-a-number"
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            with self.assertRaises(HarnessContractError) as ctx:
                load_results(harness / "results.csv")

        self.assertIn("fpr", str(ctx.exception))
        self.assertIn("non-numeric", str(ctx.exception))

    def test_unparseable_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2))
            frame["fpr_zero_div"] = frame["fpr_zero_div"].astype(object)
            frame.loc[0, "fpr_zero_div"] = "maybe"
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            with self.assertRaises(HarnessContractError) as ctx:
                load_results(harness / "results.csv")

        self.assertIn("fpr_zero_div", str(ctx.exception))

    def test_empty_results_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2)).iloc[0:0]
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            with self.assertRaises(HarnessContractError) as ctx:
                load_results(harness / "results.csv")

        self.assertIn("no configuration rows", str(ctx.exception))

    def test_missing_file_and_directory_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(HarnessContractError):
                load_harness_outputs(Path(tmp) / "nope")
            with self.assertRaises(HarnessContractError):
                load_results(Path(tmp) / "results.csv")
            with self.assertRaises(HarnessContractError):
                load_decisions(Path(tmp) / "decisions.csv")
            with self.assertRaises(HarnessContractError):
                load_run_meta(Path(tmp) / "run_meta.json")


class TestOptionalArtifacts(unittest.TestCase):

    def test_decisions_and_meta_are_loaded_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(
                Path(tmp) / "harness",
                build_results(additive_cells(0.2)),
                dataset_path=FIXTURE_DATASET,
            )
            header = ",".join(DECISIONS_REQUIRED_COLUMNS)
            (harness / "decisions.csv").write_text(
                f"{header}\nc|faq|0.70,m,faq,0.70,p1,hit,semantic_cache,0.9,1,TP\n",
                encoding="utf-8",
            )
            outputs = load_harness_outputs(harness)

        self.assertIsNotNone(outputs.decisions)
        self.assertEqual(len(outputs.decisions), 1)
        self.assertEqual(outputs.run_meta["dataset_path"], str(FIXTURE_DATASET))

    def test_absent_optional_artifacts_are_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2)))
            outputs = load_harness_outputs(harness)

        self.assertIsNone(outputs.decisions)
        self.assertEqual(outputs.run_meta, {})

    def test_decisions_missing_column_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2)))
            (harness / "decisions.csv").write_text("config_id,model\nx,y\n", encoding="utf-8")
            with self.assertRaises(HarnessContractError) as ctx:
                load_harness_outputs(harness)

        self.assertIn("outcome", str(ctx.exception))

    def test_invalid_run_meta_json_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2)))
            (harness / "run_meta.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(HarnessContractError) as ctx:
                load_harness_outputs(harness)

        self.assertIn("not valid JSON", str(ctx.exception))

    def test_non_object_run_meta_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = write_harness_dir(Path(tmp) / "harness", build_results(additive_cells(0.2)))
            (harness / "run_meta.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(HarnessContractError) as ctx:
                load_harness_outputs(harness)

        self.assertIn("expected a JSON object", str(ctx.exception))


class TestContractViolationLeavesNoOutput(unittest.TestCase):

    def test_cli_exits_non_zero_and_writes_nothing(self):
        run_analysis = load_script("run_analysis")
        with tempfile.TemporaryDirectory() as tmp:
            frame = build_results(additive_cells(0.2)).drop(columns=["hit_rate"])
            harness = write_harness_dir(Path(tmp) / "harness", frame)
            out_dir = Path(tmp) / "analysis"

            messages = []
            code = run_analysis.main(
                ["--results-dir", str(harness), "--out-dir", str(out_dir)],
                output_fn=messages.append,
            )

            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists(), "no partial bundle may be written on a contract violation")
            self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
