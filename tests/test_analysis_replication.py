"""
Tests for the +/-5% replication check (LEV-8 / 4.6).

Frozen Success Criterion 3: headline precision and recall must replicate
within +/-5%. Under mock providers the harness is byte-deterministic, so a
self-comparison must match exactly; a perturbed reference must fail with an
itemized, auditable diff.

The verdict is also written as `replication.json` (2026-08-07): the criterion is
the one about verifiability, so its outcome has to be machine-readable rather
than only printed, and a partial re-run's verdict has to say which
configurations it covers.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from analysis_fixtures import additive_cells, build_results
from test_analysis_io import load_script

from levy.analysis.io import load_results
from levy.analysis.replication import (
    ABSOLUTE_FLOOR,
    RELATIVE_TOLERANCE,
    compare_results,
    format_report,
    report_to_dict,
    tolerance_rule,
)
from levy.dataset.io import load_dataset
from levy.experiment.config import ExperimentConfig
from levy.experiment.runner import run_sweep, write_results_csv, write_run_meta

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DATASET = REPO_ROOT / "data" / "ground_truth.csv"


def _small_grid():
    return [
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.70),
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="code", threshold=0.90),
    ]


def _write_reference_run(directory: Path) -> Path:
    """A real (mock-provider) harness run to replicate against."""
    directory.mkdir(parents=True, exist_ok=True)
    configs = _small_grid()
    pairs = load_dataset(FIXTURE_DATASET)
    results, identities = run_sweep(pairs, configs=configs, embedding_provider="mock", llm_latency_seconds=0)
    write_results_csv(results, directory / "results.csv")
    write_run_meta(
        results=results,
        configs=configs,
        dataset_path=FIXTURE_DATASET,
        embedding_provider="mock",
        model_identities=identities,
        elapsed_seconds=0.0,
        path=directory / "run_meta.json",
    )
    return directory / "results.csv"


class TestToleranceRule(unittest.TestCase):

    def setUp(self):
        self.reference = build_results(additive_cells(0.2))
        self.reference["precision"] = self.reference["precision"].astype(float)
        self.reference["recall"] = self.reference["recall"].astype(float)

    def test_identical_runs_pass(self):
        report = compare_results(self.reference, self.reference.copy())

        self.assertTrue(report.passed)
        self.assertEqual(len(report.table), 60)  # 30 configurations x 2 metrics
        self.assertTrue(report.table["within_tolerance"].all())
        self.assertEqual(report.missing_configs, [])
        self.assertEqual(report.extra_configs, [])

    def test_deviation_inside_five_percent_passes(self):
        candidate = self.reference.copy()
        reference_value = float(self.reference.loc[0, "precision"])
        candidate.loc[0, "precision"] = reference_value * (1 + RELATIVE_TOLERANCE * 0.9)

        self.assertTrue(compare_results(self.reference, candidate).passed)

    def test_deviation_beyond_five_percent_fails(self):
        candidate = self.reference.copy()
        reference_value = float(self.reference.loc[0, "precision"])
        candidate.loc[0, "precision"] = reference_value * (1 + RELATIVE_TOLERANCE * 3)

        report = compare_results(self.reference, candidate)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(report.violations.iloc[0]["metric"], "precision")

    def test_absolute_floor_governs_near_zero_references(self):
        reference = self.reference.copy()
        reference["precision"] = 0.0
        candidate = reference.copy()
        candidate.loc[0, "precision"] = ABSOLUTE_FLOOR / 2  # relative rule alone would fail this
        self.assertTrue(compare_results(reference, candidate).passed)

        candidate.loc[0, "precision"] = ABSOLUTE_FLOOR * 2
        report = compare_results(reference, candidate)
        self.assertFalse(report.passed)
        self.assertEqual(report.violations.iloc[0]["rel_diff"], float("inf"))

    def test_recall_is_compared_too(self):
        candidate = self.reference.copy()
        candidate["recall"] = candidate["recall"].astype(float) + 0.5

        report = compare_results(self.reference, candidate)
        self.assertFalse(report.passed)
        self.assertEqual(set(report.violations["metric"]), {"recall"})

    def test_a_missing_configuration_fails_the_check(self):
        report = compare_results(self.reference, self.reference.iloc[1:].copy())

        self.assertFalse(report.passed)
        self.assertEqual(len(report.missing_configs), 1)
        self.assertIn("MISSING", format_report(report))

    def test_an_extra_configuration_fails_the_check(self):
        candidate = pd.concat([self.reference, self.reference.iloc[[0]].assign(config_id="extra|faq|0.70")])

        report = compare_results(self.reference, candidate)
        self.assertFalse(report.passed)
        self.assertEqual(report.extra_configs, ["extra|faq|0.70"])
        self.assertIn("UNEXPECTED", format_report(report))

    def test_rule_is_stated_in_the_report(self):
        report = compare_results(self.reference, self.reference.copy())

        self.assertEqual(report.rule, tolerance_rule())
        text = format_report(report)
        self.assertIn("5%", text)
        self.assertIn(str(ABSOLUTE_FLOOR), text)
        self.assertIn("PASSED", text)

    def test_show_all_lists_every_comparison(self):
        report = compare_results(self.reference, self.reference.copy())
        text = format_report(report, show_all=True)

        self.assertIn("All comparisons:", text)
        self.assertEqual(text.count("OK"), 60)


class TestReplicationScript(unittest.TestCase):

    def test_self_comparison_of_a_mock_run_exits_zero(self):
        check_replication = load_script("check_replication")
        with tempfile.TemporaryDirectory() as tmp:
            reference = _write_reference_run(Path(tmp) / "reference")
            messages = []
            code = check_replication.main(
                ["--reference", str(reference), "--llm-latency-seconds", "0"],
                output_fn=messages.append,
            )

        self.assertEqual(code, 0)
        joined = "\n".join(messages)
        self.assertIn("replication PASSED", joined)
        self.assertIn("Compared 4 (configuration, metric) value(s).", joined)

    def test_perturbed_reference_exits_non_zero_with_an_itemized_diff(self):
        check_replication = load_script("check_replication")
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = _write_reference_run(Path(tmp) / "reference")
            perturbed = load_results(reference_path)
            target = str(perturbed.loc[0, "config_id"])
            perturbed.loc[0, "precision"] = float(perturbed.loc[0, "precision"]) + 0.5
            perturbed.to_csv(reference_path, index=False, lineterminator="\n")

            messages = []
            code = check_replication.main(
                ["--reference", str(reference_path), "--llm-latency-seconds", "0"],
                output_fn=messages.append,
            )

        self.assertEqual(code, 1)
        joined = "\n".join(messages)
        self.assertIn("replication FAILED", joined)
        self.assertIn(target, joined)
        self.assertIn("precision", joined)
        self.assertIn("reference=", joined)
        self.assertIn("candidate=", joined)
        self.assertIn("abs_diff=", joined)
        self.assertIn("tolerance=", joined)

    def test_keep_dir_preserves_the_candidate_run(self):
        check_replication = load_script("check_replication")
        with tempfile.TemporaryDirectory() as tmp:
            reference = _write_reference_run(Path(tmp) / "reference")
            keep = Path(tmp) / "candidate"
            code = check_replication.main(
                ["--reference", str(reference), "--llm-latency-seconds", "0", "--keep-dir", str(keep)],
                output_fn=lambda _message: None,
            )

            self.assertEqual(code, 0)
            self.assertTrue((keep / "results.csv").is_file())

    def test_missing_dataset_is_reported(self):
        check_replication = load_script("check_replication")
        with tempfile.TemporaryDirectory() as tmp:
            reference_dir = Path(tmp) / "reference"
            _write_reference_run(reference_dir)
            (reference_dir / "run_meta.json").unlink()  # no recorded dataset to fall back on

            code = check_replication.main(
                ["--reference", str(reference_dir / "results.csv")],
                output_fn=lambda _message: None,
            )

        self.assertEqual(code, 1)

    def test_unreadable_reference_is_reported(self):
        check_replication = load_script("check_replication")
        with tempfile.TemporaryDirectory() as tmp:
            code = check_replication.main(
                ["--reference", str(Path(tmp) / "absent.csv")],
                output_fn=lambda _message: None,
            )

        self.assertEqual(code, 1)


class TestVerdictFile(unittest.TestCase):
    """
    Success Criterion 3 is the criterion about verifiability, so its outcome has
    to be readable, not merely printed. Before this the verdict existed only on
    stdout and in an exit code, which is what left the poster hardcoding "PASSED".
    """

    def _verdict(self, tmp: Path, extra=()):
        check_replication = load_script("check_replication")
        reference = _write_reference_run(tmp / "reference")
        code = check_replication.main(
            ["--reference", str(reference), "--llm-latency-seconds", "0", *extra],
            output_fn=lambda _message: None,
        )
        return code, tmp / "reference" / "replication.json"

    def test_passing_run_writes_a_readable_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, path = self._verdict(Path(tmp))
            self.assertEqual(code, 0)
            verdict = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["n_comparisons"], 4)
        self.assertEqual(verdict["n_out_of_tolerance"], 0)
        self.assertEqual(verdict["relative_tolerance"], RELATIVE_TOLERANCE)
        self.assertEqual(verdict["absolute_floor"], ABSOLUTE_FLOOR)
        self.assertEqual(verdict["metrics"], ["precision", "recall"])
        self.assertIn("Criterion 3", verdict["criterion"])
        self.assertRegex(verdict["generated_at_utc"], r"\d{8}T\d{6}Z")
        self.assertEqual(len(verdict["comparisons"]), 4)

    def test_verdict_records_which_configurations_it_covers(self):
        """A partial re-run's verdict must not read as covering the whole grid."""
        with tempfile.TemporaryDirectory() as tmp:
            _, path = self._verdict(Path(tmp))
            verdict = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            verdict["config_ids"],
            sorted(config.config_id for config in _small_grid()),
        )
        self.assertEqual(len(verdict["config_ids"]), 2)  # not the frozen 30

    def test_failing_run_is_recorded_as_failed_not_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            check_replication = load_script("check_replication")
            reference_path = _write_reference_run(tmp / "reference")
            perturbed = load_results(reference_path)
            target = str(perturbed.loc[0, "config_id"])
            perturbed.loc[0, "precision"] = float(perturbed.loc[0, "precision"]) + 0.5
            perturbed.to_csv(reference_path, index=False, lineterminator="\n")

            code = check_replication.main(
                ["--reference", str(reference_path), "--llm-latency-seconds", "0"],
                output_fn=lambda _message: None,
            )
            verdict = json.loads((tmp / "reference" / "replication.json").read_text())

        self.assertEqual(code, 1)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["n_out_of_tolerance"], 1)
        offending = [c for c in verdict["comparisons"] if not c["within_tolerance"]]
        self.assertEqual(len(offending), 1)
        self.assertEqual(offending[0]["config_id"], target)
        self.assertEqual(offending[0]["metric"], "precision")

    def test_existing_verdict_is_backed_up_before_being_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, path = self._verdict(tmp)
            first = path.read_bytes()
            self._verdict(tmp)  # second run over the same reference

            backups = list((tmp / "reference" / "backups").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), first)

    def test_out_json_overrides_the_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            chosen = tmp / "elsewhere" / "verdict.json"
            code, default_path = self._verdict(tmp, extra=["--out-json", str(chosen)])

            self.assertEqual(code, 0)
            self.assertTrue(chosen.is_file())
            self.assertFalse(default_path.exists())

    def test_no_json_suppresses_the_file_but_not_the_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code, path = self._verdict(tmp, extra=["--no-json"])

            self.assertEqual(code, 0)
            self.assertFalse(path.exists())

    def test_report_to_dict_is_json_serialisable(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = _write_reference_run(Path(tmp) / "reference")
            frame = load_results(reference_path)
            report = compare_results(frame, frame)
            payload = report_to_dict(
                report,
                reference_path=reference_path,
                dataset_path=FIXTURE_DATASET,
                embedding_provider="mock",
                generated_at_utc="20260807T000000Z",
            )
            # No numpy scalars: json.dumps would raise on them.
            json.dumps(payload)

        self.assertTrue(payload["passed"])
        self.assertIsInstance(payload["n_comparisons"], int)
        self.assertIsInstance(payload["comparisons"][0]["reference"], float)
        self.assertIsInstance(payload["comparisons"][0]["within_tolerance"], bool)


if __name__ == "__main__":
    unittest.main()
