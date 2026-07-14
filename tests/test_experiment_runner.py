"""
Tests for the sweep runner and output writers (LEV-4 / 4.4, 4.5).

Uses small grid subsets against the committed synthetic fixture (not the
full 30-configuration grid); latency is injected at zero so the suite
stays fast regardless of grid size.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from levy.dataset.io import load_dataset
from levy.experiment.config import ExperimentConfig
from levy.experiment.runner import run_sweep, write_decisions_csv, write_results_csv, write_run_meta

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "ground_truth.csv"


def _small_grid():
    return [
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.70),
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.90),
    ]


def _contract_grid():
    return [
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="faq", threshold=0.85),
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="code", threshold=0.85),
        ExperimentConfig(model="all-MiniLM-L6-v2", workload="chat", threshold=0.85),
    ]


class TestDeterminism(unittest.TestCase):

    def test_repeated_sweep_is_byte_identical(self):
        pairs = load_dataset(FIXTURE)
        configs = _small_grid()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run1, run2 = tmp_path / "run1", tmp_path / "run2"

            results1, identities1 = run_sweep(pairs, configs=configs, embedding_provider="mock", llm_latency_seconds=0)
            write_results_csv(results1, run1 / "results.csv")
            write_decisions_csv(results1, run1 / "decisions.csv")

            results2, identities2 = run_sweep(pairs, configs=configs, embedding_provider="mock", llm_latency_seconds=0)
            write_results_csv(results2, run2 / "results.csv")
            write_decisions_csv(results2, run2 / "decisions.csv")

            self.assertEqual(
                (run1 / "results.csv").read_bytes(),
                (run2 / "results.csv").read_bytes(),
            )
            self.assertEqual(
                (run1 / "decisions.csv").read_bytes(),
                (run2 / "decisions.csv").read_bytes(),
            )

    def test_results_and_decisions_carry_no_timestamps_or_latency(self):
        pairs = load_dataset(FIXTURE)
        configs = _small_grid()
        results, _ = run_sweep(pairs, configs=configs, embedding_provider="mock", llm_latency_seconds=0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_results_csv(results, tmp_path / "results.csv")
            write_decisions_csv(results, tmp_path / "decisions.csv")

            for name in ("results.csv", "decisions.csv"):
                header = (tmp_path / name).read_text(encoding="utf-8").splitlines()[0].lower()
                for forbidden in ("timestamp", "latency", "elapsed", "duration"):
                    self.assertNotIn(forbidden, header)


class TestDefaultGrid(unittest.TestCase):

    def test_run_sweep_without_configs_uses_full_frozen_grid(self):
        pairs = load_dataset(FIXTURE)
        results, _ = run_sweep(pairs, embedding_provider="mock", llm_latency_seconds=0)
        self.assertEqual(len(results), 30)  # 2 models x 3 workloads x 5 thresholds


class TestOutputContract(unittest.TestCase):

    def setUp(self):
        self.pairs = load_dataset(FIXTURE)
        self.configs = _contract_grid()
        self.results, self.model_identities = run_sweep(self.pairs, configs=self.configs, embedding_provider="mock", llm_latency_seconds=0)

    def test_results_csv_has_one_row_per_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            write_results_csv(self.results, path)
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

        self.assertEqual(len(rows), len(self.configs))
        required = {
            "config_id", "model", "workload", "threshold", "n", "tp", "fp", "tn", "fn",
            "precision", "recall", "f0_5", "fpr", "hit_rate",
            "precision_zero_div", "recall_zero_div", "fpr_zero_div",
        }
        self.assertTrue(required.issubset(set(rows[0].keys())))
        for row in rows:
            for col in ("precision", "recall", "f0_5", "fpr", "hit_rate", "n", "tp", "fp", "tn", "fn"):
                self.assertNotEqual(row[col], "")

    def test_decision_log_has_one_row_per_pair_per_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.csv"
            write_decisions_csv(self.results, path)
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

        expected_rows = sum(r.n for r in self.results)
        self.assertEqual(len(rows), expected_rows)

        pair_ids_by_workload = {}
        for pair in self.pairs:
            pair_ids_by_workload.setdefault(pair.workload, set()).add(pair.pair_id)

        for row in rows:
            self.assertIn(row["pair_id"], pair_ids_by_workload[row["workload"]])

    def test_run_meta_sidecar_is_separate_from_results_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_run_meta(
                results=self.results,
                configs=self.configs,
                dataset_path=FIXTURE,
                embedding_provider="mock",
                model_identities=self.model_identities,
                elapsed_seconds=1.23,
                path=tmp_path / "run_meta.json",
            )
            with (tmp_path / "run_meta.json").open(encoding="utf-8") as fh:
                meta = json.load(fh)

        self.assertEqual(meta["n_configurations"], len(self.configs))
        self.assertIn("latency", meta)
        self.assertIn("total_elapsed_seconds", meta["latency"])
        self.assertEqual(meta["embedding_provider"], "mock")


if __name__ == "__main__":
    unittest.main()
