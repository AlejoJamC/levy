"""
Tests for merging partial harness runs into one complete result set
(`levy/experiment/merge.py` + `scripts/merge_results.py`).

The reference fixture is a real sweep over the committed 15-pair synthetic
dataset with the mock providers, split by workload into separate output
directories. `run_experiment` filters pairs to its configuration's workload, so
those per-workload directories hold exactly the rows
`run_experiments.py --workloads <w>` would have written — which is what makes
the byte-identity assertion against a single full run meaningful.

All offline (mock LLM at zero latency, mock embeddings).
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from levy.dataset.io import load_dataset
from levy.experiment.config import WORKLOADS, full_grid
from levy.experiment.merge import (
    ResultsMergeError,
    check_grid_coverage,
    configs_for_ids,
    load_input,
    merge_decisions,
    merge_results,
    merge_run_meta,
    sort_by_grid,
    sort_decisions_by_grid,
    write_rows,
)
from levy.experiment.runner import (
    RESULTS_FIELDNAMES,
    run_sweep,
    write_decisions_csv,
    write_results_csv,
    write_run_meta,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DATASET = REPO_ROOT / "data" / "ground_truth.csv"

_SWEEP_CACHE = {}


def _sweep():
    """One full-grid sweep, computed once and shared across the tests."""
    if "results" not in _SWEEP_CACHE:
        pairs = load_dataset(FIXTURE_DATASET)
        results, identities = run_sweep(pairs, llm_latency_seconds=0.0)
        _SWEEP_CACHE["results"] = results
        _SWEEP_CACHE["identities"] = identities
    return _SWEEP_CACHE["results"], _SWEEP_CACHE["identities"]


def _write_run(directory: Path, results, identities, decisions=True, meta=True):
    directory.mkdir(parents=True, exist_ok=True)
    write_results_csv(results, directory / "results.csv")
    if decisions:
        write_decisions_csv(results, directory / "decisions.csv")
    if meta:
        write_run_meta(
            results=results,
            configs=[r.config for r in results],
            dataset_path=FIXTURE_DATASET,
            embedding_provider="mock",
            model_identities=identities,
            elapsed_seconds=0.0,
            path=directory / "run_meta.json",
        )
    return directory


def _per_workload_dirs(root: Path):
    """Three harness directories, one per workload, plus the full reference."""
    results, identities = _sweep()
    dirs = []
    for workload in WORKLOADS:
        subset = [r for r in results if r.config.workload == workload]
        dirs.append(_write_run(root / f"run-{workload}", subset, identities))
    full = _write_run(root / "run-full", results, identities)
    return dirs, full


def _run_cli(args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "merge_results.py"), *[str(a) for a in args]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

class TestMergeLogic(unittest.TestCase):

    def test_partial_runs_merge_to_the_full_grid(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            inputs = [load_input(d) for d in dirs]
            merged = merge_results(inputs)
            self.assertEqual(len(merged), 30)
            check_grid_coverage(merged)  # must not raise

    def test_duplicate_config_id_across_runs_is_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            inputs = [load_input(d) for d in dirs]
            with self.assertRaises(ResultsMergeError) as ctx:
                merge_results(inputs + [load_input(dirs[0])])
            message = str(ctx.exception)
            self.assertIn("duplicate configuration", message)
            self.assertIn(str(dirs[0]), message)
            self.assertIn("Nothing written", message)

    def test_duplicate_config_id_within_one_file_is_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            item = load_input(dirs[0])
            item.results.append(dict(item.results[0]))
            with self.assertRaises(ResultsMergeError) as ctx:
                merge_results([item])
            self.assertIn("twice in", str(ctx.exception))

    def test_incomplete_grid_is_rejected_and_names_what_is_missing(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            merged = merge_results([load_input(d) for d in dirs[:2]])
            with self.assertRaises(ResultsMergeError) as ctx:
                check_grid_coverage(merged)
            message = str(ctx.exception)
            self.assertIn("10 of 30 configuration(s) missing", message)
            self.assertIn(WORKLOADS[2], message)

    def test_configuration_outside_the_frozen_grid_is_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            merged = merge_results([load_input(d) for d in dirs])
            merged.append({**merged[0], "config_id": "some-other-model|faq|0.70"})
            with self.assertRaises(ResultsMergeError) as ctx:
                check_grid_coverage(merged)
            self.assertIn("outside the frozen grid", str(ctx.exception))

    def test_decisions_merge_and_a_missing_one_is_an_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            inputs = [load_input(d) for d in dirs]
            merged = merge_decisions(inputs)
            self.assertEqual(len(merged), sum(len(i.decisions) for i in inputs))

            (dirs[0] / "decisions.csv").unlink()
            with self.assertRaises(ResultsMergeError) as ctx:
                merge_decisions([load_input(d) for d in dirs])
            self.assertIn("decisions.csv", str(ctx.exception))

    def test_run_with_neither_decisions_nor_meta_still_loads(self):
        """A hand-crafted results-only directory is a legitimate input."""
        with TemporaryDirectory() as tmp:
            results, identities = _sweep()
            directory = _write_run(
                Path(tmp) / "run", results, identities, decisions=False, meta=False
            )
            item = load_input(directory)
            self.assertIsNone(merge_decisions([item]))
            self.assertEqual(item.run_meta, {})
            # Nothing to agree on, so the merged sidecar simply omits those keys.
            meta = merge_run_meta([item], "20260806T000000Z")
            self.assertNotIn("dataset_path", meta)
            self.assertEqual(meta["n_configurations"], 30)

    def test_runs_from_different_datasets_cannot_be_merged(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            inputs = [load_input(d) for d in dirs]
            inputs[1].run_meta["dataset_path"] = "data/some_other_dataset.csv"
            with self.assertRaises(ResultsMergeError) as ctx:
                merge_run_meta(inputs, "20260806T000000Z")
            self.assertIn("dataset_path", str(ctx.exception))

    def test_merged_meta_keeps_what_downstream_tools_read(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            inputs = [load_input(d) for d in dirs]
            meta = merge_run_meta(inputs, "20260806T000000Z")
        self.assertEqual(meta["dataset_path"], str(FIXTURE_DATASET))
        self.assertEqual(meta["embedding_provider"], "mock")
        self.assertEqual(meta["n_configurations"], 30)
        self.assertEqual(len(meta["grid"]), 30)
        self.assertEqual(len(meta["merged_from"]), 3)
        self.assertEqual(
            sum(item["n_configurations"] for item in meta["merged_from"]), 30
        )

    def test_missing_column_is_reported(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            directory.mkdir()
            (directory / "results.csv").write_text("config_id,model\nx,y\n", encoding="utf-8")
            with self.assertRaises(ResultsMergeError) as ctx:
                load_input(directory)
            self.assertIn("missing required column", str(ctx.exception))

    def test_empty_input_list_is_an_error(self):
        with self.assertRaises(ResultsMergeError):
            merge_results([])

    def test_absent_directory_and_absent_results_file_are_named(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ResultsMergeError) as ctx:
                load_input(Path(tmp) / "nope")
            self.assertIn("not found", str(ctx.exception))

            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(ResultsMergeError) as ctx:
                load_input(empty)
            self.assertIn("results.csv", str(ctx.exception))

    def test_header_only_results_file_is_an_error(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            directory.mkdir()
            (directory / "results.csv").write_text(
                ",".join(RESULTS_FIELDNAMES) + "\n", encoding="utf-8"
            )
            with self.assertRaises(ResultsMergeError) as ctx:
                load_input(directory)
            self.assertIn("contains no rows", str(ctx.exception))

    def test_malformed_run_meta_is_an_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, identities = _sweep()
            directory = _write_run(root / "run", results, identities)

            (directory / "run_meta.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(ResultsMergeError) as ctx:
                load_input(directory)
            self.assertIn("not valid JSON", str(ctx.exception))

            (directory / "run_meta.json").write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ResultsMergeError) as ctx:
                load_input(directory)
            self.assertIn("expected a JSON object", str(ctx.exception))

    def test_duplicate_decision_row_is_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            first, second = load_input(dirs[0]), load_input(dirs[1])
            second.decisions.append(dict(first.decisions[0]))
            with self.assertRaises(ResultsMergeError) as ctx:
                merge_decisions([first, second])
            self.assertIn("duplicate decision row", str(ctx.exception))

    def test_sorting_puts_rows_in_grid_order(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            merged = merge_results([load_input(d) for d in reversed(dirs)])
            ordered = sort_by_grid(merged)
        self.assertEqual(
            [row["config_id"] for row in ordered],
            [config.config_id for config in full_grid()],
        )

    def test_sorting_decisions_groups_by_grid_and_keeps_pair_order(self):
        with TemporaryDirectory() as tmp:
            dirs, _ = _per_workload_dirs(Path(tmp))
            merged = merge_decisions([load_input(d) for d in reversed(dirs)])
            ordered = sort_decisions_by_grid(merged)

        seen_order = []
        for row in ordered:
            if not seen_order or seen_order[-1] != row["config_id"]:
                seen_order.append(row["config_id"])
        self.assertEqual(seen_order, [config.config_id for config in full_grid()])
        # Within one configuration the pair sequence is untouched.
        first_id = seen_order[0]
        self.assertEqual(
            [row["pair_id"] for row in ordered if row["config_id"] == first_id],
            [row["pair_id"] for row in merged if row["config_id"] == first_id],
        )

    def test_rows_outside_the_grid_sort_last_rather_than_vanish(self):
        rows = [
            {"config_id": "unknown|faq|0.70"},
            {"config_id": full_grid()[0].config_id},
        ]
        self.assertEqual(
            [row["config_id"] for row in sort_by_grid(rows)],
            [full_grid()[0].config_id, "unknown|faq|0.70"],
        )

    def test_write_rows_emits_the_canonical_columns_only(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "results.csv"
            write_rows(
                path,
                RESULTS_FIELDNAMES,
                [{"config_id": "a|faq|0.70", "model": "a", "surplus": "dropped"}],
            )
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(","), RESULTS_FIELDNAMES)
        self.assertNotIn("dropped", lines[1])
        self.assertTrue(lines[1].startswith("a|faq|0.70,a,"))

    def test_configs_for_ids_returns_grid_cells_in_order(self):
        wanted = [full_grid()[5].config_id, full_grid()[1].config_id]
        self.assertEqual(
            [config.config_id for config in configs_for_ids(wanted)],
            [full_grid()[1].config_id, full_grid()[5].config_id],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestMergeCli(unittest.TestCase):

    def test_merged_output_is_byte_identical_to_a_single_full_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, full = _per_workload_dirs(root)
            out = root / "merged"
            result = _run_cli([*dirs, "--out-dir", out])
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertEqual(
                (out / "results.csv").read_bytes(), (full / "results.csv").read_bytes()
            )
            self.assertEqual(
                (out / "decisions.csv").read_bytes(), (full / "decisions.csv").read_bytes()
            )

    def test_in_place_merge_backs_up_the_existing_result_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            base = dirs[0]
            before = (base / "results.csv").read_bytes()

            result = _run_cli([*dirs, "--out-dir", base])
            self.assertEqual(result.returncode, 0, msg=result.stdout)

            backups = sorted((base / "backups").iterdir())
            self.assertEqual(len(backups), 3, msg=[b.name for b in backups])
            results_backup = [b for b in backups if b.name.startswith("results.")]
            self.assertEqual(results_backup[0].read_bytes(), before)
            # And the merged set really did land in place.
            self.assertEqual(len((base / "results.csv").read_text().splitlines()), 31)

    def test_duplicate_is_rejected_by_the_cli_without_writing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            out = root / "merged"
            result = _run_cli([*dirs, dirs[0], "--out-dir", out])
            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicate configuration", result.stdout)
            self.assertFalse(out.exists())

    def test_incomplete_grid_is_rejected_by_the_cli_without_writing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            out = root / "merged"
            result = _run_cli([dirs[0], "--out-dir", out])
            self.assertEqual(result.returncode, 1)
            self.assertIn("does not cover the frozen grid", result.stdout)
            self.assertFalse(out.exists())

    def test_allow_partial_writes_a_diagnostic_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            out = root / "merged"
            result = _run_cli([dirs[0], "--out-dir", out, "--allow-partial"])
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertEqual(len((out / "results.csv").read_text().splitlines()), 11)
            self.assertIn("not for the analysis step", result.stdout)

    def test_nothing_written_when_the_backup_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            base = dirs[0]
            before = (base / "results.csv").read_bytes()
            blocker = root / "blocked"
            blocker.write_text("a file where the backup dir should be\n", encoding="utf-8")

            result = _run_cli([*dirs, "--out-dir", base, "--backup-dir", blocker])
            self.assertEqual(result.returncode, 1)
            self.assertIn("nothing written", result.stdout)
            self.assertEqual((base / "results.csv").read_bytes(), before)

    def test_merged_set_feeds_the_analysis_pipeline(self):
        """The point of merging: the 30-row set is a valid analysis input."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            out = root / "merged"
            self.assertEqual(_run_cli([*dirs, "--out-dir", out]).returncode, 0)

            analysis = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "run_analysis.py"),
                 "--results-dir", str(out), "--out-dir", str(root / "analysis"),
                 "--dataset", str(FIXTURE_DATASET)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(analysis.returncode, 0, msg=analysis.stdout + analysis.stderr)
            self.assertTrue((root / "analysis" / "anova.csv").is_file())

    def test_merged_meta_records_every_source_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            out = root / "merged"
            self.assertEqual(_run_cli([*dirs, "--out-dir", out]).returncode, 0)
            meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["generated_by"], "scripts/merge_results.py")
        self.assertEqual(
            sorted(item["directory"] for item in meta["merged_from"]),
            sorted(str(d) for d in dirs),
        )
        self.assertRegex(meta["generated_at_utc"], r"\d{8}T\d{6}Z")


class TestResultsColumnContract(unittest.TestCase):

    def test_merge_writes_the_writers_column_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs, _ = _per_workload_dirs(root)
            out = root / "merged"
            self.assertEqual(_run_cli([*dirs, "--out-dir", out]).returncode, 0)
            header = (out / "results.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","), RESULTS_FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
