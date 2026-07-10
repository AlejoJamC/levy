"""
Tests for the ground-truth dataset platform tooling (LEV-3): schema
validation, CSV/JSON round-trip, seeded stratified sampling, blind
annotation flow, Cohen's kappa, and CLI smoke tests against the synthetic
fixtures in data/. Everything here runs fully offline.
"""

import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from levy.dataset.annotation import BlindAnnotationSession
from levy.dataset.io import (
    DatasetValidationError,
    load_csv,
    load_dataset,
    load_json,
    save_csv,
    save_dataset,
    save_json,
)
from levy.dataset.kappa import cohen_kappa, kappa_report
from levy.dataset.sampling import (
    CorpusSourceError,
    MockCorpusSource,
    sample_dataset,
    sample_workload,
)
from levy.dataset.schema import (
    QueryPair,
    QueryPairValidationError,
    WORKLOAD_CHAT,
    WORKLOAD_CODE,
    WORKLOAD_FAQ,
    WORKLOADS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CSV = REPO_ROOT / "data" / "ground_truth.csv"
FIXTURE_JSON = REPO_ROOT / "data" / "ground_truth.json"


def _make_pair(**overrides) -> QueryPair:
    defaults = dict(
        pair_id="faq-0001",
        workload=WORKLOAD_FAQ,
        source_corpus="test-corpus",
        source_pair_id="src-1",
        query_1="What is semantic caching?",
        query_2="Can you explain semantic caching?",
        original_label=1,
        author_label=None,
    )
    defaults.update(overrides)
    return QueryPair(**defaults)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestQueryPairSchema(unittest.TestCase):

    def test_valid_pair_constructs(self):
        pair = _make_pair()
        self.assertEqual(pair.workload, WORKLOAD_FAQ)

    def test_invalid_workload_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(workload="not-a-workload")

    def test_invalid_original_label_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(original_label=2)

    def test_invalid_author_label_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(author_label=5)

    def test_empty_query_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(query_1="   ")

    def test_empty_pair_id_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(pair_id="")

    def test_ground_truth_label_prefers_author_label(self):
        pair = _make_pair(original_label=1, author_label=0)
        self.assertEqual(pair.ground_truth_label(), 0)

    def test_ground_truth_label_falls_back_to_original(self):
        pair = _make_pair(original_label=1, author_label=None)
        self.assertEqual(pair.ground_truth_label(), 1)

    def test_to_dict_from_dict_roundtrip(self):
        pair = _make_pair(metadata={"k": "v"})
        restored = QueryPair.from_dict(pair.to_dict())
        self.assertEqual(pair, restored)

    def test_workloads_tuple_contents(self):
        self.assertEqual(set(WORKLOADS), {"faq", "code", "chat"})


# ---------------------------------------------------------------------------
# CSV / JSON round-trip
# ---------------------------------------------------------------------------

class TestCsvJsonRoundTrip(unittest.TestCase):

    def _sample_pairs(self):
        return [
            _make_pair(pair_id="faq-0001"),
            _make_pair(pair_id="faq-0002", original_label=0, author_label=0, workload=WORKLOAD_FAQ),
            _make_pair(pair_id="code-0001", workload=WORKLOAD_CODE, author_label=1, metadata={"note": "x"}),
            _make_pair(pair_id="chat-0001", workload=WORKLOAD_CHAT, author_label=None),
        ]

    def test_csv_roundtrip_equal(self):
        pairs = self._sample_pairs()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            save_csv(pairs, path)
            restored = load_csv(path)
        self.assertEqual(pairs, restored)

    def test_json_roundtrip_equal(self):
        pairs = self._sample_pairs()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json(pairs, path)
            restored = load_json(path)
        self.assertEqual(pairs, restored)

    def test_csv_and_json_have_identical_content(self):
        pairs = self._sample_pairs()
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            json_path = Path(tmp) / "out.json"
            save_dataset(pairs, csv_path, json_path)
            from_csv = load_csv(csv_path)
            from_json = load_json(json_path)
        self.assertEqual(from_csv, from_json)

    def test_load_dataset_dispatches_by_extension(self):
        pairs = self._sample_pairs()
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            json_path = Path(tmp) / "out.json"
            save_dataset(pairs, csv_path, json_path)
            self.assertEqual(load_dataset(csv_path), load_dataset(json_path))

    def test_load_dataset_rejects_unknown_extension(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            path.write_text("nonsense")
            with self.assertRaises(DatasetValidationError):
                load_dataset(path)

    def test_load_csv_missing_column_raises_clear_error(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("pair_id,workload\nfaq-0001,faq\n")
            with self.assertRaises(DatasetValidationError):
                load_csv(path)

    def test_load_csv_invalid_row_raises_with_row_context(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            header = "pair_id,workload,source_corpus,source_pair_id,query_1,query_2,original_label,author_label,metadata\n"
            bad_row = "faq-0001,not-a-workload,c,s,q1,q2,1,,{}\n"
            path.write_text(header + bad_row)
            with self.assertRaises(DatasetValidationError) as ctx:
                load_csv(path)
            self.assertIn(str(path), str(ctx.exception))

    def test_load_json_rejects_non_list(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"not": "a list"}))
            with self.assertRaises(DatasetValidationError):
                load_json(path)

    def test_committed_fixture_loads_and_matches_between_formats(self):
        """The data/ synthetic fixtures must themselves be valid and in sync."""
        csv_pairs = load_csv(FIXTURE_CSV)
        json_pairs = load_json(FIXTURE_JSON)
        self.assertEqual(csv_pairs, json_pairs)
        self.assertEqual(len(csv_pairs), 15)
        for pair in csv_pairs:
            self.assertEqual(pair.metadata.get("provenance"), "synthetic-fixture")
            self.assertEqual(pair.source_corpus, "synthetic-fixture")
        counts = {w: sum(1 for p in csv_pairs if p.workload == w) for w in WORKLOADS}
        self.assertEqual(counts, {"faq": 5, "code": 5, "chat": 5})


# ---------------------------------------------------------------------------
# Sampling: determinism + stratification
# ---------------------------------------------------------------------------

class TestSampling(unittest.TestCase):

    def test_same_seed_is_deterministic(self):
        source_a = MockCorpusSource(WORKLOAD_FAQ, n_candidates=40, seed=1)
        source_b = MockCorpusSource(WORKLOAD_FAQ, n_candidates=40, seed=1)
        pairs_a = sample_workload(source_a, n=10, seed=99)
        pairs_b = sample_workload(source_b, n=10, seed=99)
        self.assertEqual(pairs_a, pairs_b)

    def test_different_seed_gives_different_sample(self):
        source_a = MockCorpusSource(WORKLOAD_FAQ, n_candidates=40, seed=1)
        source_b = MockCorpusSource(WORKLOAD_FAQ, n_candidates=40, seed=1)
        pairs_a = sample_workload(source_a, n=10, seed=1)
        pairs_b = sample_workload(source_b, n=10, seed=2)
        self.assertNotEqual(
            [p.source_pair_id for p in pairs_a],
            [p.source_pair_id for p in pairs_b],
        )

    def test_stratification_counts(self):
        source = MockCorpusSource(WORKLOAD_CODE, n_candidates=100, seed=7)
        pairs = sample_workload(source, n=20, seed=7, positive_ratio=0.5)
        positives = sum(1 for p in pairs if p.original_label == 1)
        negatives = sum(1 for p in pairs if p.original_label == 0)
        self.assertEqual(positives, 10)
        self.assertEqual(negatives, 10)
        self.assertEqual(len(pairs), 20)

    def test_stratification_uneven_ratio(self):
        source = MockCorpusSource(WORKLOAD_CHAT, n_candidates=100, seed=3)
        pairs = sample_workload(source, n=10, seed=3, positive_ratio=0.3)
        positives = sum(1 for p in pairs if p.original_label == 1)
        self.assertEqual(positives, 3)

    def test_insufficient_candidates_raises(self):
        source = MockCorpusSource(WORKLOAD_FAQ, n_candidates=4, seed=1)  # 2 pos, 2 neg
        with self.assertRaises(CorpusSourceError):
            sample_workload(source, n=10, seed=1)

    def test_pair_ids_are_sequential_and_traceable(self):
        source = MockCorpusSource(WORKLOAD_FAQ, n_candidates=20, seed=5)
        pairs = sample_workload(source, n=6, seed=5)
        self.assertEqual([p.pair_id for p in pairs], [f"faq-{i:04d}" for i in range(6)])
        for pair in pairs:
            self.assertEqual(pair.source_corpus, "mock")
            self.assertTrue(pair.source_pair_id.startswith("mock-faq-"))

    def test_sample_dataset_covers_all_workloads(self):
        sources = {
            WORKLOAD_FAQ: MockCorpusSource(WORKLOAD_FAQ, n_candidates=40, seed=1),
            WORKLOAD_CODE: MockCorpusSource(WORKLOAD_CODE, n_candidates=40, seed=2),
            WORKLOAD_CHAT: MockCorpusSource(WORKLOAD_CHAT, n_candidates=40, seed=3),
        }
        pairs = sample_dataset(sources, n_per_workload=8, seed=42)
        self.assertEqual(len(pairs), 24)
        counts = {w: sum(1 for p in pairs if p.workload == w) for w in WORKLOADS}
        self.assertEqual(counts, {"faq": 8, "code": 8, "chat": 8})

    def test_sample_dataset_missing_workload_raises(self):
        sources = {WORKLOAD_FAQ: MockCorpusSource(WORKLOAD_FAQ, n_candidates=20, seed=1)}
        with self.assertRaises(CorpusSourceError):
            sample_dataset(sources, n_per_workload=5, seed=1)


# ---------------------------------------------------------------------------
# Blind annotation flow
# ---------------------------------------------------------------------------

class TestBlindAnnotation(unittest.TestCase):

    def _pairs(self):
        return [
            _make_pair(pair_id="faq-0001", original_label=1, author_label=None),
            _make_pair(pair_id="faq-0002", original_label=0, author_label=None),
            _make_pair(pair_id="faq-0003", original_label=1, author_label=None),
        ]

    def test_original_label_never_shown_to_annotator(self):
        pairs = self._pairs()
        shown_lines = []
        answers = iter(["1", "0", "1"])
        with TemporaryDirectory() as tmp:
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=lambda prompt: next(answers),
                output_fn=lambda msg: shown_lines.append(msg),
            )
            session.run()
        joined = "\n".join(shown_lines)
        self.assertNotIn("original_label", joined)
        # The literal label values (1 as a *label*) must not leak either;
        # only check that the source-identifying fields never appear.
        for pair in pairs:
            self.assertNotIn(pair.source_pair_id, joined)
            self.assertNotIn(pair.source_corpus, joined)

    def test_answers_recorded_as_author_label(self):
        pairs = self._pairs()
        answers = iter(["1", "0", "1"])
        with TemporaryDirectory() as tmp:
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=lambda prompt: next(answers),
                output_fn=lambda msg: None,
            )
            summary = session.run()
        self.assertEqual(summary.newly_labeled, 3)
        self.assertEqual([p.author_label for p in pairs], [1, 0, 1])

    def test_resume_uses_progress_file(self):
        pairs = self._pairs()
        with TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.json"

            # First session: answer only the first pair, then quit.
            answers_1 = iter(["1", "q"])
            session_1 = BlindAnnotationSession(
                pairs,
                progress_path=progress_path,
                input_fn=lambda prompt: next(answers_1),
                output_fn=lambda msg: None,
            )
            summary_1 = session_1.run()
            self.assertTrue(summary_1.quit_early)
            self.assertEqual(pairs[0].author_label, 1)
            self.assertIsNone(pairs[1].author_label)

            # New session (e.g. after a restart) reloads the same pairs +
            # progress file; already-answered pair must not be re-asked.
            fresh_pairs = self._pairs()  # simulate reloading from disk
            asked_pair_ids = []

            def _tracking_input(prompt):
                return "0"

            session_2 = BlindAnnotationSession(
                fresh_pairs,
                progress_path=progress_path,
                input_fn=_tracking_input,
                output_fn=lambda msg: asked_pair_ids.append(msg) if msg.startswith("\n---") else None,
            )
            self.assertEqual(fresh_pairs[0].author_label, 1)  # merged from progress
            summary_2 = session_2.run()
            self.assertEqual(summary_2.newly_labeled, 2)  # only pairs 2 and 3
            self.assertFalse(any("faq-0001" in line for line in asked_pair_ids))

    def test_no_overwrite_by_default(self):
        pairs = self._pairs()
        pairs[0].author_label = 1  # already annotated
        with TemporaryDirectory() as tmp:
            answers = iter(["0", "1"])  # only 2 remaining pairs asked
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=lambda prompt: next(answers),
                output_fn=lambda msg: None,
            )
            summary = session.run()
        self.assertEqual(pairs[0].author_label, 1)  # untouched
        self.assertEqual(summary.already_labeled, 1)
        self.assertEqual(summary.newly_labeled, 2)

    def test_overwrite_flag_relabels_everything(self):
        pairs = self._pairs()
        pairs[0].author_label = 1
        with TemporaryDirectory() as tmp:
            answers = iter(["0", "0", "0"])
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=lambda prompt: next(answers),
                output_fn=lambda msg: None,
                overwrite=True,
            )
            session.run()
        self.assertEqual(pairs[0].author_label, 0)  # overwritten

    def test_quit_stops_early_and_preserves_progress(self):
        pairs = self._pairs()
        answers = iter(["1", "q"])
        with TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.json"
            session = BlindAnnotationSession(
                pairs,
                progress_path=progress_path,
                input_fn=lambda prompt: next(answers),
                output_fn=lambda msg: None,
            )
            summary = session.run()
            self.assertTrue(summary.quit_early)
            self.assertTrue(progress_path.exists())
            with progress_path.open() as fh:
                saved = json.load(fh)
            self.assertEqual(saved, {"faq-0001": 1})

    def test_skip_leaves_pair_unlabeled(self):
        pairs = self._pairs()
        answers = iter(["s", "1", "0"])
        with TemporaryDirectory() as tmp:
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=lambda prompt: next(answers),
                output_fn=lambda msg: None,
            )
            summary = session.run()
        self.assertEqual(summary.skipped, 1)
        self.assertIsNone(pairs[0].author_label)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

class TestCohenKappa(unittest.TestCase):

    def test_perfect_agreement_kappa_is_one(self):
        pairs = [
            _make_pair(pair_id=f"faq-{i:04d}", original_label=lbl, author_label=lbl)
            for i, lbl in enumerate([1, 0, 1, 0, 1, 0])
        ]
        result = cohen_kappa(pairs)
        self.assertAlmostEqual(result.kappa, 1.0, places=6)
        self.assertEqual(result.n_annotated, 6)
        self.assertEqual(result.n_excluded_unannotated, 0)

    def test_chance_agreement_kappa_near_zero(self):
        # Hand-constructed 2x2 where po == pe: independent 50/50 labels.
        # original: 1,1,0,0 (2 pos / 2 neg); author: 1,0,1,0 -> tp=1,fp=1,fn=1,tn=1
        pairs = [
            _make_pair(pair_id="faq-0001", original_label=1, author_label=1),
            _make_pair(pair_id="faq-0002", original_label=1, author_label=0),
            _make_pair(pair_id="faq-0003", original_label=0, author_label=1),
            _make_pair(pair_id="faq-0004", original_label=0, author_label=0),
        ]
        result = cohen_kappa(pairs)
        # po = (1+1)/4 = 0.5; p_original_1 = 0.5; p_author_1 = 0.5
        # pe = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa = (0.5-0.5)/(1-0.5) = 0.0
        self.assertAlmostEqual(result.kappa, 0.0, places=6)

    def test_worked_2x2_example(self):
        # tp=5, fp=1, fn=2, tn=2 -> n=10
        labels = [(1, 1)] * 5 + [(0, 1)] * 1 + [(1, 0)] * 2 + [(0, 0)] * 2
        pairs = [
            _make_pair(pair_id=f"faq-{i:04d}", original_label=o, author_label=a)
            for i, (o, a) in enumerate(labels)
        ]
        result = cohen_kappa(pairs)
        po = (5 + 2) / 10  # 0.7
        p_o1 = (5 + 2) / 10  # 0.7 (original==1: tp+fn)
        p_a1 = (5 + 1) / 10  # 0.6 (author==1: tp+fp)
        pe = p_o1 * p_a1 + (1 - p_o1) * (1 - p_a1)
        expected_kappa = (po - pe) / (1 - pe)
        self.assertAlmostEqual(result.observed_agreement, po, places=6)
        self.assertAlmostEqual(result.expected_agreement, pe, places=6)
        self.assertAlmostEqual(result.kappa, expected_kappa, places=6)

    def test_empty_dataset_kappa_is_none(self):
        result = cohen_kappa([])
        self.assertIsNone(result.kappa)
        self.assertEqual(result.n_annotated, 0)

    def test_unannotated_pairs_excluded_with_count(self):
        pairs = [
            _make_pair(pair_id="faq-0001", original_label=1, author_label=1),
            _make_pair(pair_id="faq-0002", original_label=0, author_label=None),
            _make_pair(pair_id="faq-0003", original_label=0, author_label=None),
        ]
        result = cohen_kappa(pairs)
        self.assertEqual(result.n_annotated, 1)
        self.assertEqual(result.n_excluded_unannotated, 2)

    def test_degenerate_all_one_class_perfect_agreement(self):
        pairs = [
            _make_pair(pair_id=f"faq-{i:04d}", original_label=1, author_label=1)
            for i in range(4)
        ]
        result = cohen_kappa(pairs)
        self.assertEqual(result.kappa, 1.0)

    def test_all_original_one_class_partial_author_agreement(self):
        """
        original_label is 1 for every pair here, but author_label is not
        unanimous, so expected agreement pe != 1 (pe=1 requires BOTH
        annotators' marginals to be degenerate in the same direction, which
        forces po=1 too -- see kappa.py docstring). This still exercises
        the ordinary (po-pe)/(1-pe) path with a lopsided marginal.
        """
        pairs = [
            _make_pair(pair_id="faq-0001", original_label=1, author_label=1),
            _make_pair(pair_id="faq-0002", original_label=1, author_label=0),
        ]
        result = cohen_kappa(pairs)
        self.assertEqual(result.kappa, 0.0)
        self.assertLess(result.expected_agreement, 1.0)

    def test_kappa_report_per_workload(self):
        pairs = [
            _make_pair(pair_id="faq-0001", workload=WORKLOAD_FAQ, original_label=1, author_label=1),
            _make_pair(pair_id="code-0001", workload=WORKLOAD_CODE, original_label=0, author_label=0),
            _make_pair(pair_id="chat-0001", workload=WORKLOAD_CHAT, original_label=1, author_label=0),
        ]
        report = kappa_report(pairs)
        self.assertEqual(set(report.per_workload.keys()), set(WORKLOADS))
        self.assertEqual(report.per_workload["faq"].n_annotated, 1)
        self.assertEqual(report.overall.n_annotated, 3)

    def test_fixture_dataset_kappa_computes(self):
        """Sanity check against the committed synthetic fixtures."""
        pairs = load_json(FIXTURE_JSON)
        result = cohen_kappa(pairs)
        self.assertEqual(result.n_annotated, 15)
        self.assertEqual(result.n_excluded_unannotated, 0)
        self.assertIsNotNone(result.kappa)
        self.assertGreater(result.kappa, 0.0)


# ---------------------------------------------------------------------------
# CLI smoke tests (offline, against the synthetic fixtures)
# ---------------------------------------------------------------------------

class TestCliSmoke(unittest.TestCase):

    def _run(self, script: str, args, input_text: str = None):
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / script)] + args
        return subprocess.run(
            cmd, cwd=REPO_ROOT, input=input_text,
            capture_output=True, text=True, timeout=60,
        )

    def test_sample_dataset_cli_runs_offline(self):
        with TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "sampled.csv"
            out_json = Path(tmp) / "sampled.json"
            result = self._run(
                "sample_dataset.py",
                [
                    "--n-per-workload", "5", "--seed", "42",
                    "--out-csv", str(out_csv), "--out-json", str(out_json),
                ],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            pairs = load_dataset(out_csv)
            self.assertEqual(len(pairs), 15)
            for pair in pairs:
                self.assertEqual(pair.source_corpus, "mock")

    def test_compute_kappa_cli_runs_against_fixture(self):
        result = self._run(
            "compute_kappa.py",
            ["--dataset", str(FIXTURE_JSON)],
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("overall: kappa=", result.stdout)
        self.assertIn("workload=faq", result.stdout)

    def test_compute_kappa_cli_strict_exit_code(self):
        # Fixture kappa is well below a 0.99 threshold -> strict must fail (nonzero exit).
        result = self._run(
            "compute_kappa.py",
            ["--dataset", str(FIXTURE_JSON), "--strict", "--threshold", "0.99"],
        )
        self.assertNotEqual(result.returncode, 0)

    def test_export_dataset_cli_roundtrip(self):
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "converted.csv"
            result = self._run(
                "export_dataset.py",
                ["--in", str(FIXTURE_JSON), "--out", str(out_path)],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(load_csv(out_path), load_json(FIXTURE_JSON))

    def test_annotate_dataset_cli_runs_offline(self):
        with TemporaryDirectory() as tmp:
            # Start from a copy of the fixture with author_label cleared.
            pairs = load_json(FIXTURE_JSON)
            for pair in pairs:
                pair.author_label = None
            in_json = Path(tmp) / "unannotated.json"
            save_json(pairs, in_json)

            progress_path = Path(tmp) / "progress.json"
            out_csv = Path(tmp) / "annotated.csv"
            out_json = Path(tmp) / "annotated.json"

            answers = "\n".join(["1"] * 15) + "\n"
            result = self._run(
                "annotate_dataset.py",
                [
                    "--dataset", str(in_json),
                    "--progress", str(progress_path),
                    "--out-csv", str(out_csv), "--out-json", str(out_json),
                ],
                input_text=answers,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            annotated = load_dataset(out_json)
            self.assertTrue(all(p.author_label == 1 for p in annotated))
            self.assertTrue(progress_path.exists())


if __name__ == "__main__":
    unittest.main()
