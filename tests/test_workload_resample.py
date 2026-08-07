"""
Tests for per-workload re-sampling of the single ground-truth dataset:
timestamped backups, the splice that replaces one workload's rows in place, the
exclusion that keeps a re-sample disjoint from what it replaces, and the
`scripts/sample_dataset.py --workload` CLI end-to-end.

The invariant under test throughout is that there is exactly **one** ground
truth, at its canonical paths: a re-sample rewrites those paths (never a `_v2`
or a dated copy beside them), backs up what it is about to overwrite, and leaves
the other workloads' rows — `author_label` included — byte-for-byte intact.

All offline: the CLI runs against `tests/fixtures/corpora/` (real column
structure, synthetic content) with an unpinned copy of the provenance registry.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from levy.dataset.backup import (
    BackupError,
    backup_file,
    backup_files,
    backup_timestamp,
    describe_backups,
)
from levy.dataset.corpora import load_registry
from levy.dataset.io import (
    load_dataset,
    load_distribution_csv,
    save_dataset,
    save_distribution_csv,
    to_distribution_records,
)
from levy.dataset.sampling import (
    CorpusSourceError,
    MockCorpusSource,
    QuoraQQPSource,
    sample_workload,
)
from levy.dataset.schema import WORKLOADS, QueryPair
from levy.dataset.validation import validate_sources
from levy.dataset.workload_update import (
    ExcludingCorpusSource,
    WorkloadUpdateError,
    check_ids_alignment,
    clear_author_labels,
    resample_shortfall_message,
    source_pair_ids,
    splice_workload,
    verify_disjoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "corpora"
REGISTRY = REPO_ROOT / "data" / "corpora.json"

#: Names a parallel/versioned dataset would take. The whole point of the
#: in-place design is that none of these is ever created.
_FORBIDDEN_NAME = re.compile(
    r"(_v\d|_new\b|_final\b|_candidate\b|\.bak|-copy\b|\d{4}-\d{2}-\d{2})", re.IGNORECASE
)


def _make_pair(pair_id, workload, source_pair_id, label=1, author_label=None) -> QueryPair:
    return QueryPair(
        pair_id=pair_id,
        workload=workload,
        source_corpus=f"corpus-{workload}",
        source_pair_id=source_pair_id,
        query_1=f"{pair_id} q1",
        query_2=f"{pair_id} q2",
        original_label=label,
        author_label=author_label,
    )


_FIXTURE_REGISTRY: Path = None


def _fixture_registry_path() -> Path:
    """
    An unpinned copy of the committed registry, on disk, for CLI subprocesses.

    The committed registry's checksums describe the *real* corpora; every test
    here runs against the fixtures, so it must be handed a registry that makes
    no claim about their checksums.
    """
    global _FIXTURE_REGISTRY
    if _FIXTURE_REGISTRY is None:
        directory = Path(tempfile.mkdtemp(prefix="levy-resample-registry-"))
        raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for entry in raw["corpora"].values():
            for file_entry in entry.get("files", []):
                file_entry["sha256"] = None
        path = directory / "corpora.json"
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        _FIXTURE_REGISTRY = path
    return _FIXTURE_REGISTRY


def _run(script: str, args, timeout: int = 120):
    args = [str(a) for a in args]
    if "--registry" not in args:
        args += ["--registry", str(_fixture_registry_path())]
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

class TestBackup(unittest.TestCase):

    def test_backup_lands_in_backups_dir_with_a_utc_timestamp(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "ground_truth.ids.csv"
            target.write_text("original\n", encoding="utf-8")
            created = backup_file(target)
            self.assertIsNotNone(created)
            self.assertEqual(created.parent, Path(tmp) / "backups")
            self.assertEqual(created.read_text(encoding="utf-8"), "original\n")
            # `ground_truth.ids.<stamp>.csv` — the ids file stays recognisable.
            self.assertTrue(created.name.startswith("ground_truth.ids."))
            self.assertTrue(created.name.endswith(".csv"))
            self.assertRegex(created.name, r"\d{8}T\d{6}Z")

    def test_timestamp_is_utc_not_local(self):
        stamp = backup_timestamp(datetime(2026, 8, 6, 21, 52, 33, tzinfo=timezone.utc))
        self.assertEqual(stamp, "20260806T215233Z")

    def test_absent_file_needs_no_backup(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(backup_file(Path(tmp) / "nope.csv"))
            self.assertFalse((Path(tmp) / "backups").exists())

    def test_second_backup_in_the_same_second_does_not_clobber_the_first(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "d.csv"
            target.write_text("first\n", encoding="utf-8")
            first = backup_file(target, timestamp="20260806T215233Z")
            target.write_text("second\n", encoding="utf-8")
            second = backup_file(target, timestamp="20260806T215233Z")
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(encoding="utf-8"), "first\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "second\n")

    def test_unusable_backup_dir_raises(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "d.csv"
            target.write_text("x\n", encoding="utf-8")
            blocker = Path(tmp) / "blocked"
            blocker.write_text("i am a file, not a directory\n", encoding="utf-8")
            with self.assertRaises(BackupError):
                backup_file(target, backup_dir=blocker)

    def test_naive_datetime_is_read_as_utc(self):
        self.assertEqual(
            backup_timestamp(datetime(2026, 8, 6, 21, 52, 33)), "20260806T215233Z"
        )

    def test_a_directory_is_never_backed_up_as_a_file(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(BackupError):
                backup_file(Path(tmp))

    def test_exhausted_discriminators_refuse_to_overwrite(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "d.csv"
            target.write_text("x\n", encoding="utf-8")
            with unittest.mock.patch(
                "levy.dataset.backup._MAX_DISCRIMINATORS", 2
            ):
                backup_file(target, timestamp="20260806T215233Z")
                backup_file(target, timestamp="20260806T215233Z")
                with self.assertRaises(BackupError) as ctx:
                    backup_file(target, timestamp="20260806T215233Z")
            self.assertIn("refusing to overwrite an existing backup", str(ctx.exception))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores mode bits")
    def test_unwritable_backup_dir_raises_and_names_the_target(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "d.csv"
            target.write_text("x\n", encoding="utf-8")
            read_only = Path(tmp) / "ro"
            read_only.mkdir(mode=0o500)
            try:
                with self.assertRaises(BackupError) as ctx:
                    backup_file(target, backup_dir=read_only)
            finally:
                read_only.chmod(0o700)
            self.assertIn("cannot write backup", str(ctx.exception))
            self.assertIn("the current dataset is intact", str(ctx.exception))

    def test_describe_backups_is_empty_when_none_were_needed(self):
        self.assertEqual(describe_backups({}), "")

    def test_describe_backups_lists_one_line_per_file(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "d.csv"
            target.write_text("x\n", encoding="utf-8")
            made = backup_files([target])
            self.assertEqual(len(describe_backups(made).splitlines()), 1)

    def test_group_backup_shares_one_timestamp(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for name in ("a.csv", "b.json"):
                path = Path(tmp) / name
                path.write_text(name, encoding="utf-8")
                paths.append(path)
            # A repeated path is backed up once, not twice; an absent one is skipped.
            made = backup_files(paths + [paths[0], Path(tmp) / "absent.csv"])
            self.assertEqual(sorted(made), sorted(paths))
            stamps = {
                backup.name.split(".")[-2] for backup in made.values()
            }
            self.assertEqual(len(stamps), 1)


# ---------------------------------------------------------------------------
# Splicing one workload's rows into a live dataset
# ---------------------------------------------------------------------------

class TestSpliceWorkload(unittest.TestCase):

    def _dataset(self):
        return [
            _make_pair("faq-0000", "faq", "q-1", 1, author_label=1),
            _make_pair("faq-0001", "faq", "q-2", 0, author_label=0),
            _make_pair("code-0000", "code", "s-1", 1, author_label=0),
            _make_pair("chat-0000", "chat", "t-1", 1, author_label=1),
            _make_pair("chat-0001", "chat", "t-2", 0, author_label=1),
        ]

    def test_other_workloads_pass_through_by_identity(self):
        current = self._dataset()
        fresh = [_make_pair("chat-0000", "chat", "t-9", 1)]
        result = splice_workload(current, "chat", fresh)

        kept = [p for p in result.pairs if p.workload != "chat"]
        # Same objects, not copies: nothing can have re-derived their labels.
        self.assertEqual([id(p) for p in kept], [id(current[0]), id(current[1]), id(current[2])])
        self.assertEqual([p.author_label for p in kept], [1, 0, 0])

    def test_replaced_block_keeps_its_position(self):
        current = self._dataset()
        fresh = [
            _make_pair("chat-0000", "chat", "t-9", 1),
            _make_pair("chat-0001", "chat", "t-8", 0),
        ]
        result = splice_workload(current, "chat", fresh)
        self.assertEqual(
            [p.pair_id for p in result.pairs],
            ["faq-0000", "faq-0001", "code-0000", "chat-0000", "chat-0001"],
        )
        self.assertEqual([p.source_pair_id for p in result.removed], ["t-1", "t-2"])

    def test_larger_sample_appends_surplus_after_the_block(self):
        current = self._dataset()
        fresh = [_make_pair(f"chat-{i:04d}", "chat", f"t-{i + 9}", i % 2) for i in range(4)]
        result = splice_workload(current, "chat", fresh)
        self.assertEqual(sum(1 for p in result.pairs if p.workload == "chat"), 4)
        self.assertEqual([p.workload for p in result.pairs][:3], ["faq", "faq", "code"])

    def test_smaller_sample_drops_trailing_rows(self):
        current = self._dataset()
        result = splice_workload(current, "chat", [_make_pair("chat-0000", "chat", "t-9", 1)])
        self.assertEqual(sum(1 for p in result.pairs if p.workload == "chat"), 1)
        self.assertEqual(len(result.removed), 2)

    def test_absent_workload_is_appended_and_flagged(self):
        current = [p for p in self._dataset() if p.workload != "chat"]
        result = splice_workload(current, "chat", [_make_pair("chat-0000", "chat", "t-9", 1)])
        self.assertTrue(result.appended)
        self.assertEqual(result.pairs[-1].pair_id, "chat-0000")

    def test_foreign_workload_in_new_pairs_rejected(self):
        with self.assertRaises(WorkloadUpdateError):
            splice_workload(self._dataset(), "chat", [_make_pair("faq-0000", "faq", "q-9", 1)])

    def test_unknown_workload_rejected(self):
        with self.assertRaises(WorkloadUpdateError) as ctx:
            splice_workload(self._dataset(), "rag", [])
        self.assertIn("rag", str(ctx.exception))

    def test_source_pair_ids_can_be_scoped_or_global(self):
        current = self._dataset()
        self.assertEqual(source_pair_ids(current, "chat"), {"t-1", "t-2"})
        self.assertEqual(len(source_pair_ids(current)), 5)

    def test_clear_author_labels_reports_what_it_cleared(self):
        pairs = [_make_pair("chat-0000", "chat", "t-9", 1, author_label=1)]
        self.assertEqual(clear_author_labels(pairs), 1)
        self.assertIsNone(pairs[0].author_label)
        self.assertEqual(clear_author_labels(pairs), 0)


class TestExclusion(unittest.TestCase):

    def test_excluded_ids_are_hidden_from_candidates(self):
        source = MockCorpusSource("faq", n_candidates=10, seed=1)
        all_ids = [c.source_pair_id for c in source.iter_candidates()]
        excluded = ExcludingCorpusSource(source, all_ids[:4])
        remaining = [c.source_pair_id for c in excluded.iter_candidates()]
        self.assertEqual(remaining, all_ids[4:])

    def test_wrapper_delegates_every_declaration(self):
        source = QuoraQQPSource(CORPUS_FIXTURES / "quora-qqp" / "train.tsv")
        wrapped = ExcludingCorpusSource(source, [])
        self.assertEqual(wrapped.workload, "faq")
        self.assertEqual(wrapped.name, source.name)
        self.assertEqual(wrapped.corpus_key, source.corpus_key)
        self.assertEqual(wrapped.label_mapping(), source.label_mapping())
        self.assertEqual(wrapped.options(), source.options())
        self.assertEqual(wrapped.source_files(), source.source_files())
        self.assertEqual(wrapped.check_fields(), source.check_fields())

    def test_validation_counts_the_post_exclusion_pool(self):
        source = QuoraQQPSource(CORPUS_FIXTURES / "quora-qqp" / "train.tsv")
        first = sample_workload(source, n=6, seed=42)
        wrapped = ExcludingCorpusSource(source, source_pair_ids(first))

        # An unpinned registry: the committed checksums describe the real
        # corpora, not these fixtures.
        registry = load_registry(_fixture_registry_path())
        report = validate_sources(
            {"faq": wrapped}, n_per_workload=6, registry=registry, raw_root=CORPUS_FIXTURES
        )
        pool = report.pools["faq"]
        # 10 positives in the fixture, 3 of them already sampled.
        self.assertEqual(pool.positive_available, 7)
        self.assertTrue(pool.sufficient)

        short = validate_sources(
            {"faq": wrapped}, n_per_workload=16, registry=registry, raw_root=CORPUS_FIXTURES
        )
        self.assertFalse(short.ok)
        self.assertTrue(any("faq" == f.workload for f in short.findings))

    def test_resample_is_disjoint_from_what_it_replaces(self):
        source = QuoraQQPSource(CORPUS_FIXTURES / "quora-qqp" / "train.tsv")
        first = sample_workload(source, n=6, seed=42)
        previous = source_pair_ids(first)
        second = sample_workload(
            ExcludingCorpusSource(source, previous), n=6, seed=42
        )
        self.assertEqual(previous & source_pair_ids(second), set())
        verify_disjoint(second, previous)  # must not raise

    def test_verify_disjoint_catches_an_overlap(self):
        pairs = [_make_pair("faq-0000", "faq", "q-1", 1)]
        with self.assertRaises(WorkloadUpdateError):
            verify_disjoint(pairs, {"q-1"})

    def test_shortfall_message_names_workload_and_exclusion(self):
        message = resample_shortfall_message(
            "chat", 300, CorpusSourceError("twitter-pit2015/chat: requested 150, only 90 available")
        )
        self.assertIn("'chat'", message)
        self.assertIn("only 90 available", message)
        self.assertIn("300 pair(s) already in the ground truth were excluded", message)
        self.assertIn("Nothing written", message)

    def test_exhausted_pool_raises_naming_the_workload(self):
        source = QuoraQQPSource(CORPUS_FIXTURES / "quora-qqp" / "train.tsv")
        first = sample_workload(source, n=6, seed=42)
        with self.assertRaises(CorpusSourceError) as ctx:
            sample_workload(
                ExcludingCorpusSource(source, source_pair_ids(first)), n=18, seed=42
            )
        self.assertIn("faq", str(ctx.exception))


class TestIdsAlignment(unittest.TestCase):

    def _paired(self):
        pairs = [
            _make_pair("faq-0000", "faq", "q-1", 1, author_label=1),
            _make_pair("chat-0000", "chat", "t-1", 0, author_label=0),
        ]
        return pairs, to_distribution_records(pairs)

    def test_aligned_dataset_and_ids_pass(self):
        pairs, records = self._paired()
        check_ids_alignment(pairs, records, ["faq"])  # must not raise

    def test_label_only_difference_is_allowed(self):
        """Annotation writes the working dataset first; the ids file lags by design."""
        pairs, records = self._paired()
        records[0].author_label = None
        check_ids_alignment(pairs, records, ["faq"])  # must not raise

    def test_identity_mismatch_is_an_error(self):
        pairs, records = self._paired()
        records[0].source_pair_id = "q-999"
        with self.assertRaises(WorkloadUpdateError) as ctx:
            check_ids_alignment(pairs, records, ["faq"])
        self.assertIn("rehydrate_dataset.py", str(ctx.exception))

    def test_row_count_mismatch_is_an_error(self):
        pairs, records = self._paired()
        with self.assertRaises(WorkloadUpdateError):
            check_ids_alignment(pairs, records[:1], ["faq", "chat"])


# ---------------------------------------------------------------------------
# scripts/sample_dataset.py --workload, end to end
# ---------------------------------------------------------------------------

class TestSingleWorkloadSamplingCli(unittest.TestCase):
    """
    Each test builds a complete, fully annotated dataset at the canonical paths
    inside a temp dir, then re-samples one workload and inspects what moved.
    """

    N = 6

    def _paths(self, tmp: Path):
        return {
            "csv": tmp / "ground_truth.full.csv",
            "json": tmp / "ground_truth.full.json",
            "ids": tmp / "ground_truth.ids.csv",
            "meta": tmp / "ground_truth.ids.meta.json",
        }

    def _sample_full(self, tmp: Path, seed: int = 42, raw_dir: Path = CORPUS_FIXTURES):
        paths = self._paths(tmp)
        result = _run(
            "sample_dataset.py",
            ["--require-real", "--raw-dir", raw_dir, "--n-per-workload", self.N,
             "--seed", seed, "--out-csv", paths["csv"], "--out-json", paths["json"],
             "--out-ids", paths["ids"]],
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return paths

    def _annotate_everything(self, paths):
        """Stand in for the author's finished blind re-annotation."""
        pairs = load_dataset(paths["csv"])
        for index, pair in enumerate(pairs):
            pair.author_label = index % 2
        save_dataset(pairs, paths["csv"], paths["json"])
        save_distribution_csv(to_distribution_records(pairs), paths["ids"])
        return pairs

    def _resample(self, tmp: Path, workload, seed=4242, raw_dir=CORPUS_FIXTURES, extra=()):
        paths = self._paths(tmp)
        workloads = [workload] if isinstance(workload, str) else workload
        args = ["--require-real", "--raw-dir", raw_dir, "--n-per-workload", self.N,
                "--seed", seed, "--out-csv", paths["csv"], "--out-json", paths["json"],
                "--out-ids", paths["ids"]]
        for name in workloads:
            args += ["--workload", name]
        return _run("sample_dataset.py", args + list(extra))

    # -- core behaviour -------------------------------------------------

    def test_resample_replaces_only_that_workload_and_keeps_the_others_byte_for_byte(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)

            before = paths["csv"].read_text(encoding="utf-8").splitlines()
            result = self._resample(tmp, "chat")
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            after = paths["csv"].read_text(encoding="utf-8").splitlines()

            untouched_before = [line for line in before if not line.startswith("chat-")]
            untouched_after = [line for line in after if not line.startswith("chat-")]
            # Byte-for-byte, including the author_label column.
            self.assertEqual(untouched_before, untouched_after)
            self.assertEqual(len(after), len(before))

    def test_author_label_cleared_for_the_resampled_workload_only(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)

            self.assertEqual(self._resample(tmp, "chat").returncode, 0)
            pairs = load_dataset(paths["csv"])
            for pair in pairs:
                if pair.workload == "chat":
                    self.assertIsNone(pair.author_label, msg=pair.pair_id)
                else:
                    self.assertIsNotNone(pair.author_label, msg=pair.pair_id)

    def test_new_sample_is_disjoint_from_the_pairs_it_replaced(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            before = source_pair_ids(load_dataset(paths["csv"]), "chat")

            self.assertEqual(self._resample(tmp, "chat").returncode, 0)
            after = source_pair_ids(load_dataset(paths["csv"]), "chat")

            self.assertEqual(len(after), self.N)
            self.assertEqual(before & after, set())

    def test_same_seed_still_draws_fresh_pairs(self):
        """Exclusion, not the seed, is what makes a re-sample new."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp, seed=42)
            before = source_pair_ids(load_dataset(paths["csv"]), "chat")
            self.assertEqual(self._resample(tmp, "chat", seed=42).returncode, 0)
            after = source_pair_ids(load_dataset(paths["csv"]), "chat")
            self.assertEqual(before & after, set())

    def test_only_the_named_workloads_corpus_is_read(self):
        """A chat re-sample must not require the Quora TSV or the SODD shards."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)

            chat_only = tmp / "raw-chat-only"
            (chat_only / "twitter-pit2015").mkdir(parents=True)
            for name in ("train.data", "dev.data"):
                shutil.copy2(
                    CORPUS_FIXTURES / "twitter-pit2015" / name, chat_only / "twitter-pit2015" / name
                )

            result = self._resample(tmp, "chat", raw_dir=chat_only)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(len(load_dataset(paths["csv"])), self.N * len(WORKLOADS))

    def test_repeatable_flag_resamples_several_workloads(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            before = {w: source_pair_ids(load_dataset(paths["csv"]), w) for w in WORKLOADS}

            result = self._resample(tmp, ["chat", "faq"])
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            pairs = load_dataset(paths["csv"])
            for workload in ("chat", "faq"):
                self.assertEqual(before[workload] & source_pair_ids(pairs, workload), set())
            self.assertEqual(before["code"], source_pair_ids(pairs, "code"))
            self.assertTrue(
                all(p.author_label is not None for p in pairs if p.workload == "code")
            )

    # -- guards --------------------------------------------------------

    def test_pool_shortfall_names_the_workload_and_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            before = paths["csv"].read_bytes()

            # The fixture chat pool is 13 positives / 13 negatives; 6 pairs
            # (3 + 3) are already sampled, so 24 pairs (12 + 12) cannot be drawn.
            args = ["--require-real", "--raw-dir", CORPUS_FIXTURES, "--n-per-workload", 24,
                    "--seed", 7, "--workload", "chat",
                    "--out-csv", paths["csv"], "--out-json", paths["json"],
                    "--out-ids", paths["ids"]]
            result = _run("sample_dataset.py", args)

            self.assertEqual(result.returncode, 1, msg=result.stderr)
            self.assertIn("chat", result.stderr)
            self.assertIn("nothing written", result.stderr)
            self.assertEqual(paths["csv"].read_bytes(), before)

    def test_shortfall_past_validation_names_the_exclusion(self):
        """With pre-flight skipped, the sampling path itself must still refuse."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            before = paths["csv"].read_bytes()

            result = _run(
                "sample_dataset.py",
                ["--require-real", "--raw-dir", CORPUS_FIXTURES, "--n-per-workload", 24,
                 "--seed", 7, "--workload", "chat", "--skip-validation",
                 "--out-csv", paths["csv"], "--out-json", paths["json"],
                 "--out-ids", paths["ids"]],
            )
            self.assertEqual(result.returncode, 1, msg=result.stderr)
            self.assertIn("chat", result.stderr)
            self.assertIn("already in the ground truth were excluded", result.stderr)
            self.assertEqual(paths["csv"].read_bytes(), before)

    def test_missing_dataset_is_an_error_not_an_empty_start(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._paths(tmp)
            result = self._resample(tmp, "chat")
            self.assertEqual(result.returncode, 1)
            self.assertIn("--workload", result.stderr)
            self.assertFalse(paths["csv"].exists())

    def test_stale_ids_file_blocks_the_merge(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)

            records = load_distribution_csv(paths["ids"])
            for record in records:
                if record.workload == "faq":
                    record.source_pair_id = "drifted"
                    break
            save_distribution_csv(records, paths["ids"])
            before = paths["csv"].read_bytes()

            result = self._resample(tmp, "chat")
            self.assertEqual(result.returncode, 1)
            self.assertIn("stale", result.stderr)
            self.assertEqual(paths["csv"].read_bytes(), before)

    # -- annotation progress -------------------------------------------

    def _write_progress(self, tmp: Path, pairs, name="annotation_progress.json"):
        """A v1 (flat) progress file, the format the real dataset's file is in."""
        path = tmp / name
        path.write_text(
            json.dumps({p.pair_id: (p.author_label or 0) for p in pairs}), encoding="utf-8"
        )
        return path

    def test_resample_invalidates_the_progress_for_the_pairs_it_replaced(self):
        """
        The re-sampled pairs keep their `pair_id`s, so a progress file left alone
        would have the next annotation session re-apply the old answers to the new
        pairs. Since the real file is v1 and carries no fingerprints, the
        re-sample itself has to drop those entries.
        """
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            annotated = self._annotate_everything(paths)
            progress = self._write_progress(tmp, annotated)
            before = json.loads(progress.read_text(encoding="utf-8"))

            self.assertEqual(self._resample(tmp, "chat").returncode, 0)
            after = json.loads(progress.read_text(encoding="utf-8"))

        self.assertEqual(len(before), self.N * len(WORKLOADS))
        self.assertEqual(after["version"], 2)
        self.assertEqual(len(after["labels"]), self.N * 2)  # chat's entries gone
        self.assertFalse(any(k.startswith("chat-") for k in after["labels"]))
        # The other workloads' answers survive unchanged, now fingerprinted.
        for pair_id, entry in after["labels"].items():
            self.assertEqual(entry["label"], before[pair_id])
            self.assertIsNotNone(entry["source_pair_id"])

    def test_progress_file_is_backed_up_before_it_is_pruned(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            annotated = self._annotate_everything(paths)
            progress = self._write_progress(tmp, annotated)
            original = progress.read_bytes()

            self.assertEqual(self._resample(tmp, "chat").returncode, 0)

            backups = [
                p for p in (tmp / "backups").iterdir()
                if p.name.startswith("annotation_progress.")
            ]
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_progress_path_defaults_beside_the_dataset_not_into_the_repo(self):
        """
        A run writing into a temp directory must never reach into `data/` for the
        progress file — the default is derived from `--out-csv`'s directory.
        """
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            annotated = self._annotate_everything(paths)
            beside = self._write_progress(tmp, annotated)
            elsewhere = self._write_progress(tmp, annotated, name="other_progress.json")
            repo_progress_before = (
                REPO_ROOT / "data" / "annotation_progress.json"
            ).read_bytes()

            self.assertEqual(self._resample(tmp, "chat").returncode, 0)

            self.assertEqual(json.loads(beside.read_text())["version"], 2)  # pruned
            # An unrelated file, and the repo's own, are untouched. The repo's is
            # compared byte-for-byte rather than by format: it is a live file that
            # a real production run legitimately rewrites, so asserting anything
            # about its *contents* would make this test depend on the state of the
            # author's work.
            self.assertNotIn("version", json.loads(elsewhere.read_text()))
            self.assertEqual(
                (REPO_ROOT / "data" / "annotation_progress.json").read_bytes(),
                repo_progress_before,
            )

    def test_explicit_progress_path_is_honoured(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            annotated = self._annotate_everything(paths)
            chosen = self._write_progress(tmp, annotated, name="my_progress.json")

            result = self._resample(
                tmp, "chat", extra=["--progress", str(chosen)]
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(json.loads(chosen.read_text())["version"], 2)
            self.assertIn("dropped", result.stderr)

    def test_malformed_progress_file_stops_the_run_before_anything_is_written(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            before = paths["csv"].read_bytes()
            (tmp / "annotation_progress.json").write_text("[not, an, object]", encoding="utf-8")

            result = self._resample(tmp, "chat")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Nothing written", result.stderr)
            self.assertEqual(paths["csv"].read_bytes(), before)

    def test_full_run_leaves_the_progress_file_alone(self):
        """No --workload, no in-place surgery, nothing to invalidate."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            annotated = self._annotate_everything(paths)
            progress = self._write_progress(tmp, annotated)
            original = progress.read_bytes()

            self._sample_full(tmp, seed=99)
            self.assertEqual(progress.read_bytes(), original)

    # -- backups -------------------------------------------------------

    def test_every_overwritten_file_is_backed_up_first(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            snapshots = {name: path.read_bytes() for name, path in paths.items()}

            self.assertEqual(self._resample(tmp, "chat").returncode, 0)

            backups = sorted((tmp / "backups").iterdir())
            self.assertEqual(len(backups), 4, msg=[p.name for p in backups])
            # Each backup holds the pre-run bytes of its original.
            for name, original_bytes in snapshots.items():
                original = paths[name]
                match = [
                    b
                    for b in backups
                    if b.name.startswith(original.stem + ".") and b.suffix == original.suffix
                ]
                self.assertEqual(len(match), 1, msg=f"{name}: {[b.name for b in backups]}")
                self.assertEqual(match[0].read_bytes(), original_bytes)

    def test_nothing_is_written_when_the_backup_cannot_be_made(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            snapshots = {name: path.read_bytes() for name, path in paths.items()}

            blocker = tmp / "blocked-backups"
            blocker.write_text("a file where the backup dir should be\n", encoding="utf-8")
            result = self._resample(tmp, "chat", extra=["--backup-dir", str(blocker)])

            self.assertEqual(result.returncode, 1)
            self.assertIn("nothing written", result.stderr)
            for name, original_bytes in snapshots.items():
                self.assertEqual(paths[name].read_bytes(), original_bytes, msg=name)

    def test_no_versioned_or_suffixed_dataset_is_ever_created(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            self.assertEqual(self._resample(tmp, "chat").returncode, 0)
            self.assertEqual(self._resample(tmp, "faq").returncode, 0)

            in_dataset_dir = sorted(p.name for p in tmp.iterdir() if p.is_file())
            self.assertEqual(
                in_dataset_dir,
                [
                    "ground_truth.full.csv",
                    "ground_truth.full.json",
                    "ground_truth.ids.csv",
                    "ground_truth.ids.meta.json",
                ],
            )
            for name in in_dataset_dir:
                self.assertIsNone(_FORBIDDEN_NAME.search(name), msg=name)

    # -- sidecar provenance --------------------------------------------

    def test_sidecar_records_per_workload_seed_and_timestamp(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp, seed=42)
            self._annotate_everything(paths)
            first = json.loads(paths["meta"].read_text(encoding="utf-8"))
            for workload in WORKLOADS:
                self.assertEqual(first["workloads"][workload]["seed"], 42)
                self.assertRegex(
                    first["workloads"][workload]["sampled_at_utc"], r"\d{8}T\d{6}Z"
                )

            self.assertEqual(self._resample(tmp, "chat", seed=4242).returncode, 0)
            second = json.loads(paths["meta"].read_text(encoding="utf-8"))

            # The re-sampled workload carries the new seed; the others keep the
            # seed and timestamp of the run that actually produced their rows.
            self.assertEqual(second["workloads"]["chat"]["seed"], 4242)
            self.assertEqual(second["workloads"]["faq"]["seed"], 42)
            self.assertEqual(
                second["workloads"]["faq"]["sampled_at_utc"],
                first["workloads"]["faq"]["sampled_at_utc"],
            )
            self.assertNotIn("provenance_backfilled", second["workloads"]["faq"])
            self.assertEqual(second["sampled_workloads_this_run"], ["chat"])
            self.assertEqual(second["n_pairs"], self.N * len(WORKLOADS))
            for workload in WORKLOADS:
                self.assertEqual(second["workloads"][workload]["n_pairs"], self.N)

    def test_sidecar_written_before_per_workload_provenance_is_backfilled(self):
        """
        A sidecar from the previous tooling has no per-workload seed. Carrying it
        forward as-is would leave the untouched workloads with no recorded
        provenance at all, so it is filled from that sidecar's top level — which
        for a full-dataset run is exactly the right historical fact — and flagged
        as backfilled rather than passed off as an original per-workload record.
        """
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp, seed=42)
            self._annotate_everything(paths)

            legacy = json.loads(paths["meta"].read_text(encoding="utf-8"))
            for entry in legacy["workloads"].values():
                for key in ("seed", "n_per_workload", "positive_ratio", "sampled_at_utc"):
                    entry.pop(key, None)
            legacy.pop("generated_at_utc", None)
            paths["meta"].write_text(json.dumps(legacy, indent=2), encoding="utf-8")

            self.assertEqual(self._resample(tmp, "chat", seed=4242).returncode, 0)
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

        self.assertEqual(meta["workloads"]["faq"]["seed"], 42)
        self.assertTrue(meta["workloads"]["faq"]["provenance_backfilled"])
        # Unknown rather than invented: the old sidecar recorded no timestamp.
        self.assertIsNone(meta["workloads"]["faq"]["sampled_at_utc"])
        # The re-sampled workload's provenance is first-hand, not backfilled.
        self.assertEqual(meta["workloads"]["chat"]["seed"], 4242)
        self.assertNotIn("provenance_backfilled", meta["workloads"]["chat"])

    def test_sidecar_carries_no_query_text(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            self.assertEqual(self._resample(tmp, "chat").returncode, 0)
            body = paths["meta"].read_text(encoding="utf-8")
            ids_body = paths["ids"].read_text(encoding="utf-8")
            for pair in load_dataset(paths["csv"]):
                self.assertNotIn(pair.query_1, body)
                self.assertNotIn(pair.query_1, ids_body)

    def test_rehydration_still_reproduces_the_dataset_after_a_resample(self):
        """The ids file + sidecar stay a complete recipe for the spliced dataset."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            self._annotate_everything(paths)
            self.assertEqual(self._resample(tmp, "chat").returncode, 0)
            expected = paths["csv"].read_bytes()

            result = _run(
                "rehydrate_dataset.py",
                ["--ids", paths["ids"], "--raw-dir", CORPUS_FIXTURES,
                 "--out-csv", tmp / "rebuilt.csv", "--out-json", tmp / "rebuilt.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual((tmp / "rebuilt.csv").read_bytes(), expected)

    def test_full_run_without_the_flag_is_unchanged(self):
        """Backward compatibility: no --workload means sample all three afresh."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = self._sample_full(tmp)
            pairs = load_dataset(paths["csv"])
            self.assertEqual(len(pairs), self.N * len(WORKLOADS))
            self.assertEqual(
                sorted({p.workload for p in pairs}), sorted(WORKLOADS)
            )
            self.assertTrue(all(p.author_label is None for p in pairs))


if __name__ == "__main__":
    unittest.main()
