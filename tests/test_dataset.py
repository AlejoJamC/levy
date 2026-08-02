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

import pytest

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
    CorpusSource,
    CorpusSourceError,
    MockCorpusSource,
    QuoraQQPSource,
    SODDSource,
    TwitterPIT2015Source,
    make_source,
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

# Committed raw-corpus fixtures: the real column structure of each corpus with
# entirely synthetic content, laid out as a valid `--raw-dir` so the same tree
# also drives the CLI tests. Regenerate the parquet shards with
# `python tests/fixtures/corpora/make_sodd_fixture.py`.
CORPUS_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "corpora"
QQP_TSV = CORPUS_FIXTURES / "quora-qqp" / "train.tsv"
SODD_TRAIN = CORPUS_FIXTURES / "sodd" / "SODD_train.parquet.gzip"
SODD_DEV = CORPUS_FIXTURES / "sodd" / "SODD_dev.parquet.gzip"
PIT_TRAIN = CORPUS_FIXTURES / "twitter-pit2015" / "train.data"
PIT_DEV = CORPUS_FIXTURES / "twitter-pit2015" / "dev.data"
PIT_TEST = CORPUS_FIXTURES / "twitter-pit2015" / "test.data"


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

    def test_empty_query_2_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(query_2="   ")

    def test_query_text_invariant_is_not_relaxed(self):
        """
        Pins the non-empty-text invariant against future relaxation.

        The licence-safe distribution format (`DistributionRecord`) exists
        precisely because this invariant must hold: the LEV-4 replay path
        depends on a constructed `QueryPair` always having replayable text, so
        an identifiers-only dataset must be a different type, not a `QueryPair`
        with blanked columns.
        """
        for field in ("query_1", "query_2"):
            for value in ("", "   ", "\t\n"):
                with self.subTest(field=field, value=repr(value)):
                    with self.assertRaises(QueryPairValidationError):
                        _make_pair(**{field: value})

    def test_empty_pair_id_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(pair_id="")

    def test_empty_source_corpus_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(source_corpus="")

    def test_empty_source_pair_id_rejected(self):
        with self.assertRaises(QueryPairValidationError):
            _make_pair(source_pair_id="")

    def test_from_dict_invalid_field_value_wrapped(self):
        """A field that can't convert (e.g. non-integer label) raises QueryPairValidationError."""
        data = _make_pair().to_dict()
        data["original_label"] = "not-an-int"
        with self.assertRaises(QueryPairValidationError):
            QueryPair.from_dict(data)

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

    def test_load_json_rejects_malformed_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not valid json")
            with self.assertRaises(DatasetValidationError):
                load_json(path)

    def test_load_json_rejects_invalid_item(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps([{"pair_id": "faq-0001"}]))  # missing required fields
            with self.assertRaises(DatasetValidationError) as ctx:
                load_json(path)
            self.assertIn("item[0]", str(ctx.exception))

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

    def test_insufficient_negatives_raises(self):
        """Plenty of positives but too few negatives must name the negative shortfall."""
        source = MockCorpusSource(WORKLOAD_FAQ, n_candidates=20, seed=1)  # 10 pos, 10 neg
        with self.assertRaises(CorpusSourceError):
            sample_workload(source, n=30, seed=1, positive_ratio=0.1)  # needs 27 neg, only 10 available

    def test_sample_dataset_workload_mismatch_raises(self):
        """A source registered under the wrong workload key is rejected."""
        sources = {
            WORKLOAD_FAQ: MockCorpusSource(WORKLOAD_CODE, n_candidates=40, seed=1),  # wrong workload
            WORKLOAD_CODE: MockCorpusSource(WORKLOAD_CODE, n_candidates=40, seed=2),
            WORKLOAD_CHAT: MockCorpusSource(WORKLOAD_CHAT, n_candidates=40, seed=3),
        }
        with self.assertRaises(CorpusSourceError):
            sample_dataset(sources, n_per_workload=8, seed=42)

    def test_mock_corpus_source_rejects_unknown_workload(self):
        with self.assertRaises(CorpusSourceError):
            MockCorpusSource("not-a-workload")


# ---------------------------------------------------------------------------
# Real corpus source adapters (raw-file parsing, no network)
# ---------------------------------------------------------------------------

class _MinimalCorpusSource(CorpusSource):
    workload = WORKLOAD_FAQ
    name = "minimal"

    def iter_candidates(self):
        return super().iter_candidates()


class TestCorpusSourceAbstract(unittest.TestCase):

    def test_abstract_stub_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _MinimalCorpusSource().iter_candidates()


class TestQuoraQQPSource(unittest.TestCase):

    def test_parses_valid_tsv(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqp.tsv"
            path.write_text(
                "id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n"
                "1\t10\t11\tHow do I learn Python?\tHow can I learn Python?\t1\n"
                "2\t12\t13\tWhat is the capital of France?\tHow tall is Mount Everest?\t0\n"
                "3\t14\t15\t\tEmpty question one\t0\n",  # blank question1 -> skipped
                encoding="utf-8",
            )
            source = QuoraQQPSource(path)
            self.assertEqual(source.workload, "faq")
            candidates = list(source.iter_candidates())
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].label, 1)
        self.assertEqual(candidates[1].label, 0)

    def test_missing_columns_raises(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqp.tsv"
            path.write_text("id\tquestion1\n1\tHello\n", encoding="utf-8")
            source = QuoraQQPSource(path)
            with self.assertRaises(CorpusSourceError):
                list(source.iter_candidates())


    def test_rejects_label_outside_declared_domain(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqp.tsv"
            path.write_text(
                "id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n"
                "1\t10\t11\tA fixture question?\tAnother fixture question?\t7\n",
                encoding="utf-8",
            )
            with self.assertRaises(CorpusSourceError) as ctx:
                list(QuoraQQPSource(path).iter_candidates())
        self.assertIn("outside the declared domain", str(ctx.exception))

    def test_rejects_non_integer_label(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqp.tsv"
            path.write_text(
                "id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n"
                "1\t10\t11\tA fixture question?\tAnother fixture question?\tmaybe\n",
                encoding="utf-8",
            )
            with self.assertRaises(CorpusSourceError) as ctx:
                list(QuoraQQPSource(path).iter_candidates())
        self.assertIn("not an integer", str(ctx.exception))

    def test_check_fields_reports_missing_columns_without_raising(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqp.tsv"
            path.write_text("id\tquestion1\n1\tHello\n", encoding="utf-8")
            problems = QuoraQQPSource(path).check_fields()
        self.assertEqual(len(problems), 1)
        self.assertIn("missing columns", problems[0])

    def test_check_fields_reports_unreadable_file(self):
        problems = QuoraQQPSource(Path("/nonexistent/qqp.tsv")).check_fields()
        self.assertEqual(len(problems), 1)
        self.assertIn("cannot read", problems[0])


class TestSODDSource(unittest.TestCase):
    """SODD (code workload) — gzipped parquet, HTML posts, five label classes."""

    def test_parses_fixture_shards(self):
        source = SODDSource([SODD_TRAIN, SODD_DEV])
        self.assertEqual(source.workload, "code")
        self.assertEqual(source.name, "sodd")
        self.assertEqual(source.check_fields(), [])
        candidates = list(source.iter_candidates())

        # 10 duplicates + 10 different in train, 1 + 1 in dev. Classes 1, 2 and
        # 4 are in-domain but not in either pool with the default options.
        self.assertEqual(len(candidates), 22)
        self.assertEqual(sum(c.label for c in candidates), 11)

    def test_html_posts_are_normalised_and_keep_code(self):
        candidate = next(iter(SODDSource([SODD_TRAIN]).iter_candidates()))
        self.assertNotIn("<", candidate.query_1)
        self.assertIn("fixture_call_0(a, b)", candidate.query_1)  # code kept
        self.assertNotIn("  ", candidate.query_1)  # whitespace collapsed

    def test_source_pair_id_is_shard_scoped(self):
        candidates = list(SODDSource([SODD_TRAIN, SODD_DEV]).iter_candidates())
        shards = {c.source_pair_id.split(":")[0] for c in candidates}
        self.assertEqual(shards, {"SODD_train", "SODD_dev"})

    def test_hard_negatives_option_widens_the_negative_pool(self):
        default = SODDSource([SODD_TRAIN])
        hard = SODDSource([SODD_TRAIN], hard_negatives=True)

        self.assertEqual(default.label_mapping().negative, (3,))
        self.assertEqual(hard.label_mapping().negative, (1, 2, 3))
        n_default = sum(1 for c in default.iter_candidates() if c.label == 0)
        n_hard = sum(1 for c in hard.iter_candidates() if c.label == 0)
        self.assertEqual(n_hard, n_default + 2)  # the two "similar" fixtures

    def test_options_are_reported_for_the_manifest(self):
        options = SODDSource([SODD_TRAIN], hard_negatives=True).options()
        self.assertTrue(options["hard_negatives"])
        self.assertEqual(options["shards"], ["SODD_train.parquet.gzip"])
        self.assertIn("html.parser", options["normalisation"])

    def test_requires_at_least_one_shard(self):
        with self.assertRaises(CorpusSourceError):
            SODDSource([])

    def test_rejects_label_outside_declared_domain(self):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "SODD_bad.parquet.gzip"
            table = pa.table(
                {
                    "first_post": ["<p>fixture one</p>"],
                    "second_post": ["<p>fixture two</p>"],
                    "label": [9],  # outside 0..4
                }
            )
            pq.write_table(table, path, compression="gzip")
            with self.assertRaises(CorpusSourceError) as ctx:
                list(SODDSource([path]).iter_candidates())
        self.assertIn("outside the declared domain", str(ctx.exception))

    def test_missing_columns_reported_and_raised(self):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "SODD_thin.parquet.gzip"
            pq.write_table(pa.table({"first_post": ["x"]}), path, compression="gzip")
            source = SODDSource([path])
            self.assertIn("missing columns", source.check_fields()[0])
            with self.assertRaises(CorpusSourceError):
                list(source.iter_candidates())

    def test_unreadable_shard_is_reported_not_raised_by_check(self):
        source = SODDSource([Path("/nonexistent/SODD_train.parquet.gzip")])
        self.assertIn("cannot read", source.check_fields()[0])
        with self.assertRaises(CorpusSourceError):
            list(source.iter_candidates())


class TestTwitterPIT2015Source(unittest.TestCase):
    """PIT-2015 (chat workload) — vote-count labels, graded test split rejected."""

    def test_parses_fixture_splits(self):
        source = TwitterPIT2015Source([PIT_TRAIN, PIT_DEV])
        self.assertEqual(source.workload, "chat")
        self.assertEqual(source.check_fields(), [])
        candidates = list(source.iter_candidates())

        # 10 positive + 10 negative in train (the (2, 3) debatable row is
        # excluded), 3 + 3 in dev.
        self.assertEqual(len(candidates), 26)
        self.assertEqual(sum(c.label for c in candidates), 13)

    def test_debatable_vote_is_excluded_not_coerced(self):
        texts = {c.query_1 for c in TwitterPIT2015Source([PIT_TRAIN]).iter_candidates()}
        self.assertFalse(any("debatable" in t for t in texts))

    def test_vote_counts_map_to_binary_labels(self):
        mapping = TwitterPIT2015Source([PIT_TRAIN]).label_mapping()
        self.assertEqual(mapping.positive, (3, 4, 5))
        self.assertEqual(mapping.negative, (0, 1))
        self.assertIsNone(mapping.classify(2))  # debatable
        self.assertTrue(mapping.in_domain(5))
        self.assertFalse(mapping.in_domain(6))

    def test_source_pair_id_is_split_scoped(self):
        candidates = list(TwitterPIT2015Source([PIT_TRAIN, PIT_DEV]).iter_candidates())
        self.assertEqual(
            {c.source_pair_id.split(":")[0] for c in candidates}, {"train", "dev"}
        )

    def test_graded_test_split_is_rejected(self):
        source = TwitterPIT2015Source([PIT_TEST])
        problems = source.check_fields()
        self.assertEqual(len(problems), 1)
        self.assertIn("graded test split", problems[0])
        with self.assertRaises(CorpusSourceError) as ctx:
            list(source.iter_candidates())
        self.assertIn("not a PIT-2015 vote count", str(ctx.exception))

    def test_wrong_column_count_reported_and_raised(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.data"
            path.write_text("1\tTopic\tonly three columns\n", encoding="utf-8")
            source = TwitterPIT2015Source([path])
            self.assertIn("tab-separated columns", source.check_fields()[0])
            with self.assertRaises(CorpusSourceError):
                list(source.iter_candidates())

    def test_empty_file_reported(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.data"
            path.write_text("", encoding="utf-8")
            self.assertIn("is empty", TwitterPIT2015Source([path]).check_fields()[0])

    def test_unreadable_file_reported(self):
        source = TwitterPIT2015Source([Path("/nonexistent/train.data")])
        self.assertIn("cannot read", source.check_fields()[0])

    def test_requires_at_least_one_file(self):
        with self.assertRaises(CorpusSourceError):
            TwitterPIT2015Source([])

    def test_blank_lines_and_blank_sentences_are_skipped(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.data"
            path.write_text(
                "1\tT\tfixture a\tfixture b\t(4, 1)\ttag\ttag\n"
                "\n"
                "1\tT\t   \tfixture d\t(5, 0)\ttag\ttag\n",
                encoding="utf-8",
            )
            candidates = list(TwitterPIT2015Source([path]).iter_candidates())
        self.assertEqual(len(candidates), 1)


class TestMakeSource(unittest.TestCase):
    """The registry-driven adapter factory shared by sampling and rehydration."""

    def test_builds_each_adapter(self):
        self.assertIsInstance(make_source("QuoraQQPSource", [QQP_TSV]), QuoraQQPSource)
        self.assertIsInstance(make_source("SODDSource", [SODD_TRAIN]), SODDSource)
        self.assertIsInstance(
            make_source("TwitterPIT2015Source", [PIT_TRAIN]), TwitterPIT2015Source
        )

    def test_passes_adapter_options_through(self):
        source = make_source("SODDSource", [SODD_TRAIN], hard_negatives=True)
        self.assertTrue(source.hard_negatives)

    def test_unknown_adapter_named_not_guessed(self):
        with self.assertRaises(CorpusSourceError) as ctx:
            make_source("NotAnAdapter", [QQP_TSV])
        self.assertIn("NotAnAdapter", str(ctx.exception))

    def test_single_file_adapter_rejects_multiple_paths(self):
        with self.assertRaises(CorpusSourceError):
            make_source("QuoraQQPSource", [QQP_TSV, QQP_TSV])


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

    def test_unrecognized_answer_is_skipped(self):
        pairs = self._pairs()
        answers = iter(["maybe", "1", "0"])
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

    def test_eof_on_input_stops_early(self):
        """Piped stdin ending (EOFError) is treated like an early quit."""
        pairs = self._pairs()

        def _raise_eof(prompt):
            raise EOFError()

        with TemporaryDirectory() as tmp:
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=_raise_eof,
                output_fn=lambda msg: None,
            )
            summary = session.run()
        self.assertTrue(summary.quit_early)
        self.assertEqual(summary.newly_labeled, 0)

    def test_progress_entry_for_unknown_pair_id_is_ignored(self):
        """A progress file referencing a pair_id absent from `pairs` is skipped, not an error."""
        pairs = self._pairs()
        with TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.json"
            progress_path.write_text(json.dumps({"faq-9999": 1}), encoding="utf-8")
            session = BlindAnnotationSession(
                pairs,
                progress_path=progress_path,
                input_fn=lambda prompt: "0",
                output_fn=lambda msg: None,
            )
        self.assertTrue(all(p.author_label is None for p in pairs))

    def test_progress_merge_does_not_overwrite_existing_label(self):
        """A progress-file label for an already-labeled pair is ignored unless overwrite=True."""
        pairs = self._pairs()
        pairs[0].author_label = 1  # set directly, before the progress file is applied
        with TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.json"
            progress_path.write_text(json.dumps({"faq-0001": 0}), encoding="utf-8")
            BlindAnnotationSession(
                pairs,
                progress_path=progress_path,
                input_fn=lambda prompt: "0",
                output_fn=lambda msg: None,
                overwrite=False,
            )
        self.assertEqual(pairs[0].author_label, 1)  # untouched by the conflicting progress entry

    def test_remaining_count(self):
        pairs = self._pairs()
        pairs[0].author_label = 1
        with TemporaryDirectory() as tmp:
            session = BlindAnnotationSession(
                pairs,
                progress_path=Path(tmp) / "progress.json",
                input_fn=lambda prompt: "0",
                output_fn=lambda msg: None,
            )
        self.assertEqual(session.remaining_count(), 2)


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
            # An explicit empty raw dir, not the default `data/raw`: once the
            # author acquires the corpora locally, the default is populated and
            # this run would legitimately sample real data instead of falling
            # back to a synthetic source. The test is about the fallback, so it
            # has to own the condition that triggers it.
            empty_raw = Path(tmp) / "raw"
            empty_raw.mkdir()
            result = self._run(
                "sample_dataset.py",
                [
                    "--raw-dir", str(empty_raw),
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
