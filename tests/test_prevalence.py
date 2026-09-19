"""
Tests for the source-pool duplicate prevalence measurement (LEV-21):
`levy/dataset/prevalence.py`, the `native_label_counts()` adapter methods it
relies on, and `scripts/measure_prevalence.py`.

The fixtures are built here, row by row, and every expected number in the
assertions is counted by hand from the rows written — not computed with the
code under test. Prevalence A (positives over the full filtered pool, excluded
band included) and prevalence B (positives over the binary-eligible pool) are
pinned separately, and each fixture is built so the two differ.

All offline; no query text from a real corpus.
"""

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from levy.dataset.prevalence import PoolCounts, measure_pool
from levy.dataset.sampling import (
    CorpusSource,
    CorpusSourceError,
    MockCorpusSource,
    QuoraQQPSource,
    RawCandidatePair,
    SODDSource,
    TwitterPIT2015Source,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "measure_prevalence.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("measure_prevalence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_qqp(path: Path) -> Path:
    """
    QQP fixture, hand-counted:
        is_duplicate=1, both questions present  -> 3 positive
        is_duplicate=0, both questions present  -> 5 negative
        is_duplicate=0, first question blank    -> census negative, dropped by the adapter
    QQP has no excluded band.
    """
    lines = ["id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate"]
    n = 0
    for label, count in (("1", 3), ("0", 5)):
        for _ in range(count):
            n += 1
            lines.append(f"{n}\t{n}\t{n + 100}\tq{n} first\tq{n} second\t{label}")
    n += 1
    lines.append(f"{n}\t{n}\t{n + 100}\t\tq{n} second\t0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_pit(path: Path) -> Path:
    """
    PIT-2015 fixture, hand-counted by yes-votes:
        5 x1, 4 x2, 3 x1 -> 4 positive
        1 x2, 0 x3       -> 5 negative
        2 x3             -> 3 excluded (debatable band)
    """
    votes = [5] + [4] * 2 + [3] + [1] * 2 + [0] * 3 + [2] * 3
    lines = []
    for i, yes in enumerate(votes):
        lines.append(
            "\t".join(
                [str(i), "Topic", f"sent {i} a", f"sent {i} b", f"({yes}, {5 - yes})", "t", "t"]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_sodd(path: Path) -> Path:
    """
    SODD fixture, hand-counted by native label:
        0 x2                      -> 2 positive
        3 x5 (one with blank post)-> 4 negative + 1 dropped for empty text
        1 x3, 2 x2, 4 x1          -> 6 excluded
    """
    rows = []

    def add(label, first="<p>first post</p>", second="<p>second post</p>"):
        rows.append(
            {
                "first_post": first,
                "second_post": second,
                "first_author": "u",
                "second_author": "v",
                "label": label,
                "page": "stackoverflow",
            }
        )

    for _ in range(2):
        add(0)
    for _ in range(4):
        add(3)
    add(3, first="")
    for _ in range(3):
        add(1)
    for _ in range(2):
        add(2)
    add(4)
    pd.DataFrame(rows).to_parquet(path, compression="gzip")
    return path


class TestMeasurePoolHandCounted(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_qqp_no_excluded_band_so_denominators_coincide(self):
        counts = measure_pool(QuoraQQPSource(write_qqp(self.tmp / "train.tsv")))
        self.assertEqual(
            (counts.positive, counts.negative, counts.excluded), (3, 5, 0)
        )
        self.assertEqual(counts.dropped_empty_text_positive, 0)
        self.assertEqual(counts.dropped_empty_text_negative, 1)
        self.assertEqual(counts.eligible, 8)
        self.assertEqual(counts.pool_size, 8)
        self.assertEqual(counts.prevalence_a, 3 / 8)
        self.assertEqual(counts.prevalence_b, 3 / 8)
        self.assertEqual(counts.native_label_counts, {0: 6, 1: 3})

    def test_pit_excluded_band_separates_the_denominators(self):
        counts = measure_pool(TwitterPIT2015Source([write_pit(self.tmp / "train.data")]))
        self.assertEqual(
            (counts.positive, counts.negative, counts.excluded), (4, 5, 3)
        )
        self.assertEqual(counts.eligible, 9)
        self.assertEqual(counts.pool_size, 12)
        self.assertEqual(counts.prevalence_a, 4 / 12)
        self.assertEqual(counts.prevalence_b, 4 / 9)
        self.assertNotEqual(counts.prevalence_a, counts.prevalence_b)
        self.assertEqual(
            counts.native_label_counts, {0: 3, 1: 2, 2: 3, 3: 1, 4: 2, 5: 1}
        )

    def test_sodd_excluded_band_dominates_the_pool(self):
        counts = measure_pool(SODDSource([write_sodd(self.tmp / "SODD_train.parquet.gzip")]))
        self.assertEqual(
            (counts.positive, counts.negative, counts.excluded), (2, 4, 6)
        )
        self.assertEqual(counts.dropped_empty_text_positive, 0)
        self.assertEqual(counts.dropped_empty_text_negative, 1)
        self.assertEqual(counts.eligible, 6)
        self.assertEqual(counts.pool_size, 12)
        self.assertEqual(counts.prevalence_a, 2 / 12)
        self.assertEqual(counts.prevalence_b, 2 / 6)
        self.assertEqual(counts.native_label_counts, {0: 2, 1: 3, 2: 2, 3: 5, 4: 1})

    def test_sodd_hard_negatives_moves_bands_1_and_2_into_the_negative_class(self):
        source = SODDSource(
            [write_sodd(self.tmp / "SODD_train.parquet.gzip")], hard_negatives=True
        )
        counts = measure_pool(source)
        # negatives: label 3 (4 with text) + label 1 (3) + label 2 (2); excluded: label 4 only
        self.assertEqual(
            (counts.positive, counts.negative, counts.excluded), (2, 9, 1)
        )
        self.assertEqual(counts.prevalence_a, 2 / 12)
        self.assertEqual(counts.prevalence_b, 2 / 11)


class TestMeasurePoolEdges(unittest.TestCase):
    def test_mock_source_uses_the_default_census(self):
        counts = measure_pool(MockCorpusSource("faq", n_candidates=40))
        self.assertEqual(
            (counts.positive, counts.negative, counts.excluded), (20, 20, 0)
        )
        self.assertEqual(counts.prevalence_a, 0.5)
        self.assertEqual(counts.prevalence_b, 0.5)

    def test_empty_pool_reports_none_not_zero_division(self):
        empty = PoolCounts("faq", "x", 0, 0, 0, 0, 0)
        self.assertIsNone(empty.prevalence_a)
        self.assertIsNone(empty.prevalence_b)
        only_excluded = PoolCounts("code", "x", 0, 0, 5, 0, 0)
        self.assertEqual(only_excluded.prevalence_a, 0.0)
        self.assertIsNone(only_excluded.prevalence_b)

    def test_census_smaller_than_candidates_is_an_error(self):
        class Inconsistent(CorpusSource):
            workload = "faq"
            name = "inconsistent"

            def iter_candidates(self):
                yield RawCandidatePair("1", "a", "b", 1)

            def native_label_counts(self):
                return {0: 4}

        with self.assertRaisesRegex(ValueError, "smaller than the candidate stream"):
            measure_pool(Inconsistent())


class TestNativeLabelCountsValidation(unittest.TestCase):
    """The census applies the same corpus-format validation as `iter_candidates`."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_qqp_missing_column(self):
        path = self.tmp / "train.tsv"
        path.write_text("id\tquestion1\n1\tq\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusSourceError, "missing columns"):
            QuoraQQPSource(path).native_label_counts()

    def test_qqp_non_integer_label(self):
        path = self.tmp / "train.tsv"
        path.write_text(
            "id\tquestion1\tquestion2\tis_duplicate\n1\tq\tr\tmaybe\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CorpusSourceError, "not an integer"):
            QuoraQQPSource(path).native_label_counts()

    def test_qqp_label_outside_domain(self):
        path = self.tmp / "train.tsv"
        path.write_text(
            "id\tquestion1\tquestion2\tis_duplicate\n1\tq\tr\t7\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CorpusSourceError, "outside the declared domain"):
            QuoraQQPSource(path).native_label_counts()

    def test_sodd_label_outside_domain(self):
        path = self.tmp / "SODD_train.parquet.gzip"
        pd.DataFrame(
            [{"first_post": "a", "second_post": "b", "label": 9}]
        ).to_parquet(path, compression="gzip")
        with self.assertRaisesRegex(CorpusSourceError, "outside the declared domain"):
            SODDSource([path]).native_label_counts()

    def test_sodd_missing_label_column(self):
        path = self.tmp / "SODD_train.parquet.gzip"
        pd.DataFrame([{"first_post": "a", "second_post": "b"}]).to_parquet(
            path, compression="gzip"
        )
        with self.assertRaisesRegex(CorpusSourceError, "missing columns"):
            SODDSource([path]).native_label_counts()

    def test_sodd_unreadable_file(self):
        path = self.tmp / "SODD_train.parquet.gzip"
        path.write_text("not parquet", encoding="utf-8")
        with self.assertRaisesRegex(CorpusSourceError, "cannot read SODD parquet"):
            SODDSource([path]).native_label_counts()

    def test_pit_graded_test_split_is_rejected(self):
        path = self.tmp / "test.data"
        path.write_text("1\tT\ta\tb\t4\tt\tt\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusSourceError, "not a PIT-2015 vote count"):
            TwitterPIT2015Source([path]).native_label_counts()

    def test_pit_wrong_column_count(self):
        path = self.tmp / "train.data"
        path.write_text("1\tT\ta\tb\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusSourceError, "expected 7 tab-separated"):
            TwitterPIT2015Source([path]).native_label_counts()

    def test_pit_vote_outside_domain(self):
        path = self.tmp / "train.data"
        path.write_text("1\tT\ta\tb\t(9, 0)\tt\tt\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusSourceError, "outside the declared domain"):
            TwitterPIT2015Source([path]).native_label_counts()

    def test_pit_blank_lines_are_not_counted(self):
        path = self.tmp / "train.data"
        path.write_text("\n1\tT\ta\tb\t(4, 1)\tt\tt\n\n", encoding="utf-8")
        self.assertEqual(TwitterPIT2015Source([path]).native_label_counts(), {4: 1})


class TestMeasurePrevalenceScript(unittest.TestCase):
    """End to end over the hand-counted fixtures, with a registry pinned to them."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        raw = self.tmp / "raw"
        for name in ("quora-qqp", "sodd", "twitter-pit2015"):
            (raw / name).mkdir(parents=True)
        self.files = {
            "quora-qqp": write_qqp(raw / "quora-qqp" / "train.tsv"),
            "sodd": write_sodd(raw / "sodd" / "SODD_train.parquet.gzip"),
            "twitter-pit2015": write_pit(raw / "twitter-pit2015" / "train.data"),
        }
        self.raw = raw
        self.script = _load_script()

    def _sha(self, corpus):
        return hashlib.sha256(self.files[corpus].read_bytes()).hexdigest()

    def _registry(self, sodd_sha=None):
        def entry(key, workload, adapter):
            return {
                "title": key,
                "workload": workload,
                "adapter": adapter,
                "snapshot": "fixture",
                "canonical_url": "https://example.invalid",
                "licence": "fixture",
                "citation": "fixture",
                "files": [
                    {
                        "filename": self.files[key].name,
                        "url": "https://example.invalid/f",
                        "sha256": (sodd_sha if key == "sodd" and sodd_sha else self._sha(key)),
                    }
                ],
            }

        path = self.tmp / "corpora.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "corpora": {
                        "quora-qqp": entry("quora-qqp", "faq", "QuoraQQPSource"),
                        "sodd": entry("sodd", "code", "SODDSource"),
                        "twitter-pit2015": entry(
                            "twitter-pit2015", "chat", "TwitterPIT2015Source"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def _sidecar(self, sodd_mapping=None):
        path = self.tmp / "ids.meta.json"
        path.write_text(
            json.dumps(
                {
                    "workloads": {
                        "faq": {
                            "corpus": "quora-qqp",
                            "options": {},
                            "seed": 42,
                            "label_mapping": {"positive": [1], "negative": [0], "domain": [0, 1]},
                        },
                        "code": {
                            "corpus": "sodd",
                            "options": {"hard_negatives": False, "shards": ["SODD_train.parquet.gzip"]},
                            "seed": 8484,
                            "label_mapping": sodd_mapping
                            or {"positive": [0], "negative": [3], "domain": [0, 1, 2, 3, 4]},
                        },
                        "chat": {
                            "corpus": "twitter-pit2015",
                            "options": {"splits": ["train"]},
                            "seed": 4242,
                            "label_mapping": {
                                "positive": [3, 4, 5],
                                "negative": [0, 1],
                                "domain": [0, 1, 2, 3, 4, 5],
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def _run(self, out, registry=None, sidecar=None):
        return self.script.main(
            [
                "--meta", str(sidecar or self._sidecar()),
                "--raw-dir", str(self.raw),
                "--registry", str(registry or self._registry()),
                "--out-dir", str(out),
            ]
        )

    def test_writes_the_hand_counted_rows_in_workload_order(self):
        out = self.tmp / "out"
        self.assertEqual(self._run(out), 0)
        lines = (out / "prevalence.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(self.script.CSV_COLUMNS))
        self.assertEqual(
            lines[1:],
            [
                "faq,quora-qqp,8,3,5,0,8,0,1,0.375000,0.375000",
                "code,sodd,12,2,4,6,6,0,1,0.166667,0.333333",
                "chat,twitter-pit2015,12,4,5,3,9,0,0,0.333333,0.444444",
            ],
        )

    def test_meta_records_provenance_and_native_counts(self):
        out = self.tmp / "out"
        self._run(out)
        meta = json.loads((out / "prevalence_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["workloads"]["code"]["sampling_seed"], 8484)
        self.assertEqual(
            meta["workloads"]["code"]["native_label_counts"],
            {"0": 2, "1": 3, "2": 2, "3": 5, "4": 1},
        )
        self.assertEqual(
            meta["workloads"]["code"]["files"],
            [{"filename": "SODD_train.parquet.gzip", "sha256": self._sha("sodd")}],
        )
        self.assertIn("boundary", meta)
        self.assertIn("recomputed", meta["pool_counts_source"])

    def test_rerun_is_byte_identical_and_backs_up_the_previous_output(self):
        out = self.tmp / "out"
        self._run(out)
        first = (out / "prevalence.csv").read_bytes()
        self.assertFalse((out / "backups").exists())
        self._run(out)
        self.assertEqual((out / "prevalence.csv").read_bytes(), first)
        backups = sorted(p.name for p in (out / "backups").iterdir())
        self.assertEqual(len(backups), 2)
        self.assertTrue(any(name.startswith("prevalence.") and name.endswith(".csv") for name in backups))

    def test_checksum_mismatch_writes_nothing(self):
        out = self.tmp / "out"
        self.assertEqual(self._run(out, registry=self._registry(sodd_sha="0" * 64)), 1)
        self.assertFalse(out.exists())

    def test_sidecar_mapping_disagreement_writes_nothing(self):
        out = self.tmp / "out"
        wrong = {"positive": [0], "negative": [1, 2, 3], "domain": [0, 1, 2, 3, 4]}
        self.assertEqual(self._run(out, sidecar=self._sidecar(sodd_mapping=wrong)), 1)
        self.assertFalse(out.exists())

    def test_missing_corpus_files_writes_nothing(self):
        out = self.tmp / "out"
        self.files["sodd"].unlink()
        self.assertEqual(self._run(out, registry=self._registry(sodd_sha="0" * 64)), 1)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
