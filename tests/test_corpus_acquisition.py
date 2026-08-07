"""
Tests for the corpus-acquisition capability (LEV-12): the provenance registry,
pre-flight validation, the identifiers-only distribution format, and the
rehydration round-trip.

Everything here runs fully offline against the committed corpus fixtures in
`tests/fixtures/corpora/` — that directory is laid out as a valid `--raw-dir`,
so the CLIs can be exercised end to end without any real corpus. Acquisition
itself (`scripts/fetch_corpora.py`) is the one network-touching entry point in
the repository and is deliberately not exercised here; `TestAcquisitionIsOutOfBand`
below is the guard that keeps it that way.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from levy.dataset.corpora import (
    CorpusRegistryError,
    load_registry,
    pin_checksums,
    sha256_file,
)
from levy.dataset.normalize import NORMALISATION_RULE, normalize_html
from levy.dataset.io import (
    DatasetValidationError,
    load_distribution_csv,
    load_dataset,
    load_json,
    save_distribution_csv,
    to_distribution_records,
)
from levy.dataset.sampling import (
    CorpusSourceError,
    MockCorpusSource,
    QuoraQQPSource,
    SODDSource,
    TwitterPIT2015Source,
)
from levy.dataset.schema import (
    DistributionRecord,
    QueryPair,
    QueryPairValidationError,
    WORKLOAD_CHAT,
    WORKLOAD_CODE,
    WORKLOAD_FAQ,
)
from levy.dataset.validation import (
    CHECK_CHECKSUM,
    CHECK_LABELS,
    CHECK_OVERLAP,
    CHECK_POOL,
    CHECK_PRESENCE,
    validate_sources,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "data" / "corpora.json"
CORPUS_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "corpora"
QQP_TSV = CORPUS_FIXTURES / "quora-qqp" / "train.tsv"
SODD_TRAIN = CORPUS_FIXTURES / "sodd" / "SODD_train.parquet.gzip"
SODD_DEV = CORPUS_FIXTURES / "sodd" / "SODD_dev.parquet.gzip"
PIT_TRAIN = CORPUS_FIXTURES / "twitter-pit2015" / "train.data"
PIT_DEV = CORPUS_FIXTURES / "twitter-pit2015" / "dev.data"


def _real_sources(hard_negatives: bool = False):
    return {
        WORKLOAD_FAQ: QuoraQQPSource(QQP_TSV),
        WORKLOAD_CODE: SODDSource([SODD_TRAIN, SODD_DEV], hard_negatives=hard_negatives),
        WORKLOAD_CHAT: TwitterPIT2015Source([PIT_TRAIN, PIT_DEV]),
    }


def _write_registry(path: Path, mutate=None) -> Path:
    """
    A copy of the committed registry with **every checksum unpinned**, for use
    against `tests/fixtures/corpora/`.

    Unpinning is not incidental. The committed registry describes the real
    corpora, and once the author has run `fetch_corpora.py --pin` it carries
    their checksums — which the fixtures, being synthetic, will never match.
    A test that reused those pins would be asserting that a fixture is the real
    corpus, and would flip from passing to failing the day the author pins.
    So the fixture registry starts unpinned, and the tests that care about
    pinning establish it explicitly.
    """
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for entry in raw["corpora"].values():
        for corpus_file in entry["files"]:
            corpus_file["sha256"] = None
    if mutate is not None:
        mutate(raw)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def _unpinned_registry():
    """The committed registry as an in-memory object, with no checksums pinned."""
    registry = load_registry(REGISTRY)
    return replace(
        registry,
        entries={
            key: replace(
                entry, files=tuple(replace(f, sha256=None) for f in entry.files)
            )
            for key, entry in registry.entries.items()
        },
    )


# ---------------------------------------------------------------------------
# Provenance registry
# ---------------------------------------------------------------------------

class TestCorpusRegistry(unittest.TestCase):

    def test_committed_registry_loads_and_covers_every_workload(self):
        registry = load_registry(REGISTRY)
        self.assertEqual(
            sorted(registry.keys()), ["quora-qqp", "sodd", "twitter-pit2015"]
        )
        for workload, key in (("faq", "quora-qqp"), ("code", "sodd"), ("chat", "twitter-pit2015")):
            self.assertEqual(registry.for_workload(workload).key, key)

    def test_every_entry_records_the_provenance_the_datasheet_needs(self):
        for entry in load_registry(REGISTRY).entries.values():
            with self.subTest(corpus=entry.key):
                self.assertTrue(entry.canonical_url.startswith("http"))
                self.assertTrue(entry.licence)
                self.assertTrue(entry.citation)
                self.assertTrue(entry.snapshot)
                self.assertTrue(entry.files)
                # Whether a checksum is pinned depends on whether the author
                # has run a real acquisition, so it is not asserted either way.
                # What must hold is that a pin, when present, is a SHA-256.
                for corpus_file in entry.files:
                    if corpus_file.sha256 is not None:
                        self.assertRegex(corpus_file.sha256, r"^[0-9a-f]{64}$")

    def test_unknown_corpus_is_named_not_guessed(self):
        registry = load_registry(REGISTRY)
        with self.assertRaises(CorpusRegistryError) as ctx:
            registry.get("no-such-corpus")
        self.assertIn("no-such-corpus", str(ctx.exception))
        self.assertIn("known corpora", str(ctx.exception))

    def test_unknown_workload_is_reported(self):
        with self.assertRaises(CorpusRegistryError):
            load_registry(REGISTRY).for_workload("nope")

    def test_ambiguous_workload_is_reported(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json",
                lambda raw: raw["corpora"]["sodd"].update({"workload": "faq"}),
            )
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path).for_workload("faq")
        self.assertIn("more than one corpus", str(ctx.exception))

    def test_missing_file_reports_path(self):
        with self.assertRaises(CorpusRegistryError) as ctx:
            load_registry(Path("/nonexistent/corpora.json"))
        self.assertIn("cannot read", str(ctx.exception))

    def test_invalid_json_reported(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_non_object_top_level_reported(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(CorpusRegistryError):
                load_registry(path)

    def test_missing_corpora_key_reported(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.json"
            path.write_text('{"schema_version": 1}', encoding="utf-8")
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
        self.assertIn("'corpora'", str(ctx.exception))

    def test_missing_entry_field_names_corpus_and_field(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json", lambda raw: raw["corpora"]["sodd"].pop("licence")
            )
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
        self.assertIn("sodd", str(ctx.exception))
        self.assertIn("licence", str(ctx.exception))

    def test_missing_file_field_names_corpus_and_index(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json",
                lambda raw: raw["corpora"]["quora-qqp"]["files"][0].pop("sha256"),
            )
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
        self.assertIn("files[0]", str(ctx.exception))
        self.assertIn("sha256", str(ctx.exception))

    def test_empty_files_list_rejected(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json", lambda raw: raw["corpora"]["sodd"].update({"files": []})
            )
            with self.assertRaises(CorpusRegistryError):
                load_registry(path)

    def test_non_manual_corpus_without_url_rejected(self):
        def mutate(raw):
            raw["corpora"]["twitter-pit2015"]["files"][0]["url"] = None

        with TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp) / "reg.json", mutate)
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
        self.assertIn("not marked manual", str(ctx.exception))

    def test_empty_filename_rejected(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json",
                lambda raw: raw["corpora"]["sodd"]["files"][0].update({"filename": ""}),
            )
            with self.assertRaises(CorpusRegistryError):
                load_registry(path)

    def test_paths_and_missing_paths_use_the_raw_root(self):
        entry = load_registry(REGISTRY).get("sodd")
        self.assertEqual(
            [p.name for p in entry.paths(CORPUS_FIXTURES)],
            ["SODD_train.parquet.gzip", "SODD_dev.parquet.gzip"],
        )
        self.assertEqual(entry.missing_paths(CORPUS_FIXTURES), [])
        self.assertEqual(len(entry.missing_paths(Path("/nonexistent"))), 2)

    def test_provenance_carries_no_query_text_fields(self):
        provenance = load_registry(REGISTRY).get("sodd").provenance()
        self.assertEqual(
            set(provenance), {"title", "snapshot", "canonical_url", "licence", "files"}
        )

    def test_acquisition_url_prefers_the_download_location(self):
        registry = load_registry(REGISTRY)
        sodd = registry.get("sodd")
        self.assertEqual(sodd.acquisition_url, sodd.download_url)
        qqp = registry.get("quora-qqp")
        self.assertEqual(qqp.acquisition_url, qqp.canonical_url)


class TestChecksums(unittest.TestCase):

    def test_sha256_matches_hashlib(self):
        import hashlib

        expected = hashlib.sha256(QQP_TSV.read_bytes()).hexdigest()
        self.assertEqual(sha256_file(QQP_TSV), expected)

    def test_pin_records_checksums_of_present_files_only(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp) / "reg.json")
            pinned = pin_checksums(path, CORPUS_FIXTURES)

            self.assertEqual(len(pinned), 5)
            self.assertEqual(pinned["quora-qqp/train.tsv"], sha256_file(QQP_TSV))
            reloaded = load_registry(path)
            self.assertTrue(all(f.is_pinned for f in reloaded.get("sodd").files))

    def test_pin_leaves_already_pinned_entries_alone(self):
        sentinel = "0" * 64

        def mutate(raw):
            raw["corpora"]["quora-qqp"]["files"][0]["sha256"] = sentinel

        with TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp) / "reg.json", mutate)
            pinned = pin_checksums(path, CORPUS_FIXTURES)

            self.assertNotIn("quora-qqp/train.tsv", pinned)
            self.assertEqual(load_registry(path).get("quora-qqp").files[0].sha256, sentinel)

    def test_pin_with_nothing_present_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp) / "reg.json")
            before = path.read_text(encoding="utf-8")
            self.assertEqual(pin_checksums(path, Path(tmp) / "empty-raw"), {})
            self.assertEqual(path.read_text(encoding="utf-8"), before)


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_clean_inputs_pass_and_report_pool_sizes(self):
        report = validate_sources(_real_sources(), n_per_workload=6, registry=_unpinned_registry())
        self.assertTrue(report.ok, msg=report.render())
        self.assertEqual(set(report.pools), {"faq", "code", "chat"})
        self.assertEqual(report.pools["faq"].positive_available, 10)
        self.assertTrue(report.pools["faq"].sufficient)

    def test_unpinned_checksums_are_notes_not_findings(self):
        report = validate_sources(_real_sources(), n_per_workload=6, registry=_unpinned_registry())
        self.assertTrue(report.ok)
        self.assertTrue(any("no pinned checksum" in note for note in report.notes))

    def test_two_simultaneous_problems_are_both_reported(self):
        """One pass, every problem — not 'fix one, rerun, find the next'."""
        sources = _real_sources()
        sources[WORKLOAD_FAQ] = QuoraQQPSource(Path("/nonexistent/train.tsv"))
        report = validate_sources(sources, n_per_workload=1000, registry=_unpinned_registry())

        checks = {f.check for f in report.findings}
        workloads = {f.workload for f in report.findings}
        self.assertIn(CHECK_PRESENCE, checks)
        self.assertIn(CHECK_POOL, checks)
        # The missing faq file did not stop code and chat from being measured.
        self.assertEqual(workloads, {"faq", "code", "chat"})

    def test_pool_shortfall_reported_for_all_workloads_at_once(self):
        report = validate_sources(_real_sources(), n_per_workload=1000, registry=_unpinned_registry())
        self.assertFalse(report.ok)
        short = {f.workload for f in report.findings if f.check == CHECK_POOL}
        self.assertEqual(short, {"faq", "code", "chat"})
        self.assertIn("needs 500 positive pairs", report.render())

    def test_cross_workload_corpus_overlap_rejected(self):
        sources = _real_sources()
        sources[WORKLOAD_CHAT] = QuoraQQPSource(QQP_TSV)  # faq's corpus, reused
        report = validate_sources(sources, n_per_workload=4, registry=_unpinned_registry())

        overlap = [f for f in report.findings if f.check == CHECK_OVERLAP]
        self.assertEqual(len(overlap), 1)
        self.assertIn("quora-qqp", overlap[0].message)
        self.assertIn("workload contrast", overlap[0].message)

    def test_independent_mock_sources_are_not_an_overlap(self):
        """Three synthetic pools share a display name but not a corpus."""
        sources = {w: MockCorpusSource(w, n_candidates=20) for w in ("faq", "code", "chat")}
        report = validate_sources(sources, n_per_workload=4, registry=_unpinned_registry())
        self.assertTrue(report.ok, msg=report.render())

    def test_unexpected_label_value_reported_not_coerced(self):
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "train.tsv"
            bad.write_text(
                "id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n"
                "1\t1\t2\tfixture one\tfixture two\t7\n",
                encoding="utf-8",
            )
            sources = _real_sources()
            sources[WORKLOAD_FAQ] = QuoraQQPSource(bad)
            report = validate_sources(sources, n_per_workload=4, registry=_unpinned_registry())

        labels = [f for f in report.findings if f.check == CHECK_LABELS]
        self.assertEqual(len(labels), 1)
        self.assertIn("7", labels[0].message)
        # The other two workloads were still measured in the same pass.
        self.assertEqual(set(report.pools), {"code", "chat"})

    def test_checksum_mismatch_is_a_finding(self):
        def mutate(raw):
            raw["corpora"]["quora-qqp"]["files"][0]["sha256"] = "0" * 64

        with TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp) / "reg.json", mutate)
            report = validate_sources(
                _real_sources(), n_per_workload=4, registry=load_registry(path)
            )

        mismatches = [f for f in report.findings if f.check == CHECK_CHECKSUM]
        self.assertEqual(len(mismatches), 1)
        self.assertIn("does not match the pinned", mismatches[0].message)
        # A file that failed its checksum is not then scanned for pool sizes.
        self.assertNotIn("faq", report.pools)

    def test_fully_pinned_registry_validates_clean(self):
        """The post-`--pin` state: every checksum matches, no notes, no findings."""
        with TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp) / "reg.json")
            pin_checksums(path, CORPUS_FIXTURES)
            report = validate_sources(
                _real_sources(), n_per_workload=6, registry=load_registry(path)
            )
        self.assertTrue(report.ok, msg=report.render())
        self.assertEqual(report.notes, [])
        self.assertEqual(set(report.pools), {"faq", "code", "chat"})

    def test_missing_required_columns_reported(self):
        with TemporaryDirectory() as tmp:
            thin = Path(tmp) / "train.tsv"
            thin.write_text("id\tquestion1\n1\tfixture\n", encoding="utf-8")
            sources = _real_sources()
            sources[WORKLOAD_FAQ] = QuoraQQPSource(thin)
            report = validate_sources(sources, n_per_workload=4, registry=_unpinned_registry())
        self.assertTrue(any("missing columns" in f.message for f in report.findings))

    def test_count_pools_false_skips_the_corpus_scan(self):
        report = validate_sources(
            _real_sources(), n_per_workload=1000, registry=_unpinned_registry(), count_pools=False
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.pools, {})

    def test_unreadable_registry_becomes_a_finding_not_an_exception(self):
        sources = {w: MockCorpusSource(w, n_candidates=20) for w in ("faq", "code", "chat")}
        report = validate_sources(sources, n_per_workload=4, registry=None, raw_root=None)
        # The committed registry does load, so this simply must not explode.
        self.assertIsNotNone(report.render())

    def test_finding_renders_with_its_location(self):
        report = validate_sources(
            {**_real_sources(), WORKLOAD_FAQ: QuoraQQPSource(Path("/nonexistent/x.tsv"))},
            n_per_workload=4,
            registry=_unpinned_registry(),
        )
        rendered = [f.render() for f in report.findings]
        self.assertTrue(any("faq/quora-qqp" in line for line in rendered))


# ---------------------------------------------------------------------------
# Identifiers-only distribution format
# ---------------------------------------------------------------------------

def _pair(**overrides) -> QueryPair:
    defaults = dict(
        pair_id="faq-0001",
        workload=WORKLOAD_FAQ,
        source_corpus="quora-qqp",
        source_pair_id="42",
        query_1="fixture question one?",
        query_2="fixture question two?",
        original_label=1,
        author_label=None,
        metadata={"note": "fixture"},
    )
    defaults.update(overrides)
    return QueryPair(**defaults)


class TestDistributionFormat(unittest.TestCase):

    def test_round_trip_preserves_records_and_order(self):
        pairs = [
            _pair(pair_id="faq-0001"),
            _pair(pair_id="faq-0002", original_label=0, author_label=1, metadata={}),
        ]
        records = to_distribution_records(pairs)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.csv"
            save_distribution_csv(records, path)
            self.assertEqual(load_distribution_csv(path), records)
            self.assertEqual([r.pair_id for r in load_distribution_csv(path)],
                             ["faq-0001", "faq-0002"])

    def test_no_query_text_in_any_field(self):
        pairs = load_json(REPO_ROOT / "data" / "ground_truth.json")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.csv"
            save_distribution_csv(to_distribution_records(pairs), path)
            body = path.read_text(encoding="utf-8")
        for pair in pairs:
            self.assertNotIn(pair.query_1, body)
            self.assertNotIn(pair.query_2, body)
        self.assertNotIn("query_1", body)
        self.assertNotIn("query_2", body)

    def test_absent_author_label_decodes_to_none(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.csv"
            save_distribution_csv([DistributionRecord.from_query_pair(_pair())], path)
            self.assertIsNone(load_distribution_csv(path)[0].author_label)

    def test_missing_column_reported_with_file_name(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.csv"
            path.write_text("pair_id,workload\nfaq-0001,faq\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError) as ctx:
                load_distribution_csv(path)
        self.assertIn(str(path), str(ctx.exception))
        self.assertIn("source_corpus", str(ctx.exception))

    def test_invalid_row_reported_with_row_number(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.csv"
            save_distribution_csv([DistributionRecord.from_query_pair(_pair())], path)
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].replace("faq-0001,faq", "faq-0001,not-a-workload")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError) as ctx:
                load_distribution_csv(path)
        self.assertIn(":2:", str(ctx.exception))

    def test_record_validates_its_identifiers_and_labels(self):
        for bad in (
            {"pair_id": ""},
            {"workload": "nope"},
            {"source_corpus": ""},
            {"source_pair_id": ""},
            {"original_label": 7},
            {"author_label": 7},
        ):
            with self.subTest(**bad):
                with self.assertRaises(QueryPairValidationError):
                    DistributionRecord.from_query_pair(_pair()).__class__(
                        **{**DistributionRecord.from_query_pair(_pair()).to_dict(), **bad}
                    )

    def test_from_dict_reports_a_missing_field(self):
        with self.assertRaises(QueryPairValidationError) as ctx:
            DistributionRecord.from_dict({"pair_id": "faq-0001"})
        self.assertIn("Missing required field", str(ctx.exception))

    def test_from_dict_reports_an_invalid_value(self):
        data = DistributionRecord.from_query_pair(_pair()).to_dict()
        data["original_label"] = "not-an-int"
        with self.assertRaises(QueryPairValidationError):
            DistributionRecord.from_dict(data)

    def test_harness_loader_refuses_the_distribution_file(self):
        """Rejected with an instruction, not a crash on absent text."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.csv"
            save_distribution_csv([DistributionRecord.from_query_pair(_pair())], path)
            with self.assertRaises(DatasetValidationError) as ctx:
                load_dataset(path)
        self.assertIn("rehydrate_dataset.py", str(ctx.exception))

    def test_harness_loader_refuses_a_distribution_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.json"
            path.write_text(
                json.dumps([DistributionRecord.from_query_pair(_pair()).to_dict()]),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetValidationError) as ctx:
                load_dataset(path)
        self.assertIn("rehydrate_dataset.py", str(ctx.exception))

    def test_a_genuinely_short_csv_still_reports_missing_columns(self):
        """The rehydrate hint must not swallow an ordinary malformed file."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.csv"
            path.write_text("pair_id,query_1\nfaq-0001,text\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError) as ctx:
                load_dataset(path)
        self.assertIn("missing required columns", str(ctx.exception))


# ---------------------------------------------------------------------------
# CLI: sample -> distribute -> rehydrate
# ---------------------------------------------------------------------------

_FIXTURE_REGISTRY: Path = None


def _fixture_registry_path() -> Path:
    """
    An unpinned registry file on disk, for CLI subprocesses.

    The CLIs default `--registry` to the committed `data/corpora.json`, whose
    checksums describe the real corpora once the author has pinned them. Every
    CLI test here runs against `tests/fixtures/corpora/`, so it must be handed
    a registry that makes no claim about those checksums — see `_write_registry`.
    """
    global _FIXTURE_REGISTRY
    if _FIXTURE_REGISTRY is None:
        directory = Path(tempfile.mkdtemp(prefix="levy-fixture-registry-"))
        _FIXTURE_REGISTRY = _write_registry(directory / "corpora.json")
    return _FIXTURE_REGISTRY


def _run(script: str, args, timeout: int = 120):
    args = [str(a) for a in args]
    # Injected rather than repeated at ~12 call sites, so a new CLI test cannot
    # accidentally validate fixtures against the real corpora's checksums.
    if script in {"sample_dataset.py", "rehydrate_dataset.py"} and "--registry" not in args:
        args += ["--registry", str(_fixture_registry_path())]
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestSamplingCli(unittest.TestCase):

    def test_require_real_refuses_synthetic_fallback(self):
        with TemporaryDirectory() as tmp:
            empty_raw = Path(tmp) / "raw"
            empty_raw.mkdir()
            result = _run(
                "sample_dataset.py",
                ["--require-real", "--raw-dir", empty_raw, "--n-per-workload", 4,
                 "--out-csv", Path(tmp) / "o.csv", "--out-json", Path(tmp) / "o.json"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("quora-qqp", result.stderr)
            self.assertIn("fetch_corpora.py", result.stderr)
            self.assertEqual(list(Path(tmp).glob("o.*")), [])

    def test_nothing_is_written_when_validation_fails(self):
        with TemporaryDirectory() as tmp:
            result = _run(
                "sample_dataset.py",
                ["--raw-dir", CORPUS_FIXTURES, "--n-per-workload", 1000,
                 "--out-csv", Path(tmp) / "o.csv", "--out-json", Path(tmp) / "o.json"],
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("nothing written", result.stderr)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [])

    def test_sampling_against_real_fixture_corpora(self):
        with TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "sampled.csv"
            result = _run(
                "sample_dataset.py",
                ["--require-real", "--raw-dir", CORPUS_FIXTURES, "--n-per-workload", 6,
                 "--seed", 42, "--out-csv", out_csv, "--out-json", Path(tmp) / "sampled.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            pairs = load_dataset(out_csv)
            self.assertEqual(len(pairs), 18)
            self.assertEqual(
                {p.source_corpus for p in pairs},
                {"quora-qqp", "sodd", "twitter-pit2015"},
            )

    def test_manifest_records_the_inputs_and_carries_no_query_text(self):
        with TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "sampled.csv"
            result = _run(
                "sample_dataset.py",
                ["--require-real", "--raw-dir", CORPUS_FIXTURES, "--n-per-workload", 6,
                 "--seed", 42, "--sodd-hard-negatives",
                 "--out-csv", out_csv, "--out-json", Path(tmp) / "sampled.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            meta = json.loads((Path(tmp) / "sampled.ids.meta.json").read_text(encoding="utf-8"))
            body = (Path(tmp) / "sampled.ids.meta.json").read_text(encoding="utf-8")
            pairs = load_dataset(out_csv)

        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["positive_ratio"], 0.5)
        # A content-affecting adapter option is recorded, not implied.
        self.assertTrue(meta["workloads"]["code"]["options"]["hard_negatives"])
        self.assertIn("html.parser", meta["normalisation_rule"])
        # Input checksums, so a third party can prove identical inputs.
        self.assertEqual(
            meta["workloads"]["faq"]["files"][0]["sha256"], sha256_file(QQP_TSV)
        )
        self.assertEqual(sorted(meta["corpora"]), ["quora-qqp", "sodd", "twitter-pit2015"])
        for pair in pairs:
            self.assertNotIn(pair.query_1, body)
            self.assertNotIn(pair.query_2, body)


class TestRehydrationRoundTrip(unittest.TestCase):

    def _sample(self, tmp: Path, extra=()):
        result = _run(
            "sample_dataset.py",
            ["--require-real", "--raw-dir", CORPUS_FIXTURES, "--n-per-workload", 6,
             "--seed", 42, "--out-csv", tmp / "sampled.csv",
             "--out-json", tmp / "sampled.json", *extra],
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return tmp / "sampled.csv", tmp / "sampled.json", tmp / "sampled.ids.csv"

    def test_round_trip_is_byte_identical(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sampled_csv, sampled_json, ids = self._sample(tmp)
            result = _run(
                "rehydrate_dataset.py",
                ["--ids", ids, "--raw-dir", CORPUS_FIXTURES,
                 "--out-csv", tmp / "full.csv", "--out-json", tmp / "full.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                (tmp / "full.csv").read_bytes(), sampled_csv.read_bytes()
            )
            self.assertEqual(
                (tmp / "full.json").read_bytes(), sampled_json.read_bytes()
            )

    def test_round_trip_survives_a_content_affecting_option(self):
        """The sidecar carries `hard_negatives`, so rehydration reproduces it."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sampled_csv, _, ids = self._sample(tmp, extra=["--sodd-hard-negatives"])
            result = _run(
                "rehydrate_dataset.py",
                ["--ids", ids, "--raw-dir", CORPUS_FIXTURES,
                 "--out-csv", tmp / "full.csv", "--out-json", tmp / "full.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual((tmp / "full.csv").read_bytes(), sampled_csv.read_bytes())

    def test_unacquired_corpus_names_the_corpus_and_the_fetch_script(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, ids = self._sample(tmp)
            empty_raw = tmp / "raw"
            empty_raw.mkdir()
            result = _run(
                "rehydrate_dataset.py",
                ["--ids", ids, "--raw-dir", empty_raw,
                 "--out-csv", tmp / "full.csv", "--out-json", tmp / "full.json"],
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not acquired", result.stderr)
        self.assertIn("fetch_corpora.py", result.stderr)
        self.assertIn("nothing written", result.stderr)
        self.assertFalse((tmp / "full.csv").exists())

    def test_unresolvable_source_pair_id_is_named_not_dropped(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, ids = self._sample(tmp)
            records = load_distribution_csv(ids)
            records[0].source_pair_id = "does-not-exist"
            save_distribution_csv(records, ids)
            result = _run(
                "rehydrate_dataset.py",
                ["--ids", ids, "--raw-dir", CORPUS_FIXTURES,
                 "--out-csv", tmp / "full.csv", "--out-json", tmp / "full.json"],
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("does-not-exist", result.stderr)
            self.assertIn(records[0].pair_id, result.stderr)
            self.assertFalse((tmp / "full.csv").exists())

    def test_missing_sidecar_is_reported(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, ids = self._sample(tmp)
            (tmp / "sampled.ids.meta.json").unlink()
            result = _run(
                "rehydrate_dataset.py",
                ["--ids", ids, "--raw-dir", CORPUS_FIXTURES,
                 "--out-csv", tmp / "full.csv", "--out-json", tmp / "full.json"],
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("sidecar not found", result.stderr)

    def test_offline_mock_sample_also_rehydrates(self):
        """A synthetic run round-trips too, so the path is exercised with no corpus."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            empty_raw = tmp / "raw"
            empty_raw.mkdir()
            result = _run(
                "sample_dataset.py",
                ["--raw-dir", empty_raw, "--n-per-workload", 5, "--seed", 42,
                 "--out-csv", tmp / "s.csv", "--out-json", tmp / "s.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            result = _run(
                "rehydrate_dataset.py",
                ["--ids", tmp / "s.ids.csv", "--raw-dir", empty_raw,
                 "--out-csv", tmp / "r.csv", "--out-json", tmp / "r.json"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual((tmp / "r.csv").read_bytes(), (tmp / "s.csv").read_bytes())


class TestValidationErrorPaths(unittest.TestCase):
    """The paths that only fire when something on disk is wrong."""

    def test_unloadable_registry_becomes_a_finding(self):
        sources = {w: MockCorpusSource(w, n_candidates=20) for w in ("faq", "code", "chat")}
        with mock.patch(
            "levy.dataset.validation.load_registry",
            side_effect=CorpusRegistryError("registry exploded"),
        ):
            report = validate_sources(sources, n_per_workload=4)
        self.assertFalse(report.ok)
        self.assertIn("registry exploded", report.render())

    def test_corpus_absent_from_the_registry_is_a_finding(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json", lambda raw: raw["corpora"].pop("quora-qqp")
            )
            report = validate_sources(
                _real_sources(), n_per_workload=4, registry=load_registry(path)
            )
        self.assertIn("no entry for corpus 'quora-qqp'", report.render())
        self.assertNotIn("faq", report.pools)

    def test_unreadable_file_is_a_finding(self):
        with TemporaryDirectory() as tmp:
            unreadable = Path(tmp) / "train.tsv"
            unreadable.write_text("id\tquestion1\n", encoding="utf-8")
            unreadable.chmod(0o000)
            try:
                if os.access(unreadable, os.R_OK):  # running as root: chmod is a no-op
                    self.skipTest("cannot make a file unreadable as this user")
                sources = _real_sources()
                sources[WORKLOAD_FAQ] = QuoraQQPSource(unreadable)
                report = validate_sources(
                    sources, n_per_workload=4, registry=_unpinned_registry()
                )
            finally:
                unreadable.chmod(0o600)
        findings = [f for f in report.findings if f.check == CHECK_PRESENCE]
        self.assertTrue(any("not readable" in f.message for f in findings))


class TestUncommonAdapterPaths(unittest.TestCase):
    """Small surfaces the happy path never reaches."""

    def test_label_mapping_serialises_for_the_manifest(self):
        mapping = SODDSource([SODD_TRAIN], hard_negatives=True).label_mapping()
        self.assertEqual(
            mapping.to_dict(),
            {"positive": [0], "negative": [1, 2, 3], "domain": [0, 1, 2, 3, 4]},
        )

    def test_default_options_are_empty(self):
        self.assertEqual(QuoraQQPSource(QQP_TSV).options(), {})
        self.assertEqual(
            TwitterPIT2015Source([PIT_TRAIN, PIT_DEV]).options(), {"splits": ["train", "dev"]}
        )
        self.assertEqual(
            MockCorpusSource("faq", n_candidates=8, seed=3).options(),
            {"n_candidates": 8, "seed": 3},
        )

    def test_base_adapter_defaults_to_a_binary_label_mapping(self):
        """`MockCorpusSource` inherits the ABC's default rather than restating it."""
        mapping = MockCorpusSource("faq").label_mapping()
        self.assertEqual((mapping.positive, mapping.negative, mapping.domain),
                         ((1,), (0,), (0, 1)))
        self.assertEqual(MockCorpusSource("faq").source_files(), [])
        self.assertEqual(MockCorpusSource("faq").check_fields(), [])
        self.assertIsNone(MockCorpusSource("faq").corpus_key)

    def test_quora_declares_a_binary_label_domain(self):
        mapping = QuoraQQPSource(QQP_TSV).label_mapping()
        self.assertEqual(mapping.domain, (0, 1))
        self.assertEqual(mapping.classify(1), 1)
        self.assertEqual(mapping.classify(0), 0)

    def test_sodd_skips_a_post_that_normalises_to_nothing(self):
        pa = __import__("pyarrow")
        pq = __import__("pyarrow.parquet", fromlist=["parquet"])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "SODD_empty.parquet.gzip"
            pq.write_table(
                pa.table(
                    {
                        "first_post": ["<p>  </p>", "<p>fixture kept</p>"],
                        "second_post": ["<p>fixture two</p>", "<p>fixture three</p>"],
                        "label": [0, 0],
                    }
                ),
                path,
                compression="gzip",
            )
            candidates = list(SODDSource([path]).iter_candidates())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_pair_id, "SODD_empty:1")

    def test_pit_vote_count_outside_the_domain_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.data"
            path.write_text(
                "1\tT\tfixture a\tfixture b\t(7, 0)\ttag\ttag\n", encoding="utf-8"
            )
            with self.assertRaises(CorpusSourceError) as ctx:
                list(TwitterPIT2015Source([path]).iter_candidates())
        self.assertIn("outside the declared domain", str(ctx.exception))

    def test_distribution_record_rehydrates_into_a_full_pair(self):
        record = DistributionRecord.from_query_pair(_pair())
        pair = record.to_query_pair("fixture one?", "fixture two?")
        self.assertIsInstance(pair, QueryPair)
        self.assertEqual(pair.query_1, "fixture one?")
        self.assertEqual(pair.pair_id, record.pair_id)
        self.assertEqual(pair.metadata, record.metadata)

    def test_registry_membership_and_archive_flag(self):
        registry = load_registry(REGISTRY)
        self.assertIn("sodd", registry)
        self.assertNotIn("nope", registry)
        self.assertTrue(registry.get("twitter-pit2015").files[0].in_archive)
        self.assertFalse(registry.get("quora-qqp").files[0].in_archive)

    def test_registry_rejects_a_non_object_entry_and_an_empty_corpora_map(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.json"
            path.write_text('{"corpora": {"sodd": "not-an-object"}}', encoding="utf-8")
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
            self.assertIn("must be an object", str(ctx.exception))

            path.write_text('{"corpora": {}}', encoding="utf-8")
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
            self.assertIn("declares no corpora", str(ctx.exception))

    def test_registry_rejects_a_non_object_file_entry(self):
        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp) / "reg.json",
                lambda raw: raw["corpora"]["sodd"].update({"files": ["not-an-object"]}),
            )
            with self.assertRaises(CorpusRegistryError) as ctx:
                load_registry(path)
        self.assertIn("files[0] must be an object", str(ctx.exception))


class TestHtmlNormalisation(unittest.TestCase):
    """
    Pinned because the same rule runs at sampling time and, on someone else's
    machine, at rehydration time — a difference is a broken round-trip.
    """

    def test_empty_input(self):
        self.assertEqual(normalize_html(""), "")

    def test_tags_dropped_text_kept(self):
        self.assertEqual(normalize_html("<p>Hello <b>world</b></p>"), "Hello world")

    def test_code_block_content_is_kept(self):
        out = normalize_html("<p>Try</p><pre><code>x = [1, 2]\ny = x[::-1]</code></pre>")
        self.assertIn("x = [1, 2]", out)
        self.assertIn("y = x[::-1]", out)

    def test_block_boundaries_do_not_join_words(self):
        self.assertEqual(normalize_html("<p>one</p><p>two</p>"), "one two")

    def test_void_tag_becomes_a_separator(self):
        self.assertEqual(normalize_html("one<br/>two"), "one two")
        self.assertEqual(normalize_html("one<img src='x'/>two"), "one two")

    def test_entities_are_unescaped(self):
        self.assertEqual(normalize_html("<p>a &amp; b &mdash; c</p>"), "a & b — c")

    def test_whitespace_runs_collapse(self):
        self.assertEqual(normalize_html("<p>a   \n\t  b</p>"), "a b")

    def test_malformed_markup_still_yields_its_text(self):
        self.assertEqual(normalize_html("<p>unclosed <span>bad"), "unclosed bad")
        self.assertEqual(normalize_html("a <<< b"), "a <<< b")
        self.assertEqual(normalize_html("<p>x</div></p>"), "x")

    def test_rule_is_stated_for_the_manifest(self):
        self.assertIn("html.parser", NORMALISATION_RULE)
        self.assertIn("whitespace", NORMALISATION_RULE)


# ---------------------------------------------------------------------------
# The offline boundary
# ---------------------------------------------------------------------------

class TestAcquisitionIsOutOfBand(unittest.TestCase):
    """
    The repository's two network-touching entry points — `fetch_corpora.py`
    (LEV-12, acquisition) and `populate_responses.py` (LEV-14, billed provider
    calls) — must stay outside the test suite. These are structural guards, not
    conventions.
    """

    NETWORK_MODULES = {"urllib", "urllib.request", "socket", "http", "http.client", "httpx", "requests"}

    # Script stems no test may execute, and the reason each is out of band.
    OUT_OF_BAND_SCRIPTS = {
        "fetch_corpora": "acquires third-party corpora over the network",
        "populate_responses": "makes billed Anthropic API calls",
    }

    def test_no_test_module_invokes_an_out_of_band_script(self):
        """
        No test *calls* anything with one of those script names as an argument.

        Checked on the AST rather than by substring, so this module can go on
        naming the scripts in prose and in expected-output assertions — which
        is exactly how a replicator learns about them — while an actual
        invocation, through `_run`, `subprocess` or anything else, still fails.

        Importing the module and calling its `main()` is caught too: `main` is
        in `runners`, and an import of the script's name is checked separately
        below.
        """
        runners = {"_run", "run", "call", "check_call", "check_output", "Popen", "system", "main"}
        offenders = []
        for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        {alias.name for alias in node.names}
                        if isinstance(node, ast.Import)
                        else {node.module or ""}
                    )
                    if names & set(self.OUT_OF_BAND_SCRIPTS):
                        offenders.append(f"{path.name}:{node.lineno} (import)")
                    continue

                if not isinstance(node, ast.Call):
                    continue
                callee = node.func
                name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
                if name not in runners:
                    continue
                # `<module>.main(...)` where <module> is an out-of-band script.
                if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
                    if callee.value.id in self.OUT_OF_BAND_SCRIPTS:
                        offenders.append(f"{path.name}:{node.lineno} (call)")
                for argument in list(node.args) + [kw.value for kw in node.keywords]:
                    for inner in ast.walk(argument):
                        if (
                            isinstance(inner, ast.Constant)
                            and isinstance(inner.value, str)
                            and any(script in inner.value for script in self.OUT_OF_BAND_SCRIPTS)
                        ):
                            offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_latency_package_imports_nothing_that_can_reach_the_network(self):
        """
        `levy/latency/` is imported by the offline suite, so nothing in it may
        pull in a network library at module scope. The billed loop's *logic*
        lives in `levy/latency/population.py` precisely so it can be tested;
        what must not follow it into the package is the ability to open a
        socket without a caller handing one over.
        """
        for module in sorted((REPO_ROOT / "levy" / "latency").glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            with self.subTest(module=module.name):
                self.assertEqual(imported & self.NETWORK_MODULES, set())

    def test_dataset_package_imports_nothing_that_can_reach_the_network(self):
        for module in sorted((REPO_ROOT / "levy" / "dataset").glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            with self.subTest(module=module.name):
                self.assertEqual(imported & self.NETWORK_MODULES, set())


if __name__ == "__main__":
    unittest.main()
