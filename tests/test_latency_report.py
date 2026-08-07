"""
Tests for the latency artefact writers and the timed replay driver
(LEV-14 / 4.4, 4.5, 6.3, 8.2).

What these assert, beyond "the files exist":

- Every percentile field of every configuration row is populated. An empty
  cell would be a segment the lookup never reached, reported as if measured.
- The metadata sidecar states the reproducibility boundary, because the one
  mistake a reader of these numbers must not make is treating the provider
  figure as replicable.
- A savings figure never appears without the model identifier that produced
  it. A model-free savings number is a claim about caching in general, which
  this measurement does not support.
- The reference result directory is byte-identical after a full run.

Fully offline: mock providers, no network, no provider call anywhere.
"""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_latency  # noqa: E402  (scripts/ is not a package)

from levy.dataset.io import load_dataset  # noqa: E402
from levy.experiment.config import ExperimentConfig  # noqa: E402
from levy.experiment.runner import run_sweep, write_decisions_csv, write_results_csv  # noqa: E402
from levy.latency.benchmark import benchmark_configurations  # noqa: E402
from levy.latency.corpus import (  # noqa: E402
    CorpusLLMClient,
    ResponseCorpusError,
    ResponseRecord,
    append_record,
    load_corpus,
    response_key,
)
from levy.latency.report import (  # noqa: E402
    LATENCY_FIELDNAMES,
    REPRODUCIBILITY_BOUNDARY,
    price_at,
    savings_figure,
    summarise_calls,
    write_latency_csv,
    write_latency_meta,
    write_llm_calls,
)
from levy.models import LLMRequest  # noqa: E402

FIXTURE = REPO_ROOT / "data" / "ground_truth.csv"

# The ten FAQ configurations of the frozen grid: 2 models x 5 thresholds.
FAQ_CONFIGS = [
    ExperimentConfig(model=model, workload="faq", threshold=threshold)
    for model in ("all-MiniLM-L6-v2", "modernbert")
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90)
]


def _write_reference(out_dir: Path) -> Path:
    """A real harness run over the FAQ workload, used as the read-only reference."""
    pairs = load_dataset(FIXTURE)
    results, _ = run_sweep(pairs, configs=FAQ_CONFIGS, embedding_provider="mock", llm_latency_seconds=0)
    write_results_csv(results, out_dir / "results.csv")
    write_decisions_csv(results, out_dir / "decisions.csv")
    return out_dir / "results.csv"


def _hash_tree(directory: Path) -> dict:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _record(key_source: str, latency_ms: float = 900.0, **overrides) -> ResponseRecord:
    payload = dict(
        key=response_key(key_source),
        response_text=f"recorded answer for {key_source}",
        latency_ms=latency_ms,
        input_tokens=120,
        output_tokens=240,
        model="claude-haiku-4-5-20251001",
        timestamp_utc="2026-08-07T12:00:00+00:00",
        stop_reason="end_turn",
    )
    payload.update(overrides)
    return ResponseRecord(**payload)


class TestLatencyCsv(unittest.TestCase):

    def test_completed_run_writes_all_ten_faq_rows_with_no_empty_percentile(self):
        pairs = load_dataset(FIXTURE)
        results, _ = benchmark_configurations(
            FAQ_CONFIGS, pairs, embedding_provider="mock", warmup=1, repetitions=3
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency.csv"
            write_latency_csv(results, path)

            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, LATENCY_FIELDNAMES)
                rows = list(reader)

        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["config_id"] for row in rows],
            [config.config_id for config in FAQ_CONFIGS],
        )
        for row in rows:
            self.assertEqual(row["workload"], "faq")
            for column in LATENCY_FIELDNAMES:
                self.assertNotEqual(row[column], "", f"{row['config_id']}: empty {column}")
            for column in (name for name in LATENCY_FIELDNAMES if name.endswith("_ms")):
                self.assertGreaterEqual(float(row[column]), 0.0, f"{row['config_id']}: {column}")


class TestLatencyMeta(unittest.TestCase):

    def _meta(self, tmp: Path, call_summary=None) -> dict:
        pairs = load_dataset(FIXTURE)
        results, identities = benchmark_configurations(
            FAQ_CONFIGS[:2], pairs, embedding_provider="mock", warmup=1, repetitions=3
        )
        path = tmp / "latency_meta.json"
        write_latency_meta(
            path=path,
            results=results,
            dataset_path=FIXTURE,
            embedding_provider="mock",
            model_identities=identities,
            warmup=1,
            repetitions=3,
            probe_seed=42,
            call_summary=call_summary,
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_sidecar_contains_the_reproducibility_boundary_statement(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._meta(Path(tmp))

        self.assertEqual(meta["reproducibility_boundary"], REPRODUCIBILITY_BOUNDARY)
        boundary = meta["reproducibility_boundary"]
        self.assertIn("latency.csv", boundary)
        self.assertIn("llm_calls.json", boundary)
        self.assertIn("NOT reproducible", boundary)

    def test_sidecar_records_the_protocol_parameters_and_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._meta(Path(tmp))

        self.assertEqual(meta["protocol"]["warmup_iterations"], 1)
        self.assertEqual(meta["protocol"]["measured_repetitions"], 3)
        self.assertEqual(meta["protocol"]["probe_seed"], 42)
        self.assertIn("python_version", meta["host"])
        self.assertIn("numpy", meta["host"]["library_versions"])
        self.assertEqual(meta["embedding_provider"], "mock")


class TestSavingsFigureCarriesItsModel(unittest.TestCase):
    """
    LEV-14 / 8.2 — the guard against a Haiku-derived figure being read as a
    general result. Whenever an artefact states what a hit saves, the model
    that produced the avoided latency is in the same artefact.
    """

    def _results(self):
        pairs = load_dataset(FIXTURE)
        results, _ = benchmark_configurations(
            FAQ_CONFIGS[:2], pairs, embedding_provider="mock", warmup=1, repetitions=3
        )
        return results

    def test_savings_figure_names_its_model(self):
        summary = summarise_calls([_record("a"), _record("b", latency_ms=1100.0)], 1.0, 5.0)
        savings = savings_figure(self._results(), summary)

        self.assertIsNotNone(savings["net_saved_per_hit_ms"])
        self.assertEqual(savings["model"], "claude-haiku-4-5-20251001")
        self.assertIn("claude-haiku-4-5-20251001", savings["note"])

    def test_no_savings_figure_is_stated_without_provider_calls(self):
        savings = savings_figure(self._results(), None)

        self.assertIsNone(savings["provider_latency_p50_ms"])
        self.assertIsNone(savings["model"])
        self.assertNotIn("net_saved_per_hit_ms", savings)

    def test_every_savings_figure_in_the_written_sidecar_has_a_model_beside_it(self):
        summary = summarise_calls([_record("a"), _record("b")], 1.0, 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency_meta.json"
            write_latency_meta(
                path=path,
                results=self._results(),
                dataset_path=FIXTURE,
                embedding_provider="mock",
                model_identities={},
                warmup=1,
                repetitions=3,
                probe_seed=42,
                call_summary=summary,
            )
            meta = json.loads(path.read_text(encoding="utf-8"))

        savings = meta["savings"]
        stated = [key for key in savings if key in ("net_saved_per_hit_ms", "provider_latency_p50_ms")]
        self.assertTrue(stated)
        self.assertTrue(savings["model"], "a savings figure was written with no model identifier")


class TestLlmCallsSummary(unittest.TestCase):

    def test_summary_reports_observed_cost_tokens_window_and_model(self):
        records = [
            _record("a", latency_ms=800.0, timestamp_utc="2026-08-07T12:00:00+00:00"),
            _record("b", latency_ms=1200.0, timestamp_utc="2026-08-07T12:05:00+00:00"),
        ]
        summary = summarise_calls(records, input_price_per_mtok=1.0, output_price_per_mtok=5.0)

        self.assertEqual(summary["n_calls"], 2)
        self.assertEqual(summary["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(summary["latency_ms"]["min"], 800.0)
        self.assertEqual(summary["latency_ms"]["max"], 1200.0)
        self.assertEqual(summary["tokens"]["input_total"], 240)
        self.assertEqual(summary["tokens"]["output_total"], 480)
        # 240 input tokens at $1/MTok + 480 output tokens at $5/MTok
        expected = 240 / 1e6 * 1.0 + 480 / 1e6 * 5.0
        self.assertAlmostEqual(summary["observed_cost_usd"]["total"], expected)
        self.assertAlmostEqual(summary["observed_cost_usd"]["per_call"], expected / 2)
        self.assertEqual(summary["window_utc"]["first_call"], "2026-08-07T12:00:00+00:00")
        self.assertEqual(summary["window_utc"]["last_call"], "2026-08-07T12:05:00+00:00")

    def test_empty_run_reports_no_calls_rather_than_zeroed_statistics(self):
        summary = summarise_calls([], 1.0, 5.0)
        self.assertEqual(summary["n_calls"], 0)
        self.assertNotIn("observed_cost_usd", summary)

    def test_repricing_uses_the_recorded_token_totals_and_names_the_priced_model(self):
        summary = summarise_calls([_record("a"), _record("b")], 1.0, 5.0)
        projection = price_at(summary, 3.0, 15.0, "claude-sonnet-5")

        expected = 240 / 1e6 * 3.0 + 480 / 1e6 * 15.0
        self.assertEqual(projection["model_priced"], "claude-sonnet-5")
        self.assertAlmostEqual(projection["projected_cost_usd"]["total"], expected)

    def test_write_llm_calls_merges_extra_sections(self):
        summary = summarise_calls([_record("a")], 1.0, 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_calls.json"
            write_llm_calls(summary, path, extra={"repriced": price_at(summary, 3.0, 15.0, "claude-sonnet-5")})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["n_calls"], 1)
        self.assertEqual(payload["repriced"]["model_priced"], "claude-sonnet-5")


class TestResponseCorpus(unittest.TestCase):

    def test_append_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            append_record(path, _record("alpha"))
            append_record(path, _record("beta"))
            records = load_corpus(path)

        self.assertEqual(sorted(records), sorted([response_key("alpha"), response_key("beta")]))
        self.assertEqual(records[response_key("alpha")].response_text, "recorded answer for alpha")

    def test_prompt_text_is_never_stored(self):
        """The corpus is keyed by prompt hash and carries no corpus query text."""
        prompt = "what is the refund window"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            append_record(path, _record(prompt, response_text="Thirty days from delivery."))
            body = path.read_text(encoding="utf-8")

        self.assertNotIn(prompt, body)
        self.assertIn(response_key(prompt), body)

    def test_missing_corpus_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_corpus(Path(tmp) / "absent.jsonl"), {})

    def test_blank_lines_are_skipped(self):
        """A trailing newline from an interrupted append is not a corrupt corpus."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            append_record(path, _record("alpha"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n")
            self.assertEqual(len(load_corpus(path)), 1)

    def test_write_llm_calls_without_extra_sections(self):
        summary = summarise_calls([_record("a")], 1.0, 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_calls.json"
            write_llm_calls(summary, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["n_calls"], 1)

    def test_unreadable_line_names_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            path.write_text('{"key": "abc"}\n', encoding="utf-8")
            with self.assertRaises(ResponseCorpusError) as ctx:
                load_corpus(path)
        self.assertIn(":1:", str(ctx.exception))

    def test_corpus_client_serves_recorded_responses_and_delegates_the_rest(self):
        from levy.llm_client import MockLLMClient

        records = {response_key("known"): _record("known")}
        client = CorpusLLMClient(records, fallback=MockLLMClient(latency_seconds=0))

        served = client.generate(LLMRequest(prompt="known"))
        delegated = client.generate(LLMRequest(prompt="unknown"))

        self.assertEqual(served.text, "recorded answer for known")
        self.assertEqual(served.metadata["source"], "response_corpus")
        self.assertEqual(served.token_usage, 360)
        self.assertNotEqual(delegated.text, served.text)
        self.assertEqual((client.served, client.delegated), (1, 1))


class TestDriverEndToEnd(unittest.TestCase):

    def test_driver_writes_both_artefacts_and_leaves_the_reference_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference_dir = tmp_path / "run-ref"
            reference_dir.mkdir()
            reference = _write_reference(reference_dir)
            before = _hash_tree(reference_dir)

            out_dir = tmp_path / "latency-faq"
            exit_code = run_latency.main(
                [
                    "--reference", str(reference),
                    "--dataset", str(FIXTURE),
                    "--out-dir", str(out_dir),
                    "--workload", "faq",
                    "--warmup", "1",
                    "--repetitions", "3",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "latency.csv").exists())
            self.assertTrue((out_dir / "latency_meta.json").exists())

            with (out_dir / "latency.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 10)

            meta = json.loads((out_dir / "latency_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["reference_results"], str(reference))
            self.assertEqual(meta["response_corpus"]["n_records"], 0)

            self.assertEqual(_hash_tree(reference_dir), before)

    def test_driver_serves_a_response_corpus_when_one_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference_dir = tmp_path / "run-ref"
            reference_dir.mkdir()
            reference = _write_reference(reference_dir)

            pairs = [pair for pair in load_dataset(FIXTURE) if pair.workload == "faq"]
            prompts = {query for pair in pairs for query in (pair.query_1, pair.query_2)}
            corpus = tmp_path / "responses.jsonl"
            for prompt in sorted(prompts):
                append_record(corpus, _record(prompt))

            out_dir = tmp_path / "latency-faq"
            exit_code = run_latency.main(
                [
                    "--reference", str(reference),
                    "--dataset", str(FIXTURE),
                    "--out-dir", str(out_dir),
                    "--responses", str(corpus),
                    "--warmup", "1",
                    "--repetitions", "3",
                ]
            )

            self.assertEqual(exit_code, 0)
            meta = json.loads((out_dir / "latency_meta.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["response_corpus"]["n_records"], len(prompts))
        self.assertGreater(meta["response_corpus"]["served"], 0)
        # A recorded corpus makes a savings figure derivable -- with its model.
        self.assertEqual(meta["savings"]["model"], "claude-haiku-4-5-20251001")
        self.assertIsNotNone(meta["savings"]["net_saved_per_hit_ms"])

    def test_missing_reference_and_empty_selection_exit_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.assertEqual(
                run_latency.main(
                    ["--reference", str(tmp_path / "absent.csv"), "--out-dir", str(tmp_path / "out")]
                ),
                1,
            )

            reference_dir = tmp_path / "run-ref"
            reference_dir.mkdir()
            reference = _write_reference(reference_dir)
            self.assertEqual(
                run_latency.main(
                    [
                        "--reference", str(reference),
                        "--dataset", str(FIXTURE),
                        "--out-dir", str(tmp_path / "out"),
                        "--workload", "chat",  # the reference holds faq only
                    ]
                ),
                1,
            )
            self.assertFalse((tmp_path / "out").exists())

    def test_configuration_list_comes_from_the_reference_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_dir = Path(tmp) / "run-ref"
            reference_dir.mkdir()
            reference = _write_reference(reference_dir)

            configs = run_latency.read_configurations(reference, workload="faq")

        self.assertEqual(
            [config.config_id for config in configs],
            [config.config_id for config in FAQ_CONFIGS],
        )


if __name__ == "__main__":
    unittest.main()
