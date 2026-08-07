#!/usr/bin/env python
"""
Measure cache lookup overhead over a workload's configurations (LEV-14 / D1).

Reads the configuration list from a reference `results.csv` — so the measured
grid is exactly the grid that produced the reported results, rather than a
second, hand-maintained copy of it — and replays each configuration with the
timing instrumentation active. Writes `latency.csv` and `latency_meta.json` to
`--out-dir`.

Fully offline. Recorded real-provider responses are served from
`responses.jsonl` when it exists (built out-of-band by
`scripts/populate_responses.py`); anything unrecorded falls back to the mock
client. No provider call is made from here, ever.

The reference directory is opened read-only: nothing is written to it, and
`--out-dir` is required precisely so no default can resolve to a result set.

Examples:
    # Offline smoke run against the committed synthetic fixture:
    python scripts/run_latency.py --reference results/reproduce/results.csv \\
        --dataset data/ground_truth.csv --out-dir /tmp/levy_latency --workload faq

    # The study run, over the real dataset and the real encoders:
    python scripts/run_latency.py --reference results/run-003/results.csv \\
        --dataset data/ground_truth.full.csv --embedding-provider sentence-transformers \\
        --out-dir results/latency-faq --workload faq
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.io import load_dataset
from levy.experiment.config import ExperimentConfig
from levy.latency.benchmark import (
    BenchmarkError,
    DEFAULT_PROBE_SEED,
    DEFAULT_REPETITIONS,
    DEFAULT_WARMUP,
    benchmark_configurations,
)
from levy.latency.corpus import CorpusLLMClient, load_corpus
from levy.latency.report import summarise_calls, write_latency_csv, write_latency_meta
from levy.llm_client import MockLLMClient

RESPONSES_FILENAME = "responses.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="Reference results.csv supplying the configuration list (read-only)")
    parser.add_argument("--dataset", type=Path, default=Path("data/ground_truth.csv"), help="Dataset file (.csv or .json); default: data/ground_truth.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for latency.csv and latency_meta.json (never a harness result directory)")
    parser.add_argument("--workload", type=str, default="faq", help="Workload to measure (default: faq, the only workload where the cache measurably operates)")
    parser.add_argument("--responses", type=Path, default=None, help=f"Response corpus to serve from; default: <out-dir>/{RESPONSES_FILENAME}")
    parser.add_argument("--embedding-provider", type=str, default="mock", choices=["mock", "sentence-transformers", "ollama"], help="Embedding provider (default: mock, fully offline)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"Discarded warm-up lookups per configuration (default: {DEFAULT_WARMUP})")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help=f"Measured repetitions per segment per phase (default: {DEFAULT_REPETITIONS})")
    parser.add_argument("--probe-seed", type=int, default=DEFAULT_PROBE_SEED, help=f"Seed for probe selection (default: {DEFAULT_PROBE_SEED})")
    parser.add_argument("--input-price-per-mtok", type=float, default=1.0, help="USD per 1M input tokens, for costing the recorded calls")
    parser.add_argument("--output-price-per-mtok", type=float, default=5.0, help="USD per 1M output tokens, for costing the recorded calls")
    return parser


def read_configurations(reference: Path, workload: Optional[str] = None) -> List[ExperimentConfig]:
    """
    Configurations from a reference `results.csv`, in file order.

    Read-only, and deliberately from the results file rather than from
    `full_grid()`: measuring a configuration the reference run did not contain
    would report overhead for a cell that has no result to sit beside.
    """
    with reference.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    configs = []
    for row in rows:
        if workload and row["workload"] != workload:
            continue
        configs.append(
            ExperimentConfig(
                model=row["model"],
                workload=row["workload"],
                threshold=float(row["threshold"]),
            )
        )
    return configs


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.reference.exists():
        print(f"[run_latency] reference results file not found: {args.reference}", file=sys.stderr)
        return 1

    configs = read_configurations(args.reference, workload=args.workload)
    if not configs:
        print(
            f"[run_latency] no configurations for workload {args.workload!r} in {args.reference}",
            file=sys.stderr,
        )
        return 1

    pairs = load_dataset(args.dataset)

    responses_path = args.responses or (args.out_dir / RESPONSES_FILENAME)
    records = load_corpus(responses_path)
    llm_client = CorpusLLMClient(records, fallback=MockLLMClient(latency_seconds=0))
    if records:
        print(f"[run_latency] serving {len(records)} recorded response(s) from {responses_path}")
    else:
        print(f"[run_latency] no response corpus at {responses_path}; replaying with the mock client")

    try:
        results, identities = benchmark_configurations(
            configs,
            pairs,
            embedding_provider=args.embedding_provider,
            llm_client=llm_client,
            warmup=args.warmup,
            repetitions=args.repetitions,
            probe_seed=args.probe_seed,
        )
    except BenchmarkError as exc:
        print(f"[run_latency] benchmark failed: {exc}", file=sys.stderr)
        return 1

    call_summary = (
        summarise_calls(records.values(), args.input_price_per_mtok, args.output_price_per_mtok)
        if records
        else None
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_latency_csv(results, args.out_dir / "latency.csv")
    write_latency_meta(
        path=args.out_dir / "latency_meta.json",
        results=results,
        dataset_path=args.dataset,
        embedding_provider=args.embedding_provider,
        model_identities=identities,
        warmup=args.warmup,
        repetitions=args.repetitions,
        probe_seed=args.probe_seed,
        reference_results=args.reference,
        call_summary=call_summary,
        response_corpus={
            "path": str(responses_path),
            "n_records": len(records),
            "served": llm_client.served,
            "delegated_to_mock": llm_client.delegated,
        },
    )
    print(f"[run_latency] wrote {len(results)} configuration row(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
