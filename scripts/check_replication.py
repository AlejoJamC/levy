#!/usr/bin/env python
"""
Verify the frozen replication criterion (LEV-8; S&D Report Success Criterion
3: headline results replicate within +/-5%).

Re-runs the harness over exactly the grid recorded in a reference
`results.csv`, then compares the candidate run's precision and recall,
configuration by configuration, against that reference. Exits zero when
every value is within tolerance; otherwise exits non-zero and prints a
per-configuration diff table naming the configuration, the metric, both
values, and the deviation.

Tolerance rule (see `levy.analysis.replication`):
    |candidate - reference| <= max(0.01, 5% * |reference|)

The grid is taken from the reference file itself -- replication reproduces
the reference's configurations, so there is no separate grid flag to keep in
sync. The dataset defaults to the one recorded in the reference run's
`run_meta.json`.

Examples:
    python scripts/check_replication.py --reference results/run-001/results.csv

    # Keep the candidate run's outputs instead of discarding the temp dir:
    python scripts/check_replication.py --reference results/run-001/results.csv \\
        --dataset data/ground_truth.csv --keep-dir /tmp/replication-candidate
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.analysis.io import HarnessContractError, load_results, load_run_meta
from levy.analysis.replication import (
    ABSOLUTE_FLOOR,
    RELATIVE_TOLERANCE,
    compare_results,
    format_report,
)
from levy.dataset.io import load_dataset
from levy.experiment.config import ExperimentConfig
from levy.experiment.metrics import ExperimentSanityError
from levy.experiment.runner import run_sweep, write_results_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="Reference results.csv to replicate")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset to re-run against (default: the dataset_path in the reference run's run_meta.json)")
    parser.add_argument("--embedding-provider", type=str, default=None, choices=["mock", "sentence-transformers", "ollama"], help="Embedding provider for the re-run (default: the provider recorded in run_meta.json, else mock)")
    parser.add_argument("--keep-dir", type=Path, default=None, help="Write the candidate run here instead of a discarded temp directory")
    parser.add_argument("--relative-tolerance", type=float, default=RELATIVE_TOLERANCE, help=f"Relative tolerance (default: {RELATIVE_TOLERANCE}, the frozen +/-5%%)")
    parser.add_argument("--absolute-floor", type=float, default=ABSOLUTE_FLOOR, help=f"Absolute floor for near-zero reference values (default: {ABSOLUTE_FLOOR})")
    parser.add_argument("--llm-latency-seconds", type=float, default=0.5, help="Mock LLM latency for the re-run; does not affect results (default: 0.5, matching a real run)")
    parser.add_argument("--show-all", action="store_true", help="Print every comparison, not just the out-of-tolerance ones")
    return parser


def _configs_from_reference(reference) -> List[ExperimentConfig]:
    """Replicate exactly the grid the reference recorded."""
    return [
        ExperimentConfig(model=str(row["model"]), workload=str(row["workload"]), threshold=float(row["threshold"]))
        for _, row in reference.iterrows()
    ]


def _write_and_load(results, candidate_dir: Path):
    """Write the candidate run through the harness writer, then read it back
    through the same contract loader the reference went through."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "results.csv"
    write_results_csv(results, candidate_path)
    return load_results(candidate_path)


def main(argv=None, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        reference = load_results(args.reference)
    except HarnessContractError as exc:
        print(f"[check_replication] {exc}", file=sys.stderr)
        return 1

    meta_path = args.reference.parent / "run_meta.json"
    run_meta = load_run_meta(meta_path) if meta_path.is_file() else {}

    dataset_path = args.dataset or run_meta.get("dataset_path")
    if dataset_path is None:
        print(
            "[check_replication] no dataset: pass --dataset, or place the reference "
            "run's run_meta.json next to results.csv",
            file=sys.stderr,
        )
        return 1

    embedding_provider = args.embedding_provider or run_meta.get("embedding_provider") or "mock"
    configs = _configs_from_reference(reference)

    output_fn(f"[check_replication] re-running {len(configs)} configuration(s) on {dataset_path} (embeddings: {embedding_provider})")
    pairs = load_dataset(dataset_path)
    try:
        results, _ = run_sweep(
            pairs,
            configs=configs,
            embedding_provider=embedding_provider,
            llm_latency_seconds=args.llm_latency_seconds,
        )
    except ExperimentSanityError as exc:
        print(f"[check_replication] candidate run failed its sanity checks: {exc}", file=sys.stderr)
        return 1

    if args.keep_dir:
        candidate = _write_and_load(results, args.keep_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="levy-replication-") as tmp:
            candidate = _write_and_load(results, Path(tmp))

    report = compare_results(
        reference,
        candidate,
        relative=args.relative_tolerance,
        absolute_floor=args.absolute_floor,
    )
    output_fn(format_report(report, show_all=args.show_all))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
