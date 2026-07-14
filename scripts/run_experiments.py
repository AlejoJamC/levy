#!/usr/bin/env python
"""
Run the frozen experimental grid end-to-end (LEV-4 / D3).

Loads a ground-truth dataset via `levy.dataset.load_dataset`, replays it
through the offline harness (Algorithm 2) for every (model, workload,
threshold) configuration, and writes `results.csv`, `decisions.csv`, and
`run_meta.json` to the chosen output directory. Runs fully offline with
the mock LLM provider; `--embedding-provider` defaults to `mock` so the
committed synthetic fixture can be exercised with zero external
dependencies (a real study run passes `--embedding-provider
sentence-transformers` once the real 900-pair dataset, LEV-11, lands).

Examples:
    # Full 30-configuration grid on the committed synthetic fixture:
    python scripts/run_experiments.py --out-dir /tmp/levy_experiment

    # Smoke run: one model, one workload, two thresholds:
    python scripts/run_experiments.py --out-dir /tmp/levy_smoke \\
        --models all-MiniLM-L6-v2 --workloads faq --thresholds 0.70,0.90

    # Real study run once the real dataset + sentence-transformers models are available:
    python scripts/run_experiments.py --dataset data/ground_truth.csv \\
        --embedding-provider sentence-transformers --out-dir results/run-001
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.io import load_dataset
from levy.experiment.config import ExperimentConfig, full_grid
from levy.experiment.metrics import ExperimentSanityError
from levy.experiment.runner import run_sweep, write_decisions_csv, write_results_csv, write_run_meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/ground_truth.csv"), help="Dataset file (.csv or .json); default: data/ground_truth.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for results.csv, decisions.csv, run_meta.json")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model subset (default: full frozen grid, e.g. all-MiniLM-L6-v2,modernbert)")
    parser.add_argument("--workloads", type=str, default=None, help="Comma-separated workload subset (default: faq,code,chat)")
    parser.add_argument("--thresholds", type=str, default=None, help="Comma-separated threshold subset (default: 0.70,0.75,0.80,0.85,0.90)")
    parser.add_argument("--embedding-provider", type=str, default="mock", choices=["mock", "sentence-transformers", "ollama"], help="Embedding provider for the sweep (default: mock, fully offline)")
    return parser


def _filter_grid(
    models: Optional[List[str]],
    workloads: Optional[List[str]],
    thresholds: Optional[List[float]],
) -> List[ExperimentConfig]:
    configs = full_grid()
    if models:
        wanted = set(models)
        configs = [c for c in configs if c.model in wanted]
    if workloads:
        wanted = set(workloads)
        configs = [c for c in configs if c.workload in wanted]
    if thresholds:
        wanted = {round(t, 2) for t in thresholds}
        configs = [c for c in configs if round(c.threshold, 2) in wanted]
    return configs


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    pairs = load_dataset(args.dataset)

    models = args.models.split(",") if args.models else None
    workloads = args.workloads.split(",") if args.workloads else None
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else None
    configs = _filter_grid(models, workloads, thresholds)

    if not configs:
        print("[run_experiments] no configurations selected -- check --models/--workloads/--thresholds", file=sys.stderr)
        return 1

    start = time.perf_counter()
    try:
        results, model_identities = run_sweep(pairs, configs=configs, embedding_provider=args.embedding_provider)
    except ExperimentSanityError as exc:
        print(f"[run_experiments] sanity check failed: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(results, args.out_dir / "results.csv")
    write_decisions_csv(results, args.out_dir / "decisions.csv")
    write_run_meta(
        results=results,
        configs=configs,
        dataset_path=args.dataset,
        embedding_provider=args.embedding_provider,
        model_identities=model_identities,
        elapsed_seconds=elapsed,
        path=args.out_dir / "run_meta.json",
    )
    print(f"[run_experiments] wrote {len(results)} configuration result(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
