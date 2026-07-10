#!/usr/bin/env python
"""
Sample a ground-truth dataset from raw corpora (LEV-3 / D2 platform tooling).

For each workload (faq / code / chat), reads a local raw corpus file (never
downloads anything) and draws a seeded, stratified sample of query pairs.
If a workload's raw-corpus path is omitted, falls back to `MockCorpusSource`
so the whole pipeline can be exercised offline against synthetic data (e.g.
for CI smoke tests) — pairs produced this way carry `source_corpus="mock"`
and must never be mistaken for real sampled data.

Producing the real 900-pair dataset means pointing this script at the actual
downloaded Quora QQP / Stack Overflow duplicates / ConvAI2-derived files;
that data-acquisition step is an author task, not performed by this script.

Examples:
    # Fully offline smoke test (all three workloads mocked):
    python scripts/sample_dataset.py --n-per-workload 5 --seed 42 \\
        --out-csv /tmp/sample.csv --out-json /tmp/sample.json

    # Real run once raw corpora are downloaded locally:
    python scripts/sample_dataset.py \\
        --quora-tsv /data/quora_qqp/train.tsv \\
        --stackoverflow-csv /data/so_duplicates/pairs.csv \\
        --convai2-json /data/convai2_pairs/pairs.json \\
        --n-per-workload 300 --seed 42 \\
        --out-csv data/ground_truth.csv --out-json data/ground_truth.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.io import save_dataset
from levy.dataset.sampling import (
    ConvAI2Source,
    MockCorpusSource,
    QuoraQQPSource,
    StackOverflowDuplicatesSource,
    sample_dataset,
)
from levy.dataset.schema import WORKLOAD_CHAT, WORKLOAD_CODE, WORKLOAD_FAQ


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quora-tsv", type=Path, default=None, help="Raw Quora QQP TSV file (faq workload)")
    parser.add_argument("--stackoverflow-csv", type=Path, default=None, help="Raw Stack Overflow duplicates CSV file (code workload)")
    parser.add_argument("--convai2-json", type=Path, default=None, help="Raw ConvAI2-derived pairs JSON file (chat workload)")
    parser.add_argument("--n-per-workload", type=int, default=300, help="Pairs to sample per workload (default: 300, per D2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Fraction of sampled pairs with original_label=1 (default: 0.5, balanced)")
    parser.add_argument("--mock-candidates", type=int, default=None, help="Candidate pool size per workload when falling back to MockCorpusSource (default: max(40, 4 * n-per-workload))")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    mock_candidates = args.mock_candidates or max(40, 4 * args.n_per_workload)

    sources = {}
    if args.quora_tsv:
        sources[WORKLOAD_FAQ] = QuoraQQPSource(args.quora_tsv)
    else:
        print("[sample_dataset] --quora-tsv not given: falling back to MockCorpusSource for 'faq'", file=sys.stderr)
        sources[WORKLOAD_FAQ] = MockCorpusSource(WORKLOAD_FAQ, n_candidates=mock_candidates, seed=args.seed)

    if args.stackoverflow_csv:
        sources[WORKLOAD_CODE] = StackOverflowDuplicatesSource(args.stackoverflow_csv)
    else:
        print("[sample_dataset] --stackoverflow-csv not given: falling back to MockCorpusSource for 'code'", file=sys.stderr)
        sources[WORKLOAD_CODE] = MockCorpusSource(WORKLOAD_CODE, n_candidates=mock_candidates, seed=args.seed)

    if args.convai2_json:
        sources[WORKLOAD_CHAT] = ConvAI2Source(args.convai2_json)
    else:
        print("[sample_dataset] --convai2-json not given: falling back to MockCorpusSource for 'chat'", file=sys.stderr)
        sources[WORKLOAD_CHAT] = MockCorpusSource(WORKLOAD_CHAT, n_candidates=mock_candidates, seed=args.seed)

    pairs = sample_dataset(
        sources,
        n_per_workload=args.n_per_workload,
        seed=args.seed,
        positive_ratio=args.positive_ratio,
    )
    save_dataset(pairs, args.out_csv, args.out_json)
    print(f"[sample_dataset] wrote {len(pairs)} pairs to {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
