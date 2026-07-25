#!/usr/bin/env python
"""
Produce the D3 statistical-analysis bundle from a harness output directory
(LEV-8).

One invocation emits every table and figure: the two-way ANOVA on false
positive rate testing H0_1/H0_2/H0_3, the conditional Tukey HSD post-hoc
comparisons, the threshold-selection curve tables and figures, the Cohen's
kappa section (computed by the LEV-3 dataset tooling, not reimplemented
here), and an `analysis_meta.json` sidecar.

The pipeline is dataset-agnostic: it reads whatever harness output directory
it is given, whether produced from the committed synthetic fixture or from
the real 900-pair dataset. Tables are timestamp-free and byte-identical
across re-runs on identical input.

Examples:
    # Analyse a harness run; kappa dataset taken from the run's run_meta.json:
    python scripts/run_analysis.py --results-dir results/run-001 \\
        --out-dir results/run-001/analysis

    # Point kappa at an explicit dataset instead:
    python scripts/run_analysis.py --results-dir results/run-001 \\
        --dataset data/ground_truth.csv --out-dir analysis/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.analysis.hypothesis import DEFAULT_ALPHA, AnovaDesignError
from levy.analysis.io import HarnessContractError
from levy.analysis.report import KAPPA_THRESHOLD, build_analysis_bundle
from levy.dataset.schema import QueryPairValidationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Harness output directory containing results.csv (default: results/)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the analysis bundle (tables, figures, metadata)")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset file for the kappa section (default: the dataset_path recorded in run_meta.json)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Significance level for the hypothesis tests (default: {DEFAULT_ALPHA})")
    parser.add_argument("--kappa-threshold", type=float, default=KAPPA_THRESHOLD, help=f"Annotation-validity bar for kappa (default: {KAPPA_THRESHOLD}, per frozen S&D Report)")
    return parser


def main(argv=None, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        bundle = build_analysis_bundle(
            harness_dir=args.results_dir,
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            alpha=args.alpha,
            kappa_threshold=args.kappa_threshold,
        )
    except (HarnessContractError, AnovaDesignError, QueryPairValidationError) as exc:
        print(f"[run_analysis] {exc}", file=sys.stderr)
        return 1

    output_fn(f"[run_analysis] analysed {bundle.meta['inputs']['n_configurations']} configuration(s) from {args.results_dir}")
    for _, row in bundle.anova.table.iterrows():
        if not row["hypothesis"]:
            continue
        output_fn(f"  {row['hypothesis']} ({row['effect']}): p={row['p_value']:.6g} -> {row['decision']}")
    output_fn(f"  {bundle.tukey.statement}")

    kappa = bundle.kappa
    if kappa["status"] == "computed":
        provenance = "FIXTURE ONLY" if kappa["provenance"]["fixture_only"] else "released dataset"
        output_fn(f"  kappa={kappa['overall']['kappa']} (threshold {kappa['threshold']}, {provenance})")
    else:
        output_fn(f"  kappa: {kappa['status']} -- {kappa['reason']}")

    output_fn(f"[run_analysis] wrote the analysis bundle to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
