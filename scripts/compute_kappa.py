#!/usr/bin/env python
"""
Compute Cohen's kappa between original corpus labels and the author's blind
re-annotation over a ground-truth dataset (LEV-3 / D2).

Frozen success criterion (S&D Report): kappa > 0.7 computed over the full
900-pair set. This script reports overall kappa plus a per-workload
breakdown, and (with `--strict`) exits non-zero if overall kappa falls below
`--threshold` -- but only once every pair has an `author_label`; a partially
annotated dataset is reported but never fails the `--strict` gate, since it
is expected to be below threshold before annotation is complete.

Example:
    python scripts/compute_kappa.py --dataset data/ground_truth.json --strict
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.io import load_dataset
from levy.dataset.kappa import KappaResult, kappa_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset file (.csv or .json)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Kappa success threshold (default: 0.7, per frozen S&D Report)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if the fully annotated dataset's overall kappa is below --threshold")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to also write the report as JSON")
    return parser


def _format_result(label: str, result: KappaResult) -> str:
    if result.kappa is None:
        return f"{label}: no annotated pairs (n_excluded_unannotated={result.n_excluded_unannotated})"
    return (
        f"{label}: kappa={result.kappa:.4f} "
        f"(n_annotated={result.n_annotated}, excluded_unannotated={result.n_excluded_unannotated}, "
        f"po={result.observed_agreement:.4f}, pe={result.expected_agreement:.4f}, "
        f"confusion={result.confusion})"
    )


def _result_to_dict(result: KappaResult) -> dict:
    return {
        "n_annotated": result.n_annotated,
        "n_excluded_unannotated": result.n_excluded_unannotated,
        "observed_agreement": result.observed_agreement,
        "expected_agreement": result.expected_agreement,
        "kappa": result.kappa,
        "confusion": result.confusion,
    }


def main(argv=None, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)
    pairs = load_dataset(args.dataset)
    report = kappa_report(pairs)

    output_fn(f"Cohen's kappa report for {args.dataset} ({len(pairs)} total pairs)")
    output_fn(_format_result("overall", report.overall))
    for workload, result in report.per_workload.items():
        output_fn(_format_result(f"  workload={workload}", result))

    if args.out_json:
        payload = {
            "overall": _result_to_dict(report.overall),
            "per_workload": {w: _result_to_dict(r) for w, r in report.per_workload.items()},
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        output_fn(f"Wrote JSON report to {args.out_json}")

    if args.strict:
        fully_annotated = report.overall.n_excluded_unannotated == 0
        if fully_annotated and (report.overall.kappa is None or report.overall.kappa < args.threshold):
            output_fn(
                f"[compute_kappa] STRICT FAIL: overall kappa "
                f"{report.overall.kappa} < threshold {args.threshold}"
            )
            return 1
        if not fully_annotated:
            output_fn(
                "[compute_kappa] --strict requested but dataset is not fully annotated; "
                "not gating on the threshold yet."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
