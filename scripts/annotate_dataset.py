#!/usr/bin/env python
"""
Run a blind re-annotation session over a sampled ground-truth dataset
(LEV-3 / D2 platform tooling).

Shows `query_1`/`query_2` only (never the original label or source corpus)
and records the author's independent judgment as `author_label`. Progress is
persisted after every answer to `--progress`, so a 900-pair session can be
interrupted (Ctrl-C, closed terminal) and resumed later by re-running this
script with the same `--progress` path.

Performing the actual 900-pair blind re-annotation is an author task; this
script is the tool that makes that task resumable and blind, not the
annotation itself.

Examples:
    python scripts/annotate_dataset.py \\
        --dataset data/ground_truth.json \\
        --progress data/.annotation_progress.json \\
        --out-csv data/ground_truth.csv --out-json data/ground_truth.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.annotation import BlindAnnotationSession
from levy.dataset.io import load_dataset, save_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Input dataset (.csv or .json)")
    parser.add_argument("--progress", type=Path, required=True, help="Progress file (JSON, created/updated as you annotate)")
    parser.add_argument("--out-csv", type=Path, default=None, help="Where to save the merged dataset as CSV (default: --dataset with a .csv suffix)")
    parser.add_argument("--out-json", type=Path, default=None, help="Where to save the merged dataset as JSON (default: --dataset with a .json suffix)")
    parser.add_argument("--overwrite", action="store_true", help="Re-annotate pairs that already have an author_label (default: skip them)")
    return parser


def main(argv=None, input_fn=input, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)

    pairs = load_dataset(args.dataset)
    session = BlindAnnotationSession(
        pairs,
        progress_path=args.progress,
        input_fn=input_fn,
        output_fn=output_fn,
        overwrite=args.overwrite,
    )
    summary = session.run()

    # Default: keep both sibling formats in sync (they carry identical content).
    out_csv = args.out_csv or args.dataset.with_suffix(".csv")
    out_json = args.out_json or args.dataset.with_suffix(".json")
    save_dataset(pairs, out_csv, out_json)

    output_fn(
        f"\n[annotate_dataset] total={summary.total_pairs} "
        f"already_labeled_before_run={summary.already_labeled} "
        f"newly_labeled={summary.newly_labeled} skipped={summary.skipped} "
        f"quit_early={summary.quit_early} remaining={session.remaining_count()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
