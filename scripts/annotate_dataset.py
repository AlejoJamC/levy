#!/usr/bin/env python
"""
Run a blind re-annotation session over a sampled ground-truth dataset
(LEV-3 / D2 platform tooling).

Shows `query_1`/`query_2` only (never the original label or source corpus)
and records the author's independent judgment as `author_label`. Progress is
persisted after every answer to `--progress`, so a 900-pair session can be
interrupted (Ctrl-C, closed terminal) and resumed later by re-running this
script with the same `--progress` path.

Pairs are presented workload block by workload block — `faq,chat,code` by
default, code last because its posts are the longest to read — and shuffled
within each block under `--order-seed`, so neither the sampler's ordering nor a
run of similar pairs can stand in for the judgment. The resolved order is
recorded in the progress file and reused on resume. `--workload` restricts the
session to one or more workloads (repeatable), which is what makes a
single-workload re-annotation present only that workload's pairs; combined with
the default of skipping already-labeled pairs, a re-sampled workload is the only
thing you are asked about. `--session-limit` ends a sitting cleanly after N
labeled pairs, so a long run splits into short sessions without Ctrl-C.

Writes back to the dataset's canonical paths (`--out-csv` / `--out-json`, and
`--out-ids` for the published identifiers projection), backing up whatever is
already there first: there is one ground truth, so a re-annotation updates it in
place rather than creating a second copy.

Examples:
    # Full session over the working dataset:
    python scripts/annotate_dataset.py \\
        --dataset data/ground_truth.full.json \\
        --progress data/annotation_progress.json \\
        --out-csv data/ground_truth.full.csv \\
        --out-json data/ground_truth.full.json \\
        --out-ids data/ground_truth.ids.csv

    # A 50-pair sitting over the freshly re-sampled chat workload only:
    python scripts/annotate_dataset.py \\
        --dataset data/ground_truth.full.json \\
        --progress data/annotation_progress.json \\
        --workload chat --session-limit 50 \\
        --out-csv data/ground_truth.full.csv \\
        --out-json data/ground_truth.full.json \\
        --out-ids data/ground_truth.ids.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.annotation import (
    DEFAULT_WORKLOAD_ORDER,
    AnnotationOrderError,
    BlindAnnotationSession,
    parse_workload_order,
    unordered_workloads,
)
from levy.dataset.backup import BackupError, backup_files, describe_backups
from levy.dataset.io import (
    load_dataset,
    save_dataset,
    save_distribution_csv,
    to_distribution_records,
)
from levy.dataset.schema import WORKLOADS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Input dataset (.csv or .json)")
    parser.add_argument("--progress", type=Path, required=True, help="Progress file (JSON, created/updated as you annotate)")
    parser.add_argument("--out-csv", type=Path, default=None, help="Where to save the merged dataset as CSV (default: --dataset with a .csv suffix)")
    parser.add_argument("--out-json", type=Path, default=None, help="Where to save the merged dataset as JSON (default: --dataset with a .json suffix)")
    parser.add_argument("--out-ids", type=Path, default=None, help="Also refresh the identifiers-only published artifact (e.g. data/ground_truth.ids.csv). Omitted: the ids file is left as it is")
    parser.add_argument("--workload", action="append", choices=list(WORKLOADS), default=None, help="Restrict the session to this workload; repeatable (default: all three)")
    parser.add_argument("--workload-order", type=str, default=",".join(DEFAULT_WORKLOAD_ORDER), help=f"Comma-separated order the workload blocks are presented in (default: {','.join(DEFAULT_WORKLOAD_ORDER)})")
    parser.add_argument("--order-seed", type=int, default=0, help="Seed for the within-block shuffle, recorded in the progress file (default: 0)")
    parser.add_argument("--session-limit", type=int, default=None, help="End the session cleanly after this many labeled pairs (default: no limit)")
    parser.add_argument("--overwrite", action="store_true", help="Re-annotate pairs that already have an author_label (default: skip them)")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Directory for timestamped backups of the files this run overwrites (default: <file>/../backups)")
    return parser


def main(argv=None, input_fn=input, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        workload_order = parse_workload_order(args.workload_order)
    except AnnotationOrderError as exc:
        output_fn(f"[annotate_dataset] {exc}")
        return 2

    unordered = unordered_workloads(args.workload, workload_order)
    if unordered:
        output_fn(
            f"[annotate_dataset] --workload-order does not name {unordered}; "
            "those blocks are presented after the ones it does name."
        )

    pairs = load_dataset(args.dataset)

    try:
        session = BlindAnnotationSession(
            pairs,
            progress_path=args.progress,
            input_fn=input_fn,
            output_fn=output_fn,
            overwrite=args.overwrite,
            workloads=args.workload,
            workload_order=workload_order,
            order_seed=args.order_seed,
            session_limit=args.session_limit,
            backup_dir=args.backup_dir,
        )
    except AnnotationOrderError as exc:
        output_fn(f"[annotate_dataset] {exc}")
        return 2
    except BackupError as exc:
        output_fn(f"[annotate_dataset] {exc}")
        return 1

    summary = session.run()

    # Default: keep both sibling formats in sync (they carry identical content).
    out_csv = args.out_csv or args.dataset.with_suffix(".csv")
    out_json = args.out_json or args.dataset.with_suffix(".json")
    targets = [out_csv, out_json] + ([args.out_ids] if args.out_ids else [])

    try:
        made = backup_files(targets, backup_dir=args.backup_dir)
    except BackupError as exc:
        output_fn(f"[annotate_dataset] {exc}")
        output_fn(
            "[annotate_dataset] the dataset was NOT written; your answers are safe in "
            f"{args.progress} and will be re-applied on the next run."
        )
        return 1

    if made:
        output_fn(describe_backups(made))
    save_dataset(pairs, out_csv, out_json)
    if args.out_ids:
        save_distribution_csv(to_distribution_records(pairs), args.out_ids)

    output_fn(
        f"\n[annotate_dataset] selected={summary.selected_pairs} "
        f"already_labeled_before_run={summary.already_labeled} "
        f"newly_labeled={summary.newly_labeled} skipped={summary.skipped} "
        f"quit_early={summary.quit_early} "
        f"session_limit_reached={summary.session_limit_reached} "
        f"stale_progress_dropped={summary.stale_progress_dropped} "
        f"remaining={session.remaining_count()}"
    )
    output_fn(f"[annotate_dataset] wrote {out_csv} and {out_json}")
    if args.out_ids:
        output_fn(f"[annotate_dataset] refreshed {args.out_ids} (identifiers only)")
    elif summary.newly_labeled:
        output_fn(
            "[annotate_dataset] note: the identifiers-only artifact was not refreshed "
            "(no --out-ids), so it does not yet carry these labels."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
