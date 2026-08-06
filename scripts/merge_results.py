#!/usr/bin/env python
"""
Merge partial harness runs into one complete 30-configuration result set
(LEV-4 / D3).

The analysis pipeline needs all 30 rows in one `results.csv`, but a production
run often arrives in pieces: `run_experiments.py --workloads chat` writes 10
rows, one re-sampled workload gets re-run on its own, a long sweep is
interrupted. This merges those directories into a single result set, in the
canonical grid order, without re-running anything.

Two conditions fail the merge loudly, because both would otherwise produce a
file that looks complete:

  * a `config_id` present in more than one run — one of them is stale, and the
    merge does not get to pick a winner;
  * a merged set that is not exactly the frozen grid (2 models x 3 workloads x
    5 thresholds) — reported before the ANOVA assumes a balanced design.

List every input directory, including the one you are merging into. `--out-dir`
may be one of them (an in-place merge); whatever is already there is backed up
to `<out-dir>/backups/` first, and a backup that cannot be created aborts the
write.

Examples:
    # Three per-workload runs into a fresh result set:
    python scripts/merge_results.py --out-dir results/run-001 \\
        results/run-faq results/run-code results/run-chat

    # Fold a re-run of the chat workload into an existing result set in place.
    # The chat rows must not still be in results/run-001 — remove or re-run
    # that directory's cells rather than merging a duplicate.
    python scripts/merge_results.py --out-dir results/run-001 \\
        results/run-001 results/run-chat-reseeded
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.backup import (
    BackupError,
    backup_files,
    backup_timestamp,
    describe_backups,
)
from levy.experiment.merge import (
    DECISIONS_FILENAME,
    RESULTS_FILENAME,
    RUN_META_FILENAME,
    ResultsMergeError,
    check_grid_coverage,
    load_input,
    merge_decisions,
    merge_results,
    merge_run_meta,
    sort_by_grid,
    sort_decisions_by_grid,
    write_rows,
)
from levy.experiment.runner import DECISIONS_FIELDNAMES, RESULTS_FIELDNAMES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Harness output directories to merge (each holding results.csv)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where the merged result set is written; may be one of the inputs (in-place)")
    parser.add_argument("--allow-partial", action="store_true", help="Write the merged set even if it does not cover the full 30-configuration grid (diagnostics only; the analysis step will still refuse it)")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Directory for timestamped backups of the files this run overwrites (default: <out-dir>/backups)")
    return parser


def main(argv=None, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        inputs = [load_input(directory) for directory in args.inputs]
        results = merge_results(inputs)
        decisions = merge_decisions(inputs)
        if args.allow_partial:
            output_fn(
                "[merge_results] --allow-partial: grid coverage not enforced; this result "
                "set is for diagnostics, not for the analysis step."
            )
        else:
            check_grid_coverage(results)
    except ResultsMergeError as exc:
        output_fn(f"[merge_results] {exc}")
        return 1

    results = sort_by_grid(results)
    decisions = sort_decisions_by_grid(decisions) if decisions is not None else None
    timestamp = backup_timestamp()

    try:
        run_meta = merge_run_meta(inputs, timestamp)
    except ResultsMergeError as exc:
        output_fn(f"[merge_results] {exc}")
        return 1

    out_results = args.out_dir / RESULTS_FILENAME
    out_decisions = args.out_dir / DECISIONS_FILENAME
    out_meta = args.out_dir / RUN_META_FILENAME

    try:
        made = backup_files(
            [out_results, out_decisions, out_meta],
            timestamp=timestamp,
            # Default resolves to <out-dir>/backups, beside the files themselves.
            backup_dir=args.backup_dir,
        )
    except BackupError as exc:
        output_fn(f"[merge_results] {exc}")
        output_fn("[merge_results] nothing written.")
        return 1
    if made:
        output_fn(describe_backups(made))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_results, RESULTS_FIELDNAMES, results)
    if decisions is not None:
        write_rows(out_decisions, DECISIONS_FIELDNAMES, decisions)
    out_meta.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    sources = ", ".join(str(item.directory) for item in inputs)
    output_fn(
        f"[merge_results] merged {len(results)} configuration row(s) from {len(inputs)} "
        f"run(s) ({sources}) into {out_results}"
    )
    if decisions is not None:
        output_fn(f"[merge_results] merged {len(decisions)} decision row(s) into {out_decisions}")
    else:
        output_fn(f"[merge_results] no {DECISIONS_FILENAME} in any input; none written")
    output_fn(f"[merge_results] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
