#!/usr/bin/env python
"""
Sample a ground-truth dataset from raw corpora (D2 platform tooling).

For each workload (faq / code / chat) this resolves a corpus source, runs
pre-flight validation over all three at once, and — only if validation is
clean — draws a seeded, stratified sample of query pairs. It never downloads
anything: `scripts/fetch_corpora.py` populates `data/raw/`, and the corpus
each workload uses, along with its expected filenames, comes from
`data/corpora.json`.

Outputs, all written together or not at all:

    --out-csv / --out-json   the full working dataset (query text included)
    --out-ids                identifiers and labels only, no query text: this
                             is the artifact the repository publishes
    --out-meta               the sampling sidecar: per-workload seed and run
                             timestamp, adapter options, corpus versions and
                             input checksums

If a workload's raw corpus is absent, it falls back to `MockCorpusSource` so
the pipeline can be exercised offline against synthetic data — pairs produced
that way carry `source_corpus="mock"`. Pass `--require-real` for a production
run: the fallback then becomes a hard error naming the missing corpus.

**Re-sampling one workload (`--workload`).** By the time a workload needs
re-drawing, the dataset is live: 900 pairs, all annotated. `--workload chat`
therefore reads only the chat corpus and performs surgery on the existing
dataset — chat's 300 rows are replaced, the other 600 rows and their
`author_label`s pass through untouched, and the new chat pairs come back
unannotated so the annotation step presents exactly them. New pairs are drawn
from the candidate pool *minus* everything already in the dataset, so a
re-sample is disjoint from what it replaces; if the pool cannot cover
`--n-per-workload` after that exclusion, the run fails naming the workload and
the shortfall and writes nothing.

A re-sample also **invalidates the annotation progress for the pairs it
replaced** (`--progress`, default `annotation_progress.json` beside `--out-csv`).
It has to: the replaced pairs keep their `pair_id`s, so their recorded answers
would otherwise be re-applied by the next annotation session to the new pairs —
silently, and to precisely the pairs that are supposed to come back unannotated.
The other workloads' answers are untouched, and the file is backed up first.

There is one ground truth, at its canonical paths, always. Every mode here
writes back to those paths — never a `_v2`, a `_new`, or a dated copy beside
them — and backs up whatever it is about to overwrite first (`--backup-dir`,
default `data/backups/`). If a backup cannot be made, nothing is written.

Examples:
    # Fully offline smoke test (no raw corpora present -> all three mocked):
    python scripts/sample_dataset.py --n-per-workload 5 --seed 42 \\
        --out-csv /tmp/sample.csv --out-json /tmp/sample.json

    # Production run, corpora already acquired into data/raw/:
    python scripts/sample_dataset.py --require-real \\
        --n-per-workload 300 --seed 42 \\
        --out-csv data/ground_truth.full.csv \\
        --out-json data/ground_truth.full.json \\
        --out-ids data/ground_truth.ids.csv

    # Re-draw the chat workload only, in place, at a new seed:
    python scripts/sample_dataset.py --require-real --workload chat \\
        --n-per-workload 300 --seed 4242 \\
        --out-csv data/ground_truth.full.csv \\
        --out-json data/ground_truth.full.json \\
        --out-ids data/ground_truth.ids.csv
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.annotation import prune_progress
from levy.dataset.backup import (
    BackupError,
    backup_files,
    backup_timestamp,
    describe_backups,
)
from levy.dataset.corpora import (
    DEFAULT_RAW_ROOT,
    DEFAULT_REGISTRY_PATH,
    CorpusRegistryError,
    load_registry,
    sha256_file,
)
from levy.dataset.io import (
    DatasetValidationError,
    load_dataset,
    load_distribution_csv,
    save_dataset,
    save_distribution_csv,
    to_distribution_records,
)
from levy.dataset.normalize import NORMALISATION_RULE
from levy.dataset.sampling import (
    CorpusSource,
    CorpusSourceError,
    MockCorpusSource,
    make_source,
    sample_dataset,
    sample_workload,
)
from levy.dataset.schema import QueryPair, WORKLOADS
from levy.dataset.validation import validate_sources
from levy.dataset.workload_update import (
    ExcludingCorpusSource,
    WorkloadUpdateError,
    check_ids_alignment,
    clear_author_labels,
    resample_shortfall_message,
    source_pair_ids,
    splice_workload,
    verify_disjoint,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT, help="Root of the acquired raw corpora (default: data/raw)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Corpus provenance registry (default: data/corpora.json)")
    parser.add_argument("--workload", action="append", choices=list(WORKLOADS), default=None, help="Re-sample only this workload, in place, leaving the others (and their author_label values) untouched; repeatable. Default: sample all three from scratch")
    parser.add_argument("--quora-tsv", type=Path, default=None, help="Override the faq corpus file (Quora QQP TSV)")
    parser.add_argument("--sodd-parquet", type=Path, nargs="+", default=None, help="Override the code corpus shards (SODD gzipped parquet)")
    parser.add_argument("--pit2015-data", type=Path, nargs="+", default=None, help="Override the chat corpus files (PIT-2015 train/dev .data)")
    parser.add_argument("--sodd-hard-negatives", action="store_true", help="Use SODD classes 1/2 ('similar') as negatives alongside class 3, making the negative pool adversarially hard (default: off, class 3 only)")
    parser.add_argument("--n-per-workload", type=int, default=300, help="Pairs to sample per workload (default: 300, per D2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Fraction of sampled pairs with original_label=1 (default: 0.5, balanced)")
    parser.add_argument("--mock-candidates", type=int, default=None, help="Candidate pool size per workload when falling back to MockCorpusSource (default: max(40, 4 * n-per-workload))")
    parser.add_argument("--require-real", action="store_true", help="Refuse to fall back to MockCorpusSource; fail naming the missing corpus")
    parser.add_argument("--skip-validation", action="store_true", help="Skip pre-flight validation (smoke tests only; never for a production run)")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path for the full dataset")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path for the full dataset")
    parser.add_argument("--out-ids", type=Path, default=None, help="Output path for the identifiers-only distribution file (default: <out-csv>.ids.csv)")
    parser.add_argument("--out-meta", type=Path, default=None, help="Output path for the sampling sidecar (default: alongside --out-ids as .meta.json)")
    parser.add_argument("--progress", type=Path, default=None, help="Annotation progress file whose entries for re-sampled pairs must be invalidated (default: annotation_progress.json beside --out-csv). Only read/written with --workload")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Directory for timestamped backups of the files this run overwrites (default: <file>/../backups)")
    return parser


def _default_ids_path(out_csv: Path) -> Path:
    return out_csv.with_suffix("").with_suffix(".ids.csv") if out_csv.suffixes else out_csv.parent / "ids.csv"


def _default_meta_path(out_ids: Path) -> Path:
    return out_ids.with_suffix(".meta.json")


def target_workloads(selected: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    The workloads this run samples, in canonical `WORKLOADS` order.

    Canonical order rather than the order the flags were typed in, so
    `--workload chat --workload faq` and `--workload faq --workload chat`
    produce the same dataset.
    """
    if not selected:
        return tuple(WORKLOADS)
    chosen = set(selected)
    return tuple(w for w in WORKLOADS if w in chosen)


def resolve_sources(args, registry, workloads: Sequence[str]) -> Dict[str, CorpusSource]:
    """
    Pick a `CorpusSource` per requested workload: an explicit override, else the
    corpus the registry assigns to that workload if its files are all present,
    else a synthetic fallback (or a hard error under --require-real).

    Only `workloads` are resolved — a `--workload chat` run must not require the
    Quora TSV or the 1.4M-row SODD shards to be on disk at all.
    """
    overrides = {
        "faq": [args.quora_tsv] if args.quora_tsv else None,
        "code": list(args.sodd_parquet) if args.sodd_parquet else None,
        "chat": list(args.pit2015_data) if args.pit2015_data else None,
    }
    adapter_options = {"code": {"hard_negatives": args.sodd_hard_negatives}}
    mock_candidates = args.mock_candidates or max(40, 4 * args.n_per_workload)

    sources: Dict[str, CorpusSource] = {}
    for workload in workloads:
        entry = registry.for_workload(workload)
        paths: Optional[List[Path]] = overrides[workload]
        if paths is None:
            expected = entry.paths(args.raw_dir)
            missing = [p for p in expected if not p.is_file()]
            if missing:
                if args.require_real:
                    raise SystemExit(
                        f"[sample_dataset] --require-real: corpus {entry.key!r} for workload "
                        f"{workload!r} is not acquired; missing "
                        f"{[str(p) for p in missing]}. Run `python scripts/fetch_corpora.py`."
                    )
                print(
                    f"[sample_dataset] {entry.key} not present under {args.raw_dir}: "
                    f"falling back to MockCorpusSource for {workload!r}",
                    file=sys.stderr,
                )
                sources[workload] = MockCorpusSource(
                    workload, n_candidates=mock_candidates, seed=args.seed
                )
                continue
            paths = expected
        sources[workload] = make_source(
            entry.adapter, paths, **adapter_options.get(workload, {})
        )
    return sources


def workload_entry(args, source: CorpusSource, n_pairs: int, timestamp: str) -> Dict:
    """
    One workload's sidecar entry, including the seed and UTC timestamp of the
    run that produced *its* rows.

    Per-workload provenance is what lets a dataset whose workloads were sampled
    at different times still be reproducible: the top-level seed describes the
    latest invocation only, so a workload re-drawn at a different seed has to
    carry its own.
    """
    files = []
    for path in source.source_files():
        files.append(
            {
                "filename": Path(path).name,
                "sha256": sha256_file(path) if Path(path).is_file() else None,
            }
        )
    return {
        "corpus": source.name,
        "adapter": type(source).__name__,
        "synthetic": isinstance(source, MockCorpusSource),
        "options": source.options(),
        "label_mapping": source.label_mapping().to_dict(),
        "files": files,
        "n_pairs": n_pairs,
        "seed": args.seed,
        "n_per_workload": args.n_per_workload,
        "positive_ratio": args.positive_ratio,
        "sampled_at_utc": timestamp,
    }


def build_manifest(
    args,
    registry,
    sources: Dict[str, CorpusSource],
    pairs: List[QueryPair],
    timestamp: str,
    previous: Optional[Dict] = None,
) -> Dict:
    """
    The sampling sidecar: everything a third party needs to prove they hold
    byte-identical inputs, and nothing that would leak corpus text.

    For a per-workload re-sample, `previous` is the sidecar already on disk: the
    untouched workloads keep their recorded entries verbatim (their seed and
    timestamp are historical facts about how those rows were drawn, not this
    run's parameters), and only the re-sampled workloads' entries are replaced.
    """
    base = dict(previous or {})
    workloads: Dict[str, Dict] = dict(base.get("workloads") or {})
    for workload, source in sources.items():
        workloads[workload] = workload_entry(
            args,
            source,
            n_pairs=sum(1 for p in pairs if p.workload == workload),
            timestamp=timestamp,
        )
    for workload, entry in workloads.items():
        if workload in sources:
            continue
        # Keep recorded pair counts truthful for workloads this run did not draw,
        # and backfill per-workload provenance for entries written before it was
        # recorded — from the previous sidecar's top level, which for a
        # full-dataset run is exactly the right historical fact. Flagged, so a
        # backfilled seed is never mistaken for one the original run recorded
        # per workload.
        entry = dict(entry)
        entry["n_pairs"] = sum(1 for p in pairs if p.workload == workload)
        if "seed" not in entry:
            entry.update(
                seed=base.get("seed"),
                n_per_workload=base.get("n_per_workload"),
                positive_ratio=base.get("positive_ratio"),
                sampled_at_utc=base.get("generated_at_utc"),
                provenance_backfilled=True,
            )
        workloads[workload] = entry

    corpora = dict(base.get("corpora") or {})
    for source in sources.values():
        if source.corpus_key:
            corpora[source.corpus_key] = registry.get(source.corpus_key).provenance()

    return {
        "generated_by": "scripts/sample_dataset.py",
        "generated_at_utc": timestamp,
        "seed": args.seed,
        "n_per_workload": args.n_per_workload,
        "positive_ratio": args.positive_ratio,
        "n_pairs": len(pairs),
        "sampled_workloads_this_run": sorted(sources),
        "provenance_note": (
            "Top-level seed / n_per_workload / positive_ratio describe the most recent "
            "invocation. When workloads were sampled at different times, the per-workload "
            "'seed' and 'sampled_at_utc' under 'workloads' are authoritative."
        ),
        "normalisation_rule": NORMALISATION_RULE,
        "workloads": workloads,
        "corpora": corpora,
        "tool_versions": _tool_versions(),
    }


def _tool_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("pyarrow", "pandas"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - an absent optional dep is recorded as absent
            versions[name] = None
    return versions


def load_current_dataset(out_csv: Path, out_json: Path) -> List[QueryPair]:
    """
    Load the ground truth a per-workload re-sample is about to operate on.

    The CSV is preferred (it is the canonical working file); the JSON twin is a
    fallback for a dataset that only exists in that format. Neither present is
    an error, not an empty start: `--workload` means "replace this workload's
    rows in the existing dataset", and there is nothing to replace them in.
    """
    for candidate in (out_csv, out_json):
        if candidate.is_file():
            return load_dataset(candidate)
    raise WorkloadUpdateError(
        f"--workload re-samples one workload inside the existing ground truth, but neither "
        f"{out_csv} nor {out_json} exists. Sample the full dataset first (drop --workload), "
        "or rebuild the working dataset with `python scripts/rehydrate_dataset.py`."
    )


def run_partial_sample(
    args,
    sources: Dict[str, CorpusSource],
    workloads: Sequence[str],
    current: List[QueryPair],
    out_ids: Path,
) -> Tuple[List[QueryPair], Set[str]]:
    """
    Re-sample `workloads` inside `current` and return `(dataset, dropped_pair_ids)`
    — the complete spliced dataset, and the `pair_id`s whose meaning changed, so
    the caller can invalidate their annotation progress. Every guard that could
    reject the run happens here, before the caller writes anything.
    """
    untouched = [w for w in WORKLOADS if w not in set(workloads)]
    if out_ids.is_file():
        check_ids_alignment(current, load_distribution_csv(out_ids), untouched)

    pairs = current
    dropped_pair_ids: Set[str] = set()
    for workload in workloads:
        previous_ids = source_pair_ids(pairs, workload)
        excluding = ExcludingCorpusSource(sources[workload], previous_ids)
        try:
            fresh = sample_workload(
                excluding,
                n=args.n_per_workload,
                seed=args.seed,
                positive_ratio=args.positive_ratio,
            )
        except CorpusSourceError as exc:
            raise WorkloadUpdateError(
                resample_shortfall_message(workload, len(previous_ids), exc)
            ) from exc

        verify_disjoint(fresh, previous_ids)
        clear_author_labels(fresh)

        spliced = splice_workload(pairs, workload, fresh)
        pairs = spliced.pairs
        # Both sides: a `pair_id` that now names a different pair, and one that
        # no longer exists because the new block is shorter.
        dropped_pair_ids.update(p.pair_id for p in spliced.replaced)
        dropped_pair_ids.update(p.pair_id for p in spliced.removed)
        annotated_kept = sum(
            1 for p in pairs if p.workload != workload and p.author_label is not None
        )
        print(
            f"[sample_dataset] {workload}: replaced {len(spliced.removed)} pair(s) with "
            f"{len(spliced.replaced)} freshly sampled one(s) at seed {args.seed} "
            f"({len(previous_ids)} already-sampled candidate(s) excluded); "
            f"{annotated_kept} annotated pair(s) in other workloads left untouched"
            + ("; workload was absent and has been appended" if spliced.appended else ""),
            file=sys.stderr,
        )
    return pairs, dropped_pair_ids


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_ids = args.out_ids or _default_ids_path(args.out_csv)
    out_meta = args.out_meta or _default_meta_path(out_ids)
    # Derived from the dataset's own directory, never a repo-absolute default: a
    # run writing into a temp directory must not reach into `data/` and prune the
    # real annotation progress.
    progress = args.progress or args.out_csv.parent / "annotation_progress.json"
    workloads = target_workloads(args.workload)
    partial = bool(args.workload)

    try:
        registry = load_registry(args.registry)
    except CorpusRegistryError as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        return 2

    try:
        sources = resolve_sources(args, registry, workloads)
    except (CorpusSourceError, CorpusRegistryError) as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        return 2

    validation_sources: Dict[str, CorpusSource] = dict(sources)
    previous_meta: Optional[Dict] = None
    current: List[QueryPair] = []

    if partial:
        # Validation must see the *post-exclusion* pool, so a re-sample that
        # cannot be satisfied fails in pre-flight rather than mid-sample.
        try:
            current = load_current_dataset(args.out_csv, args.out_json)
        except (WorkloadUpdateError, DatasetValidationError) as exc:
            print(f"[sample_dataset] {exc}", file=sys.stderr)
            return 1
        for workload in workloads:
            validation_sources[workload] = ExcludingCorpusSource(
                sources[workload], source_pair_ids(current, workload)
            )
        # Read the progress file now, not after the dataset is written: pruning it
        # is part of this run, and a malformed one has to stop the run while
        # stopping still means "nothing written".
        if progress.is_file():
            try:
                if not isinstance(json.loads(progress.read_text(encoding="utf-8")), dict):
                    raise json.JSONDecodeError("expected a JSON object", "", 0)
            except json.JSONDecodeError as exc:
                print(
                    f"[sample_dataset] {progress}: not a readable progress file ({exc}). "
                    "A re-sample has to invalidate this workload's recorded annotations, and "
                    "cannot. Fix or move the file (or pass --progress). Nothing written.",
                    file=sys.stderr,
                )
                return 1

        if out_meta.is_file():
            try:
                previous_meta = json.loads(out_meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(
                    f"[sample_dataset] {out_meta}: invalid JSON: {exc}. The sidecar carries the "
                    "other workloads' provenance and cannot be rewritten from scratch without "
                    "losing it. Nothing written.",
                    file=sys.stderr,
                )
                return 1

    if not args.skip_validation:
        report = validate_sources(
            validation_sources,
            n_per_workload=args.n_per_workload,
            positive_ratio=args.positive_ratio,
            registry=registry,
            raw_root=args.raw_dir,
        )
        print(
            "[sample_dataset] pre-flight validation"
            + (f" ({', '.join(workloads)}, excluding already-sampled pairs):" if partial else ":"),
            file=sys.stderr,
        )
        print(report.render(), file=sys.stderr)
        if not report.ok:
            print(
                "[sample_dataset] validation failed; nothing written. Resolve every "
                "finding above before sampling.",
                file=sys.stderr,
            )
            return 1

    dropped_pair_ids: Set[str] = set()
    try:
        if partial:
            pairs, dropped_pair_ids = run_partial_sample(
                args, sources, workloads, current, out_ids
            )
        else:
            pairs = sample_dataset(
                sources,
                n_per_workload=args.n_per_workload,
                seed=args.seed,
                positive_ratio=args.positive_ratio,
            )
    except (CorpusSourceError, WorkloadUpdateError, DatasetValidationError) as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        return 1

    timestamp = backup_timestamp()
    try:
        made = backup_files(
            [args.out_csv, args.out_json, out_ids, out_meta]
            + ([progress] if dropped_pair_ids else []),
            timestamp=timestamp,
            backup_dir=args.backup_dir,
        )
    except BackupError as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        print("[sample_dataset] nothing written.", file=sys.stderr)
        return 1
    if made:
        print(describe_backups(made), file=sys.stderr)

    save_dataset(pairs, args.out_csv, args.out_json)
    save_distribution_csv(to_distribution_records(pairs), out_ids)

    if dropped_pair_ids:
        # The re-sampled workload's `pair_id`s now name different pairs. Left in
        # place, their recorded answers would be re-applied to the new pairs by
        # the next annotation session — silently, and to exactly the pairs that
        # are supposed to come back unannotated.
        pruned = prune_progress(progress, dropped_pair_ids, pairs)
        if pruned:
            print(
                f"[sample_dataset] dropped {pruned} recorded annotation(s) from {progress} "
                "for the re-sampled pairs; the other workloads' answers are untouched.",
                file=sys.stderr,
            )

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(
        json.dumps(
            build_manifest(args, registry, sources, pairs, timestamp, previous_meta),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    scope = f"re-sampled {', '.join(workloads)} in place; " if partial else ""
    print(
        f"[sample_dataset] {scope}wrote {len(pairs)} pairs to {args.out_csv} and {args.out_json}\n"
        f"[sample_dataset] wrote {out_ids} (identifiers only) and {out_meta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
