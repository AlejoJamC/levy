#!/usr/bin/env python
"""
Rebuild the full working dataset from the published identifiers (LEV-12).

`data/ground_truth.ids.csv` carries which pairs were sampled and how they are
labeled, but no query text — Quora QQP grants no redistribution right and SODD
is CC BY-NC-SA, so this repository cannot publish the text itself. This script
puts the text back, from the reader's own copy of the raw corpora.

    python scripts/fetch_corpora.py        # once, to populate data/raw/
    python scripts/rehydrate_dataset.py    # -> data/ground_truth.full.{csv,json}

The result is byte-identical to the dataset originally sampled: the sidecar
(`data/ground_truth.ids.meta.json`) supplies the adapter options and
normalisation the original run used, and the distribution file supplies the
order. Nothing is re-derived, so there is nothing to drift.

Fully offline: it reads local files only. It writes nothing on failure — a
missing corpus or an identifier that cannot be resolved is an error naming what
is wrong, never a quietly shorter dataset. An existing working dataset is backed
up to `data/backups/` before it is overwritten (a rehydration whose `author_label`
column comes from a stale ids file would otherwise discard annotation work
irrecoverably), and a backup that cannot be created aborts the write.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.backup import BackupError, backup_files, describe_backups
from levy.dataset.corpora import (
    DEFAULT_RAW_ROOT,
    DEFAULT_REGISTRY_PATH,
    CorpusRegistryError,
    load_registry,
)
from levy.dataset.io import load_distribution_csv, save_dataset, DatasetValidationError
from levy.dataset.sampling import (
    CorpusSource,
    CorpusSourceError,
    MockCorpusSource,
    make_source,
)
from levy.dataset.schema import DistributionRecord, QueryPair

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IDS = REPO_ROOT / "data" / "ground_truth.ids.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "ground_truth.full.csv"
DEFAULT_OUT_JSON = REPO_ROOT / "data" / "ground_truth.full.json"


class RehydrationError(RuntimeError):
    """Raised when the working dataset cannot be reconstructed."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS, help="Identifiers-only distribution file (default: data/ground_truth.ids.csv)")
    parser.add_argument("--meta", type=Path, default=None, help="Sampling sidecar (default: <ids>.meta.json)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT, help="Root of the acquired raw corpora (default: data/raw)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Corpus provenance registry (default: data/corpora.json)")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Output CSV path (default: data/ground_truth.full.csv)")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output JSON path (default: data/ground_truth.full.json)")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Directory for timestamped backups of the files this run overwrites (default: <file>/../backups)")
    return parser


def load_meta(path: Path) -> Dict:
    """Read the sampling sidecar; it carries the adapter options to reproduce."""
    if not path.is_file():
        raise RehydrationError(
            f"{path}: sampling sidecar not found. It records the seed, adapter options "
            "and normalisation the sampled dataset was built with, and rehydration "
            "cannot reproduce the same text without it. It is published alongside the "
            "identifiers file."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RehydrationError(f"{path}: invalid JSON: {exc}") from exc


def build_sources(
    needed: List[Tuple[str, str]], meta: Dict, registry, raw_dir: Path
) -> Dict[Tuple[str, str], CorpusSource]:
    """
    One `CorpusSource` per (corpus, workload) the distribution file references,
    configured with the options the original sampling run recorded.

    Keyed by the pair, not by corpus alone: pre-flight validation guarantees
    one corpus per workload for real corpora, but a dataset sampled offline has
    three synthetic sources all reporting `source_corpus="mock"`, and they are
    three different generators.
    """
    specs = {}
    for workload, spec in (meta.get("workloads") or {}).items():
        specs[(spec.get("corpus", ""), workload)] = dict(spec.get("options") or {})

    sources: Dict[Tuple[str, str], CorpusSource] = {}
    for key in needed:
        corpus, workload = key
        options = specs.get(key)
        if options is None:
            raise RehydrationError(
                f"corpus {corpus!r} / workload {workload!r} is referenced by the identifiers "
                "file but is absent from the sidecar's 'workloads' section, so its adapter "
                "options are unknown"
            )
        if corpus == MockCorpusSource.name:
            # A dataset sampled offline against synthetic sources rehydrates
            # from the same deterministic generator, no raw corpus involved.
            sources[key] = MockCorpusSource(workload, **options)
            continue

        entry = registry.get(corpus)
        missing = entry.missing_paths(raw_dir)
        if missing:
            raise RehydrationError(
                f"corpus {corpus!r} is not acquired: missing {[str(p) for p in missing]}.\n"
                "Acquire it first:  python scripts/fetch_corpora.py --corpus " + corpus
            )
        adapter_options = {
            key_: value
            for key_, value in options.items()
            # `shards`, `splits` and `normalisation` are recorded for the reader,
            # not accepted as constructor arguments — the files come from the registry.
            if key_ not in {"shards", "splits", "normalisation"}
        }
        sources[key] = make_source(entry.adapter, entry.paths(raw_dir), **adapter_options)
    return sources


def build_text_index(
    sources: Dict[Tuple[str, str], CorpusSource], wanted: Dict[Tuple[str, str], set]
) -> Dict[Tuple[str, str, str], Tuple[str, str]]:
    """
    `(corpus, workload, source_pair_id) -> (query_1, query_2)` for exactly the
    identifiers wanted.

    One linear pass per source: 900 identifiers against SODD's ~1.4M rows is a
    single scan, and holding only the wanted rows keeps the corpus off the heap.
    """
    index: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for (corpus, workload), source in sources.items():
        needed = wanted[(corpus, workload)]
        for candidate in source.iter_candidates():
            if candidate.source_pair_id in needed:
                index[(corpus, workload, candidate.source_pair_id)] = (
                    candidate.query_1,
                    candidate.query_2,
                )
    return index


def rehydrate(
    records: List[DistributionRecord],
    index: Dict[Tuple[str, str, str], Tuple[str, str]],
) -> List[QueryPair]:
    """Full pairs in the distribution file's order — that ordering is the byte-identity."""
    pairs: List[QueryPair] = []
    for record in records:
        key = (record.source_corpus, record.workload, record.source_pair_id)
        try:
            query_1, query_2 = index[key]
        except KeyError:
            raise RehydrationError(
                f"pair {record.pair_id!r}: source_pair_id {record.source_pair_id!r} was not "
                f"found in corpus {record.source_corpus!r}. The corpus on disk is not the "
                "snapshot this dataset was sampled from — compare its checksum against "
                "data/corpora.json."
            ) from None
        pairs.append(record.to_query_pair(query_1, query_2))
    return pairs


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    meta_path = args.meta or args.ids.with_suffix(".meta.json")

    try:
        records = load_distribution_csv(args.ids)
        meta = load_meta(meta_path)
        registry = load_registry(args.registry)

        wanted: Dict[Tuple[str, str], set] = {}
        for record in records:
            wanted.setdefault(
                (record.source_corpus, record.workload), set()
            ).add(record.source_pair_id)

        sources = build_sources(sorted(wanted), meta, registry, args.raw_dir)
        index = build_text_index(sources, wanted)
        pairs = rehydrate(records, index)
    except (RehydrationError, CorpusRegistryError, CorpusSourceError, DatasetValidationError) as exc:
        print(f"[rehydrate_dataset] {exc}", file=sys.stderr)
        print("[rehydrate_dataset] nothing written.", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[rehydrate_dataset] {exc}", file=sys.stderr)
        print("[rehydrate_dataset] nothing written.", file=sys.stderr)
        return 1

    try:
        made = backup_files(
            [args.out_csv, args.out_json], backup_dir=args.backup_dir
        )
    except BackupError as exc:
        print(f"[rehydrate_dataset] {exc}", file=sys.stderr)
        print("[rehydrate_dataset] nothing written.", file=sys.stderr)
        return 1
    if made:
        print(describe_backups(made), file=sys.stderr)

    save_dataset(pairs, args.out_csv, args.out_json)
    print(
        f"[rehydrate_dataset] rebuilt {len(pairs)} pairs from {args.ids}\n"
        f"[rehydrate_dataset] wrote {args.out_csv} and {args.out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
