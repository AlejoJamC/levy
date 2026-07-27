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
    --out-meta               the sampling sidecar: seed, ratio, adapter
                             options, corpus versions and input checksums

If a workload's raw corpus is absent, it falls back to `MockCorpusSource` so
the pipeline can be exercised offline against synthetic data — pairs produced
that way carry `source_corpus="mock"`. Pass `--require-real` for a production
run: the fallback then becomes a hard error naming the missing corpus.

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
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.corpora import (
    DEFAULT_RAW_ROOT,
    DEFAULT_REGISTRY_PATH,
    CorpusRegistryError,
    load_registry,
    sha256_file,
)
from levy.dataset.io import (
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
)
from levy.dataset.schema import WORKLOADS
from levy.dataset.validation import validate_sources


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT, help="Root of the acquired raw corpora (default: data/raw)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Corpus provenance registry (default: data/corpora.json)")
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
    return parser


def _default_ids_path(out_csv: Path) -> Path:
    return out_csv.with_suffix("").with_suffix(".ids.csv") if out_csv.suffixes else out_csv.parent / "ids.csv"


def _default_meta_path(out_ids: Path) -> Path:
    return out_ids.with_suffix(".meta.json")


def resolve_sources(args, registry) -> Dict[str, CorpusSource]:
    """
    Pick a `CorpusSource` per workload: an explicit override, else the corpus
    the registry assigns to that workload if its files are all present, else a
    synthetic fallback (or a hard error under --require-real).
    """
    overrides = {
        "faq": [args.quora_tsv] if args.quora_tsv else None,
        "code": list(args.sodd_parquet) if args.sodd_parquet else None,
        "chat": list(args.pit2015_data) if args.pit2015_data else None,
    }
    adapter_options = {"code": {"hard_negatives": args.sodd_hard_negatives}}
    mock_candidates = args.mock_candidates or max(40, 4 * args.n_per_workload)

    sources: Dict[str, CorpusSource] = {}
    for workload in WORKLOADS:
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


def build_manifest(args, registry, sources, pairs) -> Dict:
    """
    The sampling sidecar: everything a third party needs to prove they hold
    byte-identical inputs, and nothing that would leak corpus text.
    """
    workloads: Dict[str, Dict] = {}
    for workload, source in sources.items():
        files = []
        for path in source.source_files():
            files.append(
                {
                    "filename": Path(path).name,
                    "sha256": sha256_file(path) if Path(path).is_file() else None,
                }
            )
        workloads[workload] = {
            "corpus": source.name,
            "adapter": type(source).__name__,
            "synthetic": isinstance(source, MockCorpusSource),
            "options": source.options(),
            "label_mapping": source.label_mapping().to_dict(),
            "files": files,
            "n_pairs": sum(1 for p in pairs if p.workload == workload),
        }

    corpora = {}
    for source in sources.values():
        if source.corpus_key:
            corpora[source.corpus_key] = registry.get(source.corpus_key).provenance()

    return {
        "generated_by": "scripts/sample_dataset.py",
        "seed": args.seed,
        "n_per_workload": args.n_per_workload,
        "positive_ratio": args.positive_ratio,
        "n_pairs": len(pairs),
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


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_ids = args.out_ids or _default_ids_path(args.out_csv)
    out_meta = args.out_meta or _default_meta_path(out_ids)

    try:
        registry = load_registry(args.registry)
    except CorpusRegistryError as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        return 2

    try:
        sources = resolve_sources(args, registry)
    except (CorpusSourceError, CorpusRegistryError) as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        return 2

    if not args.skip_validation:
        report = validate_sources(
            sources,
            n_per_workload=args.n_per_workload,
            positive_ratio=args.positive_ratio,
            registry=registry,
            raw_root=args.raw_dir,
        )
        print("[sample_dataset] pre-flight validation:", file=sys.stderr)
        print(report.render(), file=sys.stderr)
        if not report.ok:
            print(
                "[sample_dataset] validation failed; nothing written. Resolve every "
                "finding above before sampling.",
                file=sys.stderr,
            )
            return 1

    try:
        pairs = sample_dataset(
            sources,
            n_per_workload=args.n_per_workload,
            seed=args.seed,
            positive_ratio=args.positive_ratio,
        )
    except CorpusSourceError as exc:
        print(f"[sample_dataset] {exc}", file=sys.stderr)
        return 1

    save_dataset(pairs, args.out_csv, args.out_json)
    save_distribution_csv(to_distribution_records(pairs), out_ids)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(
        json.dumps(build_manifest(args, registry, sources, pairs), indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[sample_dataset] wrote {len(pairs)} pairs to {args.out_csv} and {args.out_json}\n"
        f"[sample_dataset] wrote {out_ids} (identifiers only) and {out_meta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
