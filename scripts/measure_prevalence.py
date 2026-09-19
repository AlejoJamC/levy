#!/usr/bin/env python
"""
Measure the duplicate prevalence of the three source-corpus pools (LEV-21).

Reads the acquired raw corpora under `data/raw/` and counts, per workload, the
pool the sampler drew its 150-positive/150-negative sample from: positives,
negatives, the excluded label band, and two prevalences (see
`levy/dataset/prevalence.py` for the denominators).

    python scripts/measure_prevalence.py    # -> release/prevalence/

Adapter options and the label mapping are taken from the adapters as committed;
the sampling sidecar (`data/ground_truth.ids.meta.json`) is read, never written,
for the options and per-workload seeds the dataset was built with, and the run
aborts if its recorded label mapping differs from the adapter's. Raw-file
checksums are verified against the registry (`data/corpora.json`) before any
row is read. The 900-pair dataset is not read or modified.

Fully offline. Outputs (`prevalence.csv`, `prevalence_meta.json`) carry no query
text. An existing output is backed up to `<out-dir>/backups/` before being
overwritten; a failed backup aborts the write.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.backup import BackupError, backup_files, describe_backups
from levy.dataset.corpora import (
    DEFAULT_RAW_ROOT,
    DEFAULT_REGISTRY_PATH,
    CorpusRegistryError,
    load_registry,
    sha256_file,
)
from levy.dataset.prevalence import PoolCounts, measure_pool
from levy.dataset.sampling import CorpusSource, CorpusSourceError, make_source
from levy.dataset.schema import WORKLOADS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = REPO_ROOT / "data" / "ground_truth.ids.meta.json"
DEFAULT_OUT_DIR = REPO_ROOT / "release" / "prevalence"

CSV_NAME = "prevalence.csv"
META_NAME = "prevalence_meta.json"

CSV_COLUMNS = (
    "workload",
    "corpus",
    "pool_size",
    "positive",
    "negative",
    "excluded",
    "eligible",
    "dropped_empty_text_positive",
    "dropped_empty_text_negative",
    "prevalence_a",
    "prevalence_b",
)

DEFINITIONS = {
    "pool_size": "positive + negative + excluded",
    "eligible": "positive + negative",
    "prevalence_a": "positive / pool_size",
    "prevalence_b": "positive / eligible",
    "positive_negative": "candidates yielded by CorpusSource.iter_candidates() (native label mapped to a class, non-empty text)",
    "excluded": "rows whose native label is in the adapter's domain but in neither class, counted before the empty-text filter",
    "dropped_empty_text": "rows of the class in the native-label census minus candidates yielded",
}

BOUNDARY = "pools of the source corpora as distributed, after adapter filtering; not deployed LLM traffic"


class PrevalenceError(RuntimeError):
    """Raised when the measurement cannot be made as specified."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Sampling sidecar, read only (default: data/ground_truth.ids.meta.json)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT, help="Root of the acquired raw corpora (default: data/raw)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Corpus provenance registry (default: data/corpora.json)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory (default: release/prevalence)")
    return parser


def load_meta(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PrevalenceError(f"{path}: cannot read sampling sidecar: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PrevalenceError(f"{path}: invalid JSON: {exc}") from exc


def build_source(workload: str, meta: Dict, registry, raw_dir: Path) -> CorpusSource:
    """The adapter for `workload`, configured as the sampling sidecar records."""
    spec = (meta.get("workloads") or {}).get(workload)
    if spec is None:
        raise PrevalenceError(f"sidecar has no 'workloads.{workload}' entry")
    entry = registry.get(spec["corpus"])
    missing = entry.missing_paths(raw_dir)
    if missing:
        raise PrevalenceError(
            f"corpus {entry.key!r} is not acquired: missing {[str(p) for p in missing]}"
        )
    for corpus_file, path in zip(entry.files, entry.paths(raw_dir)):
        if corpus_file.sha256 is None:
            raise PrevalenceError(f"{entry.key}/{corpus_file.filename}: checksum is not pinned")
        actual = sha256_file(path)
        if actual != corpus_file.sha256:
            raise PrevalenceError(
                f"{path}: sha256 {actual} does not match the registry's {corpus_file.sha256}"
            )
    options = {
        key: value
        for key, value in (spec.get("options") or {}).items()
        if key not in {"shards", "splits", "normalisation"}
    }
    source = make_source(entry.adapter, entry.paths(raw_dir), **options)
    recorded = spec.get("label_mapping")
    if recorded is not None and recorded != source.label_mapping().to_dict():
        raise PrevalenceError(
            f"{workload}: sidecar label mapping {recorded} differs from the adapter's "
            f"{source.label_mapping().to_dict()}"
        )
    return source


def format_ratio(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def csv_text(counts: List[PoolCounts]) -> str:
    lines = [",".join(CSV_COLUMNS)]
    for c in counts:
        lines.append(
            ",".join(
                [
                    c.workload,
                    c.corpus,
                    str(c.pool_size),
                    str(c.positive),
                    str(c.negative),
                    str(c.excluded),
                    str(c.eligible),
                    str(c.dropped_empty_text_positive),
                    str(c.dropped_empty_text_negative),
                    format_ratio(c.prevalence_a),
                    format_ratio(c.prevalence_b),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def meta_document(counts: List[PoolCounts], sources: Dict[str, CorpusSource], meta: Dict, registry) -> Dict:
    workloads = {}
    for c in counts:
        source = sources[c.workload]
        mapping = source.label_mapping()
        spec = meta["workloads"][c.workload]
        entry = registry.get(spec["corpus"])
        workloads[c.workload] = {
            "corpus": c.corpus,
            "adapter": entry.adapter,
            "adapter_options": source.options(),
            "sampling_seed": spec.get("seed"),
            "label_mapping": mapping.to_dict(),
            "native_label_counts": {str(k): v for k, v in c.native_label_counts.items()},
            "files": [
                {"filename": f.filename, "sha256": f.sha256} for f in entry.files
            ],
        }
    return {
        "generated_by": "scripts/measure_prevalence.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampling_sidecar": "data/ground_truth.ids.meta.json",
        "pool_counts_source": "recomputed from raw corpora; not persisted in the sampling sidecar",
        "definitions": DEFINITIONS,
        "boundary": BOUNDARY,
        "workloads": workloads,
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
    }


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        meta = load_meta(args.meta)
        registry = load_registry(args.registry)
        sources = {w: build_source(w, meta, registry, args.raw_dir) for w in WORKLOADS}
        counts = [measure_pool(sources[w]) for w in WORKLOADS]
    except (PrevalenceError, CorpusRegistryError, CorpusSourceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    csv_path = args.out_dir / CSV_NAME
    meta_path = args.out_dir / META_NAME
    try:
        made = backup_files([csv_path, meta_path], backup_dir=args.out_dir / "backups")
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text(counts), encoding="utf-8")
    meta_path.write_text(
        json.dumps(meta_document(counts, sources, meta, registry), indent=2) + "\n",
        encoding="utf-8",
    )
    if made:
        print(describe_backups(made))
    print(f"wrote {csv_path}\nwrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
