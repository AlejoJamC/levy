"""
Load/save the ground-truth dataset as CSV and JSON (LEV-3 / D2).

This module is the **LEV-4 contract**: the experiment harness loads the
released dataset through `load_dataset()` and gets back a list of
`QueryPair` objects with validated fields. Keep the on-disk schema and this
loader in lockstep — any field added to `QueryPair` must be added to
`FIELDNAMES` below or CSV round-tripping silently drops it.

Both formats carry identical content: the same fields, same values, same
pair order. `metadata` (a free-form dict) is JSON-encoded into a single CSV
column and decoded back into a dict on load.

Alongside that full-dataset format, this module persists the **distribution
format** (LEV-12): the same records reduced to identifiers and labels, with no
query text, written by a separate code path into a separate file. It is what
the repository publishes, because two of the three source corpora grant no
redistribution right. It is deliberately *not* a harness input — pointing
`load_dataset` at one fails with an instruction to rehydrate first, rather than
replaying pairs with absent text.
"""

import csv
import json
from pathlib import Path
from typing import Iterable, List, Union

from levy.dataset.schema import (
    DistributionRecord,
    QueryPair,
    QueryPairValidationError,
)

PathLike = Union[str, Path]

# Canonical column order for CSV. Keep in sync with QueryPair fields.
FIELDNAMES = [
    "pair_id",
    "workload",
    "source_corpus",
    "source_pair_id",
    "query_1",
    "query_2",
    "original_label",
    "author_label",
    "metadata",
]

# Canonical column order for the identifiers-only distribution file. Its own
# list, not a filtered view of FIELDNAMES: the two formats are separate
# contracts, and a column added to one must be a deliberate decision for the other.
DISTRIBUTION_FIELDNAMES = [
    "pair_id",
    "workload",
    "source_corpus",
    "source_pair_id",
    "original_label",
    "author_label",
    "metadata",
]

_REHYDRATE_HINT = (
    "this is the identifiers-only distribution file, which carries no query "
    "text and is not a harness input. Reconstruct the working dataset first:\n"
    "    python scripts/rehydrate_dataset.py --ids {path}\n"
    "and load the dataset it writes."
)


class DatasetValidationError(ValueError):
    """Raised when a dataset file fails to load due to a schema violation."""


def save_csv(pairs: List[QueryPair], path: PathLike) -> None:
    """Write `pairs` to a CSV file with columns `FIELDNAMES`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for pair in pairs:
            row = pair.to_dict()
            row["author_label"] = "" if row["author_label"] is None else row["author_label"]
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
            writer.writerow(row)


def load_csv(path: PathLike) -> List[QueryPair]:
    """Read a CSV file written by `save_csv` back into `QueryPair` objects."""
    path = Path(path)
    pairs: List[QueryPair] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = set(FIELDNAMES) - set(fieldnames)
        if missing:
            if _looks_like_distribution(fieldnames):
                raise DatasetValidationError(
                    f"{path}: " + _REHYDRATE_HINT.format(path=path)
                )
            raise DatasetValidationError(
                f"{path}: CSV is missing required columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):  # header is line 1
            try:
                metadata_raw = row.get("metadata") or "{}"
                data = dict(row)
                data["metadata"] = json.loads(metadata_raw)
                pairs.append(QueryPair.from_dict(data))
            except (QueryPairValidationError, json.JSONDecodeError) as exc:
                raise DatasetValidationError(f"{path}:{row_number}: {exc}") from exc
    return pairs


def save_json(pairs: List[QueryPair], path: PathLike) -> None:
    """Write `pairs` to a JSON file as a list of objects (identical content to CSV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([pair.to_dict() for pair in pairs], fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_json(path: PathLike) -> List[QueryPair]:
    """Read a JSON file written by `save_json` back into `QueryPair` objects."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise DatasetValidationError(f"{path}: expected a JSON list of pair objects")
    if raw and isinstance(raw[0], dict) and _looks_like_distribution(raw[0].keys()):
        raise DatasetValidationError(f"{path}: " + _REHYDRATE_HINT.format(path=path))
    pairs: List[QueryPair] = []
    for index, item in enumerate(raw):
        try:
            pairs.append(QueryPair.from_dict(item))
        except QueryPairValidationError as exc:
            raise DatasetValidationError(f"{path}:item[{index}]: {exc}") from exc
    return pairs


def save_dataset(pairs: List[QueryPair], csv_path: PathLike, json_path: PathLike) -> None:
    """Write `pairs` to both CSV and JSON (the released dataset format, D2)."""
    save_csv(pairs, csv_path)
    save_json(pairs, json_path)


# ---------------------------------------------------------------------------
# Identifiers-only distribution format (LEV-12)
# ---------------------------------------------------------------------------

def _looks_like_distribution(fieldnames: Iterable[str]) -> bool:
    """True when a header/object carries the distribution columns but no text."""
    names = set(fieldnames)
    return set(DISTRIBUTION_FIELDNAMES) <= names and not ({"query_1", "query_2"} & names)


def to_distribution_records(pairs: Iterable[QueryPair]) -> List[DistributionRecord]:
    """
    Reduce sampled pairs to distribution records, preserving the sampled order.

    Order is load-bearing: rehydration emits records in the order the
    distribution file lists them, so the round-trip is byte-identical without
    anything having to be re-derived.
    """
    return [DistributionRecord.from_query_pair(pair) for pair in pairs]


def save_distribution_csv(records: List[DistributionRecord], path: PathLike) -> None:
    """Write the identifiers-only distribution file (`data/ground_truth.ids.csv`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DISTRIBUTION_FIELDNAMES)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row["author_label"] = "" if row["author_label"] is None else row["author_label"]
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
            writer.writerow(row)


def load_distribution_csv(path: PathLike) -> List[DistributionRecord]:
    """Read a distribution file written by `save_distribution_csv`."""
    path = Path(path)
    records: List[DistributionRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(DISTRIBUTION_FIELDNAMES) - set(reader.fieldnames or [])
        if missing:
            raise DatasetValidationError(
                f"{path}: distribution CSV is missing required columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):  # header is line 1
            try:
                data = dict(row)
                data["metadata"] = json.loads(row.get("metadata") or "{}")
                records.append(DistributionRecord.from_dict(data))
            except (QueryPairValidationError, json.JSONDecodeError) as exc:
                raise DatasetValidationError(f"{path}:{row_number}: {exc}") from exc
    return records


def load_dataset(path: PathLike) -> List[QueryPair]:
    """Load a dataset file, dispatching on file extension (`.csv` or `.json`)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".json":
        return load_json(path)
    raise DatasetValidationError(
        f"{path}: unrecognized dataset file extension {suffix!r}; expected .csv or .json"
    )
