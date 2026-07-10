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
"""

import csv
import json
from pathlib import Path
from typing import List, Union

from levy.dataset.schema import QueryPair, QueryPairValidationError

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
        missing = set(FIELDNAMES) - set(reader.fieldnames or [])
        if missing:
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
