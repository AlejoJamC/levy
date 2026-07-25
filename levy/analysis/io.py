"""
Harness output contract loader (LEV-8 / D3).

The analysis pipeline's only input is a directory produced by
`scripts/run_experiments.py` (LEV-4). This module reads it and validates the
column contract up front, so a contract violation fails loudly -- naming the
missing column -- instead of producing partial statistics downstream.

Column lists are imported from `levy.experiment.runner`, the writer, so the
reader can never drift from the writer.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from levy.experiment.runner import DECISIONS_FIELDNAMES, RESULTS_FIELDNAMES

PathLike = Union[str, Path]

RESULTS_FILENAME = "results.csv"
DECISIONS_FILENAME = "decisions.csv"
RUN_META_FILENAME = "run_meta.json"

# The contract: exactly what the harness writes (see levy/experiment/runner.py).
RESULTS_REQUIRED_COLUMNS: List[str] = list(RESULTS_FIELDNAMES)
DECISIONS_REQUIRED_COLUMNS: List[str] = list(DECISIONS_FIELDNAMES)

# Columns that must be numeric for the statistics to be meaningful.
_RESULTS_NUMERIC_COLUMNS = (
    "threshold",
    "n",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f0_5",
    "fpr",
    "hit_rate",
)
_RESULTS_BOOL_COLUMNS = ("precision_zero_div", "recall_zero_div", "fpr_zero_div")


class HarnessContractError(ValueError):
    """
    Raised when a harness output directory does not satisfy the contract:
    a missing file, a missing required column, or a non-numeric metric.
    """


@dataclass
class HarnessOutputs:
    """
    A loaded harness output directory.

    `decisions` and `run_meta` are optional: the statistics need only
    `results.csv`, and a caller may point the pipeline at a directory that
    carries just that (e.g. a hand-crafted fixture). When present they are
    validated on the same terms as `results.csv`.
    """

    results: pd.DataFrame
    source_dir: Path
    decisions: Optional[pd.DataFrame] = None
    run_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_configurations(self) -> int:
        return len(self.results)


def _require_columns(frame: pd.DataFrame, required: List[str], path: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HarnessContractError(
            f"{path}: missing required column(s): {', '.join(missing)} "
            f"(expected the harness contract {required})"
        )


def _coerce_bool(value: Any) -> bool:
    """
    CSV round-trips Python bools as the strings "True"/"False"; accept those,
    real bools, and 0/1 alike so a hand-written fixture is not second-class.
    A blank cell (which pandas reads as NaN) means "not flagged" -- the column
    itself is still required, only its value may be omitted.
    """
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0", ""):
        return False
    raise HarnessContractError(f"Not a boolean zero-division flag: {value!r}")


def load_results(path: PathLike) -> pd.DataFrame:
    """
    Load and validate a `results.csv`. Returns a DataFrame with the metric
    columns as floats/ints and the zero-division flags as real booleans.
    """
    path = Path(path)
    if not path.is_file():
        raise HarnessContractError(f"{path}: results file not found")

    frame = pd.read_csv(path)
    _require_columns(frame, RESULTS_REQUIRED_COLUMNS, path)

    if frame.empty:
        raise HarnessContractError(f"{path}: contains no configuration rows")

    for column in _RESULTS_NUMERIC_COLUMNS:
        coerced = pd.to_numeric(frame[column], errors="coerce")
        if coerced.isna().any():
            bad_rows = frame.index[coerced.isna()].tolist()
            raise HarnessContractError(
                f"{path}: column {column!r} has non-numeric value(s) at row(s) {bad_rows}"
            )
        frame[column] = coerced

    for column in _RESULTS_BOOL_COLUMNS:
        try:
            frame[column] = frame[column].map(_coerce_bool)
        except HarnessContractError as exc:
            raise HarnessContractError(f"{path}: column {column!r}: {exc}") from exc

    return frame


def load_decisions(path: PathLike) -> pd.DataFrame:
    """Load and validate a `decisions.csv` (per-pair audit log)."""
    path = Path(path)
    if not path.is_file():
        raise HarnessContractError(f"{path}: decisions file not found")
    frame = pd.read_csv(path)
    _require_columns(frame, DECISIONS_REQUIRED_COLUMNS, path)
    return frame


def load_run_meta(path: PathLike) -> Dict[str, Any]:
    """Load a `run_meta.json` sidecar."""
    path = Path(path)
    if not path.is_file():
        raise HarnessContractError(f"{path}: run metadata file not found")
    with path.open(encoding="utf-8") as fh:
        try:
            meta = json.load(fh)
        except json.JSONDecodeError as exc:
            raise HarnessContractError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise HarnessContractError(f"{path}: expected a JSON object, got {type(meta).__name__}")
    return meta


def load_harness_outputs(harness_dir: PathLike) -> HarnessOutputs:
    """
    Load a harness output directory. `results.csv` is mandatory;
    `decisions.csv` and `run_meta.json` are loaded when present (and
    validated when they are).
    """
    harness_dir = Path(harness_dir)
    if not harness_dir.is_dir():
        raise HarnessContractError(f"{harness_dir}: harness output directory not found")

    results = load_results(harness_dir / RESULTS_FILENAME)

    decisions_path = harness_dir / DECISIONS_FILENAME
    decisions = load_decisions(decisions_path) if decisions_path.is_file() else None

    meta_path = harness_dir / RUN_META_FILENAME
    run_meta = load_run_meta(meta_path) if meta_path.is_file() else {}

    return HarnessOutputs(
        results=results,
        source_dir=harness_dir,
        decisions=decisions,
        run_meta=run_meta,
    )
