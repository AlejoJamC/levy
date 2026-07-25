"""
Analysis bundle loading and validation for the results dashboard (LEV-10 / D6).

Reads whatever directory `scripts/run_analysis.py` (LEV-8) produced. Expected
filenames and columns are imported from `levy.analysis`, never re-declared,
so a change to LEV-8's writers surfaces here as a load error instead of a
silently mis-parsed table -- the same anti-drift rule LEV-8 applies between
its own reader and the harness writer (see `levy/analysis/io.py`).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from levy.analysis.curves import CURVE_COLUMNS
from levy.analysis.hypothesis import ANOVA_COLUMNS, TUKEY_COLUMNS
from levy.analysis.report import (
    ANOVA_FILENAME,
    CURVES_HIT_RATE_FILENAME,
    CURVES_PRECISION_FILENAME,
    KAPPA_FILENAME,
    META_FILENAME,
    TUKEY_FILENAME,
    TUKEY_STATUS_COLUMNS,
    TUKEY_STATUS_FILENAME,
)

PathLike = Union[str, Path]

#: The command that produces a bundle, surfaced in every actionable error.
REPRODUCE_COMMAND = "scripts/reproduce.sh"

#: CSV filename -> required columns, sourced from the analysis package itself.
REQUIRED_TABLES: Dict[str, List[str]] = {
    CURVES_HIT_RATE_FILENAME: CURVE_COLUMNS,
    CURVES_PRECISION_FILENAME: CURVE_COLUMNS,
    ANOVA_FILENAME: ANOVA_COLUMNS,
    TUKEY_FILENAME: TUKEY_COLUMNS,
    TUKEY_STATUS_FILENAME: TUKEY_STATUS_COLUMNS,
}

#: JSON sidecars the bundle also requires.
REQUIRED_JSON: List[str] = [KAPPA_FILENAME, META_FILENAME]


class BundleNotFoundError(FileNotFoundError):
    """Raised when the bundle directory, or a required file within it, is absent."""


class BundleContractError(ValueError):
    """Raised when a bundle file is present but missing an expected column."""


@dataclass
class AnalysisBundle:
    """A loaded, validated analysis bundle -- the dashboard's only input."""

    bundle_dir: Path
    hit_rate: pd.DataFrame
    precision: pd.DataFrame
    anova: pd.DataFrame
    tukey: pd.DataFrame
    tukey_status: pd.DataFrame
    kappa: Dict[str, Any]
    meta: Dict[str, Any]


def _required_paths(bundle_dir: Path) -> List[Path]:
    names = list(REQUIRED_TABLES) + list(REQUIRED_JSON)
    return [bundle_dir / name for name in names]


def _missing_paths(bundle_dir: Path) -> List[Path]:
    return [path for path in _required_paths(bundle_dir) if not path.is_file()]


def _actionable_message(missing: List[Path]) -> str:
    listed = "\n".join(f"  - {path}" for path in missing)
    return (
        "Analysis bundle is missing the following file(s):\n"
        f"{listed}\n"
        f"Generate a bundle first, e.g.: {REPRODUCE_COMMAND}"
    )


def _load_table(path: Path, required_columns: List[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise BundleContractError(
            f"{path}: missing required column(s): {', '.join(missing)} "
            f"(expected the bundle contract {required_columns}). "
            f"Regenerate the bundle, e.g.: {REPRODUCE_COMMAND}"
        )
    return frame


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle(bundle_dir: PathLike) -> AnalysisBundle:
    """
    Load and validate an analysis bundle directory.

    Raises `BundleNotFoundError` when the directory or any required file is
    absent, and `BundleContractError` when a present file lacks an expected
    column -- both name the exact gap and the command that produces a bundle,
    so the dashboard can render the message directly instead of a traceback.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise BundleNotFoundError(
            f"Analysis bundle directory not found: {bundle_dir}\n"
            f"Generate one first, e.g.: {REPRODUCE_COMMAND}"
        )

    missing = _missing_paths(bundle_dir)
    if missing:
        raise BundleNotFoundError(_actionable_message(missing))

    tables = {
        name: _load_table(bundle_dir / name, columns) for name, columns in REQUIRED_TABLES.items()
    }
    kappa = _load_json(bundle_dir / KAPPA_FILENAME)
    meta = _load_json(bundle_dir / META_FILENAME)

    return AnalysisBundle(
        bundle_dir=bundle_dir,
        hit_rate=tables[CURVES_HIT_RATE_FILENAME],
        precision=tables[CURVES_PRECISION_FILENAME],
        anova=tables[ANOVA_FILENAME],
        tukey=tables[TUKEY_FILENAME],
        tukey_status=tables[TUKEY_STATUS_FILENAME],
        kappa=kappa,
        meta=meta,
    )
