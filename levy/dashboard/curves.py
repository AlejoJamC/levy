"""
Curve selection helpers for the results dashboard (LEV-10 / D6).

Pure functions over the bundle's tidy `curves_hit_rate.csv` /
`curves_precision.csv` tables (`model, workload, threshold, metric, value,
zero_div, n` -- see `levy.analysis.curves.CURVE_COLUMNS`). No I/O, no UI.
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class CurvePoint:
    """One threshold's value for a (model, workload) curve."""

    threshold: float
    value: float
    zero_div: bool
    n: int


def available_models(table: pd.DataFrame) -> List[str]:
    """Embedding models present in a curve table, sorted."""
    return sorted(table["model"].unique())


def available_workloads(table: pd.DataFrame) -> List[str]:
    """Workloads present in a curve table, sorted."""
    return sorted(table["workload"].unique())


def available_model_workload_pairs(table: pd.DataFrame) -> List[Tuple[str, str]]:
    """Every distinct (model, workload) pair present in a curve table, sorted."""
    pairs = table[["model", "workload"]].drop_duplicates()
    return sorted((str(model), str(workload)) for model, workload in pairs.itertuples(index=False))


def select_curve(table: pd.DataFrame, model: str, workload: str) -> List[CurvePoint]:
    """
    The (model, workload) slice of a curve table, ordered by threshold, with
    each point's harness `zero_div` flag and sample size `n` preserved so a
    degenerate cell is never presented as a measured value.
    """
    subset = table[(table["model"] == model) & (table["workload"] == workload)]
    subset = subset.sort_values("threshold")
    return [
        CurvePoint(
            threshold=float(row.threshold),
            value=float(row.value),
            zero_div=bool(row.zero_div),
            n=int(row.n),
        )
        for row in subset.itertuples(index=False)
    ]
