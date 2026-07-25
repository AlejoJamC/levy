"""
Threshold-selection curves (LEV-8 / D3, evidence base for objective O3).

For each of the 6 (embedding model, workload) pairs, the frozen grid sweeps
5 similarity thresholds. This module turns `results.csv` into two tidy curve
tables -- threshold vs hit rate and threshold vs precision -- and renders one
figure per metric from those tables alone, so the figures are always
regenerable from the machine-readable source of truth.

The harness's zero-division flags travel with the values, so a degenerate
cell (e.g. precision over zero predicted positives, reported as 0.0) is
visible in the table rather than silently reading as a real zero.

Threshold-scale note: thresholds are on the `1/(1+L2)` similarity scale used
by `SemanticCache`, carried verbatim from the frozen grid (see CLAUDE.md
known-gap #3). They are not cosine similarities and are not rescaled here.
"""

from pathlib import Path
from typing import List, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # file output only; never opens a display

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import pandas as pd  # noqa: E402

PathLike = Union[str, Path]

#: Frozen economic-viability bar: hit rate must exceed 30% (S&D Report).
HIT_RATE_VIABILITY = 0.30

CURVE_COLUMNS = ["model", "workload", "threshold", "metric", "value", "zero_div", "n"]

_METRIC_HIT_RATE = "hit_rate"
_METRIC_PRECISION = "precision"

_AXIS_LABELS = {
    _METRIC_HIT_RATE: "Hit rate",
    _METRIC_PRECISION: "Precision",
}
_TITLES = {
    _METRIC_HIT_RATE: "Cache hit rate vs similarity threshold",
    _METRIC_PRECISION: "Cache precision vs similarity threshold",
}


def _curve_table(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    One tidy table for `metric`, sorted (model, workload, threshold) so the
    CSV is byte-stable regardless of the harness's row order.

    `zero_div` is the harness's own flag for precision. Hit rate has no
    harness flag (the harness discards it), so it is derived here as `n == 0`
    -- the only way hit rate's denominator can vanish.
    """
    frame = results.copy()
    if metric == _METRIC_PRECISION:
        zero_div = frame["precision_zero_div"].astype(bool)
    else:
        zero_div = frame["n"].astype(int) == 0

    table = pd.DataFrame(
        {
            "model": frame["model"].astype(str),
            "workload": frame["workload"].astype(str),
            "threshold": frame["threshold"].astype(float),
            "metric": metric,
            "value": frame[metric].astype(float),
            "zero_div": zero_div,
            "n": frame["n"].astype(int),
        },
        columns=CURVE_COLUMNS,
    )
    return table.sort_values(["model", "workload", "threshold"], kind="mergesort").reset_index(
        drop=True
    )


def build_curve_tables(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(hit_rate_table, precision_table)` in tidy long form."""
    return _curve_table(results, _METRIC_HIT_RATE), _curve_table(results, _METRIC_PRECISION)


def _render_figure(table: pd.DataFrame, metric: str, out_dir: Path) -> List[Path]:
    figure, axes = plt.subplots(figsize=(8.0, 5.0))

    for (model, workload), series in table.groupby(["model", "workload"], sort=True):
        ordered = series.sort_values("threshold")
        axes.plot(
            ordered["threshold"],
            ordered["value"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"{model} / {workload}",
        )

    if metric == _METRIC_HIT_RATE:
        axes.axhline(
            HIT_RATE_VIABILITY,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"economic viability bar ({HIT_RATE_VIABILITY:.0%})",
        )

    axes.set_xlabel("Similarity threshold  (1 / (1 + L2) scale)")
    axes.set_ylabel(_AXIS_LABELS[metric])
    axes.set_title(_TITLES[metric])
    axes.set_ylim(-0.02, 1.02)
    axes.grid(True, alpha=0.3, linewidth=0.5)
    axes.legend(fontsize="small", loc="best")
    figure.tight_layout()

    written: List[Path] = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"curve_{metric}.{suffix}"
        figure.savefig(path, dpi=200 if suffix == "png" else None)
        written.append(path)
    plt.close(figure)
    return written


def write_curve_figures(
    hit_rate_table: pd.DataFrame,
    precision_table: pd.DataFrame,
    out_dir: PathLike,
) -> List[Path]:
    """
    Render both curve figures (PNG + PDF) into `out_dir` from the tidy tables
    alone. Returns the written paths in a deterministic order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = _render_figure(hit_rate_table, _METRIC_HIT_RATE, out_dir)
    written.extend(_render_figure(precision_table, _METRIC_PRECISION, out_dir))
    return written
