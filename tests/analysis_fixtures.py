"""
Hand-crafted harness-output fixtures for the LEV-8 analysis tests.

Not a test module (pytest collects `test_*.py` only). These builders produce
`results.csv` content that satisfies the LEV-4 contract exactly, with FPR
values chosen so the ANOVA outcome is known in advance and hand-computable:

    * every (model, workload) cell holds the same 5 threshold replicates,
      offset from the cell mean by OFFSETS -- so each cell contributes
      exactly sum(OFFSETS^2) = 0.001 to the residual sum of squares, and the
      cell mean equals the caller's requested base value.

That makes the balanced 2 x 3 x 5 sums of squares computable by hand in the
tests, independently of statsmodels.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from levy.experiment.runner import RESULTS_FIELDNAMES

MODELS: Tuple[str, ...] = ("all-MiniLM-L6-v2", "modernbert")
WORKLOADS: Tuple[str, ...] = ("faq", "code", "chat")
THRESHOLDS: Tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90)

#: Within-cell replicate offsets: mean 0, sum of squares 0.001.
OFFSETS: Tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02)

#: Residual sum of squares of any fixture built with OFFSETS: 6 cells x 0.001.
RESIDUAL_SS = 0.006
RESIDUAL_DF = 24

CellMap = Dict[Tuple[str, str], float]


def additive_cells(model_effect: float = 0.0, workload_effect: float = 0.0, base: float = 0.10) -> CellMap:
    """
    Cell means with no interaction: `base` plus a per-model step and a
    per-workload step. `model_effect=0` and `workload_effect=0` gives the
    null design (identical distributions in every cell).
    """
    return {
        (model, workload): base + model_index * model_effect + workload_index * workload_effect
        for model_index, model in enumerate(MODELS)
        for workload_index, workload in enumerate(WORKLOADS)
    }


def build_results(
    cell_fpr: CellMap,
    precision_zero_div_cells: Iterable[Tuple[str, str]] = (),
    n: int = 100,
) -> pd.DataFrame:
    """
    A full 30-row `results.csv` frame honouring the harness contract.

    FPR carries the designed effects; precision and hit rate follow a
    deterministic threshold-dependent pattern so the curve tables have
    something non-flat to plot.
    """
    zero_div_cells = set(precision_zero_div_cells)
    rows = []
    for model in MODELS:
        for workload in WORKLOADS:
            base = cell_fpr[(model, workload)]
            for index, threshold in enumerate(THRESHOLDS):
                fpr = base + OFFSETS[index]
                is_zero_div = (model, workload) in zero_div_cells
                precision = 0.0 if is_zero_div else round(0.50 + 0.05 * index, 4)
                hit_rate = round(0.60 - 0.10 * index, 4)
                fp = round(fpr * 50)
                tp = round(hit_rate * n) - fp
                rows.append(
                    {
                        "config_id": f"{model}|{workload}|{threshold:.2f}",
                        "model": model,
                        "workload": workload,
                        "threshold": f"{threshold:.2f}",
                        "n": n,
                        "tp": tp,
                        "fp": fp,
                        "tn": 50 - fp,
                        "fn": n - tp - fp - (50 - fp),
                        "precision": f"{precision:.6f}",
                        "recall": f"{0.40 + 0.05 * index:.6f}",
                        "f0_5": f"{0.45 + 0.05 * index:.6f}",
                        "fpr": f"{fpr:.6f}",
                        "hit_rate": f"{hit_rate:.6f}",
                        "precision_zero_div": is_zero_div,
                        "recall_zero_div": False,
                        "fpr_zero_div": False,
                    }
                )
    return pd.DataFrame(rows, columns=RESULTS_FIELDNAMES)


def write_harness_dir(
    directory: Path,
    frame: pd.DataFrame,
    dataset_path: Optional[Path] = None,
) -> Path:
    """
    Write `frame` as `results.csv` in `directory`, plus a minimal
    `run_meta.json` when `dataset_path` is given (so the kappa section can
    resolve its dataset the way it does for a real run).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "results.csv", index=False, lineterminator="\n")
    if dataset_path is not None:
        (directory / "run_meta.json").write_text(
            '{\n  "dataset_path": "%s",\n  "embedding_provider": "mock"\n}\n' % dataset_path,
            encoding="utf-8",
        )
    return directory


def anova_frame(cell_fpr: CellMap) -> pd.DataFrame:
    """The minimal frame the hypothesis module needs (model, workload, fpr)."""
    return build_results(cell_fpr)[["model", "workload", "threshold", "fpr"]].astype(
        {"threshold": float, "fpr": float}
    )
