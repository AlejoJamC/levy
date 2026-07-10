"""
Cohen's kappa between original corpus labels and the author's blind
re-annotation (LEV-3 / D2; frozen S&D Report success criterion: kappa > 0.7,
computed over the full 900-pair set).

Implemented from the 2x2 contingency table with stdlib + numpy only (no
scipy/scikit-learn/pandas in the `levy` conda env).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from levy.dataset.schema import QueryPair, WORKLOADS


@dataclass
class KappaResult:
    """
    Cohen's kappa result for one set of pairs.

    `kappa` is `None` only when there are zero annotated pairs to compare
    (nothing to compute). See `cohen_kappa` docstring for the degenerate
    all-one-class convention.
    """

    n_annotated: int
    n_excluded_unannotated: int
    observed_agreement: Optional[float]
    expected_agreement: Optional[float]
    kappa: Optional[float]
    confusion: Dict[str, int] = field(default_factory=dict)  # tp, fp, fn, tn


@dataclass
class KappaReport:
    """Overall kappa plus a per-workload breakdown."""

    overall: KappaResult
    per_workload: Dict[str, KappaResult]


def _contingency(pairs: List[QueryPair]) -> Dict[str, int]:
    """
    2x2 contingency of original_label (rows) vs author_label (columns),
    restricted to pairs with a non-None author_label.

    tp: original=1, author=1   fp: original=0, author=1
    fn: original=1, author=0   tn: original=0, author=0
    (labels named tp/fp/fn/tn purely for a compact confusion-matrix shape;
    there is no "positive class" privilege between the two annotators here.)
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for pair in pairs:
        if pair.author_label is None:
            continue
        if pair.original_label == 1 and pair.author_label == 1:
            counts["tp"] += 1
        elif pair.original_label == 0 and pair.author_label == 1:
            counts["fp"] += 1
        elif pair.original_label == 1 and pair.author_label == 0:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def cohen_kappa(pairs: List[QueryPair]) -> KappaResult:
    """
    Compute Cohen's kappa between `original_label` and `author_label` over
    `pairs`, excluding any pair whose `author_label` is still `None`
    (not yet annotated).

    Edge cases (documented behavior, not bugs):
      - Zero annotated pairs: returns a `KappaResult` with `kappa=None` and
        `n_annotated=0`; there is nothing to compute.
      - Degenerate all-one-class (every annotated pair has the same
        `original_label` AND the same `author_label`, e.g. all "duplicate"):
        expected agreement `pe == 1.0`. By convention this method returns
        `kappa = 1.0` if observed agreement is also 1.0 (both annotators
        trivially agree on the single class), else `kappa = 0.0` (any
        disagreement against a one-class expectation is treated as no
        better than chance, avoiding a division by zero in
        `(po - pe) / (1 - pe)`). This convention MUST be stated alongside
        any reported kappa that hits it (see `data/DATASHEET.md`).
    """
    counts = _contingency(pairs)
    n = sum(counts.values())
    excluded = sum(1 for p in pairs if p.author_label is None)

    if n == 0:
        return KappaResult(
            n_annotated=0,
            n_excluded_unannotated=excluded,
            observed_agreement=None,
            expected_agreement=None,
            kappa=None,
            confusion=counts,
        )

    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

    po = (tp + tn) / n

    # Marginal proportions for "label == 1" under each annotator.
    p_original_1 = (tp + fn) / n
    p_author_1 = (tp + fp) / n
    pe = p_original_1 * p_author_1 + (1 - p_original_1) * (1 - p_author_1)

    if math.isclose(pe, 1.0, abs_tol=1e-12):
        kappa = 1.0 if math.isclose(po, 1.0, abs_tol=1e-12) else 0.0
    else:
        kappa = (po - pe) / (1 - pe)

    return KappaResult(
        n_annotated=n,
        n_excluded_unannotated=excluded,
        observed_agreement=po,
        expected_agreement=pe,
        kappa=kappa,
        confusion=counts,
    )


def kappa_report(pairs: List[QueryPair]) -> KappaReport:
    """Overall kappa plus one `KappaResult` per workload in `WORKLOADS`."""
    per_workload = {
        workload: cohen_kappa([p for p in pairs if p.workload == workload])
        for workload in WORKLOADS
    }
    return KappaReport(overall=cohen_kappa(pairs), per_workload=per_workload)
