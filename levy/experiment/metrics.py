"""
Per-configuration confusion-matrix accounting and metrics (LEV-4 / D3).

Formulas (frozen S&D Report, precision-weighted F-beta with beta=0.5):
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F0.5      = 1.25 * P * R / (0.25 * P + R)
    FPR       = FP / (FP + TN)
    hit_rate  = (TP + FP) / N

Any zero-denominator case is reported as 0.0 with an explicit flag, never
as NaN, so results.csv stays machine-readable for scipy/pandas downstream.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from levy.experiment.config import ExperimentConfig


class ExperimentSanityError(ValueError):
    """Raised when a configuration's confusion counts or rates violate the QA plan."""


@dataclass
class DecisionRecord:
    """Audit-trail row: one per replayed pair, per configuration."""

    config_id: str
    pair_id: str
    decision: str  # "hit" | "miss"
    source: str  # "exact_cache" | "semantic_cache" | "llm"
    similarity: Optional[float]
    label: int
    outcome: str  # "TP" | "FP" | "TN" | "FN"


@dataclass
class EvaluationResult:
    """Confusion counts, metrics, and decision log for one `ExperimentConfig`."""

    config: ExperimentConfig
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f0_5: float
    fpr: float
    hit_rate: float
    precision_zero_div: bool
    recall_zero_div: bool
    fpr_zero_div: bool
    decisions: List[DecisionRecord] = field(default_factory=list)


def _safe_div(numerator: int, denominator: int) -> Tuple[float, bool]:
    if denominator == 0:
        return 0.0, True
    return numerator / denominator, False


def evaluate_confusion(
    config: ExperimentConfig,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    n: int,
    decisions: Optional[List[DecisionRecord]] = None,
) -> EvaluationResult:
    """
    Compute metrics from confusion counts and run the statistical sanity
    checks (counts sum to `n`, every rate within [0, 1]) before returning.
    `n` is the number of pairs actually replayed for this configuration --
    passed independently of tp/fp/tn/fn so a bookkeeping bug is caught
    rather than silently reconciled.
    """
    precision, precision_zero_div = _safe_div(tp, tp + fp)
    recall, recall_zero_div = _safe_div(tp, tp + fn)
    fpr, fpr_zero_div = _safe_div(fp, fp + tn)

    f0_5 = 0.0 if (precision + recall) == 0 else (1.25 * precision * recall) / (0.25 * precision + recall)
    hit_rate, _ = _safe_div(tp + fp, n)

    result = EvaluationResult(
        config=config,
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f0_5=f0_5,
        fpr=fpr,
        hit_rate=hit_rate,
        precision_zero_div=precision_zero_div,
        recall_zero_div=recall_zero_div,
        fpr_zero_div=fpr_zero_div,
        decisions=list(decisions or []),
    )
    check_sanity(result)
    return result


def check_sanity(result: EvaluationResult) -> None:
    """
    Raise `ExperimentSanityError` naming the offending configuration if the
    confusion counts don't sum to the replayed pair count, or if any
    reported rate falls outside [0, 1].
    """
    total = result.tp + result.fp + result.tn + result.fn
    if total != result.n:
        raise ExperimentSanityError(
            f"{result.config.config_id}: confusion counts sum to {total}, expected {result.n} pairs replayed"
        )
    for name in ("precision", "recall", "f0_5", "fpr", "hit_rate"):
        value = getattr(result, name)
        if not (0.0 <= value <= 1.0):
            raise ExperimentSanityError(
                f"{result.config.config_id}: {name}={value} is outside the valid range [0, 1]"
            )
