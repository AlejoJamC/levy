"""
Experiment harness (LEV-4 / D3): offline replay of annotated query pairs
across the frozen 30-configuration grid, confusion-matrix accounting
against ground-truth labels, per-configuration metric computation, and
deterministic machine-readable outputs consumed by LEV-8's statistical
analysis.
"""

from levy.experiment.config import EMBEDDING_MODELS, THRESHOLDS, ExperimentConfig, full_grid
from levy.experiment.metrics import (
    DecisionRecord,
    EvaluationResult,
    ExperimentSanityError,
    check_sanity,
    evaluate_confusion,
)
from levy.experiment.merge import (
    MergeInput,
    ResultsMergeError,
    check_grid_coverage,
    load_input,
    merge_decisions,
    merge_results,
    merge_run_meta,
)
from levy.experiment.replay import run_experiment
from levy.experiment.runner import (
    run_sweep,
    write_decisions_csv,
    write_results_csv,
    write_run_meta,
)

__all__ = [
    "EMBEDDING_MODELS",
    "THRESHOLDS",
    "ExperimentConfig",
    "full_grid",
    "DecisionRecord",
    "EvaluationResult",
    "ExperimentSanityError",
    "check_sanity",
    "evaluate_confusion",
    "MergeInput",
    "ResultsMergeError",
    "check_grid_coverage",
    "load_input",
    "merge_decisions",
    "merge_results",
    "merge_run_meta",
    "run_experiment",
    "run_sweep",
    "write_decisions_csv",
    "write_results_csv",
    "write_run_meta",
]
