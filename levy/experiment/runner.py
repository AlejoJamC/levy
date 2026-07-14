"""
Sweep orchestration over the experimental grid, plus output writers
(LEV-4 / D3): `results.csv` (one row per configuration), `decisions.csv`
(per-pair audit log), and a `run_meta.json` sidecar (dataset path,
providers, model checkpoints, grid, latency labeled synthetic under the
mock LLM). Timestamps and latency never appear in the two determinism-
checked CSVs -- only in the sidecar.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from levy.dataset.schema import QueryPair
from levy.embedding_manager import EmbeddingManager
from levy.experiment.config import ExperimentConfig, full_grid
from levy.experiment.metrics import EvaluationResult
from levy.experiment.replay import run_experiment

PathLike = Union[str, Path]

RESULTS_FIELDNAMES = [
    "config_id",
    "model",
    "workload",
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
    "precision_zero_div",
    "recall_zero_div",
    "fpr_zero_div",
]

DECISIONS_FIELDNAMES = [
    "config_id",
    "model",
    "workload",
    "threshold",
    "pair_id",
    "decision",
    "source",
    "similarity",
    "label",
    "outcome",
]


def run_sweep(
    pairs: List[QueryPair],
    configs: Optional[List[ExperimentConfig]] = None,
    embedding_provider: str = "mock",
) -> Tuple[List[EvaluationResult], Dict[str, dict]]:
    """
    Run every configuration in `configs` (the full frozen grid by default),
    sharing one `EmbeddingManager` per model across its configurations so
    LEV-1's `(model_key, sha256(text))` memoization isn't defeated by a
    sweep. Returns `(results, model_identities)` where `model_identities`
    maps each model alias to its resolved checkpoint/dimension (for the
    run-metadata sidecar).
    """
    if configs is None:
        configs = full_grid()

    managers: Dict[str, EmbeddingManager] = {}
    results: List[EvaluationResult] = []

    for config in configs:
        manager = managers.get(config.model)
        if manager is None:
            manager = EmbeddingManager(model_name=config.model, provider=embedding_provider)
            managers[config.model] = manager
        results.append(run_experiment(config, pairs, embedding_manager=manager))

    model_identities = {model: manager.get_model_identity().as_dict() for model, manager in managers.items()}
    return results, model_identities


def write_results_csv(results: List[EvaluationResult], path: PathLike) -> None:
    """Write one row per configuration -- no timestamps, no latency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "config_id": result.config.config_id,
                    "model": result.config.model,
                    "workload": result.config.workload,
                    "threshold": f"{result.config.threshold:.2f}",
                    "n": result.n,
                    "tp": result.tp,
                    "fp": result.fp,
                    "tn": result.tn,
                    "fn": result.fn,
                    "precision": f"{result.precision:.6f}",
                    "recall": f"{result.recall:.6f}",
                    "f0_5": f"{result.f0_5:.6f}",
                    "fpr": f"{result.fpr:.6f}",
                    "hit_rate": f"{result.hit_rate:.6f}",
                    "precision_zero_div": result.precision_zero_div,
                    "recall_zero_div": result.recall_zero_div,
                    "fpr_zero_div": result.fpr_zero_div,
                }
            )


def write_decisions_csv(results: List[EvaluationResult], path: PathLike) -> None:
    """Write one row per pair per configuration -- no timestamps, no latency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DECISIONS_FIELDNAMES)
        writer.writeheader()
        for result in results:
            for decision in result.decisions:
                writer.writerow(
                    {
                        "config_id": decision.config_id,
                        "model": result.config.model,
                        "workload": result.config.workload,
                        "threshold": f"{result.config.threshold:.2f}",
                        "pair_id": decision.pair_id,
                        "decision": decision.decision,
                        "source": decision.source,
                        "similarity": "" if decision.similarity is None else f"{decision.similarity:.6f}",
                        "label": decision.label,
                        "outcome": decision.outcome,
                    }
                )


def write_run_meta(
    results: List[EvaluationResult],
    configs: List[ExperimentConfig],
    dataset_path: PathLike,
    embedding_provider: str,
    model_identities: Dict[str, dict],
    elapsed_seconds: float,
    path: PathLike,
) -> None:
    """
    Write run parameters and latency statistics to a sidecar, deliberately
    kept out of `results.csv` / `decisions.csv` so those two stay
    byte-identical across deterministic re-runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset_path": str(dataset_path),
        "llm_provider": "mock",
        "embedding_provider": embedding_provider,
        "n_configurations": len(configs),
        "grid": [
            {"model": config.model, "workload": config.workload, "threshold": config.threshold}
            for config in configs
        ],
        "model_identities": model_identities,
        "latency": {
            "total_elapsed_seconds": elapsed_seconds,
            "note": (
                "LLM latency is synthetic: MockLLMClient sleeps a fixed 0.5s per call and "
                "is not a measurement of any real provider's performance."
            ),
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
        fh.write("\n")
