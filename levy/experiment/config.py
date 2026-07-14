"""
Experiment configuration and the frozen 30-configuration grid (LEV-4 / D3).

The S&D Report's experimental grid is fixed: 2 embedding models
(`all-MiniLM-L6-v2`, `modernbert`) x 3 workloads (`faq`, `code`, `chat`) x
5 similarity thresholds (0.70-0.90, step 0.05) = 30 configurations.
Thresholds are carried verbatim on the `1/(1+L2)` similarity scale used by
`SemanticCache` -- no rescaling (see CLAUDE.md known-gaps note #3).
"""

from dataclasses import dataclass
from typing import List, Tuple

from levy.dataset.schema import WORKLOADS

# Frozen study models (registry keys in levy.embedding_manager._REGISTRY).
EMBEDDING_MODELS: Tuple[str, ...] = ("all-MiniLM-L6-v2", "modernbert")

# Frozen sweep: 0.70-0.90 step 0.05, thresholds carried verbatim.
THRESHOLDS: Tuple[float, ...] = tuple(round(0.70 + 0.05 * i, 2) for i in range(5))


@dataclass(frozen=True)
class ExperimentConfig:
    """One cell of the frozen experimental grid."""

    model: str
    workload: str
    threshold: float

    @property
    def config_id(self) -> str:
        return f"{self.model}|{self.workload}|{self.threshold:.2f}"


def full_grid() -> List[ExperimentConfig]:
    """
    Enumerate the frozen 30-configuration grid: model-major order so a sweep
    runner can share one EmbeddingManager per model across its 15 configs.
    """
    return [
        ExperimentConfig(model=model, workload=workload, threshold=threshold)
        for model in EMBEDDING_MODELS
        for workload in WORKLOADS
        for threshold in THRESHOLDS
    ]
