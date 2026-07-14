"""
Offline replay protocol, Algorithm 2 of the frozen S&D Report (LEV-4 / D3).

Per configuration: start from a fresh, empty cache (a new `LevyEngine`),
then for each `QueryPair` of the configuration's workload, in dataset
order -- submit `query_1` (a miss by construction; stored), then submit
`query_2` and record the cache's hit/miss decision. The cache accumulates
entries across pairs within the configuration and is never reset between
pairs. Submissions go through the engine's production lookup path (exact
cache, then semantic cache) so an exact-duplicate `query_2` is legitimately
decided by the exact cache, not only the semantic index.
"""

from typing import List, Optional

from levy.config import LevyConfig
from levy.dataset.schema import QueryPair
from levy.embedding_manager import EmbeddingManager
from levy.engine import LevyEngine
from levy.experiment.config import ExperimentConfig
from levy.experiment.metrics import DecisionRecord, EvaluationResult, evaluate_confusion


def run_experiment(
    config: ExperimentConfig,
    pairs: List[QueryPair],
    embedding_manager: Optional[EmbeddingManager] = None,
    embedding_provider: str = "mock",
) -> EvaluationResult:
    """
    Replay `pairs` (filtered to `config.workload`) against a fresh engine
    configured for `config.model` / `config.threshold`, and return the
    confusion-matrix metrics plus a per-pair decision log.

    `embedding_manager`, when given, is used as-is (e.g. one manager shared
    across a model's 15 configurations by the sweep runner, so LEV-1's
    memoization is not defeated). Otherwise a manager is constructed from
    `embedding_provider` / `config.model` for standalone use.
    """
    workload_pairs = [pair for pair in pairs if pair.workload == config.workload]

    engine_config = LevyConfig(
        llm_provider="mock",
        embedding_provider=embedding_provider,
        embedding_model=config.model,
        enable_exact_cache=True,
        enable_semantic_cache=True,
        similarity_threshold=config.threshold,
        cache_store_type="memory",
    )
    engine = LevyEngine(engine_config, embedding_manager=embedding_manager)

    tp = fp = tn = fn = 0
    decisions: List[DecisionRecord] = []

    for pair in workload_pairs:
        engine.generate(pair.query_1)
        result = engine.generate(pair.query_2)

        is_hit = result.source in ("exact_cache", "semantic_cache")
        label = pair.ground_truth_label()

        if is_hit and label == 1:
            tp += 1
            outcome = "TP"
        elif is_hit and label == 0:
            fp += 1
            outcome = "FP"
        elif not is_hit and label == 0:
            tn += 1
            outcome = "TN"
        else:
            fn += 1
            outcome = "FN"

        decisions.append(
            DecisionRecord(
                config_id=config.config_id,
                pair_id=pair.pair_id,
                decision="hit" if is_hit else "miss",
                source=result.source,
                similarity=result.similarity_score,
                label=label,
                outcome=outcome,
            )
        )

    return evaluate_confusion(
        config=config,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n=len(workload_pairs),
        decisions=decisions,
    )
