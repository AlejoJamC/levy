"""
Engine pool for the Levy API (LEV-7 design.md decision 2).

The engine binds `embedding_model` and `similarity_threshold` at construction,
while the frozen contract makes both per-request via `cache_config`. This pool
resolves the mismatch: one `LevyEngine` per distinct (embedding_model, threshold)
pair, built from a base `LevyConfig` with those two fields overridden. Instances
are reused across requests with the same pair (their caches accumulate); a
different pair is a different cache universe by design (a threshold defines the
cache decision boundary, and stored entries don't re-partition on the fly).

Embedding managers (and their loaded models) are shared across pool keys that
share an embedding_model, so switching only the threshold never reloads a model.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple

from levy.config import LevyConfig
from levy.embedding_manager import EmbeddingManager
from levy.engine import LevyEngine
from levy.metrics import LevyMetrics

PoolKey = Tuple[str, float]


class PoolCapExceededError(Exception):
    """Raised when a request's cache_config would create more engines than the cap."""

    def __init__(self, cap: int):
        self.cap = cap
        super().__init__(
            f"Engine pool cap of {cap} reached; cannot create another "
            f"(embedding_model, threshold) instance."
        )


class EnginePool:
    def __init__(self, base_config: LevyConfig, max_engines: int = 8):
        self.base_config = base_config
        self.max_engines = max_engines
        self._engines: Dict[PoolKey, LevyEngine] = {}
        self._managers: Dict[str, EmbeddingManager] = {}

    def _resolve_key(
        self, embedding_model: Optional[str], threshold: Optional[float]
    ) -> PoolKey:
        model = embedding_model or self.base_config.embedding_model
        thresh = (
            threshold if threshold is not None else self.base_config.similarity_threshold
        )
        return (model, thresh)

    def get(
        self, embedding_model: Optional[str] = None, threshold: Optional[float] = None
    ) -> LevyEngine:
        key = self._resolve_key(embedding_model, threshold)
        engine = self._engines.get(key)
        if engine is not None:
            return engine

        if len(self._engines) >= self.max_engines:
            raise PoolCapExceededError(self.max_engines)

        model, thresh = key
        cfg = dataclasses.replace(
            self.base_config, embedding_model=model, similarity_threshold=thresh
        )

        manager = self._managers.get(model)
        if manager is None:
            manager = EmbeddingManager.from_config(cfg)
            self._managers[model] = manager

        engine = LevyEngine(cfg, embedding_manager=manager)
        self._engines[key] = engine
        return engine

    def all_engines(self) -> List[LevyEngine]:
        return list(self._engines.values())

    def clear_all(self) -> Dict[str, Dict[str, int]]:
        """Empty every pooled engine's caches and reset its metrics.

        ExactCache.clear() is intentionally a no-op (its store may be shared);
        the underlying InMemoryStore's own clear() is the real accessor.
        """
        report: Dict[str, Dict[str, int]] = {}
        for key, engine in self._engines.items():
            exact_count = len(engine.store.entries)
            semantic_count = engine.semantic_cache.size()

            engine.store.clear()
            engine.semantic_cache.clear()
            engine.metrics = LevyMetrics()

            report[f"{key[0]}::{key[1]}"] = {
                "exact_entries": exact_count,
                "semantic_entries": semantic_count,
            }
        return report
