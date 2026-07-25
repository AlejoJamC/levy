"""
Live query decision for the dashboard's query panel (LEV-10 / D6).

Builds an in-process `SemanticCache` for a chosen embedding model, populated
from a dataset's queries, and evaluates user-supplied text under the
production `1/(1+L2)` semantics -- the same `VectorIndex.search()` call and
the same `similarity = 1/(1+distance)` formula `SemanticCache.get()` uses
internally. Unlike `get()`, which discards a sub-threshold nearest neighbour
entirely, `evaluate_query` always reports the nearest match and its
similarity, so a miss can still show "closest was X at 0.62".

The threshold is supplied per call rather than baked into the cache, so a
UI can flip the decision by re-evaluating the same index -- no re-embedding
of the dataset, matching the frozen `1/(1+L2)` scale used verbatim (never
rescaled).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from levy.cache.semantic_cache import SemanticCache
from levy.cache.vector_index import _l2_normalize
from levy.dataset.schema import QueryPair
from levy.embedding_manager import EmbeddingManager
from levy.models import LLMRequest

#: Exact search by default: the dataset is small, and exactness avoids
#: explaining HNSW approximation in a UI (design.md decision 5).
DEFAULT_BACKEND = "brute_force"


@dataclass
class QueryDecision:
    """The dashboard's report for one user query against one index."""

    hit: bool
    threshold: float
    similarity: Optional[float]
    matched_text: Optional[str]


class _BoundEmbeddingClient:
    """Binds an `EmbeddingManager` to one model: the interface `SemanticCache` expects."""

    def __init__(self, manager: EmbeddingManager, model_name: str) -> None:
        self._manager = manager
        self._model_name = model_name

    def embed(self, text: str) -> List[float]:
        return self._manager.embed_with(self._model_name, text)


def build_query_index(
    embedding_manager: EmbeddingManager,
    model_name: str,
    pairs: List[QueryPair],
    backend: str = DEFAULT_BACKEND,
) -> SemanticCache:
    """
    Build a `SemanticCache` for `model_name`, populated with every unique
    query text (`query_1` and `query_2`) across `pairs`.

    The cache's own `threshold` is irrelevant here -- `evaluate_query` takes
    its own threshold argument and never calls `SemanticCache.get()` -- so it
    is fixed at 0.0 purely to satisfy the constructor.
    """
    cache = SemanticCache(
        embedding_client=_BoundEmbeddingClient(embedding_manager, model_name),
        threshold=0.0,
        backend=backend,
    )

    seen = set()
    for pair in pairs:
        for text in (pair.query_1, pair.query_2):
            if text in seen:
                continue
            seen.add(text)
            cache.set(LLMRequest(prompt=text), response_text=text)

    return cache


def evaluate_query(cache: SemanticCache, text: str, threshold: float) -> QueryDecision:
    """
    Evaluate `text` against `cache`'s index at `threshold`.

    Reuses the production path exactly: `cache.embedding_client.embed`, the
    same L2 normalisation (`levy.cache.vector_index._l2_normalize`), the same
    `VectorIndex.search(k=1)`, and the same `similarity = 1/(1+distance)`
    formula `SemanticCache.get()` computes internally -- extended only to
    surface the nearest match and its similarity on a miss too.
    """
    if cache.size() == 0:
        return QueryDecision(hit=False, threshold=threshold, similarity=None, matched_text=None)

    raw = cache.embedding_client.embed(text)
    q_vec = _l2_normalize(np.array(raw, dtype=np.float32))
    ids, distances = cache._index.search(q_vec.tolist(), k=1)
    if not ids:
        return QueryDecision(hit=False, threshold=threshold, similarity=None, matched_text=None)

    distance = distances[0]
    similarity = 1.0 / (1.0 + distance)
    entry = cache._entries.get(ids[0])
    matched_text = entry.prompt if entry is not None else None

    return QueryDecision(
        hit=bool(similarity >= threshold),
        threshold=threshold,
        similarity=float(similarity),
        matched_text=matched_text,
    )
