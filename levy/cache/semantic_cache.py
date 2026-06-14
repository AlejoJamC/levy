"""
SemanticCache — Faiss-backed (or brute-force) semantic cache.

Retrieval follows Algorithm 1 from the frozen S&D Report exactly:
  embed query → L2-normalise → k=1 ANN search → distance → similarity = 1/(1+distance)
  → hit iff similarity >= threshold.

Internal-id→entry mapping (spec: "separate metadata dictionary mapping internal IDs
to (query_text, response, embedding_model)") is held in self._entries.
"""

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

from levy.cache.base import CacheInterface
from levy.cache.vector_index import VectorIndex, _l2_normalize, make_vector_index
from levy.models import CacheEntry, LLMRequest

logger = logging.getLogger(__name__)


class SemanticCache(CacheInterface):
    """
    Semantic cache backed by a VectorIndex.

    Parameters
    ----------
    embedding_client : object with embed(text) -> list[float] and get_dimension() -> int
        Typically an EmbeddingManager instance.
    vector_index : VectorIndex, optional
        Pre-constructed index (for tests). If None, built from backend/hnsw params.
    backend, m, ef_construction, ef_search : forwarded to make_vector_index when
        vector_index is None.
    threshold : similarity threshold in 1/(1+L2) space.
    """

    def __init__(
        self,
        embedding_client,
        threshold: float = 0.85,
        vector_index: Optional[VectorIndex] = None,
        backend: str = "auto",
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        self.embedding_client = embedding_client
        self.threshold = threshold

        self._index: VectorIndex = (
            vector_index
            if vector_index is not None
            else make_vector_index(
                backend=backend,
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
            )
        )
        # spec "separate metadata dictionary mapping internal IDs to (query_text, response, embedding_model)"
        self._entries: Dict[int, CacheEntry] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # CacheInterface
    # ------------------------------------------------------------------

    def get(self, request: LLMRequest) -> Optional[CacheEntry]:
        if self._index.size() == 0:
            return None

        raw = self.embedding_client.embed(request.prompt)
        q_vec = _l2_normalize(np.array(raw, dtype=np.float32))

        ids, distances = self._index.search(q_vec.tolist(), k=1)
        if not ids:
            return None

        distance = distances[0]
        similarity = 1.0 / (1.0 + distance)

        if similarity < self.threshold:
            return None

        entry = self._entries.get(ids[0])
        if entry is None:
            return None

        entry.access_count += 1
        entry.metadata["last_similarity_score"] = float(similarity)
        return entry

    def set(
        self,
        request: LLMRequest,
        response_text: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if embedding is None:
            embedding = self.embedding_client.embed(request.prompt)

        vec = _l2_normalize(np.array(embedding, dtype=np.float32))
        entry_id = self._next_id
        self._next_id += 1

        key_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        entry = CacheEntry(
            key_hash=key_hash,
            prompt=request.prompt,
            response_text=response_text,
            embedding=vec.tolist(),
            metadata=metadata or {},
        )
        self._index.add(vec.tolist(), entry_id)
        self._entries[entry_id] = entry

    def clear(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Per-configuration reset (LEV-4 calls this between experiment runs)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Empty the index and id→entry map; restart id counter."""
        self._index.reset()
        self._entries.clear()
        self._next_id = 0
