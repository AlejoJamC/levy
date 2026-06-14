"""
VectorIndex abstraction for the semantic cache (LEV-2).

Two implementations:
- BruteForceVectorIndex  — numpy exact k-NN by L2, offline default and correctness oracle.
- FaissHNSWVectorIndex   — faiss.IndexHNSWFlat(dim, M) wrapped in IndexIDMap, as prescribed
                           by the frozen S&D Report.

Both implementations accept and return raw (un-normalised) vectors; normalisation is the
caller's responsibility (SemanticCache normalises before calling add/search, per design.md D3).

Factory: make_vector_index(backend, dim, **hnsw_params) selects the backend per config.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return unit-L2-norm copy of vec. Returns zero vector unchanged if norm == 0."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec.copy()
    return vec / norm


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VectorIndex(ABC):
    """
    Minimal ANN index interface used by SemanticCache.

    Vectors passed to add/search MUST already be unit-L2-normalised by the caller.
    Distances returned are always L2 distances (not similarities).
    """

    @abstractmethod
    def add(self, vector: List[float], entry_id: int) -> None:
        """Index a single (already-normalised) embedding with the given external id."""

    @abstractmethod
    def search(self, vector: List[float], k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Return (ids, L2_distances) for the k nearest neighbours.
        Both lists have length min(k, size()). Empty index returns ([], []).
        """

    @abstractmethod
    def reset(self) -> None:
        """Empty the index entirely, as if freshly constructed."""

    @abstractmethod
    def size(self) -> int:
        """Number of indexed vectors."""


# ---------------------------------------------------------------------------
# Brute-force implementation (numpy exact k-NN)
# ---------------------------------------------------------------------------

class BruteForceVectorIndex(VectorIndex):
    """
    Exact nearest-neighbour search by L2 distance over stored numpy vectors.

    This is:
    (a) the offline/fallback backend when Faiss is unavailable, and
    (b) the correctness oracle that FaissHNSWVectorIndex is validated against.

    Lazy dimension init: dimension is inferred from the first added vector.
    """

    def __init__(self) -> None:
        self._vectors: List[np.ndarray] = []
        self._ids: List[int] = []
        self._dim: int = 0

    def add(self, vector: List[float], entry_id: int) -> None:
        v = np.array(vector, dtype=np.float32)
        if self._dim == 0:
            self._dim = len(v)
        self._vectors.append(v)
        self._ids.append(entry_id)

    def search(self, vector: List[float], k: int = 1) -> Tuple[List[int], List[float]]:
        if not self._vectors:
            return [], []
        q = np.array(vector, dtype=np.float32)
        mat = np.stack(self._vectors)  # (n, dim)
        diffs = mat - q
        sq_dists = np.sum(diffs ** 2, axis=1)
        l2_dists = np.sqrt(sq_dists)
        k_eff = min(k, len(self._ids))
        top_idx = np.argsort(l2_dists)[:k_eff]
        return (
            [self._ids[i] for i in top_idx],
            [float(l2_dists[i]) for i in top_idx],
        )

    def reset(self) -> None:
        self._vectors.clear()
        self._ids.clear()
        self._dim = 0

    def size(self) -> int:
        return len(self._vectors)


# ---------------------------------------------------------------------------
# Faiss HNSW implementation
# ---------------------------------------------------------------------------

class FaissHNSWVectorIndex(VectorIndex):
    """
    Faiss IndexHNSWFlat (L2) wrapped in IndexIDMap so external entry ids are preserved.

    Lazy dimension init: the Faiss index is created on first add once dim is known.
    HNSW params (M, efConstruction, efSearch) are set at construction time.
    """

    def __init__(
        self,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._index = None  # created lazily
        self._size = 0

    def _ensure_index(self, dim: int):
        if self._index is None:
            import faiss  # guarded: caller must check availability
            hnsw = faiss.IndexHNSWFlat(dim, self._m)
            hnsw.hnsw.efConstruction = self._ef_construction
            hnsw.hnsw.efSearch = self._ef_search
            self._index = faiss.IndexIDMap(hnsw)

    def add(self, vector: List[float], entry_id: int) -> None:
        v = np.array(vector, dtype=np.float32).reshape(1, -1)
        self._ensure_index(v.shape[1])
        ids = np.array([entry_id], dtype=np.int64)
        self._index.add_with_ids(v, ids)
        self._size += 1

    def search(self, vector: List[float], k: int = 1) -> Tuple[List[int], List[float]]:
        if self._index is None or self._size == 0:
            return [], []
        q = np.array(vector, dtype=np.float32).reshape(1, -1)
        k_eff = min(k, self._size)
        sq_distances, ids = self._index.search(q, k_eff)
        # Faiss IndexHNSWFlat returns squared L2 distances; take sqrt for consistency
        # with BruteForceVectorIndex and the spec's "L2 distance" formula.
        return (
            [int(i) for i in ids[0] if i >= 0],
            [float(math.sqrt(max(d, 0.0))) for d, i in zip(sq_distances[0], ids[0]) if i >= 0],
        )

    def reset(self) -> None:
        self._index = None
        self._size = 0

    def size(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_vector_index(
    backend: str = "auto",
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> VectorIndex:
    """
    Select and construct a VectorIndex backend.

    backend: "auto" | "faiss" | "brute_force"
        auto  — Faiss if importable, else brute-force (with a warning).
        faiss — Faiss; raises ImportError if unavailable.
        brute_force — always numpy exact-NN.
    """
    if backend == "brute_force":
        return BruteForceVectorIndex()

    if backend == "faiss":
        import faiss  # noqa: F401  raises ImportError if absent
        return FaissHNSWVectorIndex(m=m, ef_construction=ef_construction, ef_search=ef_search)

    # auto
    try:
        import faiss  # noqa: F401
        return FaissHNSWVectorIndex(m=m, ef_construction=ef_construction, ef_search=ef_search)
    except ImportError:
        logger.warning(
            "faiss-cpu is not installed; falling back to BruteForceVectorIndex. "
            "Install via `conda install -c conda-forge faiss-cpu` for the full study setup."
        )
        return BruteForceVectorIndex()
