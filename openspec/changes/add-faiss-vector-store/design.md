# Design: add-faiss-vector-store

## Context

The frozen S&D Report defines the semantic-cache retrieval engine precisely:

- Component: "Vector Store (Faiss): HNSW index over embedding vectors (**L2 distance**), with a separate metadata dictionary mapping internal IDs to `(query_text, response, embedding_model)`."
- Algorithm 1 (`cache_lookup`): embed query → `faiss_index.search(emb, k=1)` → `best_distance` → `similarity = 1.0 / (1.0 + best_distance)` → return the matched entry iff `similarity >= threshold`.

Current state (after LEV-1):

- `levy/cache/semantic_cache.py` does an O(n) brute-force **cosine** scan over `store.get_all_with_embeddings()` and writes `last_similarity_score` into the entry metadata.
- `levy/engine.py` computes the embedding via `EmbeddingManager`, stamps `get_model_identity().as_dict()` into `CacheEntry.metadata`, and on a miss calls only `exact_cache.set(...)`. The semantic path currently piggybacks on the shared `InMemoryStore.vector_index` list.
- `EmbeddingManager` (LEV-1) exposes `embed()`, `embed_with()`, `get_dimension()`, `get_model_identity()` and memoizes embeddings. It loads checkpoints lazily.
- `LevyConfig.similarity_threshold` defaults to 0.85; the study sweep is 0.70–0.90 step 0.05.

Constraints (repo conventions): synchronous; dataclasses in core; every external dependency has an ABC + offline fallback; everything must run offline with mocks; conda env `levy`. The frozen docs win on research scope — implement their formula literally and flag conflicts rather than resolving them.

## Goals / Non-Goals

**Goals:**
- Faiss HNSW index over L2 distance, with the spec's `similarity = 1/(1+distance)` transform and k=1 lookup, behind the existing `SemanticCache` surface.
- A `VectorIndex` abstraction so the mock/offline path needs no Faiss, and so a numpy exact-NN implementation can serve as the correctness oracle.
- An internal-id → entry metadata mapping holding `(query_text, response, embedding_model)`.
- Per-configuration `reset()` (empty index per experiment run).
- Distance scale comparable across the two study models so the model comparison is fair.

**Non-Goals:**
- Experiment harness / 30-config sweep (LEV-4) — this change only provides `reset()` and the metadata map it will use.
- Statistical analysis (LEV-8), API layer (LEV-7), Anthropic connector (LEV-6).
- Redis-backed vector search, on-disk index persistence/serialization, GPU Faiss, multi-vector or k>1 retrieval.
- Re-tuning the frozen threshold range.

## Decisions

### D1. `VectorIndex` ABC with Faiss and brute-force implementations

Add `levy/cache/vector_index.py`:

```
class VectorIndex(ABC):
    def add(self, vector: list[float], entry_id: int) -> None
    def search(self, vector: list[float], k: int = 1) -> tuple[list[int], list[float]]  # (ids, L2 distances)
    def reset(self) -> None
    def size(self) -> int
```

- `FaissHNSWVectorIndex` — `faiss.IndexHNSWFlat(dim, M)` (L2), wrapped in an `IndexIDMap` so external ids are preserved. Lazily created on first `add` once the dimension is known.
- `BruteForceVectorIndex` — numpy: stores vectors, returns exact k-NN by L2. This is (a) the offline default when Faiss is unavailable, and (b) the oracle the Faiss index is validated against.

**Why an abstraction:** mirrors the repo's ABC+mock convention; keeps the test suite and CI Faiss-free; gives an exact-NN oracle. **Alternative considered:** call Faiss directly inside `SemanticCache` — rejected: no offline fallback, nothing to validate approximate-NN against.

### D2. Similarity = `1/(1+distance)`, identical formula in both index paths

`SemanticCache` does: `ids, distances = index.search(q, k=1)`; `similarity = 1.0/(1.0+distances[0])`; hit iff `similarity >= threshold`. This is Algorithm 1 verbatim. The brute-force oracle returns the SAME L2 distances, so both paths feed the SAME transform — they can only disagree when HNSW's approximation returns a non-nearest neighbour (see D6/Risks).

**BREAKING (internal):** this replaces cosine similarity. Scores and the operative meaning of `similarity_threshold` change. Intentional and spec-mandated.

### D3. L2-normalize embeddings at index time and query time

Normalize every vector to unit L2 norm before `add` and before `search`.

- **Rationale (central to research validity):** the primary research question is whether *model choice* affects false positives. all-MiniLM-L6-v2 and `nomic-ai/modernbert-embed-base` can emit embeddings with different raw magnitudes; on raw vectors the same threshold would mean different things per model, confounding the comparison. Normalizing makes the L2 distance scale identical across models, so a single threshold sweep is comparable.
- **Consequence (must be carried downstream, not hidden):** for unit vectors, `distance = sqrt(2 - 2·cosine)`, so `similarity = 1/(1+sqrt(2-2·cosine))`. This compresses the range — e.g. cosine 0.95 → sim ≈ 0.76, cosine 0.90 → sim ≈ 0.69, cosine 0.99 → sim ≈ 0.88. Thus the frozen sweep 0.70–0.90 (in this similarity space) corresponds to a high-cosine band (~0.91–0.998). LEV-4 and LEV-8 must interpret thresholds in THIS space.
- **Flag, do not resolve:** the frozen docs define both the `1/(1+distance)` formula and the 0.70–0.90 range. Implement them literally. If experiments later show this band makes the >30% hit-rate viability bar unreachable, that is a research-scope finding to surface to the supervisor — NOT a reason for the implementer to switch to cosine or rescale thresholds.
- **Alternative considered:** no normalization — rejected: breaks cross-model comparability, the experiment's core requirement.

### D4. `SemanticCache` owns the index + an id→entry map; engine indexes on miss

`SemanticCache` holds a `VectorIndex` plus `self._entries: dict[int, CacheEntry]` keyed by a monotonic internal id (the spec's "separate metadata dictionary mapping internal IDs to (query_text, response, embedding_model)" — `CacheEntry` already carries prompt, response_text, and the embedding-model identity in `metadata` from LEV-1).

- `SemanticCache.set(request, response_text, embedding)`: normalize → `index.add(vec, id)` → `self._entries[id] = entry`.
- `SemanticCache.get(request)`: embed via the manager → normalize → `search(k=1)` → transform → threshold → on hit, set `last_similarity_score` and return `self._entries[id]`.
- **Engine change:** on a miss with semantic enabled, call `semantic_cache.set(...)` in addition to `exact_cache.set(...)`. The semantic index becomes independent of `InMemoryStore.vector_index` (which may then be left for the exact path only). This matches the spec diagram's "Store embedding + response → Vector Store" arrow.

**Alternative considered:** keep piggybacking on `InMemoryStore.vector_index` and rebuild the Faiss index from it on each `get` — rejected: O(n) rebuilds defeat the index; incremental `add` is the point of HNSW.

### D5. Per-configuration `reset()`

`SemanticCache.reset()` (and `clear()`) empties both the `VectorIndex` and the id→entry map and resets the id counter. LEV-4 calls this between the 30 configurations so each run starts cold, per the spec's experimental procedure ("Initialise an empty Levy cache instance").

### D6. HNSW parameters with conservative defaults, exposed in config

Defaults: `M=32`, `efConstruction=200`, `efSearch=64`, added to `LevyConfig` (e.g. `hnsw_m`, `hnsw_ef_construction`, `hnsw_ef_search`). For the study corpus (≤300 stored vectors per configuration) these make HNSW effectively exact. Also add `vector_index_backend: "auto" | "faiss" | "brute_force"` (default `"auto"`: Faiss if importable, else brute-force).

### D7. `faiss-cpu` dependency, import-guarded

Add `faiss-cpu` to `environment.yml` and `pyproject.toml`. Guard the import; if it fails (or backend is `brute_force`), construct `BruteForceVectorIndex` and log a warning. Tests run on brute-force without requiring Faiss; a separate test exercises Faiss when present.

### D8. Lazy dimension init

The index needs the vector dimension at construction. Initialize it on first `add` from `len(embedding)` (or eagerly from `EmbeddingManager.get_dimension()` — but lazy keeps engine construction cheap, preserving LEV-1's lazy-load design and avoiding a model load when semantic cache is disabled).

## Risks / Trade-offs

- [Approximate NN (HNSW) returns a non-nearest neighbour near the threshold boundary, diverging from exact-NN] → high `efSearch`; corpus is tiny per config so HNSW is effectively exact; an acceptance test asserts Faiss agrees with the brute-force oracle on a fixture (hit/miss identical). If they ever diverge in experiments, brute-force is the defined oracle.
- [Threshold-scale shift (D3) makes the frozen 0.70–0.90 band catch only very-high-cosine pairs, risking the >30% hit-rate viability bar] → documented above; carried into LEV-4/LEV-8; flagged for the supervisor as a research-scope matter. Implementer must NOT silently rescale.
- [Engine now calls `semantic_cache.set()` — double storage vs the old shared-store path] → exact and semantic stores are intentionally separated (D4); verify the existing `test_semantic_cache_mock_behavior` assertion (`len(engine.store.entries) == 1`) still holds (exact_cache.set still writes to the store) and adjust the test only if its intent is preserved.
- [`faiss-cpu` wheel availability on Apple Silicon] → `faiss-cpu` ships arm64 wheels; if install fails the `auto` backend falls back to brute-force so the suite stays green; record the resolution.
- [Normalizing a zero vector] → guard: if norm == 0, skip normalization (or treat as max distance); covered by a unit test.

## Migration Plan

Additive modules plus a `SemanticCache` rewrite and a small engine change. No persisted data to migrate (cache is in-memory). Rollback = revert the change; the mock path and exact cache are unaffected. The behavior change (D2/D3) is the migration: any consumer reading `similarity_score` now sees `1/(1+L2)` values — only the engine and (future) router read it, both updated here / aware.

## Open Questions

- None blocking. The one substantive research-scope item (threshold scale under D3) is deliberately left for the supervisor/LEV-8 to interpret, not for this change to resolve. If `faiss-cpu` cannot be installed in the env at implementation time, ship with the brute-force backend (exact-NN, same formula) and note that Faiss validation is pending — do not block LEV-4 on it.
