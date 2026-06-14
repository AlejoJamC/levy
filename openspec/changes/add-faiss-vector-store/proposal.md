# Proposal: add-faiss-vector-store

**Linear issue:** [LEV-2](https://linear.app/levy-project/issue/LEV-2) — Vector Store: Faiss HNSW index replacing the brute-force cosine scan
**Maps to:** Objective O4, Deliverable D1 (frozen baseline: `docs/Specification_and_Design_Report.md` §B "Vector Store (Faiss)" + Algorithm 1 `cache_lookup`; Johnson et al. 2021)
**Depends on:** LEV-1 `add-embedding-manager` (done) — consumes `EmbeddingManager.get_dimension()` / `get_model_identity()`.

## Why

The spec prescribes a Faiss HNSW vector index (L2 distance, `similarity = 1/(1+distance)`) as the semantic-cache retrieval engine. The current `levy/cache/semantic_cache.py` instead does an O(n) brute-force **cosine** scan over every stored embedding. Two problems: (1) it does not match the frozen design, and (2) cosine similarity is a different metric on a different numeric scale than the spec's L2-derived similarity, so the threshold sweep (0.70–0.90) cannot be interpreted consistently with the spec until the retrieval math matches. The experiment harness (LEV-4) and statistical analysis (LEV-8) both sit directly downstream and depend on this retrieval semantics being correct and fixed.

## What Changes

- Introduce a `VectorIndex` abstraction with two implementations: `FaissHNSWVectorIndex` (the spec's index) and `BruteForceVectorIndex` (numpy, exact-NN — the offline default/fallback *and* the validation oracle for Faiss).
- `SemanticCache` retrieval switches to: nearest-neighbour search (k=1) → L2 distance → `similarity = 1/(1+distance)` → threshold compare, exactly per Algorithm 1. **BREAKING (internal):** semantic similarity scores change from cosine to `1/(1+L2)`; the numeric meaning of `similarity_threshold` changes accordingly.
- L2-normalize embeddings at index and query time so the distance scale is comparable across the two study models (required for a fair model comparison — the central research question).
- `SemanticCache` owns a metadata mapping from internal index id → cache entry `(query_text, response, embedding_model)`, per the spec's "separate metadata dictionary".
- Per-configuration index reset so each of the 30 experiment runs starts from an empty cache.
- `faiss-cpu` added as a dependency, import-guarded: if Faiss is absent, the engine falls back to `BruteForceVectorIndex` with a warning (preserves the offline-first / mock-first convention).

## Capabilities

### New Capabilities

- `vector-store`: Approximate-nearest-neighbour retrieval for the semantic cache — index construction (Faiss HNSW over L2), the spec's L2→similarity transform and threshold decision, the internal-id → entry metadata mapping, per-configuration reset, and the exact-NN brute-force fallback used both offline and as the correctness oracle.

### Modified Capabilities

<!-- none synced to openspec/specs/ yet. This change supersedes the unspecified cosine-scan behavior in levy/cache/semantic_cache.py; once add-embedding-manager and this change are archived, the synced specs will carry these requirements. -->

## Impact

- **Code:** new `levy/cache/vector_index.py` (ABC + Faiss + brute-force); rewrite of `levy/cache/semantic_cache.py` retrieval/storage to use a `VectorIndex` + id→entry map; `levy/engine.py` calls `semantic_cache.set()` on miss (so vectors are indexed) and lazily initializes the index dimension from `EmbeddingManager`; `levy/config.py` gains HNSW params + an index-backend toggle.
- **Dependencies:** `faiss-cpu` in `environment.yml` and `pyproject.toml` (conda env `levy`, CPU-only on Apple Silicon). No GPU.
- **Behavior change:** semantic similarity values and the operative meaning of `similarity_threshold` change (cosine → `1/(1+L2)`). This is intentional and matches the frozen spec — see `design.md` for the threshold-scale implications, which must be carried into LEV-4/LEV-8 and flagged to the human if they conflict with intended research semantics rather than silently "corrected".
- **Tests:** new offline tests (mock embeddings) for index add/search/reset, the similarity transform, brute-force↔Faiss agreement, and threshold boundary behavior. Existing 2 tests + LEV-1 tests must stay green.
- **Downstream:** LEV-4 (harness) uses `reset()` per configuration and the (model-labelled) metadata map; LEV-7 (router) reports similarity from this path in the `X-Cache-Similarity` header.
- **Not in scope:** the experiment harness, statistical analysis, API layer, Redis vector search, persistence/serialization of the index to disk.
