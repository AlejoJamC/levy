# Tasks: add-faiss-vector-store

Implementation context for the executing agent: read `proposal.md`, `design.md`, and `specs/vector-store/spec.md` in this directory first. This change builds on LEV-1 (`EmbeddingManager` is already wired into the engine). Run every Python command inside the conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy`. Tracking issue: Linear LEV-2.

Frozen-scope guardrails: implement the spec's `similarity = 1/(1+distance)` and the 0.70–0.90 threshold range literally. The cosine→L2 change (design.md D2) and the normalization-induced threshold-scale shift (design.md D3) are intentional — document them, do NOT rescale thresholds or revert to cosine. If `faiss-cpu` will not install, ship the brute-force backend and report it; do not block.

## 1. Dependencies and environment

- [x] 1.1 Add `faiss-cpu` to `environment.yml` and `pyproject.toml`; install into the `levy` conda env and confirm `import faiss` works on this machine (Apple Silicon arm64 wheel). If it fails, record the error and proceed with the brute-force backend (design.md D7/Open Questions)
- [x] 1.2 Smoke-test in the env: build a tiny `IndexHNSWFlat`, add a few vectors, search k=1, confirm L2 distances are returned

## 2. VectorIndex abstraction

- [x] 2.1 Create `levy/cache/vector_index.py` with the `VectorIndex` ABC: `add(vector, entry_id)`, `search(vector, k=1) -> (ids, distances)`, `reset()`, `size()` (design.md D1)
- [x] 2.2 Implement `BruteForceVectorIndex` (numpy, exact k-NN by L2) — this is the offline default and the correctness oracle. Lazy dimension init on first `add` (design.md D8)
- [x] 2.3 Implement `FaissHNSWVectorIndex` using `faiss.IndexHNSWFlat(dim, M)` wrapped in `IndexIDMap` to preserve external ids; params `M`, `efConstruction`, `efSearch` from constructor args (design.md D6); lazy dimension init on first `add`
- [x] 2.4 Add a factory/selector honoring `vector_index_backend` = `auto | faiss | brute_force` (`auto`: Faiss if importable else brute-force, with a warning) (design.md D6/D7)
- [x] 2.5 Implement unit-norm L2 normalization applied in both `add` and `search` paths, with a zero-norm guard (design.md D3; spec "Embeddings normalized…")

## 3. Configuration

- [x] 3.1 Extend `levy/config.py`: add `vector_index_backend: str = "auto"`, `hnsw_m: int = 32`, `hnsw_ef_construction: int = 200`, `hnsw_ef_search: int = 64`. Leave `similarity_threshold` default at 0.85 but update its comment to note it is now `1/(1+L2)` similarity, not cosine (design.md D2)

## 4. SemanticCache rewrite

- [x] 4.1 Rewrite `levy/cache/semantic_cache.py` to own a `VectorIndex` (selected from config) plus `self._entries: dict[int, CacheEntry]` keyed by a monotonic internal id (spec "Internal-id to entry metadata mapping"); accept the `EmbeddingManager` as today
- [x] 4.2 `set(request, response_text, embedding)`: if embedding is None embed via the manager; normalize; `index.add(vec, id)`; store the `CacheEntry` (carrying prompt, response, and the LEV-1 model-identity metadata) in `self._entries[id]`
- [x] 4.3 `get(request)`: embed via the manager → normalize → `search(k=1)` → `similarity = 1/(1+distance)` → hit iff `>= threshold`; on hit set `entry.metadata['last_similarity_score']` and return the resolved entry; empty index → miss (spec "Similarity from L2 distance")
- [x] 4.4 Implement `reset()`/`clear()` to empty the index, the id→entry map, and the id counter (spec "Per-configuration reset")

## 5. Engine integration

- [x] 5.1 Update `levy/engine.py`: on a miss with `enable_semantic_cache`, call `semantic_cache.set(request, llm_response.text, embedding=embedding)` (in addition to `exact_cache.set(...)`) so vectors are indexed in the new path (design.md D4). Construct `SemanticCache` with the configured backend + HNSW params
- [x] 5.2 Confirm the semantic hit path still surfaces `similarity_score` into `LevyResult` (now an L2-similarity value) and that `enable_semantic_cache=False` skips index construction/use entirely

## 6. Tests (offline by default; mock embeddings)

- [x] 6.1 `VectorIndex`: add/search/reset/size on `BruteForceVectorIndex`; normalization applied; zero-vector guard (spec scenarios "Vectors are indexed…", "Empty index", "Zero vector is safe", "Reset empties the index")
- [x] 6.2 Similarity transform + threshold: assert hit at/above threshold and miss below, using controlled vectors with known L2 distances → known `1/(1+d)` (spec "Similarity from L2 distance")
- [x] 6.3 id→entry resolution: a returned nearest id resolves to the entry whose prompt/response/model-identity match what was stored (spec "Internal-id to entry metadata mapping")
- [x] 6.4 Backend agreement: store the same fixture vectors in both `BruteForceVectorIndex` and `FaissHNSWVectorIndex`, query, assert identical hit/miss + same nearest entry. Skip (don't fail) this test if Faiss is unavailable (spec "Brute-force and Faiss agree…")
- [x] 6.5 Engine end-to-end with mock provider: store on miss, retrieve on a near-duplicate, confirm `source == "semantic_cache"`; verify the existing `test_semantic_cache_mock_behavior` assertion still holds or update it preserving intent (design.md Risks)
- [x] 6.6 Run the full suite green: `python -m unittest discover -s tests -p "test_*.py"` (or pytest if LEV-5 has landed)

## 7. Documentation and wrap-up

- [x] 7.1 Update `CLAUDE.md`: code-architecture entry for `vector_index.py` + rewritten `semantic_cache.py`; flip known-gaps item 3 (Faiss HNSW) to resolved; note the cosine→`1/(1+L2)` semantics and the threshold-scale implication (design.md D3) so LEV-4/LEV-8 inherit it
- [x] 7.2 Run `examples/simple_replay.py` in the env as an end-to-end sanity check
- [x] 7.3 Update Linear LEV-2: tick acceptance-criteria checkboxes, note the threshold-scale flag for the supervisor, set status (In Review/Done per workflow)
