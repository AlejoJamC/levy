# Tasks: add-embedding-manager

Implementation context for the executing agent: read `proposal.md`, `design.md`, and `specs/embedding-management/spec.md` in this directory first. Run every Python command inside the conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy`. Tracking issue: Linear LEV-1.

## 1. Dependencies and environment

- [ ] 1.1 Bump `sentence-transformers` to `>=3.0` in `environment.yml` and `pyproject.toml`; ensure `transformers>=4.48` is satisfied in the `levy` conda env (`pip install -U "sentence-transformers>=3.0"` inside the env)
- [ ] 1.2 Smoke-test both checkpoints load and embed inside the env: `sentence-transformers/all-MiniLM-L6-v2` and `nomic-ai/modernbert-embed-base` (one short text each; first run downloads weights). If the ModernBERT checkpoint cannot load, STOP and report — do not substitute another checkpoint (design.md D7 / frozen-docs scope)

## 2. Model registry and configuration

- [ ] 2.1 Create the study-model registry (per design.md D4): canonical aliases `all-minilm`/`all-MiniLM-L6-v2` → (`sentence-transformers`, `sentence-transformers/all-MiniLM-L6-v2`, no prefix) and `modernbert` → (`sentence-transformers`, `nomic-ai/modernbert-embed-base`, prefix `search_query: `); unknown names raise an error listing known models
- [ ] 2.2 Update `levy/config.py` defaults: `embedding_provider="sentence-transformers"`, `embedding_model="all-MiniLM-L6-v2"`; keep Ollama settings intact for the demo path
- [ ] 2.3 Verify `examples/ollama_demo.py` and both existing tests still pass providers explicitly (they do today — just confirm nothing relies on the old defaults)

## 3. EmbeddingManager

- [ ] 3.1 Create `levy/embedding_manager.py` with `EmbeddingManager`: resolves a model name via the registry, lazily constructs and caches one `EmbeddingClient` per resolved model (design.md D3), and exposes `embed(text)`, `embed_with(model_name, text)`, `get_dimension()`, `get_model_identity()` (canonical name + resolved checkpoint id)
- [ ] 3.2 Implement memoization keyed by `(resolved_model_key, sha256(text))` (design.md D5): repeated embeds return the cached vector without calling the client; per-model independence; `clear_memoization()` method
- [ ] 3.3 Implement symmetric prefix handling inside the manager/client boundary (design.md D2): registry prefix is prepended to every text for that model, callers never pass prefixes; all-MiniLM path passes raw text
- [ ] 3.4 Support the mock provider through the manager (config `embedding_provider="mock"` bypasses the registry checkpoint and uses `MockEmbeddingClient`), preserving the offline-first convention

## 4. Engine integration

- [ ] 4.1 Update `levy/engine.py` to construct an `EmbeddingManager` from `LevyConfig` and route all embedding calls (semantic cache lookups and post-LLM stores) through it; `SemanticCache` keeps receiving an object with an `embed()`/`get_dimension()` interface so its code change is minimal
- [ ] 4.2 Ensure the manager's model identity is available on stored entries (e.g., in `CacheEntry.metadata`) so LEV-2 (Faiss store) and LEV-4 (harness) can label vectors by model later

## 5. Tests (offline, mock-based; spec scenarios are the test list)

- [ ] 5.1 Test runtime selection and switching: manager with model A then model B in one process; unknown model name raises with known-model list (spec: "Runtime study-model selection")
- [ ] 5.2 Test alias resolution exposes resolved checkpoint ids (spec: "Study-model alias registry")
- [ ] 5.3 Test memoization: same (model, text) computes once (count client calls via mock), different models create independent entries, `clear_memoization()` forces recompute (spec: "Embedding memoization")
- [ ] 5.4 Test dimension and identity exposure (spec: "Model identity and dimension exposure")
- [ ] 5.5 Test prefix handling: ModernBERT-registered model receives prefixed text on both store and lookup paths; baseline receives raw text (assert on the text reaching the mock client) (spec: "Symmetric task-prefix handling")
- [ ] 5.6 Test default config values (spec: "Default configuration matches the study baseline")
- [ ] 5.7 Run the full existing suite: `python -m unittest discover -s tests -p "test_*.py"` — all green (use pytest instead if LEV-5 has landed by then)

## 6. Documentation and wrap-up

- [ ] 6.1 Update `CLAUDE.md` (code architecture section + known-gaps item 5) and `README.md` configuration example to reflect the new defaults and the manager
- [ ] 6.2 Run `examples/simple_replay.py` in the env as an end-to-end sanity check (uses sentence-transformers when installed)
- [ ] 6.3 Update Linear LEV-1: tick acceptance-criteria checkboxes and set status (In Review/Done per workflow)
