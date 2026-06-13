# Design: add-embedding-manager

## Context

The S&D Report (frozen) defines an "Embedding Manager" component: "loads the configured model (all-MiniLM-L6-v2 or ModernBERT), generates embeddings, caches them in memory to avoid redundant recomputation during replay experiments." Current state of the code:

- `levy/embeddings.py` has an `EmbeddingClient` ABC with three implementations: `MockEmbeddingClient` (text-seeded random, normalized), `SentenceTransformerClient` (loads any sentence-transformers checkpoint), `OllamaEmbeddingClient`.
- `levy/config.py` defaults to `embedding_provider="mock"` and `embedding_model="mxbai-embed-large"` — neither matches the study.
- `levy/engine.py` instantiates one client at construction time and calls `embed()` directly; nothing is memoized, and there is no notion of "which study model produced this vector".
- Downstream consumers that will need model identity and dimension: the Faiss vector store (LEV-2, one index per model/dimension) and the experiment harness (LEV-4, 30 configurations = each model used in 15 runs, so model loading and embedding reuse matter).

Constraints: synchronous code, dataclasses only in the core package, ABC + mock pattern for every external dependency, everything runnable offline on Apple Silicon without GPU, conda env `levy`.

## Goals / Non-Goals

**Goals:**
- Single entry point (`EmbeddingManager`) for embedding generation with runtime model selection.
- Both study models usable by name: `all-MiniLM-L6-v2` and ModernBERT.
- Memoization per (model, text) so replay experiments never recompute an embedding.
- Config defaults that match the study baseline.
- Model identity + dimension exposed to downstream consumers.

**Non-Goals:**
- Faiss index or any vector-store change (LEV-2).
- Experiment harness (LEV-4).
- API layer / `cache_config.embedding_model` plumbing (LEV-7) — the manager only has to make that mapping trivial later.
- Async support, batch embedding optimization, or removing the Ollama/mock/OpenAI providers.
- Persisting the memoization cache to disk.

## Decisions

### D1. ModernBERT checkpoint: `nomic-ai/modernbert-embed-base`

The frozen docs name "ModernBERT (149M parameters)" as the study model, but raw ModernBERT (`answerdotai/ModernBERT-base`) is a masked-LM encoder, not a sentence-embedding model. The study needs a sentence-embedding checkpoint built on the ModernBERT backbone.

- **Chosen:** `nomic-ai/modernbert-embed-base` — trained for sentence embeddings on the ModernBERT-base (149M) backbone, loadable through sentence-transformers, runs on CPU/MPS.
- **Alternatives considered:** raw `answerdotai/ModernBERT-base` with mean pooling (no contrastive training → poor similarity quality, would misrepresent the model in the study); `lightonai/modernbert-embed-large` (different parameter count than the frozen docs describe).
- The concrete checkpoint id MUST be recorded in config and in experiment outputs so the dissertation can state precisely what was evaluated.

### D2. Task prefixes applied symmetrically

`modernbert-embed-base` expects task prefixes (`search_query: `, `search_document: `). Semantic-cache lookups are query-to-query comparisons (a new prompt against stored prompts), i.e. a **symmetric** similarity task.

- **Chosen:** apply the **same prefix (`search_query: `) to every text**, both at store time and at lookup time, implemented inside the client so callers never see prefixes. A per-model `prefix` setting in the model registry (empty for all-MiniLM) keeps this declarative.
- **Alternative considered:** asymmetric prefixes (`search_document:` for stored, `search_query:` for lookups) — designed for retrieval, not duplicate detection; would also make similarity scores order-dependent, which contaminates the threshold sweep.
- **Why it matters:** prefixes change the embedding space and therefore similarity magnitudes; inconsistency here would silently bias the study's core metric.

### D3. Manager wraps `EmbeddingClient`, does not replace it

`EmbeddingManager` holds a registry of named model configurations and lazily constructs/caches one `EmbeddingClient` per model on first use. The ABC and all existing clients stay as-is.

- Keeps the repo's provider-abstraction convention (mock stays first-class: a manager configured with the mock provider works offline).
- Lets the harness later hold one manager and switch models between configurations without re-loading checkpoints (each checkpoint loads once per process, then is reused across the 15 configurations that need it).
- **Alternative considered:** making `SentenceTransformerClient` itself multi-model — rejected; conflates one-model client responsibility with cross-model orchestration and doesn't cover mock/Ollama.

### D4. Study-model aliases resolved in one place

A small registry maps canonical study names to (provider, checkpoint, prefix):

- `all-minilm` / `all-MiniLM-L6-v2` → sentence-transformers, `sentence-transformers/all-MiniLM-L6-v2`, no prefix.
- `modernbert` → sentence-transformers, `nomic-ai/modernbert-embed-base`, `search_query: ` prefix.

The spec's API contract uses `"embedding_model": "modernbert"`, so the alias spelling is part of the public research contract. Unknown names fail fast with the list of known models — silent fallback could invalidate experiment results.

### D5. Memoization: plain dict keyed by (model_key, sha256(text))

- In-memory `dict`, no eviction, hashed text as key (avoids holding full prompt strings twice).
- Scale justification: the study embeds ≤ 1,800 unique texts per model (900 pairs); at 384–768 floats each this is a few MB — eviction logic is unjustified complexity.
- A `clear()` method exists for tests and for the harness's per-configuration resets (note: clearing memoization is *optional* between configs since embeddings are deterministic per model; the harness decides).
- **Alternative considered:** `functools.lru_cache` — rejected: cache key must include the model and the cache must be inspectable/clearable for tests.

### D6. Config changes are additive and study-aligned

- `embedding_provider` default: `"mock"` → `"sentence-transformers"`; `embedding_model` default: `"mxbai-embed-large"` → `"all-MiniLM-L6-v2"`.
- Tests and examples that relied on mock defaults MUST pass providers explicitly (they already do).
- Ollama settings remain for the demo path; `examples/ollama_demo.py` keeps working by passing its own config.

### D7. Dependency pins

`nomic-ai/modernbert-embed-base` needs `transformers>=4.48`; bump `sentence-transformers` to `>=3.0` in `environment.yml` and `pyproject.toml`. Verify inside the conda env (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy`).

## Risks / Trade-offs

- [ModernBERT checkpoint fails to load on the pinned stack] → bump pins per D7 and smoke-test both models in the env before integrating; if the nomic checkpoint is unusable, escalate — checkpoint substitution changes the study and must be a human decision (frozen-docs scope).
- [Prefix handling biases similarity scores] → prefixes are declarative per model, applied symmetrically (D2), covered by a unit test asserting the prefix is applied to both store and lookup paths.
- [Default-config change breaks existing demos/tests] → existing tests pass providers explicitly; run the full suite plus `examples/simple_replay.py` after the config change.
- [First call latency: checkpoint download + load] → lazy loading keeps engine construction fast; document that the first embed per model downloads the checkpoint (network needed once; cached in HF cache afterwards).
- [Memoization returns stale vectors if a model id is reused with different settings] → model registry entries are immutable; memoization key includes the resolved model key.

## Migration Plan

Pure addition plus config-default change; no data migration. Rollback = revert the change. Engine behavior with mock providers is unchanged, so existing tests are the regression net.

## Open Questions

- None blocking. If `nomic-ai/modernbert-embed-base` is unavailable at implementation time (D7 risk), stop and surface the decision rather than substituting silently.
