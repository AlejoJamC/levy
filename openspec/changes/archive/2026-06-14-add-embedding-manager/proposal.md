# Proposal: add-embedding-manager

**Linear issue:** [LEV-1](https://linear.app/levy-project/issue/LEV-1) — Embedding Manager: runtime switching between all-MiniLM-L6-v2 and ModernBERT
**Maps to:** Objective O2, Deliverable D1 (frozen baseline: `docs/Project_Proposal.md`, `docs/Specification_and_Design_Report.md` §B "Embedding Manager")

## Why

The entire research study compares two embedding models — `all-MiniLM-L6-v2` (general-purpose baseline) and ModernBERT — across 30 experimental configurations, but the current code cannot run either of them as study models: `LevyConfig` defaults to `mxbai-embed-large` via Ollama (not part of the experimental grid), there is no concept of selecting a study model at runtime, and embeddings are recomputed on every call, which the S&D Report explicitly forbids for replay experiments ("caches them in memory to avoid redundant recomputation during replay experiments"). Without this component, no other experiment-critical work (Faiss vector store, experiment harness) can start.

## What Changes

- New `EmbeddingManager` component that loads a configured study model and generates embeddings, as defined in the S&D Report component architecture.
- Runtime model switching: selecting `all-MiniLM-L6-v2` or ModernBERT requires only a configuration value (no code changes), matching the spec's "supporting both embedding models via runtime configuration".
- In-memory embedding memoization keyed by (text, model): repeated embedding of the same text under the same model returns the cached vector without recomputation.
- `LevyConfig` defaults aligned with the study: default embedding provider becomes `sentence-transformers` with `all-MiniLM-L6-v2` (the baseline model) instead of Ollama/`mxbai-embed-large`.
- ModernBERT support via a sentence-transformers-compatible checkpoint (`nomic-ai/modernbert-embed-base`), including its required query/document prefixing.
- Existing `EmbeddingClient` ABC, mock client, Ollama client, and current `SentenceTransformerClient` remain working; the manager wraps clients rather than replacing the abstraction.

## Capabilities

### New Capabilities

- `embedding-management`: Loading, selecting, and switching embedding models at runtime; generating embeddings through a single entry point; memoizing embeddings per (text, model); exposing model identity and dimension to downstream consumers (vector store, experiment harness).

### Modified Capabilities

<!-- none: no existing specs in openspec/specs/ yet; current embedding behavior was never specified -->

## Impact

- **Code:** new `levy/embedding_manager.py` (or equivalent); changes to `levy/config.py` (defaults, study-model settings); `levy/engine.py` switches from a bare `EmbeddingClient` to the manager; `levy/embeddings.py` gains ModernBERT prefix handling.
- **Dependencies:** `sentence-transformers` is already a dependency; ModernBERT checkpoint requires `transformers>=4.48` (pulled in by sentence-transformers update if needed). All models run locally on Apple Silicon, no GPU.
- **Tests:** new unit tests for model switching and memoization using mock clients (offline). Existing 2 tests must keep passing.
- **Downstream:** LEV-2 (`add-faiss-vector-store`) consumes the manager's dimension/model identity; LEV-4 (`add-experiment-harness`) relies on memoization for replay performance; the FastAPI layer (LEV-7) will map `cache_config.embedding_model` onto this capability.
- **Not in scope:** Faiss index, experiment harness, API layer, removal of Ollama/mock providers.
