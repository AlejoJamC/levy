---
name: project-lev1-done
description: LEV-1 EmbeddingManager implementation complete — what was built and what's next
metadata:
  type: project
---

LEV-1 (add-embedding-manager) is implemented and moved to "In Review" on Linear.

**What was built:**
- `levy/embedding_manager.py`: `EmbeddingManager` with study-model registry (all-MiniLM-L6-v2 → `sentence-transformers/all-MiniLM-L6-v2`, modernbert → `nomic-ai/modernbert-embed-base`), lazy client loading, `(model_key, sha256(text))` memoization, symmetric `search_query: ` prefix for ModernBERT, Ollama passthrough for demo path.
- `levy/config.py`: defaults now `embedding_provider="sentence-transformers"`, `embedding_model="all-MiniLM-L6-v2"`.
- `levy/engine.py`: uses `EmbeddingManager.from_config(config)` exclusively; model identity stored in `CacheEntry.metadata` on every LLM-path store.
- `tests/test_embedding_manager.py`: 20 offline unit tests covering all spec scenarios.
- 22 total tests green.

**Why:** The entire 30-configuration experiment grid requires model switching via config. Memoization prevents redundant recomputation during replay harness (LEV-4).

**How to apply:** LEV-1 is the critical-path predecessor: LEV-2 (Faiss store) needs `manager.get_dimension()` and `get_model_identity()`. Remind user to archive this OpenSpec change with `/opsx:archive add-embedding-manager` when ready to move to LEV-2.
