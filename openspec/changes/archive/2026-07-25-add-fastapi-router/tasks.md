# Tasks: add-fastapi-router

## 1. Dependencies

- [x] 1.1 Add `fastapi`, `uvicorn`, `pydantic` to `environment.yml` (conda-forge) and `pyproject.toml`; install into the `levy` env and verify imports.

## 2. Engine surface (additive accessors only)

- [x] 2.1 Expose semantic-index size and per-model cached-entry counts on the engine (reading `CacheEntry.metadata` model identity); no behavior changes; existing tests untouched and green.

## 3. API package

- [x] 3.1 `levy/api/schemas.py`: Pydantic v2 models — chat request (`messages`, `model`, `cache_config{threshold, embedding_model}`), Anthropic-format response body, stats response, clear response, structured error body.
- [x] 3.2 `levy/api/pool.py`: engine pool keyed by `(embedding_model, threshold)` built from the base `LevyConfig` with overrides; defaults fall back to config; configurable cap (default 8) → clear client error when exceeded.
- [x] 3.3 `levy/api/app.py`: FastAPI app; sync (`def`) endpoints so the blocking engine runs in the threadpool; `POST /v1/chat/completions` (extract prompt from messages, engine.generate, Anthropic-format body for hit and miss, `X-Cache-Status` / `X-Cache-Similarity` headers per `LevyResult`), `GET /admin/cache/stats` (aggregated snapshot + hit rate + index size + per-model breakdown), `POST /admin/cache/clear` (all pooled engines + metrics reset, report per-key counts); `BudgetExceededError` → structured 402-style JSON; endpoint descriptions written so auto-generated OpenAPI (`/docs`, `/openapi.json`) accurately documents the contract incl. header semantics.
- [x] 3.4 Replay-grade structured logging: one JSON log record per chat request (request_id echoed in response metadata, arrival/completion timestamps, resolved embedding_model+threshold, prompt, cache decision source, similarity, latency ms) via stdlib logging — records alone sufficient to replay the request sequence.

## 4. Tests (offline, TestClient + mock providers)

- [x] 4.1 Hit/miss flows: miss → MISS header + provider body; exact hit → HIT + similarity 1.0; semantic hit → HIT + reported similarity; body shape identical across hit/miss.
- [x] 4.2 Per-request config: distinct `(model, threshold)` pairs isolate caches; same pair accumulates; omitted fields use defaults; pool-cap breach → clear client error.
- [x] 4.3 Admin: stats consistency (hit rate matches counters, index size, per-model counts), clear empties caches + resets counters and a re-request misses.
- [x] 4.4 Errors: malformed body → 422 with field detail; budget-guard halt (mock transport tripping the LEV-6 guard) → structured error naming cap; unknown provider errors → structured 5xx JSON.
- [x] 4.5 Logging: capture the logger over a scripted sequence (exact hit, semantic hit, miss, distinct cache_configs) and assert every record field is present and the sequence is reconstructable (ordered prompts + per-request config); request_id in record matches response metadata.
- [x] 4.6 OpenAPI: `/openapi.json` serves and contains the three routes with request/response schemas.
- [x] 4.7 Gated suite green: `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` — no new pragmas.

## 5. Docs & sync

- [x] 5.1 README: run command (`uvicorn levy.api.app:app`), endpoint reference with curl examples demonstrating exact-hit / semantic-hit / miss with mock providers (incl. cache_config), header semantics, pointer to `/docs` OpenAPI UI, logging format, budget-error note. CLAUDE.md: architecture map entry for `levy/api/`, mark known-gap #1 resolved, record the async-at-boundary/threadpool decision.
- [x] 5.2 `openspec validate --all` passes; sync Linear LEV-7 (reference this change, tick delivered acceptance criteria, record the async decision).
