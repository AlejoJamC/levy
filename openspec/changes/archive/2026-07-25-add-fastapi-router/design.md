# Design: add-fastapi-router

## Context

Verified state on the LEV-7 branch: `LevyEngine.generate(prompt) -> LevyResult` is synchronous and returns `source ∈ {llm, exact_cache, semantic_cache}` plus `similarity_score` — a direct source for the contract's cache headers. `MetricsSnapshot` carries totals/hits/misses/tokens-saved/latency but not index size or per-model breakdown. `clear()` exists on exact and semantic caches; `SemanticCache.reset()` exists for full re-init. The engine binds `similarity_threshold` and `embedding_model` at construction (`LevyConfig`), while the frozen contract makes both per-request via `cache_config`. The Anthropic connector (LEV-6, merged) is sync by recorded decision, deferring async to this layer. `fastapi`/`uvicorn`/`pydantic` are not yet dependencies. LEV-5's 90% branch-coverage gate is enforced; CLAUDE.md conventions allow Pydantic at the API layer only. The frozen contract (S&D "Intended interface"): `POST /v1/chat/completions` (messages/model/cache_config; Anthropic-format body; `X-Cache-Status`, `X-Cache-Similarity`), `GET /admin/cache/stats` (hit rate, index size, per-model breakdown). EPIC-001 adds request logging/validation and routing intent; known-gap #1 adds `/admin/cache/clear`.

## Goals / Non-Goals

**Goals:**
- Implement the frozen contract verbatim: endpoint paths, request shape, Anthropic-format response body, and the two cache headers.
- Honour per-request `cache_config` without breaking the engine's construction-time model.
- Admin observability: stats with hit rate, index size, per-model breakdown; cache clear.
- Offline, in-process test coverage (TestClient + mock providers) keeping the 90% gate green.
- Settle and record the async question deferred by LEV-6.

**Non-Goals:**
- No streaming responses, no auth/multi-tenancy, no rate limiting — research prototype surface, single-operator use.
- No dashboard (LEV-10), no Docker packaging (LEV-9).
- No change to experiment-harness behavior — experiments keep calling the engine as a library; the router is a parallel consumer.
- No async rewrite of the core engine or providers.

## Decisions

1. **Async at the boundary, sync engine in the threadpool.** Endpoints are declared `def` (sync) so FastAPI runs them in its threadpool automatically — the whole call chain (engine, caches, LEV-6 Anthropic client) is synchronous and blocking it on the event loop would serialize all requests. This satisfies the frozen "asynchronous wrapper" intent at the HTTP layer (concurrent request handling) without an `AsyncAnthropic` migration; if a later ticket needs true async I/O, the SDK's `AsyncAnthropic` mirrors the sync surface and the swap is confined to this layer. *Alternative rejected:* `async def` endpoints calling the sync engine directly — blocks the event loop; `AsyncAnthropic` now — would fork the LEV-6 client into two implementations for no measured need.
2. **Per-request `cache_config` via a bounded engine pool keyed by `(embedding_model, threshold)`.** The engine binds both at construction, and that is correct for experiment isolation — so the router owns a small pool: first request for a key constructs an engine from the base `LevyConfig` with those two fields overridden; subsequent requests reuse it (embedding models stay loaded via LEV-1's manager; caches accumulate per key, which is the correct semantic — a different threshold or model is a different cache universe per the frozen experiment design). Omitted `cache_config` fields fall back to `LevyConfig` defaults. The pool is capped (config, default 8 keys) with clear 400 on exceeding it — the study grid needs at most 2 models × a handful of thresholds. *Alternative rejected:* mutating one engine's threshold per request — cross-request races and semantically wrong (a threshold defines the cache decision universe, and stored entries don't re-partition); re-constructing per request — reloads models and empties caches every call, destroying the very caching being measured.
3. **Response body = Anthropic Messages format, assembled by the router.** The contract says the body "matches the Anthropic response format" for hits and misses alike. On miss the LEV-6 client's metadata (model, stop_reason, input/output tokens) populates it faithfully; on hit the router synthesizes the same shape from the cached entry (cached text, model identity from entry metadata, zeroed usage) so clients parse one format regardless of source. Cache identity lives in the headers, not the body, per the contract.
4. **Headers from `LevyResult`, uppercase per contract.** `source != "llm"` → `X-Cache-Status: HIT` + `X-Cache-Similarity: <similarity_score>` (exact hits report 1.0, which the engine already sets); `source == "llm"` → `X-Cache-Status: MISS`, no similarity header.
5. **Stats endpoint = `MetricsSnapshot` + two additive accessors.** Hit rate computed from existing counters; index size from the semantic cache's vector-index size (exists, needs a public accessor on the engine); per-model breakdown from cache-entry model identity (already stored in `CacheEntry.metadata` by LEV-1/LEV-2) aggregated per pooled engine and reported per pool key. Additive accessors only — no specced behavior changes, so no Modified Capabilities.
6. **Clear endpoint clears every pooled engine's caches** via existing `clear()` methods and resets metrics counters; responds with what was cleared (per-key entry counts). `POST` (state-changing), matching known-gap #1's naming.
7. **Pydantic v2 models in `levy/api/schemas.py` only**; conversion to/from core dataclasses at the boundary. Request validation errors surface as FastAPI's standard 422 with field detail (EPIC-001's "validate payload").
8. **Replay-grade structured logging (issue acceptance criterion).** Every chat request emits one machine-parseable JSON log record via stdlib `logging`: `request_id` (uuid, echoed in the response body metadata), arrival and completion timestamps, the resolved `(embedding_model, threshold)` key, the prompt content, cache decision (`source`), similarity score, and end-to-end latency ms. "Enough to replay any request sequence" is interpreted literally: the records alone must let an operator reconstruct the ordered prompt list and per-request cache config and re-drive it (e.g. through the harness or curl). Tests assert record completeness by capturing the logger. *Alternative rejected:* free-text access logs — not machine-replayable; a separate log file/DB — stdlib logging keeps the prototype simple and the operator chooses the sink via logging config.
9. **OpenAPI docs are a deliverable (D5), not a freebie.** FastAPI auto-generates `/docs` and `/openapi.json`; the schemas are written so the generated spec is accurate (typed models for request, response, stats, clear, and error bodies; header semantics documented in endpoint descriptions). A test asserts `/openapi.json` serves and includes the three routes.
8. **Layout:** `levy/api/app.py` (FastAPI app + routes), `levy/api/schemas.py` (Pydantic), `levy/api/pool.py` (engine pool). Server run documented as `uvicorn levy.api.app:app`; `uvicorn` is a runtime/dev dependency, not imported by library code, so it stays out of the coverage denominator naturally.

## Risks / Trade-offs

- [Engine pool memory: each key holds caches + (for real providers) loaded embedding models] → Cap (default 8) with explicit 400 beyond it; study usage needs ≤ 2 models × few thresholds; documented limitation for a research prototype.
- [Threadpool concurrency: default pool size bounds concurrent blocking requests] → Acceptable for single-operator research use; documented; tunable via uvicorn/anyio settings if ever needed.
- [Synthesized Anthropic-format body on cache hits can drift from the real format] → Shape is pinned by tests against the same schema used for the miss path; usage fields on hits are explicitly zeroed (that's the point of the cache) and the body carries the cached model identity.
- [Per-request `embedding_model` requests under `sentence-transformers` provider download models on first use] → In the offline suite, providers are mocks; README notes first-request latency for real providers.
- [Budget guard (LEV-6) can halt the miss path mid-session] → Router maps `BudgetExceededError` to a clear 402-style JSON error rather than a 500, so the operator sees the cap, not a stack trace.

## Migration Plan

Pure addition: new package + dependencies + docs. Nothing existing changes behavior; the library surface (engine, harness, CLIs) is untouched. Rollback = remove `levy/api/` and the dependency entries.

## Open Questions

- None blocking. Whether LEV-9's Docker image runs uvicorn directly or behind a process manager is LEV-9's decision; this change only documents the bare run command.
