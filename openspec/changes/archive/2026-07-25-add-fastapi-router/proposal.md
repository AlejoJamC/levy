# Proposal: add-fastapi-router

**Linear:** LEV-7 | **Maps to:** Objective O4, Deliverables D1 + D5 (API docs) | **Spec basis:** S&D Report "Intended interface (API contract)" + §B "FastAPI Router" component; EPIC-001 (client/proxy layer, working seed); CLAUDE.md known-gap #1.

## Why

The frozen S&D names the FastAPI router as one of D1's three components and defines its contract precisely (`POST /v1/chat/completions` with cache headers, `GET /admin/cache/stats`), but no API layer exists — the engine is only reachable as a Python library (CLAUDE.md known-gap #1, the last unresolved D1 component now that LEV-1/2/6 are merged). The router is the transparent interception point EPIC-001 defines: capture requests, route through the cache, expose observability.

## What Changes

- New `levy/api/` package: a FastAPI application exposing the frozen contract:
  - `POST /v1/chat/completions` — request body `{messages[], model, cache_config{threshold, embedding_model}}`; response body in the Anthropic response format; headers `X-Cache-Status: HIT|MISS` and, on hits, `X-Cache-Similarity` (mapped directly from `LevyResult.source`/`similarity_score`).
  - `GET /admin/cache/stats` — hit rate, index size, per-model breakdown (per the frozen contract; requires exposing index size and per-model counts, which `MetricsSnapshot` lacks today).
  - `POST /admin/cache/clear` — empties exact + semantic caches via their existing `clear()` methods (from known-gap #1/EPIC-001; additive to the frozen contract, which specifies only stats).
- Pydantic request/response models at the API layer only — sanctioned by CLAUDE.md conventions; the core package stays dataclasses.
- Per-request `cache_config` support: the frozen contract puts `threshold` and `embedding_model` on each request, while the engine binds both at construction — the router resolves this with a bounded pool of engine instances keyed by `(embedding_model, threshold)` (design.md records the decision and its limits).
- Async decision (deferred here by LEV-6's recorded design): endpoints are `async` at the FastAPI boundary with the synchronous engine executed via FastAPI's threadpool; the Anthropic backend stays the sync LEV-6 client. Recorded as the documented resolution of the frozen "asynchronous wrapper" wording.
- Structured request/response logging per request — request_id, arrival/completion timestamps, cache decision (source), similarity, latency — machine-parseable and complete enough to replay any request sequence (issue acceptance criterion, seeded by EPIC-001's logger/validator).
- API documentation (D5): FastAPI's auto-generated OpenAPI docs (`/docs`, `/openapi.json`) with accurate Pydantic schemas, plus curl examples in the README demonstrating exact-hit / semantic-hit / miss behavior with mock providers.
- Dependencies added: `fastapi`, `uvicorn`, `pydantic` (currently declared nowhere) in `environment.yml` + `pyproject.toml`; a run command documented (`uvicorn levy.api.app:app`).
- Fully offline tests via FastAPI's in-process `TestClient` with mock providers; the enforced 90% branch-coverage gate stays green with no new pragmas. Mock/OpenAI/Ollama providers keep working; the router serves whatever `llm_provider` is configured, Anthropic being the spec-named backend.

## Capabilities

### New Capabilities

- `api-router`: HTTP interface to the caching engine — the chat-completions proxy endpoint with cache-status headers, per-request cache configuration, admin observability/maintenance endpoints (stats, clear), replay-grade structured request logging, and auto-generated OpenAPI documentation.

### Modified Capabilities

_None. Existing capabilities are consumed through their current requirements; exposing index size / per-model counts is additive surface on the engine/metrics, not a change to any specced behavior._

## Impact

- **New code:** `levy/api/` (FastAPI app, Pydantic schemas, engine-pool/dependency wiring), `tests/test_api_router.py`.
- **Touched code:** minimal additive accessors for stats (semantic-cache index size, per-model entry counts) on existing classes; no behavior changes to engine/caches/providers.
- **Dependencies:** `fastapi`, `uvicorn`, `pydantic` (new); FastAPI's TestClient uses httpx, already present.
- **Docs:** README (run the server, endpoint reference, curl examples), CLAUDE.md (architecture map, mark known-gap #1 resolved).
- **Downstream:** completes the last D1 component; LEV-9 (release packaging / Docker) and LEV-10 (dashboard) build on it.
- **Frozen-doc drift note (flagged, not resolved):** the contract's example model `claude-3-sonnet-20240229` is retired; the router defaults to LEV-6's configured `anthropic_model`.
