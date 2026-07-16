# api-router

Capability: HTTP interface to the caching engine — the chat-completions proxy endpoint with cache-status headers and per-request cache configuration, plus admin observability (stats) and maintenance (clear) endpoints.

## ADDED Requirements

### Requirement: Chat completions endpoint per the frozen contract
The system SHALL expose `POST /v1/chat/completions` accepting a JSON body with `messages` (role/content list), optional `model`, and optional `cache_config` carrying `threshold` and/or `embedding_model`, and SHALL serve the response through the engine's production flow (exact cache → semantic cache → LLM provider).

#### Scenario: Miss serves a fresh provider response
- **WHEN** a prompt not present in any cache is posted
- **THEN** the configured LLM provider generates the answer and the response body is returned in the Anthropic response format

#### Scenario: Hit serves the cached response
- **WHEN** a prompt matching a cached entry (exactly or semantically above threshold) is posted
- **THEN** the cached answer is returned without calling the LLM provider, in the same Anthropic response format

#### Scenario: Invalid request is rejected with detail
- **WHEN** the body is missing `messages` or malformed
- **THEN** the endpoint returns a validation error identifying the offending field, and nothing reaches the engine

### Requirement: Cache-status response headers
The system SHALL set `X-Cache-Status: HIT` on responses served from either cache and `X-Cache-Status: MISS` on responses served by the LLM provider, and SHALL set `X-Cache-Similarity` to the match's similarity score on hits (1.0 for exact-cache hits), omitting it on misses.

#### Scenario: Semantic hit reports similarity
- **WHEN** a request is served from the semantic cache
- **THEN** the response carries `X-Cache-Status: HIT` and `X-Cache-Similarity` with the match's similarity score

#### Scenario: Miss reports status only
- **WHEN** a request is served by the LLM provider
- **THEN** the response carries `X-Cache-Status: MISS` and no `X-Cache-Similarity` header

### Requirement: Per-request cache configuration
The system SHALL honour `cache_config.threshold` and `cache_config.embedding_model` per request by routing to an engine instance bound to that `(embedding_model, threshold)` pair, reusing instances across requests with the same pair (their caches accumulate), falling back to configured defaults for omitted fields, and rejecting requests that would exceed the configured pool cap with a clear client error naming the cap.

#### Scenario: Distinct configs use distinct cache universes
- **WHEN** the same prompt is stored under threshold 0.85 and then requested under threshold 0.70
- **THEN** each threshold's requests are decided against its own cache instance, not the other's

#### Scenario: Same config reuses accumulated cache
- **WHEN** two requests with the same `cache_config` arrive in sequence for similar prompts
- **THEN** the second is decided against the cache populated by the first

#### Scenario: Pool cap exceeded
- **WHEN** a request's `cache_config` would create more engine instances than the configured cap
- **THEN** the endpoint returns a client error naming the cap and no new instance is created

### Requirement: Admin cache statistics
The system SHALL expose `GET /admin/cache/stats` returning hit rate, semantic-index size, and a per-model breakdown of cached entries, alongside the existing counters (requests, exact/semantic hits, misses, tokens saved, average latency), aggregated across all pooled engine instances.

#### Scenario: Stats reflect activity
- **WHEN** requests have produced hits and misses
- **THEN** the stats response reports a hit rate consistent with the counters, the current index size, and entry counts broken down by embedding model

### Requirement: Admin cache clear
The system SHALL expose `POST /admin/cache/clear` that empties the exact and semantic caches of every pooled engine instance and resets metrics counters, reporting what was cleared.

#### Scenario: Clear empties caches
- **WHEN** entries exist and the clear endpoint is called
- **THEN** a previously-hitting prompt misses on the next request and the stats report zeroed counters

### Requirement: Provider-agnostic serving with clear failure mapping
The router SHALL serve whichever `llm_provider` is configured (mock, OpenAI, Ollama, Anthropic) unchanged, and SHALL map provider failures to structured JSON errors — in particular, a budget-guard halt (LEV-6) returns a payment-required-style error naming the cap rather than a generic server error.

#### Scenario: Budget cap surfaces cleanly
- **WHEN** the Anthropic budget guard refuses a call because the cap is reached
- **THEN** the endpoint returns a structured error naming the cap and estimated spend, not an unhandled 500

### Requirement: Replay-grade structured request logging
The system SHALL emit one machine-parseable structured log record per chat request carrying request_id, arrival and completion timestamps, the resolved cache configuration (embedding model, threshold), the prompt, the cache decision (source), the similarity score when present, and end-to-end latency, such that the log records alone suffice to replay any request sequence in order with identical cache configuration.

#### Scenario: Record completeness
- **WHEN** a request is served (hit or miss)
- **THEN** exactly one structured record is logged containing every field above, with the request_id matching the one echoed in the response

#### Scenario: Sequence is replayable from logs
- **WHEN** a sequence of requests has been served
- **THEN** the ordered log records contain the prompts and per-request cache configuration needed to re-drive the same sequence

### Requirement: OpenAPI documentation
The system SHALL serve auto-generated OpenAPI documentation (`/openapi.json`, interactive `/docs`) whose schemas accurately describe the chat, stats, clear, and error bodies, and project documentation SHALL include curl examples demonstrating exact-hit, semantic-hit, and miss behavior with mock providers.

#### Scenario: OpenAPI reflects the contract
- **WHEN** `/openapi.json` is fetched
- **THEN** it includes the three endpoints with their request/response schemas

### Requirement: Offline test coverage through the HTTP surface
The full router SHALL be exercised in-process (FastAPI TestClient) with mock providers and no network: hit/miss headers, per-request config routing, stats, clear, validation errors, and error mapping — keeping the enforced 90% branch-coverage gate green with no new coverage pragmas.

#### Scenario: Suite runs offline
- **WHEN** the gated test command runs with no network access
- **THEN** all router tests pass through the HTTP layer (including logging-record and OpenAPI assertions) and the coverage gate passes
