# Proposal: add-anthropic-connector

**Linear:** LEV-6 | **Maps to:** Objective O4, Deliverable D1 | **Spec basis:** S&D Report §B "LLM Connector: asynchronous wrapper around the Anthropic SDK with retry and token-accounting logic"; Proposal budget plan (~$50 expected, $200 cap).

## Why

The frozen spec names the Anthropic API as the artefact's LLM backend, but only mock/OpenAI/Ollama clients exist (CLAUDE.md known-gap #2). The connector is needed to populate caches realistically on misses during experiments and to serve the FastAPI router (LEV-7); it is not on the experiments critical path — hit/miss decisions are embedding-only — but is required for O4 completeness and blocks LEV-9 (release packaging).

## What Changes

- New `AnthropicLLMClient` in `levy/llm_client.py` implementing the existing synchronous `LLMClient` ABC via the official `anthropic` SDK (`client.messages.create`), added to the engine's provider factory as `llm_provider = "anthropic"`.
- **Retry:** use the SDK's built-in exponential-backoff retry (connection errors, 408/409/429/5xx; `max_retries` configurable, exposed through `LevyConfig`) rather than a hand-rolled loop; non-retryable API errors propagate as typed exceptions, never swallowed.
- **Token accounting:** read `response.usage.input_tokens` / `output_tokens` from every call; populate `LLMResponse.token_usage` (total) and carry the input/output split in `LLMResponse.metadata`, so `LevyMetrics`' tokens-saved accounting reflects real usage.
- **Budget guard:** a request counter + estimated-cost accumulator (tokens × configurable per-MTok prices) with a configurable hard stop honouring the frozen $200 cap (~$50 expected); once the cap is reached, further Anthropic calls raise a dedicated error instead of spending.
- **Config:** `ANTHROPIC_API_KEY` via `.env` (gitignored; `.env.example` gains the line), model name configurable with a current-model default — the frozen S&D's example model string (`claude-3-sonnet-20240229`) is retired, so the connector defaults to `claude-opus-4-8` (flagged as frozen-doc drift, not silently resolved).
- **Sync-now decision (assigned to this change by the issue):** the spec says "asynchronous wrapper" but the core engine is synchronous; the connector is implemented synchronously against the existing ABC, with async deferred to the FastAPI layer (LEV-7) where the SDK's `AsyncAnthropic` client fits naturally. Recorded in design.md.
- **Tests:** fully offline via the SDK's injectable HTTP transport (`httpx.MockTransport`) — success, usage accounting, retry-then-success, non-retryable error propagation, budget-guard halt — keeping the 90% coverage gate green without new pragmas. Mock/OpenAI/Ollama providers unchanged.
- `anthropic` added to `environment.yml` (pip section) and `pyproject.toml` dependencies.

## Capabilities

### New Capabilities

- `anthropic-connector`: Anthropic SDK backend for the engine — synchronous `LLMClient` implementation with configured retry, per-call token accounting, and a hard budget guard, selectable via configuration and testable fully offline.

### Modified Capabilities

_None. No existing capability's spec-level requirements change; the provider factory gains one branch behind the existing `llm_provider` switch._

## Impact

- **Code:** `levy/llm_client.py` (+`AnthropicLLMClient`, budget guard), `levy/config.py` (+`anthropic_api_key`, `anthropic_model`, `anthropic_max_retries`, budget-cap and price settings), `levy/engine.py` (factory branch), `tests/test_anthropic_client.py`.
- **Dependencies:** `anthropic` SDK (pip, in `environment.yml` + `pyproject.toml`).
- **Secrets:** `ANTHROPIC_API_KEY` in `.env` only; `.env.example` documents it; nothing committed.
- **Downstream:** LEV-7 router consumes the connector; LEV-4 harness can optionally use it to populate caches realistically (decisions unaffected); LEV-9 packaging unblocked.
- **Coverage gate:** client covered via mocked transport — no coverage-denominator exclusions added.
