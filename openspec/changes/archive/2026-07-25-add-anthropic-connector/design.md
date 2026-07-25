# Design: add-anthropic-connector

## Context

The engine's LLM surface is the synchronous `LLMClient` ABC (`generate(LLMRequest) -> LLMResponse`) with mock/OpenAI/Ollama implementations selected by `LevyConfig.llm_provider` in the engine factory (`levy/engine.py:30-41`). `LLMResponse` carries `token_usage: int` and a free-form `metadata` dict; `LevyMetrics` tracks tokens saved. Secrets load from `.env` via python-dotenv. LEV-5's 90% branch-coverage gate is enforced, with pragmas reserved for genuinely network-only code. The frozen S&D names the Anthropic API as the backend and the frozen budget is ~$50 expected / $200 hard cap. Verified SDK facts (claude-api reference): the `anthropic` Python SDK retries connection errors and 408/409/429/5xx with exponential backoff (`max_retries`, default 2, configurable per client), exposes typed exceptions (`RateLimitError`, `APIStatusError`, `APIConnectionError`, …), reports usage as `response.usage.input_tokens`/`output_tokens`, and accepts an injectable `http_client` (`DefaultHttpxClient`, which can wrap `httpx.MockTransport`) — the repo already depends on httpx.

## Goals / Non-Goals

**Goals:**
- `llm_provider="anthropic"` generates end-to-end through the production engine path.
- Retry on transients with exponential backoff; typed, unswallowed propagation of non-retryable errors.
- Real token accounting per call, wired into the existing `LLMResponse`/`LevyMetrics` flow.
- Hard budget stop honouring the frozen $200 cap, with request/cost visibility.
- Fully offline test coverage via mocked transport; the 90% gate stays green with no new pragmas.

**Non-Goals:**
- No FastAPI router (LEV-7) and no async engine rewrite — the async decision is settled below, not implemented broadly.
- No streaming, no extended-thinking configuration, no prompt caching — the connector sends plain single-turn `messages.create` calls; responses never affect cache hit/miss decisions.
- No change to mock/OpenAI/Ollama providers or to experiment-harness behavior.
- No real-API integration tests in the suite (no key in CI; a manual smoke script path is documented instead).

## Decisions

1. **Official `anthropic` SDK, synchronous client.** `AnthropicLLMClient` wraps `anthropic.Anthropic().messages.create(...)` behind the existing sync ABC. *The issue explicitly assigns the sync/async decision here:* the whole core is synchronous (engine, caches, harness), so an async connector would need an event-loop shim used by nothing; the FastAPI layer (LEV-7) is the natural async boundary and the SDK offers `AsyncAnthropic` with the same surface when that lands. The frozen spec's "asynchronous wrapper" intent is satisfied at the router layer, and this is recorded as the documented resolution rather than silent drift. *Alternative rejected:* raw httpx like the OpenAI client — hand-rolls retry/backoff and error taxonomy the SDK already provides and the spec explicitly names the SDK.
2. **Retry = SDK built-in, configured, not reimplemented.** The SDK already does exponential backoff on exactly the transient classes the issue lists (connection errors, 408/409/429/5xx, honouring `retry-after`). Expose `anthropic_max_retries` in `LevyConfig` (default: SDK default 2) and pass it to the client constructor. Non-retryable errors (4xx except the above) propagate as the SDK's typed exceptions after being counted by the budget guard. *Alternative rejected:* custom retry loop — duplicates tested SDK behavior and risks double-retrying (SDK retries × wrapper retries).
3. **Token accounting: total in `token_usage`, split in `metadata`.** `LLMResponse.token_usage = usage.input_tokens + usage.output_tokens` (keeps the existing int contract every consumer uses); `metadata` carries `{"input_tokens": ..., "output_tokens": ..., "model": response.model, "stop_reason": ...}` for auditability. `LevyMetrics` needs no interface change — real usage now flows through the same fields the mock populates.
4. **Budget guard as a small in-process accumulator owned by the client.** Tracks request count and estimated cost = `input_tokens × input_price + output_tokens × output_price` (per-MTok prices configurable in `LevyConfig`, defaulted for the default model). Before each call, if estimated spend ≥ `anthropic_budget_cap_usd` (default 200.0 per the frozen plan), raise a dedicated `BudgetExceededError` — no request is sent. Counter state is per-client-instance (per engine/process); that matches the experiment usage pattern (one process per run) and is documented as such. *Alternative rejected:* persistent cross-process ledger — out of scope for a prototype; the frozen cap is an operational safety net, and the Console remains the authoritative spend record.
5. **Model default `claude-opus-4-8`, configurable via `anthropic_model`.** The frozen S&D's API-contract example (`claude-3-sonnet-20240229`) is a retired model ID and would 404 — this is flagged as frozen-doc drift per CLAUDE.md rules (the doc is not edited). Default to the current recommended model `claude-opus-4-8`; the config knob lets cost-sensitive cache-population runs choose e.g. `claude-haiku-4-5`. Price defaults follow the default model and are config-overridable so the estimate stays honest when the model changes.
6. **Offline testability via injectable transport, not pragmas.** `AnthropicLLMClient` accepts an optional `http_client` passed through to the SDK constructor. Tests build `anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))` and script wire-level responses: a success with a usage block, a 429-then-success sequence (proves SDK retry engages), a 400 (propagates typed, uncounted as spend), and usage large enough to trip the budget guard. This keeps every line of the connector inside the coverage denominator — consistent with LEV-5's "intentional exclusions only" requirement. *Alternative rejected:* `# pragma: no cover` like the httpx-based clients — unnecessary here precisely because the SDK exposes the transport seam.
7. **Placement:** `AnthropicLLMClient` and `BudgetExceededError` live in `levy/llm_client.py` beside the other providers; config fields in `LevyConfig` (`anthropic_api_key` from `ANTHROPIC_API_KEY`, `anthropic_model`, `anthropic_max_retries`, `anthropic_budget_cap_usd`, `anthropic_input_price_per_mtok`, `anthropic_output_price_per_mtok`); one `elif config.llm_provider == "anthropic"` branch in the engine factory. `.env.example` gains `ANTHROPIC_API_KEY=sk-ant-placeholder`.

## Risks / Trade-offs

- [Cost estimate drifts from real billing (price changes, cache/batch pricing nuances)] → The guard is a conservative safety net, not an invoice: prices are config-visible, the estimate is labeled an estimate in the metrics/metadata, and the $200 cap sits 4× above expected spend.
- [Per-process budget state doesn't survive restarts] → Documented limitation; acceptable for a research prototype where runs are supervised and the Anthropic Console tracks true spend. LEV-7 can revisit persistence if the router goes long-lived.
- [SDK majors can change surface] → Pin a floor version in `pyproject.toml`/`environment.yml`; the wrapper touches a minimal surface (`messages.create`, `usage`, typed errors, `http_client`).
- [Safety-classifier refusals (`stop_reason: "refusal"`) return 200 with empty content on some current models] → The client checks `stop_reason` before reading content and raises a clear error naming the refusal rather than caching an empty response; cache population simply skips that pair.
- [Sync connector under a future async router] → LEV-7 decision point; the SDK's `AsyncAnthropic` mirrors the sync surface, so porting is mechanical and confined to the router layer.

## Migration Plan

Pure addition behind a new `llm_provider` value; default provider remains unchanged, so no existing behavior shifts. Rollback = remove the branch and class. No data migration.

## Open Questions

- None blocking. Whether experiment cache-population runs actually use the connector (vs the mock) is LEV-4/LEV-8 operational choice; the harness contract is unaffected either way.
