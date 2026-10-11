## 1. Default and engine hooks

- [x] 1.1 In `levy/config.py`, change the `llm_provider` default from `mock` to `anthropic`. Update the tests that relied on the old default to state their provider.
- [x] 1.2 Add the `SemanticLookupObservation` dataclass to `levy/models.py`
- [x] 1.3 Add an optional `observation` argument to `SemanticCache.get()`. When it is passed, record reached, index size, threshold, and the nearest prompt with its distance and similarity, on a miss as well as a hit. Leave the return path unchanged.
- [x] 1.4 Add `SEGMENT_LLM_CALL` and `SEGMENT_STORE` to `levy/latency/timing.py`, outside `SEGMENTS`
- [x] 1.5 In `LevyEngine.generate()`, add an optional `observation` argument and forward it to the semantic lookup. Mark exact hits as not reached. Wrap the provider call and the cache write in `segment(timing, …)`.

## 2. Console options and pool (`levy/console/`)

- [x] 2.1 `options.py`: embedding models per provider, LLM models per provider, and per-MTok prices per Anthropic model, seeded per design D3. No mock in any list. Build the `Literal` types from them.
- [x] 2.2 The console's default configuration: `LevyConfig()` with the vector index backend resolved to the one in use.
- [x] 2.3 `ConsolePool`:
  - keyed on a frozen dataclass of the nine settings;
  - embedding managers shared per `(embedding_provider, embedding_model)`;
  - its own cap and structured error;
  - each LLM client built with the request's model and that model's prices.
- [x] 2.4 `PrefixedRedisStore`: prefixes keys with the configuration id, counts its own keys, and clears only its prefix. The console pool swaps it into redis engines and rebuilds their `ExactCache`.

## 3. Trace recorder (`levy/console/tracing.py`)

- [x] 3.1 Implement `TraceRecorder`:
  - lock-guarded and bounded, oldest evicted first;
  - global version advanced on every create/update;
  - an epoch replaced on clear;
  - `since(version)` returning deep copies in arrival order.
- [x] 3.2 Implement `LiveTimingCollector(TimingCollector)`. On segment start it publishes the current stage; on segment end it publishes the span with its start offset and duration.
- [x] 3.3 Build the trace record per design D7:
  - configuration and request-set settings;
  - skipped caches;
  - outcome and spans;
  - semantic observation with cosine `1 − d²/2`;
  - embedding identity, LLM block, `cache_after`, error block.
- [x] 3.4 Derive provider location and endpoint per D8.
- [x] 3.5 Compute the call cost with `_BudgetGuard`'s formula and the request model's configured prices (`null` when none), and cumulative spend over the console engines' clients.
- [x] 3.6 Implement origin attribution per D9: record on a completed miss, resolve on exact and semantic hits, clear on console cache clear.

## 4. Console routes (`levy/console/routes.py`)

- [x] 4.1 `APIRouter(prefix="/console", tags=["console"])`, included by `create_app()`. Leave the delivered routes and schemas as they are.
- [x] 4.2 `POST /console/chat`:
  - takes `prompt` and the nine settings as query parameters typed by the lists, each defaulting to the console default;
  - has a dependency that rejects a model outside its provider's list, naming the parameter and the valid models;
  - opens the trace before resolving the engine;
  - passes a `LiveTimingCollector` and an observation;
  - returns the delivered response model and headers;
  - on an exception, finishes the trace as `error` with its kind and stage, then re-raises into the existing handlers.
- [x] 4.3 `GET /console/traces?after=<version>` returns traces, version, epoch, the console default configuration, aggregate console stats and cumulative spend. `POST /console/traces/clear` empties the history.
- [x] 4.4 `POST /console/cache/clear` empties every console engine's caches and the origins, and reports what was cleared.
- [x] 4.5 `GET /console` and the `/console/assets/` static mount, excluded from the OpenAPI schema. Add the static assets to `pyproject.toml` package data.

## 5. Visual design

- [x] 5.1 Produce the console design in Claude Design. Cover:
  - the live view at half-screen width beside Swagger's console section, with its selectable settings;
  - map drawn from each request's configuration with the host/internet boundary;
  - history;
  - trace detail with configuration, waterfall, semantic decision, cost and savings, identity;
  - the session summary;
  - the exact-hit, semantic-hit, miss, in-flight, error and cache-switched-off states;
  - Anthropic (remote) and Ollama (local).

## 6. Console page (`levy/console/static/`)

- [x] 6.1 Port the design into a static page (vanilla JS, inline SVG, no framework, bundler, CDN or remote font).
- [x] 6.2 Poll `/console/traces` with the version cursor. Reset on an epoch change. Show the connection state.
- [x] 6.3 Map: draw the nodes from the selected trace's configuration (store, embedding provider and model, backend, threshold, LLM provider, model and location; switched-off caches shown as skipped). Animate each request hop by hop along its outcome's path, park in-flight requests at `current_stage`, stop errors at their failing stage.
- [x] 6.4 History, newest first, with outcome colour, prompt start, LLM provider and latency. Selecting an entry replays it and opens its detail. Follow live traffic without overriding a viewer's selection.
- [x] 6.5 Trace detail:
  - configuration, with request-set settings distinguished from defaults;
  - linear waterfall with printed values and the lookup-to-call ratio;
  - semantic decision with nearest prompt, distance, similarity, cosine and threshold;
  - cost and savings with a link to the origin request;
  - identity;
  - prompt and answer.
- [x] 6.6 Session summary: served, hits by kind, misses, hit rate, spend, savings.

## 7. Tests (offline)

- [x] 7.1 Console chat and options:
  - each setting routes to its own engine;
  - cross-provider models are rejected with the parameter named;
  - `mock` is rejected as a provider;
  - the OpenAPI schema carries the enums under the `console` tag;
  - the delivered endpoints' OpenAPI entries are unchanged;
  - console requests leave the delivered caches untouched.
- [x] 7.2 `PrefixedRedisStore`: two configurations storing the same prompt do not collide; counts and clear are per prefix.
- [x] 7.3 Engine hooks:
  - observation on miss, hit, exact hit and empty index;
  - `llm_call`/`store` recorded on a miss only;
  - lookup decomposition still within the total;
  - one prompt sequence through engines with and without observation and timing yields identical results.
- [x] 7.4 Trace feed:
  - complete miss, exact-hit and semantic-hit traces, with the semantic similarity equal to `X-Cache-Similarity`;
  - configuration and request-set settings;
  - skipped caches;
  - provider location for Ollama and Anthropic;
  - token and cost fields through `httpx.MockTransport`.
- [x] 7.5 Attribution, cursor and buffer:
  - an exact hit references its origin;
  - origins cleared with the console cache;
  - the cursor returns only changes;
  - the buffer bound evicts the oldest;
  - clear starts a new epoch.
- [x] 7.6 Errors and in-flight:
  - budget-guard and pool-cap error traces, with client responses as the existing handlers produce them;
  - a blocking injected LLM client shows the trace in flight at `llm_call` mid-call.
- [x] 7.7 Page route: `/console` and its assets are served, and the static assets reference no external origin.
- [x] 7.8 Full suite passes, and the gated run (`--cov=levy --cov-branch --cov-fail-under=90`) is green with no new pragmas.

## 8. Verification

- [ ] 8.1 Run `uvicorn levy.api.app:app` with the console page and `/docs` side by side at half width. From Swagger's console section, fire requests to Anthropic and to Ollama on localhost, varying threshold, embedding model, cache switches, backend and store. In the browser, check exact hit, semantic hit, miss, in-flight wait, pool-cap error and history replay.

## 9. Documentation

- [x] 9.1 `README.md`: the console section in Swagger and the page, side by side.
- [x] 9.2 `docs/ARCHITECTURE.md`: the console section, its pool, tracing path and page, as a layer beside the delivered endpoints.
- [x] 9.3 `CLAUDE.md`:
  - the `llm_provider` default;
  - code map entries for `levy/console/`, the engine hooks and the new tests;
  - the LEV-23 row in the Linear ↔ OpenSpec table.

## 10. Validation

- [x] 10.1 `openspec validate --all` passes

## 11. Remove the OpenAI provider (design D13, 2026-10-11)

- [x] 11.1 Remove `OpenAILLMClient` from `levy/llm_client.py` and the engine's `openai` branch and import from `levy/engine.py`. Make the engine's final branch build the mock only for `mock` and raise, naming the accepted providers, for anything else, so a removed or mistyped provider never silently becomes the mock.
- [x] 11.2 Remove `openai_api_key` and `openai_base_url` from `LevyConfig`, and `OPENAI_API_KEY` from `.env.example`.
- [x] 11.3 Remove the tests that only checked the OpenAI client's construction (`tests/test_engine.py`, `tests/test_providers.py`), and the OpenAI mentions in test docstrings.
- [x] 11.4 Remove OpenAI from `README.md`, `docs/ARCHITECTURE.md` and `CLAUDE.md`. Leave `scripts/audit_release.sh`'s key patterns and the archived OpenSpec changes as they are.
- [x] 11.5 Remove `openai` from the console's lists, pool and tracing, and from their tests.
- [x] 11.6 Full suite and gated coverage run pass.

## 12. No model or price in the code (design D14, 2026-10-11)

- [x] 12.1 `LevyConfig` reads the numbered `LEVY_ANTHROPIC_MODEL_<n>` with `_INPUT_USD_PER_MTOK` / `_OUTPUT_USD_PER_MTOK`, `LEVY_OLLAMA_MODEL_<n>` and `LEVY_OLLAMA_EMBEDDING_MODEL_<n>`. The lowest number is the default. A price that is not a plain number with a dot, a model without both prices, or a price without its model fails naming the variable.
- [x] 12.2 Remove the default model and prices from `AnthropicLLMClient`, `OllamaLLMClient` and `OllamaEmbeddingClient`; constructing one without them fails naming the variable.
- [x] 12.3 Build the console's model lists and prices from the configuration (`ConsoleOptions`), re-reading `.env` on each request and Swagger load (`current_model_catalog`, `publish_choices`), so a new model needs no restart. Offer no provider without models, and answer 503 when no LLM model is configured.
- [x] 12.4 `scripts/populate_responses.py`, `scripts/run_throughput.py` and `examples/ollama_demo.py` take models from the environment or their flags.
- [x] 12.5 Document the variables in `.env.example` (placeholders only), `README.md`, `docs/ARCHITECTURE.md` and `CLAUDE.md`.
- [x] 12.6 Tests use placeholder model ids; environment parsing tested in `tests/test_config.py`; full suite and gated coverage run pass.

## 13. Real embeddings by default (design D15, 2026-10-11)

- [x] 13.1 `sentence-transformers` is the default embedding provider in `scripts/reproduce.sh`, `docker-compose.yml`, `run_experiments.py`, `run_latency.py`, `check_replication.py`, the harness and benchmark functions, and the examples; the CLIs list only `sentence-transformers` and `ollama`.
- [x] 13.2 The dashboard's query demo embeds with `sentence-transformers` only.
- [x] 13.3 Tests that need no model pass the mock explicitly; the suite passes with no model weights available.
- [x] 13.4 `README.md`, `docs/REPRODUCTION.md`, `Dockerfile` and `CLAUDE.md` describe the new defaults.
