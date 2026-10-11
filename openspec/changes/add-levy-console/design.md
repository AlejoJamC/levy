## Context

The delivered API (`levy/api/app.py`, LEV-7) exposes `POST /v1/chat/completions`, `GET /admin/cache/stats` and `POST /admin/cache/clear`. A chat request resolves through an `EnginePool` engine and returns an Anthropic-shaped body with `X-Cache-Status` / `X-Cache-Similarity`. FastAPI serves Swagger at `/docs`.

The delivered `cache_config` is a JSON body carrying only `threshold` and `embedding_model`, and the pool is keyed on that pair. Everything else comes from the process's `LevyConfig`: LLM provider and model, embedding provider, cache switches, index backend and store. `LevyConfig.llm_provider` defaults to `mock`, a client that sleeps 0.5 s and echoes the prompt reversed.

Most of what a viewer needs to understand a request is computed somewhere, but none of it reaches a screen:

| Information | Where it exists today | Gap |
|---|---|---|
| Lookup timings (exact, embed, index search, total) | `TimingCollector` (LEV-14), passed to `generate()` by the latency benchmark only | The API never passes one; the provider call and the store are not segments |
| Nearest neighbour and similarity | Computed in `SemanticCache.get()` | Discarded on a miss; on a hit only the score survives, in entry metadata |
| Tokens, stop reason, provider model | `LLMResponse.metadata` (Anthropic connector) | Not traced |
| Estimated spend | `_BudgetGuard`, one per `AnthropicLLMClient`, hence one per pooled engine | Not exposed |
| Which miss populated a later hit | Nowhere | — |

Constraints from the repo: synchronous core, dataclasses outside the API layer, offline tests, and ≥90 % branch coverage on `levy/` without pragmas.

## Goals / Non-Goals

**Goals:**
- A `console` section in the API, beside the delivered endpoints, where each request chooses its configuration from closed lists that Swagger shows as selectable fields.
- Show each console request's configuration, real path, timing, decision rationale, location, cost and savings, live, in a page Levy serves itself.
- Keep the delivered endpoints, schemas and caches as delivered.
- Remove the mock as the default LLM provider.

**Non-Goals:**
- Changing or tracing the delivered endpoints.
- Authentication, multi-user access or remote deployment of the console. It is a local tool, like the API itself.
- Persisting traces. They are memory-only and lost on restart.
- Distributed-tracing standards (OpenTelemetry export). The trace shape is Levy's own.

## Decisions

### D1. Observation hooks in the engine, everything else in `levy/console/`

The engine gains two opt-in surfaces, following the LEV-14 pattern (`timing=None`):

- **`observation: Optional[SemanticLookupObservation] = None`** on `LevyEngine.generate()`, forwarded to `SemanticCache.get()`. When present, `get()` writes `reached`, `index_size`, `threshold`, `nearest_prompt`, `distance`, `similarity` and `hit` before returning. The entry is resolved from `_entries` for the nearest id whether or not it clears the threshold. The return path is otherwise untouched.
- **Two segment constants, `SEGMENT_LLM_CALL` and `SEGMENT_STORE`,** in `levy/latency/timing.py`, recorded through `segment(timing, …)` around the provider call and the cache write on a miss. They are deliberately left out of `SEGMENTS`, and the latency benchmark only reads named lookup segments, so its CSV columns cannot change.

These are the only edits to delivered core code, and with neither argument passed the behaviour is identical. Everything else lives in `levy/console/`: option lists, engine pool, recorder, routes, page.

*Alternatives considered:*
- **Peek at the index from the console before or after `generate()`.** Before: the peek embeds the prompt, so the engine's own embed becomes a memo hit and the waterfall would show a warm ~0.003 ms embed instead of the real cold one. After: on a miss the index already contains the prompt itself. Both falsify what is shown.
- **Parse the structured log.** It carries no segments and no neighbour, and cannot show in-flight state.

### D2. The console section and its settings as query parameters

`levy/console/routes.py` defines an `APIRouter(prefix="/console", tags=["console"])`, included by `create_app()`. Swagger therefore shows it as its own section, and nothing in the delivered routes changes.

`POST /console/chat` takes `prompt` and the nine settings as **query parameters**:
- Each list-type setting is typed by a `Literal` built from the option lists (D3). FastAPI publishes it as an `enum`, and Swagger UI renders an enum parameter as a selectable field with the default pre-selected.
- Booleans render as a true/false selector.

*Why query parameters, not a JSON body:* Swagger UI renders a request body as an editable JSON text area, with no option lists, which is exactly the problem with the delivered `cache_config`. Form fields would also render as selectors, but they need `python-multipart`, which the project does not declare. Query parameters need nothing.

*Cross-field rule:* Swagger cannot narrow one selector by another, so each model selector lists every configured model. A dependency validates that the chosen model belongs to the chosen provider and returns a 422 naming the parameter and the models valid for that provider.

The response reuses the delivered response model and the same two headers, so the console's answer looks exactly like the delivered one.

### D3. Option lists: what Levy implements in code, which models exist from `.env`

`levy/console/options.py` keeps two kinds of list apart:
- **What Levy implements**, in code: the provider types, the vector index backends, the cache stores, and the study's two embedding models (`all-MiniLM-L6-v2`, `modernbert`, from the `EmbeddingManager` registry).
- **Which models exist and what they cost**, from the `LevyConfig` the app was built with, i.e. from `.env` (D14):
  - Anthropic models with their per-MTok prices;
  - Ollama LLM models;
  - Ollama embedding models.

`ConsoleOptions.from_config()` builds the lists from the configuration current at the time of the request. Without an explicit configuration (the module-level app), that is `.env` re-read (D14). The parameters are plain optional strings, validated against those lists on each request, and the OpenAPI schema is rewritten with the current lists and defaults each time it is served (`publish_choices`). An enum is written even for a single option, which a one-value `Literal` would publish as `const` and Swagger would render as a text box. A provider with no configured model is not offered. If no LLM model is configured at all, `POST /console/chat` answers 503 naming the variables. No list contains `mock`.

The console's default configuration is the base `LevyConfig`, with `vector_index_backend` resolved to the backend actually in use and every value the lists do not offer replaced by the first configured option.

### D4. The console engine pool

`ConsolePool` (in `levy/console/`) keys engines on a frozen dataclass of the nine settings. Each engine is built with `LevyEngine(dataclasses.replace(base_config, …), embedding_manager=…)`. Embedding managers are shared per `(embedding_provider, embedding_model)`, so a model is never loaded twice. The pool has its own cap and a structured error, mirroring the delivered one, and is a separate instance from the delivered `EnginePool`, so the two never share caches.

**Redis.** The delivered `RedisStore` keys entries by `sha256(prompt)` in one shared database and has no `entries` attribute. To keep it untouched, the console pool replaces a redis engine's store after construction with a console-owned `PrefixedRedisStore`. It is a small wrapper that:
- prefixes every key with the configuration's id;
- counts its keys for stats;
- clears only its own prefix.

The engine's `ExactCache` is rebuilt over that store. Console configurations then never collide with each other, nor with the delivered endpoints.

### D5. In-flight state through a collector subclass

`LiveTimingCollector(TimingCollector)` overrides `measure()`:
1. Before yielding, it tells the recorder that segment *X* started at offset *t* from arrival.
2. After the segment, it records the duration as usual and publishes the completed span.

This gives in-flight stage reporting and per-span start offsets, which the waterfall needs because `TimingCollector` stores durations only. It needs no hook beyond D1.

### D6. Polling with a version cursor, not SSE or WebSockets

The page polls `GET /console/traces?after=<version>` about every 250 ms. The recorder holds a global version that every trace create/update advances, and each trace carries the version of its last change.

*Why:* the stack is synchronous and the endpoints are `def` routes in FastAPI's threadpool. A sync SSE route would hold a threadpool worker for as long as the page is open. An async generator would bring asyncio into a codebase that has none. A polled endpoint is a plain sync route, testable with `TestClient`. The cost is up to ~250 ms of display lag, invisible next to provider calls that take seconds.

### D7. Trace shape

One JSON object per console request:

```
id, seq, version, status (in_flight | done), outcome (exact_hit | semantic_hit | miss | error),
arrived_at (UTC ISO), total_ms, current_stage,
config {threshold, embedding_provider, embedding_model, llm_provider, llm_model,
        exact_cache, semantic_cache, vector_index_backend, cache_store},
config_from_request [setting names the request set], skipped [exact_cache | semantic_cache],
prompt, answer,
spans [{name, start_ms, duration_ms}],
semantic {reached, index_size, threshold, nearest_prompt, distance, similarity, cosine, hit} | null,
embedding {provider, canonical_name, checkpoint, dimension} | null,
llm {provider, endpoint, location, model, input_tokens, output_tokens, stop_reason, cost_usd} | null,
savings {origin_id, llm_ms, input_tokens, output_tokens, cost_usd} | null,
cache_after {exact_entries, index_size},
error {kind, detail, stage} | null
```

`cosine` is computed server-side as `1 − d²/2`, valid because every vector is L2-normalised (known-gap #3). The page therefore never re-implements the similarity scale.

### D8. Provider location and cost

**Location** is derived from the request's configuration:

| Provider | Location | Endpoint |
|---|---|---|
| `ollama` | `local` when the host of `ollama_base_url` is a loopback address, else `remote` | that URL |
| `anthropic` | `remote` | the SDK client's `base_url` |

**Call cost** uses `_BudgetGuard.record`'s formula with the prices configured for the request's model (D3). A model with no configured prices reports `null`, shown as "no price configured", never as zero. **Cumulative spend** is the sum of `estimated_cost_usd` over the console engines' clients.

### D9. Origin attribution

On a completed miss, the recorder stores `origins[(config, sha256(prompt))] = {trace id, llm_ms, tokens, cost}`. On a hit, the matched prompt is looked up:
- exact hit: the request's own prompt;
- semantic hit: `observation.nearest_prompt`.

`POST /console/cache/clear` also clears the origins. The traces themselves are kept, because history is not cache.

### D10. Page architecture

- **Assets:** `levy/console/static/` holds `index.html` plus its CSS and JS. Vanilla JS and inline SVG, no framework, no bundler, no CDN. They ship as package data and are served by `GET /console` and a static mount at `/console/assets/`, both excluded from the OpenAPI schema.
- **Map:** a fixed SVG topology, whose node labels, skipped caches and LLM placement come from the selected trace's configuration.
- **Animation:** a request animates hop by hop along SVG paths. Replay is time-compressed, with real milliseconds printed. A live in-flight request parks at the node named by `current_stage` and continues when its trace completes.
- **Waterfall:** a linear scale from 0 to `total_ms`, with a 1-pixel minimum mark for sub-pixel spans and the exact value printed.
- **Visual design:** produced in Claude Design and ported to this static form. A framework or remote font in the design is reimplemented in vanilla JS, with the font vendored (with its licence file) or replaced by the system stack.

### D11. The mock default

`LevyConfig.llm_provider` changes from `mock` to `anthropic`. Tests that relied on the old default state their provider. Nothing else in the delivered endpoints changes.

### D12. Verification

- **Python side** (options, cross-field validation, console pool, prefixed redis store, recorder, console endpoints, engine hooks): offline unit tests with test doubles, plus `httpx.MockTransport` for Anthropic.
- **Delivered contract:** a test compares the delivered endpoints' OpenAPI entries before and after.
- **In-flight:** a test injects a blocking LLM client and reads the feed mid-call.
- **Observation-only:** a test serves one prompt sequence through two engines, with and without observation and timing, and compares every result.
- **Offline page:** a test scans the static assets for any `http(s)://` reference in `src`, `href`, `url()` or `fetch`.
- **Browser:** the page is verified in a browser against Anthropic and against Ollama on localhost, choosing settings per request in Swagger's console section.

### D13. The OpenAI provider is removed (decided 2026-10-11)

Reviewing where the console's lists came from showed that Levy carried an OpenAI provider that had never been part of the work:

- **Never used.** The study design names Anthropic as the only remote provider, and the budget was approved for Anthropic. No run of record used OpenAI: the published runs used the mock (D3 grid) and Anthropic (latency, throughput).
- **Never worked as configured.**
  - `OpenAILLMClient` was excluded from coverage and exercised by no test.
  - It defaulted to `gpt-3.5-turbo`, a model from the initial scaffold commit.
  - It sent `max_tokens` and a fixed `temperature`, which current OpenAI reasoning models reject.
  - The engine built it with `config.model_name`, Ollama's model, so a configured OpenAI engine would have requested an Ollama model from OpenAI.
- **Harmful to keep.** Listed beside Anthropic and Ollama, it presented dead code as a valid option, here and in the console's lists.

Removed:
- `OpenAILLMClient`;
- the engine's `openai` branch;
- `openai_api_key` / `openai_base_url` in `LevyConfig`;
- `OPENAI_API_KEY` in `.env.example`;
- the three tests that only checked its construction;
- every documentation and current-spec mention of it.

Kept:
- `scripts/audit_release.sh`'s `sk-…` patterns, because they detect leaked keys of that format whatever their origin;
- the archived OpenSpec changes, which are history.

Anthropic and Ollama are untouched.

The engine also stops treating an unrecognised `llm_provider` as the mock. Its last branch built a `MockLLMClient` for any value it did not recognise, so after the removal `openai` would have silently become the mock. It now raises, naming the accepted providers.

### D14. No model or price is written in the code (decided 2026-10-11)

Review showed model ids and prices written as code defaults: `claude-haiku-4-5-20251001` at $1 / $5 and `qwen3` in `LevyConfig`, `claude-opus-4-8` at $5 / $25 in `AnthropicLLMClient`, `qwen3` in `OllamaLLMClient`, and `nomic-embed-text` in `OllamaEmbeddingClient`. The console copied them into its lists. Models change every few months, so a default written in code goes stale and silently prices or calls a model nobody chose. Two of these defaults already named models that are not served on the author's machine.

Decision:
- **Models and prices come from `.env`:**
  - One fact per line, one model per number `<n>`:
    - `LEVY_ANTHROPIC_MODEL_<n>` with `LEVY_ANTHROPIC_MODEL_<n>_INPUT_USD_PER_MTOK` and `_OUTPUT_USD_PER_MTOK` (the LEV-14 triple);
    - `LEVY_OLLAMA_MODEL_<n>`;
    - `LEVY_OLLAMA_EMBEDDING_MODEL_<n>`.
  - The lowest number is the default. Numbers need not be consecutive, and adding a model means adding a new number; nothing is replaced.
  - **Prices:** the currency and unit are in the variable name (US dollars per million tokens sent, and per million written). The value is only the number, with a dot for decimals. `$1`, `1usd`, `3eur` or `1,25` are rejected with an error naming the variable and showing `1.25`. A model without both prices, or a price without its model, is rejected the same way.
  - **Why this shape and not a list or `model:price:price`:** model ids contain `-`, `.` and `:`, which cannot be part of a variable name or split safely, so a number is the key. One line per fact also keeps every `.env` diff exact.
  - **Read again on use:** `levy.config.current_model_catalog()` re-reads `.env` on every console request and every Swagger load, so a model added there appears after refreshing Swagger, without a restart. A variable set in the process environment wins over `.env`, as usual.
  - `.env.example` documents the format with placeholders only.
- **No defaults in the clients.** `AnthropicLLMClient`, `OllamaLLMClient` and `OllamaEmbeddingClient` have no default model or price. Constructing one without them fails, naming the variable to set.
- **What stays in code:** what Levy implements — the provider types, the index backends, the cache stores, and the study's two embedding models, which belong to the experiment design.
- **Scripts:** `scripts/populate_responses.py` and `scripts/run_throughput.py` take their model from those variables or from their flags, and stop with a clear message when neither is set. Their docstrings keep the model ids that produced the published results, as provenance.
- **Published results are unaffected.**
  - The D3 harness builds its engines with the mock LLM and the study's embedding models.
  - The latency benchmark replays recorded responses through an injected client.
  - Nothing writes to `release/`.

### D15. Real embeddings by default, mock only in tests (decided 2026-10-11)

Levy is a semantic cache: mock embeddings carry no meaning, so a run with them is not Levy. They were the default of `scripts/reproduce.sh`, the container, `run_experiments.py` and `run_latency.py`, and a choice in the dashboard. The published results all used `sentence-transformers`.

- Default everywhere a user runs Levy: `sentence-transformers`, the study's two models. Their weights download once on first use.
- Mock embeddings stay only as a test double for the offline suite, passed explicitly by the tests. They are not listed by any CLI, documented, or offered by the dashboard.
- The mock LLM stays where the answer is irrelevant (the harness replay decides hit or miss, not the answer).

## Risks / Trade-offs

- **Prompts and answers are served unauthenticated at `/console/traces`.** → Same exposure as the existing per-request log and the unauthenticated API. Memory-only, bounded, never written to disk, local tool.
- **Page polling adds roughly four light requests per second, visible in the uvicorn access log.** → Accepted. The polled route does no engine work.
- **The first request with a model includes that model's lazy load inside its `embed` segment (seconds, for sentence-transformers).** → Shown as measured. It is what Levy really does on a cold start.
- **Swagger lists every model in each model selector.** → Invalid pairs are rejected with the valid models named (D2).
- **Long prompts in a query string.** → Fine for demonstration-sized prompts. The delivered JSON endpoint remains for anything else.
- **JavaScript is not covered by pytest.** → The logic the page needs (similarity, cosine, cost, attribution, location) is computed server-side and tested there. The page only renders, and is verified in the browser.

## Migration Plan

- The delivered endpoints and schemas do not change; the console section is additive.
- The one default that changes is `LevyConfig.llm_provider`, from `mock` to `anthropic`. A caller that relied on the mock default now names it in its own `LevyConfig`.

## Open Questions

None.
