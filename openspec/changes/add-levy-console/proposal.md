## Why

Everything Levy does to a request happens out of sight. Swagger (`/docs`) can fire `POST /v1/chat/completions`, but it shows only the response body and two headers. The rest is computed and then thrown away:
- which path the request took, and where each step ran;
- how long each step cost;
- which cached prompt a semantic hit matched, or nearly matched on a miss;
- what a hit saved.

The delivered endpoint also lets a request choose only its threshold and embedding model, through a raw JSON body. A practitioner arriving at Levy cannot see which providers, models, caches, backends and stores exist, and cannot pick them. There is no way to watch the artefact working, live, under a configuration chosen per request. The 10-minute artefact video demonstration needs exactly that (LEV-23).

## What Changes

- **A new `console` section in the API, beside the delivered endpoints.** The delivered `POST /v1/chat/completions`, `GET /admin/cache/stats` and `POST /admin/cache/clear` are not modified. The console section is the dynamic test environment:
  - `POST /console/chat`: a prompt plus every setting as a parameter with its closed list of options, shown by Swagger as selectable fields:

    | Setting | Options |
    |---|---|
    | threshold | number |
    | embedding provider | `sentence-transformers`, `ollama` |
    | embedding model | registered models and the Ollama list |
    | LLM provider | `anthropic`, `ollama` |
    | LLM model | models per provider |
    | exact cache | `true`, `false` |
    | semantic cache | `true`, `false` |
    | vector index backend | `faiss`, `brute_force` |
    | cache store | `memory`, `redis` |

    No list contains the mock. The response has the same shape and headers as the delivered endpoint.
  - `GET /console/traces?after=<version>` and `POST /console/traces/clear`: the live trace feed and its reset.
  - `POST /console/cache/clear`: empties the console's caches.
  - `GET /console`: the observability page.

  The console runs its own engines, one per resolved configuration, so the delivered endpoints' caches are never touched.
- **Request tracing for console requests.** Every console chat request produces a trace:
  - when it arrived, and its resolved configuration;
  - the path it took (exact hit, semantic hit, miss, error), including caches switched off;
  - per-segment timings, including the provider call and the cache write after a miss;
  - the semantic lookup observation: the nearest cached prompt, distance, similarity and threshold, on a miss as well as a hit;
  - where each step ran (localhost or remote provider);
  - the resolved embedding and LLM identities, tokens and estimated cost;
  - for a hit, the latency, tokens and cost it avoided, attributed to the request that populated the entry.

  Traces live in a bounded in-memory buffer and are visible while a request is still in flight.
- **Opt-in observation hooks in the engine.** Showing the real lookup requires an optional `observation` argument on `LevyEngine.generate()`, forwarded to `SemanticCache.get()`. When a timing collector is passed, the provider call and the store are recorded as two new segments. With neither passed, behaviour is identical to today. These are the only changes to delivered core code.
- **Levy Console page**, served at `/console`:
  - a request map of the real topology with the machine/internet boundary, built from each request's configuration and animated along its actual path;
  - a clickable request history;
  - a trace detail with configuration, timing waterfall, semantic decision, cost and savings, and identity;
  - a session summary.

  The page works with no network access and no build step. Its visual design is produced in Claude Design.
- **No mock default.** `LevyConfig.llm_provider` defaults to `anthropic` instead of `mock`.
- **No model or price is written in the code (decided 2026-10-11, design D14).**
  - `.env` names each model on its own numbered lines: `LEVY_ANTHROPIC_MODEL_<n>` with `_INPUT_USD_PER_MTOK` and `_OUTPUT_USD_PER_MTOK` (plain numbers, dot decimals), `LEVY_OLLAMA_MODEL_<n>` and `LEVY_OLLAMA_EMBEDDING_MODEL_<n>`. The lowest number is the default, and adding a model means adding a new number.
  - The console re-reads `.env` on each request and Swagger load, so a new model appears without a restart.
  - The clients have no default model or price, and fail naming the variable when one is missing.
  - The console's model lists come from the same variables.
- **The OpenAI provider is removed from Levy (decided 2026-10-11).
  - Removed: `OpenAILLMClient`, the engine's `openai` branch, `LevyConfig.openai_api_key` / `openai_base_url`, `OPENAI_API_KEY` in `.env.example`, and the tests and docs that describe them.
  - Why: it was never part of the study. Anthropic is the only remote provider in the design and the only one with a budget, and no run of record ever used OpenAI. The client was dead code that would not have worked with a current model.
  - Anthropic and Ollama are untouched.

## Capabilities

### New Capabilities

- `request-tracing`: trace capture for console requests and its read-only feed. Covers:
  - configuration, path, segment timings, semantic observation, provider location and identity, tokens and cost;
  - hit-to-origin savings attribution, error traces and in-flight state;
  - the bounded buffer with its version cursor and clear;
  - the engine-level opt-in hooks, and the guarantee that they change no decision.
- `levy-console`: the console section of the API and its page. Covers:
  - `POST /console/chat` with per-request settings from closed lists, its own engines and the cache clear;
  - the live request map, history, trace detail and session summary;
  - offline operation, half-screen legibility, and the Claude Design source of its visual design.

### Modified Capabilities

- `api-router`: the LLM provider no longer defaults to the mock, and the router serves Anthropic and Ollama (OpenAI removed). The delivered endpoints are otherwise unchanged.
- `anthropic-connector`: the Anthropic client sits beside the Ollama client and the mock test double; the OpenAI provider no longer exists. The model and its prices come from the environment, never from a default in code.
- `test-infrastructure`: the coverage exclusions no longer name an OpenAI client.
- `release-packaging`: the reproduction pipeline and container default to the study's embedding models, not mock embeddings.
- `results-dashboard`: the query demo embeds with the study's models only.

## Impact

- **Code:**
  - New package `levy/console/`: option lists and prices, the console engine pool, trace recorder, routes and static page.
  - `levy/api/app.py`: includes the console router.
  - `levy/config.py`: `llm_provider` default `anthropic`; `openai_api_key` and `openai_base_url` removed.
  - `levy/llm_client.py` and `levy/engine.py`: `OpenAILLMClient` and the engine's `openai` branch removed.
  - `.env.example`: `OPENAI_API_KEY` removed; the numbered `LEVY_ANTHROPIC_MODEL_<n>` (with its two `_USD_PER_MTOK` prices), `LEVY_OLLAMA_MODEL_<n>` and `LEVY_OLLAMA_EMBEDDING_MODEL_<n>` documented with placeholders.
  - `levy/config.py`, `levy/llm_client.py`, `levy/embeddings.py`: model ids and prices read from the environment; no defaults in the clients.
  - `scripts/populate_responses.py`, `scripts/run_throughput.py`, `examples/ollama_demo.py`: take models from the environment or their flags.
  - `levy/engine.py` and `levy/cache/semantic_cache.py`: the optional `observation` argument, and the `llm_call`/`store` segments when timing is requested.
  - `levy/models.py`: a `SemanticLookupObservation` dataclass.
  - `levy/latency/timing.py`: two segment constants, kept out of `SEGMENTS`.
  - `pyproject.toml`: package data for the static page.
- **APIs:** a new `console` tag with four endpoints and the page route. Delivered endpoints, schemas, headers and the structured log record are unchanged.
- **Tests:**
  - New offline suites for the console endpoints, the option lists, tracing and the page route.
  - Tests that relied on the old `mock` default state their provider explicitly.
  - The ≥90 % branch-coverage gate stays green.
- **Dependencies:** none added. The settings are query parameters, so no form-parsing dependency is needed.
- **Docs:** `README.md`, `docs/ARCHITECTURE.md` and `CLAUDE.md` describe the console section and the new default.
