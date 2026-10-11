## ADDED Requirements

### Requirement: Console served by the Levy process
The running Levy API SHALL serve the console at `GET /console` as a self-contained page. The page SHALL need no build step and no external tool. It SHALL load no resource from outside the Levy process: no CDN, no remote font, no remote script. It SHALL fetch data only from Levy's own endpoints. The route SHALL be excluded from the OpenAPI schema.

#### Scenario: Console opens from the running API
- **WHEN** a browser opens `/console` on a running Levy API
- **THEN** the console page is returned and renders without any request to a host other than the Levy process

### Requirement: Console section of the API
The API SHALL expose a `console` section, grouped under its own tag in the OpenAPI schema and shown as its own section in Swagger:
- `POST /console/chat`;
- `GET /console/traces` and `POST /console/traces/clear`;
- `POST /console/cache/clear`.

The delivered endpoints (`POST /v1/chat/completions`, `GET /admin/cache/stats`, `POST /admin/cache/clear`) SHALL NOT be modified by this section.

#### Scenario: Swagger shows the console as its own section
- **WHEN** `/docs` is opened
- **THEN** the console endpoints appear under a `console` section, and the delivered endpoints appear unchanged in theirs

### Requirement: Console chat with settings from closed lists
`POST /console/chat` SHALL take the prompt and each setting as a query parameter. Every list-type setting SHALL be typed by its closed list, so that Swagger renders it as a selectable field with its options:

| Parameter | Options |
|---|---|
| `threshold` | number in [0, 1] |
| `embedding_provider` | `sentence-transformers`, `ollama` |
| `embedding_model` | the registered embedding models and the configured Ollama embedding models |
| `llm_provider` | `anthropic`, `ollama` |
| `llm_model` | the configured models of every LLM provider |
| `exact_cache` | `true`, `false` |
| `semantic_cache` | `true`, `false` |
| `vector_index_backend` | `faiss`, `brute_force` |
| `cache_store` | `memory`, `redis` |

Each parameter SHALL default to the console's default configuration, shown as the pre-selected value. No list SHALL contain a mock provider. A model that does not belong to the chosen provider SHALL be rejected with a validation error naming the parameter and the models valid for that provider. The response SHALL have the same body shape and `X-Cache-Status` / `X-Cache-Similarity` headers as the delivered chat endpoint.

The model lists SHALL come from the configuration (`LEVY_ANTHROPIC_MODELS`, `LEVY_OLLAMA_MODELS`, `LEVY_OLLAMA_EMBEDDING_MODELS`), never from model ids written in the code. A provider with no configured model SHALL NOT be offered. When no LLM model is configured at all, the endpoint SHALL answer 503 naming those variables.

#### Scenario: No LLM model configured
- **WHEN** neither `LEVY_ANTHROPIC_MODELS` nor `LEVY_OLLAMA_MODELS` names a model and `POST /console/chat` is called
- **THEN** the endpoint answers 503 with a message naming both variables

#### Scenario: Settings are selectable in Swagger
- **WHEN** `POST /console/chat` is opened in Swagger
- **THEN** each list-type setting is a selectable field showing its options, with the default pre-selected

#### Scenario: Provider and model chosen per request
- **WHEN** two console requests choose different `llm_provider` and `llm_model` values
- **THEN** each miss is sent to the provider and model its request chose

#### Scenario: Model of another provider is rejected
- **WHEN** a request pairs `embedding_provider=ollama` with `embedding_model=modernbert`
- **THEN** the endpoint returns a validation error naming `embedding_model` and the models valid for `ollama`, and nothing reaches an engine

#### Scenario: Mock is not an option
- **WHEN** a request names `mock` as its LLM or embedding provider
- **THEN** the endpoint returns a validation error listing the accepted providers

### Requirement: Console engines are separate from the delivered ones
The console SHALL serve its requests from its own engine pool. The pool holds one engine per resolved configuration (all the settings above), reuses an engine across requests with the same configuration so its caches accumulate, and enforces a configured cap with a structured client error. Embedding models SHALL be loaded once per `(embedding_provider, embedding_model)` and shared across configurations. When `cache_store` is `redis`, each configuration's exact-cache keys SHALL be kept apart from every other configuration's and from the delivered endpoints' keys.

#### Scenario: Delivered caches are untouched
- **WHEN** prompts are served through the console
- **THEN** the delivered endpoint's caches and `GET /admin/cache/stats` are unchanged by them

#### Scenario: Distinct configurations use distinct caches
- **WHEN** the same prompt is stored under one console configuration and requested under one differing in any setting
- **THEN** each is decided against its own cache

#### Scenario: Redis configurations do not collide
- **WHEN** two console configurations that both use `redis` store the same prompt
- **THEN** a hit under one is never served from the other's entry

### Requirement: Console cache clear
`POST /console/cache/clear` SHALL empty the exact and semantic caches of every console engine and the hit-to-origin attributions, report what was cleared, and leave the delivered endpoints' caches and the trace history untouched.

#### Scenario: Clear empties console caches only
- **WHEN** console and delivered caches both hold entries and the console cache clear is called
- **THEN** a previously-hitting console prompt misses on its next request, and the delivered caches still hold their entries

### Requirement: Live request map built from each request's configuration
The console SHALL show a request map of the real serving topology: client, Levy router, exact cache, semantic cache and the LLM provider. A visible boundary SHALL separate the Levy host from the internet. The map SHALL be drawn from the selected request's resolved configuration:
- the exact cache node shows its store (`memory` or `redis`), or is shown as switched off and skipped;
- the semantic cache node shows its embedding provider and model, vector index backend and threshold, or is shown as switched off and skipped;
- the LLM node shows its provider, model and endpoint, and is placed by its location: on the Levy host as a separate local service for Ollama, beyond the boundary for a remote provider.

Nodes SHALL also show their live entry counts.

#### Scenario: Remote provider is shown beyond the boundary
- **WHEN** the selected request was sent to Anthropic
- **THEN** the LLM node appears beyond the host boundary, labelled with the Anthropic endpoint and the request's model

#### Scenario: Local provider is shown on the host
- **WHEN** the selected request was sent to Ollama on localhost
- **THEN** the LLM node appears on the Levy host, labelled with its local endpoint and model

#### Scenario: A switched-off cache is shown as skipped
- **WHEN** the selected request ran with the semantic cache switched off
- **THEN** the semantic node is shown as off, and the request's path bypasses it

### Requirement: Requests animate along the path they actually took
Each traced request SHALL be shown travelling along its actual path:
- an exact hit returns from the exact cache;
- a semantic hit returns from the semantic cache;
- a miss travels to the LLM node and returns, with the caches marked as written on store;
- an error stops at the node where it failed and is marked as failed.

A request in flight at its provider call SHALL be shown waiting at the LLM node until it completes.

#### Scenario: A request fired from Swagger appears without a refresh
- **WHEN** `POST /console/chat` is executed from `/docs` while the console page is open
- **THEN** the console shows it on the map and in the history without a manual reload

#### Scenario: A pending provider call is shown waiting
- **WHEN** a request is in flight at its provider call
- **THEN** the map shows it waiting at the LLM node, and it returns along the response path when the call completes

### Requirement: Request history
The console SHALL list every traced request in the current buffer epoch, newest first. Each row shows the outcome as a colour coding, the start of the prompt, its LLM provider, and the latency. Selecting a request SHALL replay its path on the map and open its trace detail. New requests SHALL be selected automatically while the console follows live traffic, and SHALL NOT replace a selection the viewer made.

#### Scenario: Selecting a past request replays it
- **WHEN** a request earlier in the history is selected
- **THEN** the map replays that request's path and the detail panel shows that request's trace

### Requirement: Trace detail
For the selected request, the console SHALL show:
- **Configuration:** every resolved console setting, marking which ones the request set and which came from the defaults.
- **Timing waterfall:** each recorded segment positioned by start offset and sized by duration on one linear scale, with exact values printed, and the lookup overhead expressed as a share of the provider call it avoided or made.
- **Semantic decision:** the nearest cached prompt (on a miss too), the L2 distance, the similarity, the equivalent cosine `1 − d²/2`, the threshold, and the resulting decision.
- **Cost:** tokens, estimated call cost, cumulative spend against the budget cap and, for a hit, the latency, tokens and cost avoided with a link to the originating request.
- **Identity:** request id, arrival time, embedding identity, provider endpoint and model.
- **Content:** the prompt and the answer.

#### Scenario: A false positive shows what it was confused with
- **WHEN** a semantic hit is selected
- **THEN** the detail shows the cached prompt it matched, the similarity and the threshold it cleared

#### Scenario: A hit links to its origin
- **WHEN** a hit with an attributed origin is selected
- **THEN** the detail shows the avoided latency, tokens and cost, and selecting the origin opens that request

#### Scenario: Configuration is visible per request
- **WHEN** two requests with different settings are selected in turn
- **THEN** the detail shows each one's own resolved configuration

### Requirement: Session summary
The console SHALL show a persistent summary of:
- request count;
- exact hits, semantic hits and misses;
- hit rate;
- cumulative estimated spend and savings;
- the state of the live connection to the trace endpoint.

#### Scenario: Summary tracks the session
- **WHEN** requests produce hits and misses
- **THEN** the summary's counts and hit rate match the trace endpoint's aggregate statistics

### Requirement: Legible at half-screen width
The console SHALL remain fully usable at half the width of a laptop screen, side by side with Swagger. At that width no content is hidden behind horizontal scrolling. Text and figures SHALL stay legible in a screen recording.

#### Scenario: Side by side with Swagger
- **WHEN** the console occupies half of a 1440-pixel-wide screen
- **THEN** the map, history and trace detail are all visible and readable without horizontal scrolling

### Requirement: Visual design from Claude Design
The console's visual design SHALL be produced in Claude Design and implemented from it.

#### Scenario: Implementation follows the design
- **WHEN** the console is implemented
- **THEN** its layout, typography and colour system follow the design made in Claude Design
