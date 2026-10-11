## ADDED Requirements

### Requirement: Opt-in semantic lookup observation
`SemanticCache.get()` and `LevyEngine.generate()` SHALL accept an optional observation object. When one is passed, the semantic lookup SHALL record into it:
- whether the semantic cache was reached;
- the index size at lookup time and the threshold applied;
- the nearest cached prompt, its L2 distance and its similarity on the `1/(1+L2)` scale;
- whether it was a hit.

The nearest neighbour SHALL be recorded on a miss as well as on a hit. When no observation is passed, the lookup's return value, threshold decision and side effects SHALL be identical to the behaviour without this argument.

#### Scenario: Miss still reports its nearest neighbour
- **WHEN** a prompt is looked up against a non-empty index and its nearest neighbour falls below the threshold, with an observation passed
- **THEN** the observation records the nearest cached prompt, its distance and similarity, the threshold and `hit = false`
- **AND** the lookup returns no entry

#### Scenario: Exact hit never reaches the semantic cache
- **WHEN** a prompt is served by the exact cache with an observation passed
- **THEN** the observation records that the semantic cache was not reached

#### Scenario: Empty index
- **WHEN** a prompt is looked up against an empty index with an observation passed
- **THEN** the observation records that the semantic cache was reached, an index size of zero and no nearest neighbour

#### Scenario: Decisions are unchanged by observation
- **WHEN** the same prompt sequence is served by two identically configured engines, one with an observation per request and one without
- **THEN** every request resolves to the same source with the same similarity score and the same answer on both engines

### Requirement: Provider call and store are timed when timing is requested
When `LevyEngine.generate()` is given a timing collector and the request misses both caches, the engine SHALL record the provider call and the subsequent cache write as two segments (`llm_call`, `store`). These are separate from the lookup-path segments. The lookup decomposition SHALL stay as defined by the latency-measurement capability: the total lookup segment still ends where the lookup does, and the latency benchmark's reported columns are unchanged.

#### Scenario: A miss records the call and the store
- **WHEN** a request misses both caches with a timing collector passed
- **THEN** the collector holds one non-negative `llm_call` sample and one non-negative `store` sample
- **AND** the lookup-path segments still sum to no more than the total lookup segment

#### Scenario: A hit records neither
- **WHEN** a request is served from either cache with a timing collector passed
- **THEN** no `llm_call` and no `store` sample is recorded

### Requirement: Every console chat request produces a trace
The console SHALL record one trace per `POST /console/chat` request that reaches its handler. Each trace carries:
- request id and arrival timestamp;
- the resolved configuration (every console setting), marking which settings the request set and which came from the defaults;
- the prompt and the answer;
- outcome: `exact_hit`, `semantic_hit`, `miss` or `error`;
- the caches the configuration switched off, recorded as skipped;
- end-to-end server-side latency;
- every recorded segment with its start offset from arrival and its duration;
- the semantic lookup observation;
- the embedding identity (provider, canonical name, checkpoint, dimension);
- for a provider call: the provider, its endpoint, its location (`local` or `remote`), the model it reported, input and output tokens, stop reason, and the estimated cost when prices are configured for that model;
- the cache sizes after the request.

The delivered `POST /v1/chat/completions` is not traced and its behaviour is unchanged.

#### Scenario: Miss trace is complete
- **WHEN** a console prompt misses both caches
- **THEN** its trace has outcome `miss`, timed exact-lookup, embed (when the index is non-empty), index-search, `llm_call` and `store` segments, the provider identity and location, and token counts

#### Scenario: Semantic hit trace explains the decision
- **WHEN** a console prompt is served by the semantic cache
- **THEN** its trace has outcome `semantic_hit`, and its observation names the matched cached prompt with a similarity equal to the `X-Cache-Similarity` header returned to the client

#### Scenario: Provider location reflects where the LLM runs
- **WHEN** traces are recorded for requests sent to Ollama on localhost and to Anthropic
- **THEN** their provider location is `local` and `remote` respectively

#### Scenario: A switched-off cache is recorded as skipped
- **WHEN** a request runs with the exact or the semantic cache switched off
- **THEN** its trace records that cache as skipped and carries no segment for it

### Requirement: Hits are attributed to the request that populated them
When a miss stores an entry, the trace recorder SHALL remember that request as the entry's origin, keyed by configuration and prompt. A later hit SHALL carry the origin's trace id together with the provider latency, tokens and estimated cost the hit avoided. For an exact hit the matched prompt is the request's own; for a semantic hit it is the observed nearest prompt. Clearing the console's caches SHALL also clear the origins.

#### Scenario: Exact hit names its origin
- **WHEN** a prompt misses and the same prompt is then requested again under the same configuration
- **THEN** the second trace's savings reference the first trace's id and its provider-call latency and tokens

#### Scenario: Origins do not survive a cache clear
- **WHEN** the console's caches are cleared and a prompt that was previously cached misses again
- **THEN** no trace after the clear references an origin recorded before it

### Requirement: Failed requests are traced
A console chat request that fails after reaching its handler SHALL produce a trace with outcome `error`, the error kind matching the structured error the client receives (`pool_cap_exceeded`, `budget_exceeded`, `provider_refusal`, `provider_error`), and the stage at which it failed.

#### Scenario: Budget guard halt is traced
- **WHEN** the Anthropic budget guard refuses a console call
- **THEN** the client receives the structured budget error
- **AND** a trace with outcome `error`, kind `budget_exceeded` and failing stage `llm_call` is recorded

#### Scenario: Pool cap breach is traced
- **WHEN** a console request's configuration would exceed the console engine pool cap
- **THEN** a trace with outcome `error` and kind `pool_cap_exceeded` is recorded

### Requirement: In-flight requests are visible
A trace SHALL be readable from the moment the request reaches the handler. While the request runs, the trace SHALL report the segment currently executing. A request waiting on its provider call SHALL be observable as in flight at `llm_call` before it completes.

#### Scenario: Pending provider call is observable
- **WHEN** the trace feed is read while a request's provider call has started and not returned
- **THEN** that request's trace has status in flight and current stage `llm_call`

### Requirement: Bounded trace buffer with a version cursor
Traces SHALL be held in memory only, in a buffer bounded to a configured maximum. The oldest traces are evicted first. Every creation or update of a trace SHALL advance a monotonically increasing version. `GET /console/traces?after=<version>` SHALL return:
- every trace created or updated after that version, in arrival order;
- the current version;
- a buffer epoch;
- the console's default configuration;
- the console's aggregate cache statistics and cumulative estimated spend.

`POST /console/traces/clear` SHALL empty the buffer and start a new epoch. Traces SHALL never be written to disk.

#### Scenario: Cursor returns only what changed
- **WHEN** a client reads the feed, then one more request is served, then the client reads again with the version it received
- **THEN** the second read returns only that request's trace

#### Scenario: Buffer is bounded
- **WHEN** more requests are served than the configured maximum
- **THEN** the feed returns at most the maximum, and the oldest are the ones dropped

#### Scenario: Clear starts a new epoch
- **WHEN** the history is cleared
- **THEN** the next read returns no traces and a different epoch than before

### Requirement: Tracing changes no decision
Tracing SHALL be observation only: cache decisions, `LevyMetrics` counters, response bodies, cache-status headers and the structured request log record are unchanged by it.

#### Scenario: Existing suites pass unchanged
- **WHEN** the full test suite runs with the console in place
- **THEN** every pre-existing test passes; only tests that relied on the old `mock` default change, to name their provider

### Requirement: Offline test coverage for tracing
Trace capture, the trace feed, error traces, in-flight visibility, attribution, the buffer bound and the observation-only guarantees SHALL be tested in-process with test doubles and no network. Anthropic scenarios use the existing `httpx.MockTransport` injection. The ≥90 % branch-coverage gate stays green with no new coverage pragmas.

#### Scenario: Suite runs offline
- **WHEN** the gated test command runs with no network access
- **THEN** all tracing tests pass and the coverage gate passes
