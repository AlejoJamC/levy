# anthropic-connector

Capability: Anthropic SDK backend for the engine — a synchronous `LLMClient` implementation with configured exponential-backoff retry, per-call token accounting, and a hard budget guard, selectable via configuration and fully testable offline.

## ADDED Requirements

### Requirement: Anthropic provider behind the existing client abstraction
The system SHALL provide an `AnthropicLLMClient` implementing the existing `LLMClient` ABC using the official `anthropic` SDK, and the engine SHALL construct it when configuration selects `llm_provider = "anthropic"`. Mock, OpenAI, and Ollama providers SHALL keep working unchanged.

#### Scenario: End-to-end generation via configuration
- **WHEN** the engine is configured with `llm_provider = "anthropic"` and a prompt misses the cache
- **THEN** the response returned to the caller is produced by the Anthropic client through the same engine flow used by every other provider

#### Scenario: Other providers unaffected
- **WHEN** the engine is configured with `mock`, `openai`, or `ollama`
- **THEN** behavior is identical to before this change

### Requirement: Retry with exponential backoff on transient errors
The system SHALL retry transient failures (connection errors, HTTP 408/409/429 and 5xx) with exponential backoff, with the maximum retry count configurable, and SHALL propagate non-retryable API errors as typed exceptions without swallowing or masking them.

#### Scenario: Transient failure then success
- **WHEN** the API returns a retryable error (e.g. 429) followed by a successful response
- **THEN** the call succeeds transparently and returns the successful response

#### Scenario: Non-retryable error propagates
- **WHEN** the API returns a non-retryable error (e.g. 400 invalid request)
- **THEN** a typed exception reaches the caller identifying the failure, and no silent fallback response is fabricated

### Requirement: Token accounting from API usage data
The system SHALL read the API's usage report on every successful call, populate `LLMResponse.token_usage` with the total (input + output tokens), and carry the input/output split and the serving model in `LLMResponse.metadata`, so existing metrics consume real usage through the unchanged interface.

#### Scenario: Usage recorded on success
- **WHEN** a generation succeeds with reported input and output token counts
- **THEN** `token_usage` equals their sum and the metadata carries the individual counts and model identity

### Requirement: Budget guard with hard stop
The system SHALL accumulate the request count and an estimated cost (tokens × configurable per-MTok input/output prices) across calls, and SHALL refuse to send further Anthropic requests once the estimate reaches the configurable hard cap (default 200 USD per the frozen budget), raising a dedicated budget error instead.

#### Scenario: Cap halts spending
- **WHEN** accumulated estimated cost reaches the configured cap and another generation is requested
- **THEN** no API request is sent and a budget-exceeded error naming the cap and current estimate is raised

#### Scenario: Spend visibility
- **WHEN** calls have been made
- **THEN** the client exposes the request count and current estimated cost for inspection

### Requirement: Configuration and secret handling
The system SHALL read the API key from `ANTHROPIC_API_KEY` (loaded via `.env`, never committed; `.env.example` documents the variable), and SHALL make the model name, retry count, budget cap, and token prices configurable, defaulting the model to a current, non-retired model identifier.

#### Scenario: Key absent
- **WHEN** `llm_provider = "anthropic"` is selected without an API key configured
- **THEN** construction fails with an error naming the missing variable, rather than failing obscurely at first call

#### Scenario: Model configurable
- **WHEN** configuration specifies a different Anthropic model name
- **THEN** requests are sent with that model and the response metadata reports it

### Requirement: Refusal responses are surfaced, not cached
When the API returns a successful response whose stop reason is a refusal, the system SHALL raise a clear error identifying the refusal rather than returning (and thereby caching) empty or partial content.

#### Scenario: Refusal stop reason
- **WHEN** a response arrives with a refusal stop reason
- **THEN** the client raises an error naming the refusal and no cache entry is stored for that prompt

### Requirement: Fully offline test coverage via injectable transport
The client SHALL accept an injectable HTTP transport so the test suite exercises success, usage accounting, retry-then-success, non-retryable propagation, refusal handling, and the budget stop with no network access and no API key, keeping the connector inside the coverage denominator (no new coverage pragmas).

#### Scenario: Suite runs offline
- **WHEN** the full test suite runs with no network and no `ANTHROPIC_API_KEY`
- **THEN** all connector tests pass via mocked transport and the 90% branch-coverage gate still passes
