## MODIFIED Requirements

### Requirement: Anthropic provider behind the existing client abstraction
The system SHALL provide an `AnthropicLLMClient` implementing the existing `LLMClient` ABC using the official `anthropic` SDK, and the engine SHALL construct it when configuration selects `llm_provider = "anthropic"`. The Ollama provider and the mock test double SHALL keep working unchanged. Levy has no OpenAI provider: it was removed on 2026-10-11, having never been used by the study (design D13 of `add-levy-console`).

#### Scenario: End-to-end generation via configuration
- **WHEN** the engine is configured with `llm_provider = "anthropic"` and a prompt misses the cache
- **THEN** the response returned to the caller is produced by the Anthropic client through the same engine flow used by every other provider

#### Scenario: Other providers unaffected
- **WHEN** the engine is configured with `ollama` or with the `mock` test double
- **THEN** behavior is identical to before this change

#### Scenario: An unknown provider is rejected, never replaced by the mock
- **WHEN** the engine is configured with an `llm_provider` that is not `anthropic`, `ollama` or `mock` (for example `openai`)
- **THEN** construction fails with an error naming the accepted providers, instead of silently falling back to the mock client

### Requirement: Configuration and secret handling
The system SHALL read the API key from `ANTHROPIC_API_KEY` and each Anthropic model from `.env` (never committed; `.env.example` documents the variables with placeholders only), one fact per line: `LEVY_ANTHROPIC_MODEL_<n>` with `LEVY_ANTHROPIC_MODEL_<n>_INPUT_USD_PER_MTOK` and `LEVY_ANTHROPIC_MODEL_<n>_OUTPUT_USD_PER_MTOK`. The prices are US dollars per million tokens sent and written, given as plain numbers with a dot for decimals. The lowest `<n>` is the default. The retry count and budget cap SHALL be configurable. No Anthropic model identifier or price SHALL be written in the code: models change every few months (design D14 of `add-levy-console`).

#### Scenario: Key absent
- **WHEN** `llm_provider = "anthropic"` is selected without an API key configured
- **THEN** construction fails with an error naming the missing variable, rather than failing obscurely at first call

#### Scenario: Model absent
- **WHEN** an Anthropic client is constructed with no model or no prices configured
- **THEN** construction fails with an error naming the `LEVY_ANTHROPIC_MODEL_<n>` variables

#### Scenario: Model configurable
- **WHEN** `.env` names a different model with its two prices
- **THEN** requests are sent with that model, the response metadata reports it, and the budget guard prices calls at its prices

#### Scenario: Price that is not a plain number rejected
- **WHEN** a `_USD_PER_MTOK` value carries a currency sign, a currency code or a decimal comma (for example `$1`, `1usd`, `3eur`, `1,25`)
- **THEN** loading the configuration fails with an error naming the variable and showing the expected form `1.25`

#### Scenario: Model and prices go together
- **WHEN** a `LEVY_ANTHROPIC_MODEL_<n>` lacks one of its two prices, or a price has no `LEVY_ANTHROPIC_MODEL_<n>`
- **THEN** loading the configuration fails naming the missing variable
