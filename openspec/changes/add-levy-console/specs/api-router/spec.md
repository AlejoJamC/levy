## MODIFIED Requirements

### Requirement: Provider-agnostic serving with clear failure mapping
The router SHALL serve whichever `llm_provider` is configured (Anthropic, Ollama, or the mock test double) unchanged, and SHALL map provider failures to structured JSON errors — in particular, a budget-guard halt (LEV-6) returns a payment-required-style error naming the cap rather than a generic server error. There is no OpenAI provider (removed 2026-10-11, design D13 of `add-levy-console`).

#### Scenario: Budget cap surfaces cleanly
- **WHEN** the Anthropic budget guard refuses a call because the cap is reached
- **THEN** the endpoint returns a structured error naming the cap and estimated spend, not an unhandled 500

## ADDED Requirements

### Requirement: No mock provider by default
`LevyConfig.llm_provider` SHALL default to `anthropic`. The mock LLM client remains available only to code that constructs its own configuration with it, such as the offline test suite. The delivered endpoints, their request and response schemas, and their headers are otherwise unchanged.

#### Scenario: Default provider
- **WHEN** the application is built with the default configuration and a prompt misses both caches on `POST /v1/chat/completions`
- **THEN** the miss is sent to Anthropic

#### Scenario: Delivered contract unchanged
- **WHEN** the OpenAPI schema of the delivered endpoints is compared before and after this change
- **THEN** `POST /v1/chat/completions`, `GET /admin/cache/stats` and `POST /admin/cache/clear` have the same parameters, request bodies and responses
