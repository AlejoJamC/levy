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
