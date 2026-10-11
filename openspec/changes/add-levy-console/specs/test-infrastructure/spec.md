## MODIFIED Requirements

### Requirement: Intentional exclusions only
Code excluded from the coverage denominator SHALL be limited to network-only provider internals (the Ollama LLM client, Ollama/sentence-transformers embedding client network paths), marked with explicit inline `# pragma: no cover` comments. Whole-file omission of modules containing offline-testable logic SHALL NOT be used.

#### Scenario: Exclusions are visible in the diff
- **WHEN** a reviewer inspects why a line is not counted
- **THEN** an inline pragma marks that exact line or block, rather than a configuration entry hiding an entire file

#### Scenario: Offline-testable logic stays counted
- **WHEN** coverage is computed
- **THEN** mock providers, configuration plumbing, and all engine/cache/dataset/experiment logic are included in the denominator
