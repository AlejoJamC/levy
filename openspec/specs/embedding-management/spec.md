# embedding-management

Capability: loading, selecting, and switching embedding models at runtime; generating embeddings through a single entry point with per-(model, text) memoization; exposing model identity and dimension to downstream consumers.

## ADDED Requirements

### Requirement: Runtime study-model selection
The system SHALL load the embedding model named in configuration and SHALL allow switching between the two study models (`all-MiniLM-L6-v2` and ModernBERT) through configuration alone, with no code changes.

#### Scenario: Select the baseline model
- **WHEN** the manager is configured with model `all-MiniLM-L6-v2` and asked to embed a text
- **THEN** the embedding is produced by the `sentence-transformers/all-MiniLM-L6-v2` checkpoint and the result reports that model identity

#### Scenario: Select ModernBERT
- **WHEN** the manager is configured with model `modernbert` and asked to embed a text
- **THEN** the embedding is produced by the `nomic-ai/modernbert-embed-base` checkpoint and the result reports that model identity

#### Scenario: Switch models within one process
- **WHEN** the manager embeds a text with model A and is then asked to embed with model B
- **THEN** both calls succeed and each uses its own checkpoint, without reconstructing the manager

#### Scenario: Unknown model name
- **WHEN** the manager is configured with a model name not present in the registry
- **THEN** it raises an error that names the unknown model and lists the known model names, and it MUST NOT silently fall back to another model

### Requirement: Study-model alias registry
The system SHALL resolve canonical study aliases to concrete checkpoints in a single registry: `all-minilm` and `all-MiniLM-L6-v2` SHALL resolve to `sentence-transformers/all-MiniLM-L6-v2`; `modernbert` SHALL resolve to `nomic-ai/modernbert-embed-base`. The resolved checkpoint identifier SHALL be observable by callers (for experiment records).

#### Scenario: Alias resolution
- **WHEN** the manager is configured with the alias `modernbert`
- **THEN** it loads `nomic-ai/modernbert-embed-base` and exposes that resolved checkpoint id

### Requirement: Embedding memoization
The system SHALL cache embeddings in memory keyed by (resolved model, text) and SHALL return the cached vector for a repeated (model, text) pair without recomputing it. Memoized entries for different models SHALL be independent. The cache SHALL be clearable.

#### Scenario: Repeated text is not recomputed
- **WHEN** the same text is embedded twice with the same model
- **THEN** the underlying embedding client computes at most once and both calls return equal vectors

#### Scenario: Same text under two models
- **WHEN** the same text is embedded with model A and then with model B
- **THEN** two independent computations occur and two independent cache entries exist

#### Scenario: Cache clearing
- **WHEN** the memoization cache is cleared and a previously embedded text is embedded again
- **THEN** the embedding is recomputed

### Requirement: Model identity and dimension exposure
The system SHALL expose, for the configured model, its canonical name, its resolved checkpoint identifier, and its embedding dimension, so that downstream consumers (vector index, experiment harness) can label vectors and size indexes.

#### Scenario: Dimension query
- **WHEN** a consumer asks for the embedding dimension of the configured model
- **THEN** the manager returns the model's output dimension as a positive integer

### Requirement: Symmetric task-prefix handling
For models whose checkpoint requires task prefixes (ModernBERT via `nomic-ai/modernbert-embed-base`), the system SHALL apply the same prefix (`search_query: `) to every text it embeds — both texts being stored and texts being looked up — and callers SHALL NOT need to supply or see prefixes. Models without prefixes (all-MiniLM) SHALL receive the text unchanged.

#### Scenario: Prefix applied consistently
- **WHEN** ModernBERT embeds a text during cache store and the same text during cache lookup
- **THEN** both embeddings are computed from identically prefixed input and are equal

#### Scenario: Baseline model receives raw text
- **WHEN** all-MiniLM-L6-v2 embeds a text
- **THEN** the text is passed to the model without any prefix

### Requirement: Offline operation with mock provider
The system SHALL support a mock embedding provider through the same manager interface, requiring no network access or model downloads, so that the test suite and demos run fully offline.

#### Scenario: Mock provider end-to-end
- **WHEN** the manager is configured with the mock provider and used by the engine to store and look up cache entries
- **THEN** all operations succeed without network access and tests can assert memoization and model-identity behavior

### Requirement: Default configuration matches the study baseline
The default `LevyConfig` SHALL select the `sentence-transformers` provider with `all-MiniLM-L6-v2` as the embedding model, so that an unconfigured engine uses the study's baseline rather than a model outside the experimental grid.

#### Scenario: Unconfigured engine uses the baseline
- **WHEN** a `LevyConfig` is created without embedding settings
- **THEN** `embedding_provider` is `sentence-transformers` and `embedding_model` is `all-MiniLM-L6-v2`
