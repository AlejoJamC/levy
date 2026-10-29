## MODIFIED Requirements

### Requirement: Literal reproduction guide
The repository SHALL contain a step-by-step reproduction guide covering environment setup, dataset build, the 30-configuration harness run, the analysis bundle, and the replication check, written as commands that can be executed verbatim, defaulting to the committed fixture dataset and the study's embedding models (`sentence-transformers`), so it requires no API key. The guide SHALL state that the first run downloads the two models' weights. Mock embeddings SHALL NOT be a default or a documented option.

#### Scenario: Guide followed in a clean environment
- **WHEN** a third party follows the guide literally from a fresh checkout
- **THEN** the pipeline completes and the analysis bundle (tables and figures) is produced

#### Scenario: Commands match the shipped CLIs
- **WHEN** the guide's commands are compared against the scripts' actual arguments
- **THEN** every flag shown exists and is spelled as the script defines it

### Requirement: One-command containerised pipeline
The repository SHALL provide a container definition and a compose service such that a single documented command executes the full evaluation pipeline end-to-end with the study's embedding models, requiring no API key; the first run downloads the models' weights, which the image does not contain. The container environment SHALL be built from the repository's single dependency specification. The Anthropic backend SHALL be opt-in via environment variables, with its billing implications documented.

#### Scenario: Single command
- **WHEN** the documented container command is run with no API key
- **THEN** the pipeline completes and writes the analysis bundle

#### Scenario: Environment parity
- **WHEN** the container image is built
- **THEN** its environment derives from the repository's declared dependency specification, not a separately maintained list

#### Scenario: Existing services preserved
- **WHEN** the compose file is extended
- **THEN** the previously documented Redis service still starts and behaves as before, and the default pipeline path does not require it
