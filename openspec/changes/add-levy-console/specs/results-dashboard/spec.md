## MODIFIED Requirements

### Requirement: Local, offline-capable operation
The dashboard SHALL run locally inside the project environment with no deployment step. Its query demo SHALL embed with the study's `sentence-transformers` models, documented as downloading their weights on first use; it SHALL NOT offer mock embeddings.

#### Scenario: Query demo with the study's models
- **WHEN** the dashboard runs with the models' weights available locally
- **THEN** curve exploration and the query demo both work, with no network access
