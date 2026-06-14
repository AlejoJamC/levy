# vector-store

Capability: approximate-nearest-neighbour retrieval for the semantic cache — Faiss HNSW index over L2 distance, the spec's L2→similarity transform and threshold decision, the internal-id → entry metadata mapping, per-configuration reset, and an exact-NN brute-force fallback used both offline and as the correctness oracle.

## ADDED Requirements

### Requirement: HNSW vector index over L2 distance
The system SHALL store semantic-cache embeddings in a Faiss HNSW index using L2 distance. When Faiss is unavailable or the configured backend is brute-force, the system SHALL fall back to an exact-nearest-neighbour numpy index that returns the same L2 distances, so the engine operates with or without Faiss installed.

#### Scenario: Vectors are indexed and retrievable
- **WHEN** an embedding is stored via the semantic cache and a near-identical query embedding is later searched
- **THEN** the index returns that stored entry as the nearest neighbour with its L2 distance

#### Scenario: Faiss-absent fallback
- **WHEN** Faiss cannot be imported (or the backend is configured as brute-force)
- **THEN** the system uses the exact-nearest-neighbour numpy index and all semantic-cache operations still succeed

### Requirement: Similarity from L2 distance
The system SHALL compute the match similarity from the nearest-neighbour L2 distance as `similarity = 1.0 / (1.0 + distance)`, perform a k=1 search, and treat a query as a cache hit if and only if `similarity >= threshold`. This SHALL replace the previous cosine-similarity computation.

#### Scenario: Hit above threshold
- **WHEN** the nearest neighbour's `1/(1+distance)` similarity is greater than or equal to the configured threshold
- **THEN** the system returns the matched cache entry and records the similarity score on it

#### Scenario: Miss below threshold
- **WHEN** the nearest neighbour's `1/(1+distance)` similarity is below the configured threshold
- **THEN** the system returns no match (cache miss)

#### Scenario: Empty index
- **WHEN** a query is searched against an index containing no vectors
- **THEN** the system returns a cache miss without error

### Requirement: Embeddings normalized for cross-model comparability
The system SHALL L2-normalize every embedding to unit norm before indexing and before searching, so the distance scale is identical across the study's embedding models. A zero-norm vector SHALL be handled without error.

#### Scenario: Store and query use normalized vectors
- **WHEN** a vector is added to the index and later queried
- **THEN** both the stored and the query vector are unit-normalized before the L2 distance is computed

#### Scenario: Zero vector is safe
- **WHEN** a zero-norm vector is added or searched
- **THEN** the operation completes without raising and without producing NaN distances

### Requirement: Internal-id to entry metadata mapping
The system SHALL maintain a mapping from each index entry's internal id to its cache entry, carrying at least the query text, the response, and the embedding-model identity. A retrieved nearest-neighbour id SHALL resolve to the correct entry.

#### Scenario: Retrieved id resolves to its entry
- **WHEN** a search returns the internal id of a stored entry
- **THEN** the system resolves that id to the cache entry whose query text, response, and embedding-model identity match what was stored

### Requirement: Per-configuration reset
The system SHALL provide a reset operation that empties the vector index and its id→entry mapping and restarts id assignment, so each experiment configuration begins with an empty cache.

#### Scenario: Reset empties the index
- **WHEN** entries have been stored and the cache is reset
- **THEN** the index size is zero, the next query is a miss, and a subsequently stored entry is retrievable again

### Requirement: Brute-force and Faiss agree on hit/miss decisions
For a given set of stored embeddings and a query, the Faiss HNSW index and the exact-nearest-neighbour brute-force index SHALL produce the same hit/miss decision at a given threshold (the brute-force index being the defined correctness oracle).

#### Scenario: Backends agree on a fixture
- **WHEN** the same embeddings are stored in both the Faiss and brute-force indexes and the same query is searched at the same threshold
- **THEN** both backends yield the same hit/miss outcome and resolve to the same nearest entry
