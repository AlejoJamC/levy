# results-dashboard

Capability: local interactive exploration of an analysis bundle — threshold-performance curves per (model, workload), a hypothesis/κ summary, and a live query demo reporting similarity and the cache decision under the production threshold semantics.

## ADDED Requirements

### Requirement: Bundle-agnostic loading with validation
The dashboard SHALL take an analysis bundle directory as input and render any bundle produced by the analysis pipeline, whether its results came from the fixture dataset or a real one. It SHALL validate that the required bundle files and columns are present, sourcing those expectations from the analysis package's own definitions rather than a duplicate declaration.

#### Scenario: Any bundle renders
- **WHEN** the dashboard is pointed at a directory produced by the analysis command
- **THEN** the curve, hypothesis, and κ panels populate from that directory's files

#### Scenario: Schema drift surfaces as a load error
- **WHEN** a bundle file lacks an expected column
- **THEN** loading fails with an error naming the file and column instead of rendering a mis-parsed table

### Requirement: Missing bundle is handled actionably
When the bundle directory is absent or incomplete, the dashboard SHALL report what is missing and the exact command that produces a bundle, and SHALL stop cleanly without a traceback; the underlying loader SHALL raise a typed error carrying the same information so this path is testable headlessly.

#### Scenario: Fresh clone with no results
- **WHEN** the dashboard runs in a checkout where no bundle has been generated
- **THEN** it displays the missing paths and the command to generate them, and exits without an unhandled exception

### Requirement: Threshold-performance exploration
The dashboard SHALL let a user select an embedding model and workload and view hit-rate and precision against threshold for that pair, showing the analysis bundle's zero-division flags for degenerate points and marking the 30% hit-rate viability reference.

#### Scenario: Curves per model and workload
- **WHEN** a (model, workload) pair is selected
- **THEN** both metric curves for that pair render across the bundle's thresholds

#### Scenario: Degenerate points stay visible
- **WHEN** a point carries a zero-division flag from the harness
- **THEN** the dashboard marks it as degenerate rather than presenting it as a measured value

### Requirement: Hypothesis and agreement summary
The dashboard SHALL display the hypothesis-test outcomes, post-hoc status, and κ report read from the bundle, and SHALL NOT recompute any statistic.

#### Scenario: Summary reflects the bundle
- **WHEN** the hypothesis panel renders
- **THEN** the reported effects, p-values, post-hoc status, and κ match the bundle's files exactly

### Requirement: Live query decision under production semantics
The dashboard SHALL accept user-supplied text, embed it with the selected model, search an index built in-process from the dataset's queries, and report the nearest match, its similarity, and the hit/miss decision at the selected threshold — computed by the production semantic-cache path (the `1/(1+L2)` similarity on normalised vectors with the threshold applied verbatim), not by a reimplementation.

#### Scenario: Near-duplicate query reports a hit
- **WHEN** the user submits text closely matching an indexed query at a threshold below its similarity
- **THEN** the dashboard reports a hit, the matched entry, and the similarity score

#### Scenario: Threshold change flips the decision without reindexing
- **WHEN** only the threshold is changed after a query has been run
- **THEN** the decision is re-evaluated against the same index and no re-embedding of the dataset occurs

#### Scenario: Unrelated query reports a miss
- **WHEN** the user submits text unrelated to any indexed query
- **THEN** the dashboard reports a miss with the nearest similarity shown

### Requirement: Provenance is visible
The dashboard SHALL surface the bundle's provenance (including fixture-only labelling and the run's providers) so demonstration data cannot be mistaken for research results.

#### Scenario: Fixture bundle is labelled
- **WHEN** a bundle derived from the synthetic fixture is loaded
- **THEN** the interface shows its fixture provenance alongside the metrics

### Requirement: Local, offline-capable operation
The dashboard SHALL run locally inside the project environment with no deployment step, and SHALL operate fully offline with mock embeddings; using real sentence-transformers models SHALL be documented as requiring a first-run model download.

#### Scenario: Fully offline demo
- **WHEN** the dashboard runs with mock embeddings and no network access
- **THEN** curve exploration and the query demo both work

### Requirement: Testable core outside the UI framework
Bundle loading, curve selection, and the query decision SHALL live in framework-independent modules covered by unit tests, with the UI entry point kept thin; the enforced branch-coverage gate SHALL remain green without adding coverage exclusions.

#### Scenario: Logic tested headlessly
- **WHEN** the test suite runs
- **THEN** loading, validation, curve selection, and decision logic are exercised without starting a UI process

#### Scenario: Gate stays green
- **WHEN** the gated coverage command runs
- **THEN** it passes with no new pragmas or omit entries
