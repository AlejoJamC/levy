# latency-measurement Specification

Capability: the second half of the Proposal's economic-viability test — decomposed measurement of what a cache lookup costs (offline, reproducible) and of what a cache hit avoids (real provider, not reproducible), their artefact contract, and their isolation from the deterministic D3 result set.

## Purpose

Hit rate alone does not settle economic viability. The frozen Project Proposal
assesses it on hit rate **and** "latency measurements (cache lookup overhead vs
LLM call savings)", and before this capability existed the second half was
measured nowhere: every timing the repository could produce came from
`MockLLMClient`'s fixed sleep, and the engine timed only the whole `generate()`
path with a single clock read, so the cache's own cost could not be separated
from the response it avoids.

This capability supplies both numbers and, just as importantly, keeps them
apart. Lookup overhead is measured offline, decomposed by segment, with
embedding cost reported **cold and warm as separate figures** that are never
averaged — both are true of the artefact and answer different questions. It
replicates on other hardware. Provider latency is measured against a real
model and does **not** replicate; it belongs to the model, provider, region and
moment that produced it, so every figure derived from it carries its resolved
model identifier and the artefacts state that boundary explicitly.

The measurement never writes into an existing harness result directory, and no
latency value enters `results.csv`, `decisions.csv` or the +/-5% replication
criterion — those stay byte-identical across re-runs, which is what the
criterion rests on. The billed provider path is a separate out-of-band entry
point, guarded the same way corpus acquisition is.

## Requirements
### Requirement: Decomposed cache lookup timing

The engine SHALL expose per-request timing for each segment of the lookup path — embedding generation, vector index search, exact-cache lookup, and the total lookup path — as an additive surface. The existing `LevyResult` fields, the `LevyMetrics` contract and the default behaviour of `LevyEngine.generate()` SHALL remain unchanged when timing collection is not requested.

#### Scenario: Segment timings are collected on a semantic cache hit

- **WHEN** a request resolves through the semantic cache with timing collection enabled
- **THEN** timings for embedding generation, index search and total lookup path are each recorded as a non-negative duration
- **AND** the sum of the recorded segments does not exceed the recorded total lookup path

#### Scenario: Default behaviour is unaffected

- **WHEN** a request is served with timing collection disabled
- **THEN** the returned `LevyResult` carries the same fields and values it carried before this change
- **AND** no segment timing is recorded

### Requirement: Cold and warm embedding cost are measured separately

The benchmark SHALL report embedding generation cost twice: **cold**, with `EmbeddingManager` memoisation cleared immediately before the measured call, and **warm**, with the memoised value present. The two figures SHALL be reported as distinct fields and SHALL NOT be averaged together.

#### Scenario: Cold and warm are distinguishable in the output

- **WHEN** the offline benchmark completes over a workload
- **THEN** the output contains separate cold and warm embedding figures for each configuration
- **AND** neither field is empty

#### Scenario: Memoisation is cleared before a cold sample

- **WHEN** a cold embedding sample is taken for a text already embedded earlier in the run
- **THEN** the memoisation entry for that text is cleared before the measured call
- **AND** the measured call invokes the underlying embedding client

### Requirement: Repeated sampling with warm-up

The benchmark SHALL discard a configured number of warm-up iterations before recording, take a configured number of measured repetitions per segment, and report the median (p50) and 95th percentile (p95) of each segment. A single-sample figure SHALL NOT be reported as a segment result.

#### Scenario: Percentiles are derived from the recorded repetitions

- **WHEN** the benchmark runs with N measured repetitions for a segment
- **THEN** the reported p50 and p95 are computed from exactly those N recorded values
- **AND** the warm-up iterations are absent from that computation

#### Scenario: Repetition and warm-up counts are recorded

- **WHEN** the benchmark writes its metadata sidecar
- **THEN** the warm-up count and the measured repetition count appear in it

### Requirement: Real-provider latency sample

The system SHALL provide an entry point that calls the configured Anthropic model once per unique prompt of the target workload and records, per call, wall-clock end-to-end latency, input and output token counts, the resolved model identifier and a UTC timestamp. The recorded response corpus SHALL be keyed by `sha256(prompt)` so a single population serves every configuration of that workload without repeating the spend.

#### Scenario: A call is recorded with its measurement fields

- **WHEN** a real provider call completes
- **THEN** its latency, input token count, output token count, resolved model identifier and UTC timestamp are recorded
- **AND** the response is stored under the SHA-256 of its prompt

#### Scenario: A prompt already present is not called again

- **WHEN** the population runs over a prompt whose SHA-256 key already exists in the response corpus
- **THEN** no provider call is made for that prompt

#### Scenario: The budget guard halts the run

- **WHEN** the accumulated estimated cost reaches the configured cap during population
- **THEN** the run stops before sending the next request
- **AND** the calls recorded up to that point remain valid and readable

### Requirement: Observed cost is recorded, not estimated

The provider run SHALL record observed total cost, observed cost per call, and the token totals the cost was derived from, using the per-MTok prices configured for the model actually used. A pre-run estimate SHALL NOT be presented as the result.

#### Scenario: Cost fields are populated from the run

- **WHEN** the provider run completes
- **THEN** observed total cost, observed cost per call and the input/output token totals appear in the output
- **AND** the resolved model identifier those figures belong to appears alongside them

### Requirement: Every reported latency figure names its model

Any reported provider-latency or savings figure SHALL carry the resolved model identifier that produced it. A savings figure SHALL NOT be presented as model-independent.

#### Scenario: Savings figure carries its model

- **WHEN** a headline savings figure is written to an artefact
- **THEN** the resolved model identifier appears in the same artefact and is associated with that figure

### Requirement: Latency artefact contract

The measurement SHALL write its artefacts to a dedicated output directory containing: a per-configuration table of segment percentiles; a provider-call summary with latency distribution, token totals, model identifier, UTC window and observed cost; and a metadata sidecar recording host specification, operating system, Python and library versions, embedding provider, seeds, warm-up and repetition counts. The raw response corpus SHALL be written to the same directory and SHALL be excluded from version control.

#### Scenario: All artefacts are present after a completed run

- **WHEN** a run completes for a workload
- **THEN** the per-configuration table, the provider-call summary and the metadata sidecar all exist in the output directory
- **AND** the per-configuration table contains one row per configuration of that workload

#### Scenario: The response corpus is not committed

- **WHEN** the response corpus file is written
- **THEN** it matches an entry in `.gitignore`
- **AND** it is not tracked by version control

### Requirement: Reproducibility boundary is stated in the artefacts

The metadata sidecar SHALL state explicitly that the lookup-overhead measurement is reproducible on other hardware and that the provider-latency measurement is provider-, region- and time-dependent and does not replicate.

#### Scenario: The boundary statement is present

- **WHEN** the metadata sidecar is written
- **THEN** it contains an explicit statement distinguishing the reproducible measurement from the non-reproducible one

### Requirement: Isolation from the deterministic result set

The measurement SHALL NOT create, modify, rename or delete any file in an existing harness result directory. Its only permitted interaction with one is reading `results.csv` to obtain the configuration list. Latency values SHALL NOT be written into `results.csv` or `decisions.csv`, and SHALL NOT participate in the ±5 % replication criterion.

#### Scenario: An existing result directory is untouched

- **WHEN** a full measurement run completes against a reference result directory
- **THEN** every file in that directory is byte-identical to its state before the run

#### Scenario: The harness output contract is unchanged

- **WHEN** the harness writes `results.csv` and `decisions.csv` after this change
- **THEN** their column sets are unchanged
- **AND** neither contains a latency or timestamp column

#### Scenario: Replication is unaffected

- **WHEN** the replication check runs against the reference result directory after a measurement run
- **THEN** it reports the same pass result and the same number of comparisons as before the measurement run

### Requirement: The networked path stays out of the offline test suite

The provider-population entry point SHALL NOT be invoked by any test, and no module under `levy/latency/` that the offline suite imports SHALL import a network library at module scope. The offline benchmark and the artefact writers SHALL be testable without network access.

#### Scenario: No test invokes the networked entry point

- **WHEN** the test suite is analysed for references to the provider-population entry point
- **THEN** no test invokes it

#### Scenario: The benchmark runs offline

- **WHEN** the offline benchmark is executed with mock providers and no network access
- **THEN** it completes and writes its per-configuration table and metadata sidecar

