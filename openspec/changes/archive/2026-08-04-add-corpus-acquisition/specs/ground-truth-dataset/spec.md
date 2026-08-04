## MODIFIED Requirements

### Requirement: Seeded, stratified sampling from a corpus source
The system SHALL provide a `CorpusSource` abstraction that yields candidate
labeled pairs for one workload from one corpus, without performing any
network access, plus a sampling function that draws a deterministic,
stratified sample of `n` pairs given a `CorpusSource`, a sample size, a
random seed, and a target positive-label ratio. Concrete adapters SHALL be
provided for exactly the corpora the study uses: Quora Question Pairs for
the `faq` workload, the Stack Overflow Duplicity Dataset for the `code`
workload, and the Twitter PIT-2015 paraphrase corpus for the `chat`
workload. Every adapter SHALL declare which label values it treats as
positive and which as negative, and any adapter choice that changes which
pairs are produced or how their text is normalised SHALL be an explicit
option with a documented default rather than an implicit behaviour.
Sampling SHALL be preceded by the pre-flight validation defined by the
`corpus-acquisition` capability, and SHALL offer a mode in which falling
back to a synthetic source is a hard error rather than a warning.

#### Scenario: Same seed is deterministic
- **WHEN** the same `CorpusSource` (same underlying candidates) is sampled
  twice with the same `n` and the same seed
- **THEN** both samples are identical, in the same order

#### Scenario: Different seed differs
- **WHEN** the same `CorpusSource` is sampled with two different seeds
- **THEN** the resulting samples are not identical

#### Scenario: Stratification honored
- **WHEN** sampling `n` pairs with a target positive ratio `r`
- **THEN** the resulting sample contains `round(n * r)` pairs with
  `original_label == 1` and the remainder with `original_label == 0`

#### Scenario: Insufficient candidates reported
- **WHEN** a `CorpusSource` does not have enough candidates in a required
  class to satisfy the requested stratified sample size
- **THEN** sampling raises an error naming the shortfall, rather than
  silently returning fewer pairs than requested

#### Scenario: Adapter declares its label mapping
- **WHEN** an adapter reads a corpus whose native labels are not binary
- **THEN** it maps a declared value to `original_label == 1` and a declared
  value to `original_label == 0`, and never mixes two different label
  scales from the same corpus into one sample

#### Scenario: Content-affecting adapter option is explicit
- **WHEN** an adapter choice changes which candidate pairs are produced or
  how their text is normalised
- **THEN** that choice is an explicit option with a documented default, and
  its value is recorded in the run manifest

#### Scenario: Production run refuses synthetic fallback
- **WHEN** sampling runs in the mode that requires real corpora and a raw
  corpus path is absent
- **THEN** the run fails naming the missing corpus, rather than substituting
  a synthetic source and emitting pairs whose `source_corpus` is synthetic

### Requirement: CSV/JSON persistence with round-trip and cross-format equality
The system SHALL save and load ground-truth datasets in both CSV and JSON,
such that loading either format for the same underlying data yields
identical `QueryPair` objects (same fields, same values, same order,
including `metadata` and an absent `author_label` decoding to `None` in
both formats). The system SHALL reject a dataset file missing a required
column/field with an error identifying the file and, for CSV, the row
number. In addition to this full-dataset format, the system SHALL persist
the identifiers-and-labels distribution format defined by the
`corpus-acquisition` capability. The two formats SHALL be distinct files
written by distinct code paths, and the full-dataset format SHALL remain
the only input contract the experiment harness loads.

#### Scenario: CSV round-trip
- **WHEN** a list of `QueryPair`s is saved to CSV and reloaded
- **THEN** the reloaded list equals the original list

#### Scenario: JSON round-trip
- **WHEN** a list of `QueryPair`s is saved to JSON and reloaded
- **THEN** the reloaded list equals the original list

#### Scenario: CSV and JSON carry identical content
- **WHEN** the same list of `QueryPair`s is saved to both CSV and JSON
- **THEN** loading either file yields the same list of `QueryPair`s

#### Scenario: Missing column reported clearly
- **WHEN** a CSV file is loaded that is missing one of the required columns
- **THEN** loading raises an error naming the file and the missing
  column(s), and does not silently drop or default the field

#### Scenario: Distribution format is not a harness input
- **WHEN** the experiment harness is pointed at the identifiers-and-labels
  distribution file
- **THEN** loading fails with an error directing the caller to rehydrate
  first, rather than replaying pairs with absent query text

### Requirement: Offline CLIs over the dataset platform
The system SHALL expose command-line entry points for sampling, blind
annotation, kappa computation, and rehydration of a distributed
identifiers-and-labels dataset that operate fully offline against
locally available files (including the committed synthetic fixtures), with
no network access required to exercise any of them. Corpus acquisition is
the sole entry point permitted to perform network access; it SHALL be
excluded from the automated test suite, and adapter behaviour SHALL instead
be covered by committed fixtures.

#### Scenario: Sampling CLI runs offline
- **WHEN** the sampling CLI is invoked without any real corpus file paths
- **THEN** it falls back to a synthetic candidate source per workload and
  produces a valid dataset file, without any network access

#### Scenario: Kappa CLI reports and can gate
- **WHEN** the kappa CLI is invoked against a fully annotated dataset with
  `--strict` and a threshold higher than the dataset's actual kappa
- **THEN** it reports the kappa figures and exits with a non-zero status

#### Scenario: Kappa CLI does not gate on incomplete annotation
- **WHEN** the kappa CLI is invoked with `--strict` against a dataset that
  still has unannotated pairs
- **THEN** it reports the current kappa figures but does not fail on the
  threshold, since the dataset is not yet ready for a pass/fail judgment

#### Scenario: Rehydration CLI runs offline
- **WHEN** the rehydration CLI is invoked with the distribution file and a
  populated raw-corpus directory, with network access unavailable
- **THEN** it reconstructs the working dataset from local files alone

#### Scenario: Acquisition is not in the test suite
- **WHEN** the automated test suite runs
- **THEN** no test performs corpus acquisition or any other network access,
  and adapter behaviour is covered by committed fixtures instead
