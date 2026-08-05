# ground-truth-dataset

Capability: the schema, persistence, sampling, blind re-annotation, and
inter-annotator-agreement tooling for the D2 ground-truth dataset (900
query pairs across 3 workloads). This capability specifies the
data-agnostic platform; it does not specify the content of the real 900-pair
dataset itself (a data-production task, not a spec requirement).

## ADDED Requirements

### Requirement: Query pair schema and validation
The system SHALL represent each ground-truth record as a `QueryPair` with
fields `pair_id`, `workload` (one of `faq`, `code`, `chat`), `source_corpus`,
`source_pair_id`, `query_1`, `query_2`, `original_label` (0 or 1),
`author_label` (0, 1, or absent/unset), and `metadata`. The system SHALL
reject construction of a `QueryPair` whose `workload`, `original_label`, or
`author_label` is outside its allowed values, or whose `pair_id`,
`source_corpus`, `source_pair_id`, `query_1`, or `query_2` is empty.

#### Scenario: Valid pair constructs
- **WHEN** a `QueryPair` is constructed with a valid workload, non-empty
  identifying fields, non-empty queries, and `original_label` in `{0, 1}`
- **THEN** construction succeeds

#### Scenario: Invalid workload rejected
- **WHEN** a `QueryPair` is constructed with a `workload` not in
  `{faq, code, chat}`
- **THEN** construction raises a validation error naming the field and the
  offending value

#### Scenario: Ground-truth label resolution
- **WHEN** a `QueryPair` has an `author_label` set
- **THEN** `ground_truth_label()` returns `author_label`
- **WHEN** a `QueryPair` has no `author_label` set
- **THEN** `ground_truth_label()` returns `original_label`

### Requirement: CSV/JSON persistence with round-trip and cross-format equality
The system SHALL save and load ground-truth datasets in both CSV and JSON,
such that loading either format for the same underlying data yields
identical `QueryPair` objects (same fields, same values, same order,
including `metadata` and an absent `author_label` decoding to `None` in
both formats). The system SHALL reject a dataset file missing a required
column/field with an error identifying the file and, for CSV, the row
number.

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

### Requirement: Seeded, stratified sampling from a corpus source
The system SHALL provide a `CorpusSource` abstraction that yields candidate
labeled pairs for one workload from one corpus, without performing any
network access, plus a sampling function that draws a deterministic,
stratified sample of `n` pairs given a `CorpusSource`, a sample size, a
random seed, and a target positive-label ratio.

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

### Requirement: Blind, resumable, non-destructive re-annotation
The system SHALL provide an annotation flow that presents an annotator with
only `query_1` and `query_2` for each unlabeled pair — never
`original_label`, `source_corpus`, or `source_pair_id` — records the
annotator's binary judgment as `author_label`, persists progress after each
recorded judgment so a session can be resumed after interruption without
losing already-recorded judgments, and never overwrites an existing
`author_label` unless the caller explicitly requests an overwrite.

#### Scenario: Original label withheld
- **WHEN** an annotation session presents a pair to the annotator
- **THEN** neither `original_label`, `source_corpus`, nor `source_pair_id`
  appears in anything shown to the annotator

#### Scenario: Session resumes after interruption
- **WHEN** a session records judgments for some pairs, stops (interruption
  or explicit quit), and a new session is started against the same pairs
  and the same progress file
- **THEN** previously recorded judgments are present without being asked
  again, and only the remaining unlabeled pairs are presented

#### Scenario: Existing label not overwritten by default
- **WHEN** a session runs over a pair that already has an `author_label`
  and the caller has not requested an overwrite
- **THEN** that pair's `author_label` is left unchanged and the pair is not
  re-presented to the annotator

### Requirement: Cohen's kappa with documented edge-case behavior
The system SHALL compute Cohen's kappa between `original_label` and
`author_label` over a set of pairs, excluding pairs without an
`author_label` and reporting how many were excluded, and SHALL provide a
per-workload breakdown alongside the overall figure. The system SHALL define
and document its behavior for the zero-annotated-pairs case and for the
case where expected agreement equals 1.

#### Scenario: Perfect agreement
- **WHEN** every annotated pair's `author_label` equals its
  `original_label`
- **THEN** the computed kappa is 1.0

#### Scenario: Chance-level agreement
- **WHEN** annotated labels are constructed such that observed agreement
  equals expected agreement under the annotators' marginal distributions
- **THEN** the computed kappa is 0.0

#### Scenario: Unannotated pairs excluded and counted
- **WHEN** the input set includes pairs with no `author_label`
- **THEN** those pairs are excluded from the kappa computation and their
  count is reported separately from the count of annotated pairs used

#### Scenario: Zero annotated pairs
- **WHEN** no pair in the input set has an `author_label`
- **THEN** the result reports no computable kappa rather than raising or
  returning a misleading numeric value

#### Scenario: Per-workload breakdown
- **WHEN** a kappa report is requested over pairs spanning multiple
  workloads
- **THEN** the report includes one kappa result per workload in addition to
  the overall result

### Requirement: Offline CLIs over the dataset platform
The system SHALL expose command-line entry points for sampling, blind
annotation, and kappa computation that operate fully offline against
locally available files (including the committed synthetic fixtures), with
no network access required to exercise any of them.

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
