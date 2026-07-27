## ADDED Requirements

### Requirement: One-command corpus acquisition into a fixed location
The system SHALL provide an executable that acquires every raw corpus the study depends on into a fixed, per-corpus directory under `data/raw/`, verifying each acquired file against a recorded SHA-256 checksum. The executable SHALL be idempotent: files already present whose checksum matches SHALL be skipped rather than re-downloaded. Where a corpus cannot be acquired without human action, the executable SHALL exit non-zero and print the canonical URL, the expected filename and the expected SHA-256, and SHALL NOT leave a partially acquired corpus behind.

#### Scenario: Fresh acquisition
- **WHEN** the acquisition executable runs against an empty `data/raw/`
- **THEN** every automatically acquirable corpus file is downloaded into its per-corpus directory and its checksum verified

#### Scenario: Re-run skips verified files
- **WHEN** the acquisition executable runs again with all files already present and checksum-matching
- **THEN** no file is re-downloaded and the run exits zero

#### Scenario: Checksum mismatch is a failure
- **WHEN** an acquired file's SHA-256 does not match the recorded checksum
- **THEN** the run exits non-zero identifying the file and both checksums, and the mismatched file is not treated as acquired

#### Scenario: Human acquisition step is actionable
- **WHEN** a corpus requires a human step (for example accepting terms on a hosting platform)
- **THEN** the run exits non-zero printing the canonical URL, the expected filename and the expected SHA-256, and no other corpus is left half-acquired

### Requirement: Raw corpus directory present in the tree, absent from the remote
The repository SHALL contain the `data/raw/` directory and one subdirectory per corpus in a clean clone, while their contents SHALL be excluded from version control. Directory presence SHALL be achieved with tracked placeholder files that survive the ignore rules.

#### Scenario: Clean clone has the structure
- **WHEN** the repository is cloned fresh
- **THEN** `data/raw/` and each per-corpus subdirectory exist and contain only their tracked placeholder file

#### Scenario: Acquired corpora do not appear as changes
- **WHEN** the acquisition executable has populated `data/raw/`
- **THEN** the working tree reports no untracked or modified files under `data/raw/`

### Requirement: Machine-readable corpus provenance registry
The system SHALL record, in a single machine-readable registry, for every corpus: the canonical source URL, the snapshot or version identifier, the licence, the expected filenames, the expected SHA-256 checksums, and the citation. The acquisition executable, the pre-flight validation, and the release audit SHALL read this registry rather than embedding their own copies of the same facts.

#### Scenario: Registry is the single source
- **WHEN** a corpus's expected filename or checksum changes in the registry
- **THEN** the acquisition executable and the pre-flight validation both observe the change without any other file being edited

#### Scenario: Unknown corpus is rejected
- **WHEN** a corpus is requested that has no entry in the registry
- **THEN** the run exits non-zero naming the missing entry, rather than guessing a URL or filename

### Requirement: Pre-flight validation reports every problem in one pass
The system SHALL validate raw corpus inputs before any sampling occurs, and SHALL report every detected problem in a single pass rather than aborting on the first. Validation SHALL cover: file presence and readability, checksum agreement with the registry, presence of the fields each adapter requires, sufficiency of both the positive and the negative candidate pool for every workload simultaneously, label values confined to the domain the adapter declares, and absence of corpus overlap between workloads. When validation fails, no output file SHALL be written.

#### Scenario: All problems reported together
- **WHEN** two different workloads each have a distinct validation problem
- **THEN** the report names both problems in one run, not only the first

#### Scenario: Nothing written on failure
- **WHEN** validation fails for any reason
- **THEN** no dataset or manifest file is created or modified

#### Scenario: Insufficient class pool detected before sampling
- **WHEN** a workload's positive or negative candidate pool is smaller than the requested stratified sample requires
- **THEN** validation reports the shortfall for that workload, alongside any shortfalls in the other workloads

#### Scenario: Cross-workload corpus overlap rejected
- **WHEN** two workloads would draw from the same underlying corpus
- **THEN** validation fails, because a shared corpus removes the workload contrast the study's workload hypothesis tests

#### Scenario: Unexpected label value rejected
- **WHEN** a raw corpus row carries a label outside the domain its adapter declares
- **THEN** validation reports the offending value rather than coercing it

### Requirement: Licence-safe distribution as identifiers and labels
The system SHALL provide a distribution record type that carries a pair's identifiers and labels but no query text, and SHALL persist it as its own file distinct from the full dataset. This record type SHALL be a distinct type rather than a full pair with emptied text fields, because the full pair type's non-empty-text invariant is relied upon downstream and SHALL NOT be weakened.

#### Scenario: Distribution file carries no query text
- **WHEN** the distribution file is written for a sampled dataset
- **THEN** it contains the pair identifier, workload, source corpus, source pair identifier, original label, author label and metadata, and no query text in any field

#### Scenario: Full pair type keeps its invariant
- **WHEN** a full pair is constructed with empty or whitespace-only query text
- **THEN** construction still fails, unchanged by the existence of the distribution record type

### Requirement: Rehydration reconstructs the working dataset losslessly
The system SHALL provide an executable that reconstructs the full working dataset from the distribution file plus a populated `data/raw/`. The reconstruction SHALL be lossless: a dataset sampled, reduced to the distribution file, and rehydrated SHALL be byte-identical to the originally sampled dataset. The rehydrated working dataset SHALL be written to a location excluded from version control.

#### Scenario: Round-trip is byte-identical
- **WHEN** a dataset is sampled, reduced to the distribution file, and rehydrated
- **THEN** the rehydrated dataset file is byte-identical to the originally sampled dataset file

#### Scenario: Missing raw corpus reported
- **WHEN** rehydration runs with `data/raw/` unpopulated or incomplete
- **THEN** it exits non-zero naming the missing corpus and the executable that acquires it, and writes no partial dataset

#### Scenario: Unresolvable source pair identifier reported
- **WHEN** a distribution record's source pair identifier is not present in the raw corpus
- **THEN** rehydration exits non-zero naming the record and the corpus, rather than dropping the pair

### Requirement: Run manifest records everything needed to reproduce the inputs
The system SHALL write a manifest for each acquisition and sampling run recording the raw-file checksums, the random seed, the target positive ratio, every adapter option that affects the sampled content, and the tool versions used. The manifest SHALL be sufficient for a third party to prove they hold byte-identical inputs without any party redistributing corpus text.

#### Scenario: Manifest records adapter options that change content
- **WHEN** an adapter option that changes which pairs are produced or how their text is normalised is used
- **THEN** the manifest records that option's value

#### Scenario: Manifest enables input verification
- **WHEN** a third party acquires the corpora and compares against the manifest
- **THEN** they can confirm every raw input checksum, the seed and the sampling parameters without receiving any corpus text
