# experiment-harness

Capability: offline replay of annotated query pairs across the frozen 30-configuration experimental grid — per-configuration confusion-matrix accounting against ground-truth labels, metric computation, and deterministic machine-readable outputs consumed by the statistical analysis.

## ADDED Requirements

### Requirement: Experiment configuration and frozen grid enumeration
The system SHALL define an `ExperimentConfig` carrying an embedding model, a workload, and a similarity threshold, and SHALL enumerate the frozen experimental grid: 2 embedding models (`all-MiniLM-L6-v2`, `modernbert`) × 3 workloads (`faq`, `code`, `chat`) × 5 thresholds (0.70, 0.75, 0.80, 0.85, 0.90) = 30 configurations. Thresholds SHALL be applied exactly as configured on the `1/(1+L2_distance)` similarity scale, with no rescaling or transformation.

#### Scenario: Grid enumeration
- **WHEN** the full grid is enumerated
- **THEN** exactly 30 distinct configurations are produced covering every (model, workload, threshold) combination of the frozen study

#### Scenario: Threshold passed through unmodified
- **WHEN** a configuration with threshold 0.80 is run
- **THEN** the semantic cache decides hit/miss by comparing the `1/(1+L2)` similarity against 0.80 exactly

### Requirement: Replay protocol per Algorithm 2
For each configuration the system SHALL initialise a fresh, empty cache (exact and semantic), then for each query pair of the configuration's workload, in dataset order: submit `query_1` (a miss by construction; its embedding-response pair is stored), then submit `query_2` and record the cache's hit/miss decision. The cache SHALL accumulate entries across pairs within a configuration and SHALL NOT be reset between pairs. Submissions SHALL go through the engine's production lookup path (exact cache, then semantic cache), and the decision source SHALL be recorded.

#### Scenario: First query stored, second query decided
- **WHEN** a pair is replayed against an empty cache
- **THEN** `query_1` produces a miss and is stored, and `query_2` produces a hit or miss decided by the configured threshold

#### Scenario: Cache accumulates within a configuration
- **WHEN** the second pair of a workload is replayed
- **THEN** its `query_2` is compared against all entries stored so far in this configuration, not only its own `query_1`

#### Scenario: Fresh cache per configuration
- **WHEN** a new configuration begins
- **THEN** the cache contains zero entries and no state from any previous configuration influences its decisions

### Requirement: Confusion-matrix accounting against ground-truth labels
The system SHALL compare each `query_2` decision against the pair's ground-truth label and increment exactly one counter per pair: hit with duplicate label → TP; hit with non-duplicate label → FP; miss with non-duplicate label → TN; miss with duplicate label → FN.

#### Scenario: True positive
- **WHEN** the label marks the pair as duplicate and the cache returns a hit for `query_2`
- **THEN** TP is incremented and no other counter changes

#### Scenario: False positive
- **WHEN** the label marks the pair as non-duplicate and the cache returns a hit for `query_2`
- **THEN** FP is incremented and no other counter changes

#### Scenario: True negative and false negative
- **WHEN** the cache returns a miss for `query_2`
- **THEN** TN is incremented if the label is non-duplicate, and FN is incremented if the label is duplicate

### Requirement: Per-configuration metrics
After replaying all pairs of a configuration the system SHALL compute: precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, F0.5 `1.25·P·R/(0.25·P+R)`, false positive rate `FP/(FP+TN)`, and hit rate `(TP+FP)/N` where `N` is the number of pairs replayed. Any zero-denominator case SHALL be reported as `0.0` together with an explicit flag, never as NaN.

#### Scenario: Known counts produce known metrics
- **WHEN** a configuration ends with TP=8, FP=2, TN=7, FN=3
- **THEN** precision=0.8, recall≈0.7273, F0.5≈0.7843, FPR≈0.2222, hit rate=0.5

#### Scenario: Zero denominator is flagged
- **WHEN** a configuration produces no hits at all (TP+FP=0)
- **THEN** precision is reported as 0.0 with the zero-division flag set, and the output remains machine-readable

### Requirement: Machine-readable outputs
The system SHALL write one results CSV with exactly one row per configuration (configuration identity, TP/FP/TN/FN, all five metrics, zero-division flags) and one per-pair decision log (configuration identity, pair id, decision, decision source, matched similarity when present, ground-truth label, confusion outcome). Run parameters and latency statistics SHALL be written to a separate run-metadata sidecar, not into the results CSV or decision log.

#### Scenario: Results CSV shape
- **WHEN** the full grid is run
- **THEN** the results CSV contains exactly 30 data rows, one per configuration, with the confusion counts and metric columns populated

#### Scenario: Decision log audits every pair
- **WHEN** a configuration replays N pairs
- **THEN** the decision log contains exactly N rows for that configuration, each resolvable to its `QueryPair` by pair id

### Requirement: Offline deterministic execution
The full grid SHALL run end-to-end with no network access using the mock LLM provider, and re-running with identical inputs SHALL reproduce the results CSV and decision log byte-identically. Timestamps and wall-clock measurements SHALL NOT appear in these two outputs.

#### Scenario: Deterministic re-run
- **WHEN** the harness is run twice on the same dataset with the same configuration grid
- **THEN** both runs produce byte-identical results CSV and decision log files

#### Scenario: Fully offline
- **WHEN** the harness runs with mock providers and no network access
- **THEN** all 30 configurations complete and produce outputs

### Requirement: Statistical sanity checks
The system SHALL verify, per configuration, that TP+FP+TN+FN equals the number of pairs replayed and that every reported rate lies within [0, 1], and SHALL fail loudly (non-zero exit, explicit error naming the configuration) if any check is violated.

#### Scenario: Violated invariant aborts the run
- **WHEN** a configuration's confusion counts do not sum to the pair count
- **THEN** the run fails with an error identifying the offending configuration rather than writing silently corrupt results

### Requirement: One-command full sweep
The system SHALL provide a single command that loads the dataset via the `levy/dataset` loader, runs all 30 configurations offline, and writes the results CSV, decision log, and run-metadata sidecar to a chosen output directory.

#### Scenario: Single command end-to-end
- **WHEN** the run-experiments command is invoked with a dataset path and output directory
- **THEN** it completes offline and the output directory contains the results CSV (30 rows), the decision log, and the run metadata
