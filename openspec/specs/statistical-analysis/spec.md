# statistical-analysis Specification

Capability: the D3 evidence pipeline — hypothesis tests (two-way ANOVA on false positive rate + Tukey HSD), threshold-selection curve tables and figures, kappa reporting consumed from the dataset capability, and the +/-5% replication check, all driven by one command over the experiment harness outputs.

## Purpose

Turn experiment harness outputs into the study's evidence bundle: a two-way ANOVA on false positive rate with explicit reject/retain decisions for the three null hypotheses, conditional Tukey HSD post-hoc tests that always state whether they ran and why, threshold-selection curve tables and figures, annotator-agreement reporting consumed from the dataset capability rather than reimplemented, and the +/-5% replication check — all from one command, dataset-agnostically, with degenerate or undefined results reported as such rather than laundered into retained nulls.

## Requirements
### Requirement: Harness output contract as the only input
The pipeline SHALL consume the harness output directory (`results.csv` with the per-configuration confusion counts, metrics, and zero-division flags; `decisions.csv`; `run_meta.json`) and SHALL validate the expected columns on load, failing with a clear error naming any missing column rather than producing partial statistics.

#### Scenario: Valid harness output loads
- **WHEN** the pipeline is pointed at a directory produced by the harness
- **THEN** all 30 configuration rows load with their metrics and flags

#### Scenario: Contract violation fails loudly
- **WHEN** `results.csv` lacks a required column
- **THEN** the pipeline exits non-zero naming the missing column and writes no output tables

### Requirement: Two-way ANOVA testing the three frozen hypotheses
The pipeline SHALL fit a two-way ANOVA on false positive rate with factors (embedding model, workload) including their interaction, and SHALL report, for H0₁ (no model main effect), H0₂ (no workload main effect), and H0₃ (no interaction): degrees of freedom, sum of squares, F-statistic, p-value, and an explicit reject/retain statement at α = 0.05, as a machine-readable table.

#### Scenario: Constructed model effect is detected
- **WHEN** the input fixture has one model's FPR uniformly higher across workloads
- **THEN** the ANOVA table rejects H0₁ with p below 0.05 and reports the supporting F-statistic

#### Scenario: Null fixture retains all hypotheses
- **WHEN** the input fixture has identical FPR distributions across all cells
- **THEN** all three H0 are explicitly retained with their p-values reported

### Requirement: Tukey HSD post-hoc where effects are significant
The pipeline SHALL run Tukey HSD comparisons for significant effects (over the 6 model×workload cells when the interaction is significant), SHALL report pairwise differences with confidence intervals and adjusted p-values as a table, and SHALL always state whether post-hoc ran and why.

#### Scenario: Post-hoc follows a significant effect
- **WHEN** the ANOVA rejects an H0 on the fixture
- **THEN** the Tukey table contains the pairwise comparisons for that effect with adjusted p-values

#### Scenario: Post-hoc skipped is stated
- **WHEN** no effect is significant
- **THEN** the output records that Tukey HSD was not run and for which effects

### Requirement: Threshold-selection curves as tables and figures
The pipeline SHALL emit, per (model, workload), threshold-vs-hit-rate and threshold-vs-precision as machine-readable tables carrying the harness zero-division flags, and SHALL render corresponding figures to files, including the frozen 30% hit-rate viability reference on the hit-rate figure.

#### Scenario: Curve tables cover the grid
- **WHEN** the full 30-configuration results load
- **THEN** each of the 6 (model, workload) pairs contributes 5 threshold points to both curve tables, flags preserved

#### Scenario: Figures are produced
- **WHEN** the pipeline runs
- **THEN** figure files for both metrics exist, non-empty, regenerable from the tables alone

### Requirement: Kappa reported by consuming the dataset capability
The D3 bundle SHALL include the Cohen's kappa result sourced from the existing dataset tooling (LEV-3) over the released dataset — κ value, contingency, and pass/fail against the 0.7 bar — and SHALL NOT reimplement the kappa computation. Fixture-derived values SHALL be labeled as fixture-only.

#### Scenario: Kappa section present and sourced
- **WHEN** the pipeline runs with a dataset path available
- **THEN** the bundle contains the kappa report produced by the dataset tooling, labeled with its data provenance

### Requirement: Replication check within ±5%
A replication script SHALL re-run the released harness on a given dataset and grid, compare headline precision and recall per configuration against a reference results file using a documented tolerance rule (±5% relative with a stated absolute floor), exit zero when every value is within tolerance, and on failure exit non-zero with a per-configuration difference table.

#### Scenario: Deterministic re-run replicates exactly
- **WHEN** the script re-runs the harness on identical inputs with mock providers and compares against that run's own reference
- **THEN** every compared value matches and the script exits zero

#### Scenario: Violation is itemized
- **WHEN** a reference value differs beyond tolerance
- **THEN** the script exits non-zero and the diff table names the configuration, metric, both values, and the computed deviation

### Requirement: One command, deterministic outputs
A single command SHALL consume the harness output directory and emit all tables (ANOVA, Tukey, curves, kappa) and figures into a chosen output directory; tables SHALL contain no timestamps and SHALL be byte-identical across re-runs on identical inputs, with environment/version details confined to a metadata sidecar.

#### Scenario: Full bundle from one invocation
- **WHEN** the analysis command runs over a harness output directory
- **THEN** the output directory contains every table, every figure, and the metadata sidecar

#### Scenario: Deterministic tables
- **WHEN** the command runs twice on the same input
- **THEN** all emitted tables are byte-identical

### Requirement: Offline validation on fixtures
The full pipeline SHALL be exercised offline against hand-crafted fixtures with known expected outcomes (including hand-verifiable ANOVA sums of squares for the balanced design), with no network access, keeping the enforced 90% branch-coverage gate green.

#### Scenario: Suite runs offline
- **WHEN** the gated test command runs with no network
- **THEN** all analysis tests pass and the coverage gate passes

