# Proposal: add-statistical-analysis

**Linear:** LEV-8 | **Maps to:** Deliverable D3, hypotheses H0₁–H0₃ | **Spec basis:** S&D Report "Statistical analysis" + "Hypotheses to be tested".

## Why

The harness (LEV-4) produces raw per-configuration results, but nothing turns them into the statistical evidence the dissertation needs: the three hypothesis tests on false positive rate, post-hoc comparisons, threshold-selection curves (O3's evidence base), and the ±5% replication check that is Success Criterion 3. This is the last engineering piece of M2; LEV-9 (packaging) and LEV-10 (dashboard) build on its outputs.

**Platform-vs-data split (LEV-3/LEV-11 precedent):** the pipeline is data-agnostic and is built and validated now against the harness output contract using fixtures; the real D3 tables and figures are produced by running the same command once LEV-11 delivers the real 900-pair dataset. The pipeline never waits for the data.

## What Changes

- New `levy/analysis/` package consuming the verified harness contract (`results.csv`: config_id, model, workload, threshold, n, tp/fp/tn/fn, precision, recall, f0_5, fpr, hit_rate + zero-division flags; `decisions.csv`; `run_meta.json`):
  - **Two-way ANOVA** on FPR with factors (embedding model, workload), explicitly testing and reporting p-values for H0₁ (no model main effect), H0₂ (no workload main effect), H0₃ (no interaction).
  - **Tukey HSD** post-hoc comparisons where effects are significant.
  - **Threshold-vs-hit-rate and threshold-vs-precision curves** per (model, workload) as machine-readable tables + publication-quality matplotlib figures.
  - **Kappa reporting**: the D3 bundle includes the Cohen's kappa result by consuming LEV-3's existing, tested `levy/dataset/kappa.py` output — explicitly *not* reimplemented; the κ > 0.7 bar is noted and the real value lands with LEV-11.
  - **Replication check**: a script that re-runs the released harness on the released dataset and verifies headline precision/recall reproduce within ±5% (LEV-4 determinism already guarantees byte-identical re-runs on identical inputs; the tolerance covers real-provider variance).
- One command (`scripts/run_analysis.py`) consumes the harness output directory and emits every table and figure.
- Dependencies (measured in the env): `pandas`, `statsmodels`, `matplotlib` added to `environment.yml` + `pyproject.toml`; `scipy` and `scikit-learn` are already present.
- Fully offline tests on synthetic/hand-crafted results fixtures with known expected outcomes; deterministic tables (no timestamps); the 90% branch-coverage gate stays green.

## Capabilities

### New Capabilities

- `statistical-analysis`: D3 evidence pipeline — hypothesis tests (two-way ANOVA on FPR + Tukey HSD), threshold-selection curves (tables + figures), kappa reporting from the dataset capability's output, and the ±5% replication check, all driven by one command over the harness outputs.

### Modified Capabilities

_None. The pipeline is a pure consumer of the harness and dataset outputs through their existing contracts._

## Impact

- **New code:** `levy/analysis/` (loaders for the harness contract, ANOVA/Tukey, curves, figure writers, replication compare), `scripts/run_analysis.py`, `tests/test_analysis_*.py`.
- **Dependencies:** `pandas`, `statsmodels`, `matplotlib` (new); `scipy`/`scikit-learn` already in the env.
- **Docs:** README (analysis command, outputs, replication check), CLAUDE.md (architecture map; the pipeline is dataset-agnostic and validated on fixtures).
- **Downstream:** LEV-9 packages the pipeline + outputs; LEV-10 visualizes the curve tables.
- **Nothing here is blocked by data production.** This change delivers a complete, tested pipeline. Publishing the real D3 numbers is a later *execution* of the same command against real harness results, tracked as the production run in LEV-11.
