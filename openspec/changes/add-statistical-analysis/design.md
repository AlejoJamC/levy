# Design: add-statistical-analysis

## Context

Verified inputs (live smoke run this session): the LEV-4 harness writes `results.csv` with columns `config_id, model, workload, threshold, n, tp, fp, tn, fn, precision, recall, f0_5, fpr, hit_rate, precision_zero_div, recall_zero_div, fpr_zero_div` (one row per configuration, deterministic, no timestamps), `decisions.csv` (per-pair audit), and `run_meta.json` (providers, checkpoints, grid). The frozen S&D prescribes: two-way ANOVA on FPR with factors (embedding model, workload); Tukey HSD where interactions are significant; Cohen's kappa over the 900 pairs (κ > 0.7); replication of headline precision/recall within ±5%. Kappa computation already exists and is tested (LEV-3, `levy/dataset/kappa.py` + `scripts/compute_kappa.py`). Env facts: scipy and scikit-learn installed; pandas, statsmodels, matplotlib absent. LEV-5's 90% branch-coverage gate is enforced. LEV-11 (real data) is the only blocker, and it gates numbers, not code.

## Goals / Non-Goals

**Goals:**
- One command: harness output directory in → all D3 tables (CSV) and figures (files) out.
- H0₁/H0₂/H0₃ explicitly tested on FPR with reported F-statistics and p-values; Tukey HSD when significant.
- Threshold-vs-hit-rate and threshold-vs-precision curve tables + figures per (model, workload).
- Kappa result included in the bundle by consuming LEV-3's tooling output — no reimplementation.
- Replication script validating the ±5% criterion against a reference results file.
- Pipeline fully validated offline on fixtures now; identical invocation later on real results.

**Non-Goals:**
- No dashboard (LEV-10) and no packaging (LEV-9).
- No harness changes — pure consumer of its contract.
- No new statistics beyond the frozen plan (no extra tests, corrections, or effect-size inventions); anything the frozen plan under-specifies is surfaced, not silently added.
- No real-data conclusions in this change — fixture outputs validate machinery only.

## Decisions

1. **statsmodels for ANOVA + Tukey; pandas as the table backbone; matplotlib for figures.** Two-way ANOVA *with interaction* needs a linear-model ANOVA table — `statsmodels` `ols('fpr ~ C(model) * C(workload)')` + `anova_lm` is the standard, auditable path, and `statsmodels` also provides `pairwise_tukeyhsd`. scipy alone covers neither the two-factor interaction test nor labeled post-hoc output cleanly. These three are exactly the additions the issue names; scipy/sklearn stay as already-installed deps. *Alternative rejected:* hand-rolled ANOVA on numpy — reviewable-formula appeal, but error-prone for interaction sums of squares and impossible to justify over the reference implementation in a dissertation's methods section.
2. **The ANOVA design follows the frozen plan literally.** Observations = the 30 per-configuration FPR values; factors = model (2 levels) × workload (3 levels); the 5 thresholds per cell are the replicates the frozen grid provides. The pipeline reports the ANOVA table (df, sum-sq, F, p per effect + interaction) and rejects/retains each H0 at α=0.05, stated explicitly in the output. Any statistical caveat about this design is dissertation-space (supervisor), not code-space — the code implements the frozen plan.
3. **Tukey HSD runs conditionally, per the frozen wording**: executed for factors whose effect (or the interaction) is significant; the output table always states whether it ran and why. Groups for the interaction case are the 6 (model, workload) cells.
4. **Curves come straight from `results.csv`**: for each (model, workload), threshold vs hit_rate and threshold vs precision as tidy CSV tables plus one figure per metric (2 models × 3 workloads as labeled series; horizontal reference line at the frozen 30% hit-rate viability bar on the hit-rate figure). Figures are written as files (PNG + PDF); no interactive display. Zero-division flags from the harness are carried into the curve tables so degenerate cells are visible, never hidden.
5. **Kappa: consume, don't compute.** The bundle includes a kappa section sourced from LEV-3's tooling (invoke `levy.dataset.kappa.kappa_report` over the released dataset file, or ingest `compute_kappa.py`'s JSON report), reporting κ, the 2×2 contingency, and pass/fail against 0.7. On the synthetic fixture this is labeled fixture-only; the real value arrives with LEV-11. *Rejected:* reimplementing kappa in the analysis package — duplicate logic, drift risk, and LEV-3's is already hand-verified.
6. **Replication check = re-run + tolerance compare, as a script.** `scripts/check_replication.py`: runs the harness CLI on the given dataset/grid into a temp directory, then compares headline precision and recall per configuration against a reference `results.csv`, passing when every value is within ±5% (relative, with an absolute floor for near-zero values — both reported); non-zero exit and a per-config diff table on failure. Byte-identical determinism (LEV-4-tested) makes the mock-path comparison exact; the tolerance exists for real-provider runs per Success Criterion 3.
7. **Determinism and layout mirror LEV-4's conventions:** all tables timestamp-free and byte-stable for identical inputs; outputs under a single `--out-dir` (`anova.csv`, `tukey.csv`, `curves_hit_rate.csv`, `curves_precision.csv`, `kappa.json`, `figures/`); an `analysis_meta.json` sidecar records input paths and library versions (the one place versions/timestamps may live). Package layout: `levy/analysis/io.py` (contract loader + validation), `hypothesis.py` (ANOVA/Tukey), `curves.py`, `report.py` (bundle assembly), driven by `scripts/run_analysis.py`.
8. **Fixture strategy for tests:** hand-crafted `results.csv` fixtures with known outcomes — a constructed dataset where model A's FPR is uniformly higher (H0₁ must reject; direction visible in Tukey), a null dataset with identical cell means (all H0 retained; Tukey skipped), and degenerate cells exercising the zero-division flags. ANOVA/Tukey expected values are cross-checked against statsmodels' own worked outputs and, where feasible, hand-computed sums of squares for the balanced 2×3×5 design. Figures are asserted by file existence + non-emptiness and by the underlying table values (not pixel comparison).

## Risks / Trade-offs

- [Adding pandas/statsmodels/matplotlib grows the env] → They are the issue-named, spec-implied stack for D3; recorded in `environment.yml` so the env stays reproducible.
- [Figure aesthetics are subjective ("publication-quality")] → Tables are the machine-readable source of truth; figures are regenerable from them, and styling is confined to one module so the author can iterate without touching statistics.
- [Fixture ANOVA expectations could encode a wrong hand computation] → Expectations are double-sourced (hand-computed for the balanced design AND cross-checked against statsmodels on independent input); disagreement fails loudly in test setup.
- [Real-data run may violate ANOVA assumptions (normality/variance)] → The pipeline reports the standard diagnostics available from the model residuals in `analysis_meta.json`; interpreting them is dissertation-space. Nothing is auto-"corrected" beyond the frozen plan.
- [±5% comparison ambiguity near zero] → Explicit rule: relative tolerance with documented absolute floor; both the rule and every compared value appear in the script's output, so the criterion is auditable.

## Migration Plan

Pure addition (`levy/analysis/`, two scripts, three dependencies, docs). Nothing existing changes. Rollback = remove the package/scripts/dependency lines.

## Open Questions

- None blocking. Which harness run becomes the "released reference" for the replication check is decided at LEV-11/LEV-9 time; the script takes the reference path as an argument, so it does not depend on that decision.
