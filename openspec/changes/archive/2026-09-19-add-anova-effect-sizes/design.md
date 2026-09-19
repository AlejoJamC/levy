## Context

`run_two_way_anova` (`levy/analysis/hypothesis.py`) fits `fpr ~ C(model) * C(workload)` with `anova_lm(typ=2)` and emits one row per hypothesis term plus a Residual row. `levy/analysis/report.py` writes that table to `anova.csv` (`float_format="%.10g"`) and embeds only the reject/retain decisions — not the table rows — in `analysis_meta.json`. `levy/dashboard/bundle.py` imports `ANOVA_COLUMNS` as the set of columns a bundle is *required* to have.

The published D3 grid lives in `release/d3-results/` (`results.csv`, `decisions.csv`, `run_meta.json`) with its analysis bundle in `release/d3-results/analysis/`. `results/` is read-only and is neither read nor written by this change.

## Goals / Non-Goals

**Goals:**
- η², partial η² and ω² for model, workload and model:workload, computed where the ANOVA is computed and shipped in `anova.csv`.
- Formulas pinned by a hand-computed, double-sourced test.
- The published analysis regenerated with every pre-existing statistic provably unchanged.

**Non-Goals:**
- No confidence intervals on effect sizes, power analysis, or Cohen's f — beyond what LEV-20 asks.
- No change to the ANOVA model, SS type, α, Tukey logic or diagnostics.
- No new output file or format; no change to the dashboard.

## Decisions

**D1 — SS_total is the corrected total sum of squares of the response (`fitted.centered_tss`), not the sum of the table's rows.** It is the textbook definition and holds for any SS type. For this balanced grid the two coincide (Type II SS partition the total exactly), so the published numbers are the same either way; the choice only matters if a future unbalanced input reaches the function. *Alternative:* summing the Type II rows — rejected, because in an unbalanced design those rows do not partition the total and η² would silently use the wrong denominator.

**D2 — Effect sizes are appended as `eta_sq`, `partial_eta_sq`, `omega_sq`, declared in a separate `EFFECT_SIZE_COLUMNS` list; `ANOVA_COLUMNS` is unchanged.** The emitted table's columns are `ANOVA_COLUMNS + EFFECT_SIZE_COLUMNS`. Appending keeps every existing column at its position, and leaving `ANOVA_COLUMNS` alone keeps the dashboard's required-column contract identical, so bundles written before this change still load. *Alternative:* extending `ANOVA_COLUMNS` — rejected, it would make every existing bundle fail dashboard validation for columns the dashboard never displays.

**D3 — `anova.csv` is the only home for the values.** `analysis_meta.json` stores decisions, not rows; duplicating the numbers there would create two sources that could drift. The regenerated `analysis_meta.json` changes only because it is re-emitted (timestamp, input path).

**D4 — ω² is not clamped.** For a null effect SS_effect can fall below df_effect × MS_residual, giving a negative ω²; that says the term explains less than chance would, which is exactly the information a retained H0₁ needs. Clamping to 0 would hide it.

**D5 — Degenerate and Residual rows carry NaN, written as empty cells.** A zero-variance response makes SS_total = 0; every ratio is then undefined and reported as such, matching the existing rule for undefined F-tests. The Residual row has no effect size by definition. `to_csv` already writes NaN as an empty field, as it does for the Residual row's F and p today.

**D6 — Re-run to scratch, verify, then place only what changed.** The analysis is re-run with the published grid as input (`--results-dir release/d3-results`) into a scratch directory. Before anything touches `release/`:
- `anova.csv`'s pre-existing columns must be byte-identical to the published file (df, SS, MS, F, p, decision unchanged);
- `tukey.csv`, `tukey_status.csv`, `curves_*.csv`, `kappa.json` and the PNG figures must be byte-identical to the published ones.

Only `anova.csv` and `analysis_meta.json` — the files whose content legitimately changes — are then copied into `release/d3-results/analysis/`, and `release/checksums.sha256` is regenerated. The PDF figures are not replaced: `savefig` embeds a creation timestamp, so a re-run changes their bytes with no change in content. *Alternative:* writing the whole bundle straight into `release/` — rejected, it would churn published figures and give no point at which a moved statistic is caught before publication.

## Risks / Trade-offs

- **The residual is not replication error.** The five within-cell observations are the five thresholds, a systematically varied factor, so the residual term absorbs a structured threshold effect the model does not name. Every effect size here inherits that: MS_residual is inflated, so partial η² and ω² for the named terms are pulled down. The values are a bound on what this design could detect, not estimates of a population effect, and must be read that way.
- **η² is upward-biased at small n** (5 observations per cell) → ω² is reported alongside it as the less biased figure, which is why it is included at all.
- **A statistic moves on re-run** → D6's byte-identity gate on the pre-existing columns stops publication and surfaces the difference.
- **An older bundle lacks the new columns** → D2 keeps it loadable; consumers that want effect sizes read them when present.
