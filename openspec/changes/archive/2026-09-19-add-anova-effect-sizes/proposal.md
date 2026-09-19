## Why

The D3 headline is a retained H0₁ — F(1,24) = 0.553, p = 0.465: no effect of embedding model on false positive rate was detected. A p-value alone cannot say how large an effect the design would have missed, and the analysis stage computes no effect size at all, so the central claim currently has no bound on the difference that went undetected (LEV-20).

## What Changes

- The two-way ANOVA stage computes three effect sizes per term (model, workload, model:workload) from the sums of squares it already produces:
  - η² = SS_effect / SS_total
  - partial η² = SS_effect / (SS_effect + SS_residual)
  - ω² = (SS_effect − df_effect × MS_residual) / (SS_total + MS_residual), reported unclamped — a negative ω² for a null effect is informative, not an error
- The values ship in the existing machine-readable ANOVA table (`anova.csv`) as additional columns; no new output file or format. The Residual row carries no effect size.
- A degenerate response (zero variance in FPR) reports the effect sizes as undefined (empty), consistent with the existing rule that undefined F-tests are never laundered into retained nulls.
- A unit test pins all three formulas against a hand-computed balanced-design fixture, double-sourced like the existing sum-of-squares check.
- The analysis is re-run over the published D3 grid (`release/d3-results/`), regenerating `release/d3-results/analysis/`; df, sum of squares, F and p must come out unchanged.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `statistical-analysis`: the two-way ANOVA requirement additionally reports η², partial η² and ω² per hypothesis term in the machine-readable table, with defined behaviour for negative ω² and for a degenerate response.

## Impact

- **Code:** `levy/analysis/hypothesis.py` (`run_two_way_anova`). The effect-size columns are appended to the emitted table but kept out of `ANOVA_COLUMNS`, which `levy/dashboard/bundle.py` uses as its *required* column set — so bundles written before this change still load there. No column is removed or renamed.
- **Tests:** `tests/test_analysis_hypothesis.py` gains the effect-size fixture check; existing tests unchanged.
- **Published artefacts:** `release/d3-results/analysis/anova.csv` (new columns) and `analysis_meta.json` (regenerated), with `release/checksums.sha256` regenerated to match. `results/` is read only and never written.
- **No dependency changes** — the values are arithmetic over quantities `anova_lm` already returns.
