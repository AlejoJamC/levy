## 1. Effect-size computation

- [x] 1.1 Add `EFFECT_SIZE_COLUMNS = ["eta_sq", "partial_eta_sq", "omega_sq"]` to `levy/analysis/hypothesis.py`, leaving `ANOVA_COLUMNS` unchanged
- [x] 1.2 In `run_two_way_anova`, take SS_total from `fitted.centered_tss` and MS_residual from the Residual row, and compute η², partial η² and ω² per term (ω² unclamped)
- [x] 1.3 Set all three to NaN on the Residual row, and on every row when the response is degenerate (SS_total = 0)
- [x] 1.4 Emit the table with columns `ANOVA_COLUMNS + EFFECT_SIZE_COLUMNS`

## 2. Tests

- [x] 2.1 Pin η², partial η² and ω² for model, workload and interaction against hand-computed values on the balanced model-effect fixture (SS 0.30 / 0.20 / 0.0, residual 0.006 on 24 df)
- [x] 2.2 Double-source SS_total: `fitted.centered_tss` equals the sum of the hand-computed sums of squares on a non-additive design
- [x] 2.3 Assert a negative ω² is reported, not clamped (the zero-SS interaction term)
- [x] 2.4 Assert the degenerate fixture reports every effect size as NaN and the Residual row carries none
- [x] 2.5 Assert the emitted table keeps every pre-existing column in its original position, and that a bundle without the new columns still loads in `levy/dashboard/bundle.py`
- [x] 2.6 Gated suite passes (`--cov=levy --cov-branch --cov-fail-under=90`)

## 3. Re-run over the published D3 grid

- [x] 3.1 Run `scripts/run_analysis.py --results-dir release/d3-results` into a scratch directory
- [x] 3.2 Verify the regenerated `anova.csv`'s pre-existing columns are identical to `release/d3-results/analysis/anova.csv` (df, sum of squares, mean square, F, p-value, decision)
- [x] 3.3 Verify `tukey.csv`, `tukey_status.csv`, `curves_*.csv`, `kappa.json` and the PNG figures are byte-identical to the published copies
- [x] 3.4 Copy the regenerated `anova.csv` and `analysis_meta.json` into `release/d3-results/analysis/`; leave the published PDF figures in place
- [x] 3.5 Regenerate and verify `release/checksums.sha256`, and confirm nothing under `results/` was written or referenced

## 4. Validation

- [x] 4.1 `openspec validate --all` passes
