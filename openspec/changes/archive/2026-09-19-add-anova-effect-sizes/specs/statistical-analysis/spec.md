## MODIFIED Requirements

### Requirement: Two-way ANOVA testing the three frozen hypotheses
The pipeline SHALL fit a two-way ANOVA on false positive rate with factors (embedding model, workload) including their interaction, and SHALL report, for H0₁ (no model main effect), H0₂ (no workload main effect), and H0₃ (no interaction): degrees of freedom, sum of squares, F-statistic, p-value, and an explicit reject/retain statement at α = 0.05, as a machine-readable table. The same table SHALL report, for each of the three terms, the effect sizes η² = SS_effect / SS_total, partial η² = SS_effect / (SS_effect + SS_residual), and ω² = (SS_effect − df_effect × MS_residual) / (SS_total + MS_residual), computed from the fitted sums of squares. ω² SHALL be reported unclamped, including negative values. The Residual row SHALL carry no effect size, and a degenerate response (zero variance in false positive rate) SHALL report every effect size as undefined rather than as a number.

#### Scenario: Constructed model effect is detected
- **WHEN** the input fixture has one model's FPR uniformly higher across workloads
- **THEN** the ANOVA table rejects H0₁ with p below 0.05 and reports the supporting F-statistic

#### Scenario: Null fixture retains all hypotheses
- **WHEN** the input fixture has identical FPR distributions across all cells
- **THEN** all three H0 are explicitly retained with their p-values reported

#### Scenario: Effect sizes match the hand-computed formulas
- **WHEN** the ANOVA runs on a balanced fixture whose sums of squares are known by hand
- **THEN** η², partial η² and ω² for model, workload and interaction equal the values computed by hand from those sums of squares

#### Scenario: Negative omega squared is reported, not clamped
- **WHEN** a term's sum of squares is smaller than df_effect × MS_residual
- **THEN** its ω² is reported as the negative value the formula yields

#### Scenario: Degenerate response leaves effect sizes undefined
- **WHEN** every configuration reports the same false positive rate
- **THEN** every effect size in the table is undefined (empty), alongside the undefined F-tests

#### Scenario: Adding effect sizes changes no existing statistic
- **WHEN** the analysis is re-run over a harness output directory that was analysed before this change
- **THEN** df, sum of squares, mean square, F, p-value and decision for every row are identical to the earlier table
