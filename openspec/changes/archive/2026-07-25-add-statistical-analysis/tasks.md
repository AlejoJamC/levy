# Tasks: add-statistical-analysis

## 1. Dependencies

- [x] 1.1 Add `pandas`, `statsmodels`, `matplotlib` to `environment.yml` (conda-forge) and `pyproject.toml`; install into the `levy` env and verify imports (scipy/scikit-learn already present — verify, don't re-add).

## 2. Analysis package

- [x] 2.1 `levy/analysis/io.py`: harness-contract loader — read `results.csv`/`decisions.csv`/`run_meta.json`, validate required columns, clear error naming any missing column.
- [x] 2.2 `levy/analysis/hypothesis.py`: two-way ANOVA on FPR (`ols('fpr ~ C(model) * C(workload)')` + `anova_lm`) reporting df/sum-sq/F/p per effect + interaction and explicit reject/retain at α=0.05 for H0₁/H0₂/H0₃; conditional Tukey HSD (`pairwise_tukeyhsd`, 6 model×workload cells for the interaction case) with a ran/skipped statement; residual diagnostics into metadata.
- [x] 2.3 `levy/analysis/curves.py`: per-(model, workload) threshold-vs-hit-rate and threshold-vs-precision tidy tables (zero-division flags carried through) + matplotlib figures (PNG + PDF; 30% viability reference line on hit-rate; regenerable from tables).
- [x] 2.4 `levy/analysis/report.py`: bundle assembly — `anova.csv`, `tukey.csv`, `curves_*.csv`, `kappa.json` (sourced from `levy.dataset.kappa` over the dataset path, provenance-labeled; NOT reimplemented), `figures/`, `analysis_meta.json` (inputs, library versions — the only place versions live); all tables timestamp-free and byte-stable.

## 3. Scripts

- [x] 3.1 `scripts/run_analysis.py` (argparse): harness output dir (default `results/` or explicit), dataset path for kappa, `--out-dir`; one invocation emits every table + figure; non-zero exit on contract violation.
- [x] 3.2 `scripts/check_replication.py`: re-run the harness CLI on a given dataset/grid into a temp dir, compare headline precision/recall per configuration against a reference `results.csv` with the documented ±5% relative tolerance + absolute floor; exit zero in-tolerance, else non-zero with per-config diff table.

## 4. Tests (offline, fixtures with known outcomes)

- [x] 4.1 Loader: valid harness dir loads 30 rows; missing column → non-zero, named, no partial outputs.
- [x] 4.2 ANOVA fixtures: constructed model-effect fixture rejects H0₁ (p<0.05); null fixture retains all three; expected values double-sourced (hand-computed balanced-design sums of squares vs statsmodels on independent input, disagreement fails setup).
- [x] 4.3 Tukey: runs with pairwise table on significant fixture; skipped-with-statement on null fixture.
- [x] 4.4 Curves: 6 pairs × 5 thresholds in both tables, flags preserved; figure files exist and are non-empty.
- [x] 4.5 Kappa section: sourced from dataset tooling over the synthetic fixture, provenance labeled fixture-only.
- [x] 4.6 Replication script: self-comparison on a mock-provider run exits zero; perturbed reference exits non-zero with itemized diff.
- [x] 4.7 Determinism: run the analysis command twice on the same input → all tables byte-identical.
- [x] 4.8 Gated suite green: `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` — no new pragmas.

## 5. Docs & sync

- [x] 5.1 README: analysis command, output bundle reference, replication-check usage; state that the pipeline is dataset-agnostic (any harness output directory, fixture or real). CLAUDE.md: architecture map entry for `levy/analysis/` + scripts.
- [x] 5.2 `openspec validate --all` passes; sync Linear LEV-8 (reference this change; tick the acceptance criteria — all of them are satisfiable here, since the pipeline is complete and fixture-validated without any dataset dependency).
