# Tasks: add-results-dashboard

> **Desirable deliverable (D6).** Lowest priority in the plan — do not start or continue this at the expense of any essential deliverable; it is safe to drop entirely.

## 1. Dependency

- [x] 1.1 Add `streamlit` to `environment.yml` (conda-forge) and mirror as a `dashboard` optional extra in `pyproject.toml`; install into the `levy` env and verify `streamlit --version`. Add nothing else (pandas/matplotlib already ship with the analysis pipeline).

## 2. Testable core (`levy/dashboard/`, no UI imports)

- [x] 2.1 `levy/dashboard/bundle.py`: load and validate an analysis bundle (`curves_hit_rate.csv`, `curves_precision.csv`, `anova.csv`, `tukey.csv`, `tukey_status.csv`, `kappa.json`, `analysis_meta.json`), taking expected files/columns from the analysis package's own definitions; raise a typed `BundleNotFoundError`/`BundleContractError` naming missing paths/columns and the command that generates a bundle (`scripts/reproduce.sh`).
- [x] 2.2 `levy/dashboard/curves.py`: selection helpers — available models/workloads/thresholds from the bundle, and the (model, workload) slice for each metric with `zero_div` flags and `n` preserved.
- [x] 2.3 `levy/dashboard/query.py`: build a `SemanticCache` for a chosen model + threshold, populate it from the dataset's queries (in-process index; brute-force exact backend by default), and evaluate user text → nearest match, similarity, hit/miss. Reuse the production path; do not reimplement similarity or rescale thresholds.
- [x] 2.4 Confirm no module under `levy/dashboard/` imports `streamlit`.

## 3. UI shell (`scripts/`, outside the coverage source)

- [x] 3.1 `scripts/dashboard.py`: Streamlit app with `--bundle` (and optional `--dataset`) parsed from `sys.argv` after the `--` separator; three panels — threshold explorer (both metrics per selected model/workload, degenerate points marked, 30% viability reference), hypothesis/κ summary read from the bundle, and the query box.
- [x] 3.2 Memoise the in-process index with `st.cache_resource` keyed on the selected model so a threshold change re-evaluates without re-embedding.
- [x] 3.3 Missing/incomplete bundle → render the typed error's message (missing paths + `scripts/reproduce.sh`) and stop cleanly; never surface a traceback.
- [x] 3.4 Render the bundle provenance (fixture-only labelling, providers from `analysis_meta.json`) as a visible banner.

## 4. Tests (offline, headless)

- [x] 4.1 Bundle loading: valid bundle (use a generated fixture bundle) loads all sections; missing file and missing column each raise the typed error with the actionable message.
- [x] 4.2 Curve selection: available options enumerated from the bundle; each (model, workload) slice returns the expected thresholds with `zero_div`/`n` preserved.
- [x] 4.3 Query decision: constructed near-duplicate → hit with reported similarity; unrelated text → miss with nearest similarity; the same index re-evaluated at a stricter threshold flips hit→miss with no re-embedding (assert via a counting/mock embedding client).
- [x] 4.4 Semantics parity: the decision matches a direct `SemanticCache` query for the same inputs (guards against divergence from production).
- [x] 4.5 Gated suite green: `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` — no new pragmas or omit entries.

## 5. Author verification & docs

- [x] 5.1 Run the app locally against `results/reproduce/analysis` with mock embeddings (fully offline): confirm curve exploration, the summary panel, and the query demo; then spot-check with a real sentence-transformers model and note first-run download time.
- [x] 5.2 Document the run command (`streamlit run scripts/dashboard.py -- --bundle <dir>`), the mock-vs-real offline story, and the "generate a bundle first" prerequisite in README; link from `docs/REPRODUCTION.md` and add the component to `docs/ARCHITECTURE.md` + the CLAUDE.md map. Keep D6's desirable status explicit in user-facing docs.
- [x] 5.3 `openspec validate --all` passes; sync Linear LEV-10 (reference this change, tick delivered acceptance criteria, record the Streamlit-over-Gradio decision and the in-process-index correction).
