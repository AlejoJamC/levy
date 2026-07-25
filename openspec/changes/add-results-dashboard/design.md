# Design: add-results-dashboard

## Context

Everything through LEV-9 is merged; `openspec/specs/` holds 8 synced capabilities. The analysis bundle contract is concrete and on disk (`results/reproduce/analysis/`): `curves_hit_rate.csv` and `curves_precision.csv` with columns `model, workload, threshold, metric, value, zero_div, n`, plus `anova.csv`, `tukey.csv`, `tukey_status.csv`, `kappa.json`, `analysis_meta.json`; the harness directory above it holds `results.csv`, `decisions.csv`, `run_meta.json`. `results/` is gitignored, so no bundle ships with the repo. `pandas` and `matplotlib` are already installed (LEV-8); `streamlit`, `gradio`, `plotly`, `altair` are all absent. `EmbeddingManager` exposes `embed`, `embed_with`, `get_dimension`, `get_model_identity`, `clear_memoization`. The vector index has **no persistence** — no save/load in `levy/cache/vector_index.py` or `semantic_cache.py`. Coverage is enforced at ≥90% branch over `source = ["levy"]`, with LEV-5's rule that exclusions are intentional and narrow. D6 is the frozen plan's lowest-priority, desirable-only deliverable.

## Goals / Non-Goals

**Goals:**
- Explore threshold-performance curves per (model, workload) from any analysis bundle, with degenerate cells visible and the 30% viability line shown.
- Summarise the hypothesis tests and κ from the same bundle.
- Let a user type a query and see the similarity and hit/miss decision under the production threshold semantics.
- Run locally in the `levy` env, offline-capable, with a clear message when no bundle exists.
- Keep the coverage gate green without pragmas.

**Non-Goals:**
- No deployment, hosting, authentication, or multi-user concerns (local-only per the frozen scope).
- No new statistics — the dashboard displays what LEV-8 computed; it never recomputes ANOVA/Tukey/κ.
- No index persistence feature, no engine/cache/analysis behaviour changes.
- No second charting stack; no replacement for LEV-8's static figures (those remain the publication artefacts).
- Not a substitute for any essential deliverable — this ships only if time permits.

## Decisions

1. **Framework: Streamlit.** The frozen doc allows either; the work here is *exploration* (selectors driving tables and charts across three panels), which is Streamlit's core idiom, whereas Gradio is built around function-in/function-out demo I/O and would fit only the query panel. Two concrete tie-breakers: `st.cache_resource` gives exactly the "build the index once per selected model" behaviour this needs, and `st.pyplot` renders LEV-8's existing matplotlib output directly, so no new charting dependency. *Alternative rejected:* Gradio (would need Blocks scaffolding for the tabular panels and adds queue machinery that a local app does not need).
2. **The core must not import the UI framework.** `levy/dashboard/` holds pure functions — bundle loading/validation, curve selection, query decision — with no `streamlit` import, so it is testable headlessly and the framework stays a swappable shell. The UI entry point lives at `scripts/dashboard.py` (run as `streamlit run scripts/dashboard.py -- --bundle <dir>`; the `--` separator is a Streamlit requirement worth documenting). This mirrors the repo's established split: logic in `levy/`, thin shells in `scripts/`.
3. **Coverage is satisfied structurally, not by exemption.** Because `source = ["levy"]`, the tested logic in `levy/dashboard/` counts and is covered; the UI shell in `scripts/` was never in the denominator — the same position the dataset/harness/analysis CLIs already occupy. No `# pragma: no cover`, no `omit` entry. If any UI-only helper starts accumulating logic, it moves into `levy/dashboard/` and gets tests rather than an exclusion.
4. **The query demo reuses the production `SemanticCache`, it does not reimplement similarity.** The app instantiates a real `SemanticCache` with the selected model and threshold, stores the dataset's queries through it, then submits the user's text — so `1/(1+L2)` on L2-normalised vectors, the k=1 search, and the `similarity >= threshold` comparison are literally the production code path. Thresholds are used verbatim on that scale (never rescaled), consistent with the frozen sweep. *Alternative rejected:* computing cosine similarity in the UI — would silently diverge from the semantics every reported number in the study depends on.
5. **Index built in-process at startup, memoised per model.** Since nothing persists an index, the app embeds the dataset's queries when a model is selected and caches that index with `st.cache_resource`, so changing only the threshold does not rebuild it (thresholds are a comparison, not an index property). Brute-force exact search is the default backend for the demo: the dataset is small, and exactness avoids explaining HNSW approximation in a UI. *Alternative rejected:* rebuilding per interaction (unusable latency with real models).
6. **Bundle schema has a single source of truth.** The loader takes its expected columns/files from the analysis package's own definitions rather than re-declaring them, so a change in LEV-8's writers surfaces as a load error here instead of a silently mis-parsed table — the same anti-drift rule LEV-8 applied between its reader and the harness writer.
7. **Missing or partial bundle is a first-class state.** Because `results/` is gitignored, "no bundle" is the *default* experience for a fresh clone: the app validates the directory and, if files are missing, shows what is absent and the exact command to produce it (`scripts/reproduce.sh`), then stops cleanly. The headless loader raises a typed error carrying the same information so tests assert it without the UI.
8. **Provenance is displayed, not assumed.** The fixture bundle's metrics are degenerate under mock embeddings (zero hit rates, `zero_div` flags set, κ labelled `FIXTURE ONLY` by LEV-8). The UI surfaces the provenance from `analysis_meta.json` / `kappa.json` as a visible banner and renders the `zero_div` flags on the curves, so a demo can never be mistaken for research results.
9. **Dependency placement:** `streamlit` goes in `environment.yml` (conda-forge) so the acceptance criterion "runs inside the `levy` conda env" holds with no extra step, mirrored as a `dashboard` optional extra in `pyproject.toml` for pip users. Only the chosen framework is added — no plotly/altair.

## Risks / Trade-offs

- [Streamlit's rerun-per-interaction model makes an unmemoised index build feel broken] → `st.cache_resource` keyed on the model; first load with real sentence-transformers is documented as slow (model download + embedding pass), instant thereafter.
- [Adding Streamlit pulls a non-trivial dependency tree into the env and the Docker image] → Accepted for a desirable deliverable and small next to the torch already present; kept to a single framework, and the pipeline container path does not invoke the dashboard.
- [A viewer over degenerate fixture numbers could be screenshotted as if it were results] → Decision 8 makes provenance and zero-division flags visible in the UI itself, not just in the files.
- [Scope creep into "another analysis tool"] → The dashboard only displays what the bundle contains; any new statistic belongs to LEV-8. Stated as a non-goal so review can enforce it.
- [D6 competing for time with essentials] → Proposal and tasks both state its desirable-only status; it is last in the queue by design and safe to drop.

## Migration Plan

Purely additive: new package, new UI script, one dependency, docs. No existing behaviour changes and nothing else imports the dashboard. Rollback = delete `levy/dashboard/`, the UI script, and the dependency line.

## Open Questions

- None blocking. Whether the query panel ever offers a side-by-side comparison of both embedding models is a nice-to-have deferred until the base panels are working.
