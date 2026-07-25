# Proposal: add-results-dashboard

**Linear:** LEV-10 | **Maps to:** Deliverable D6 (**desirable — only if time permits**) | **Spec basis:** S&D Report §C D6.

## Why

D6 asks for an interactive view of the study results: threshold-performance curves a reader can explore, plus a query box that shows the cache decision for their own text. The analysis pipeline already emits every table this needs, but they are static CSVs and PNG/PDF figures — a reviewer cannot vary a threshold or try a query without re-running scripts.

**Priority discipline:** this is the frozen plan's lowest-priority, explicitly *desirable* deliverable. It must never displace an essential one; if time runs short, dropping it costs nothing in D1–D5.

**No data dependency.** The dashboard is a viewer over the analysis bundle *contract*, not over particular numbers: it renders any directory produced by `scripts/run_analysis.py`. A complete sample bundle already exists on disk from the LEV-9 reproduce run, so this is fully buildable and demonstrable now.

## What Changes

- **`levy/dashboard/`** — the testable core: bundle loading and validation (`curves_hit_rate.csv` / `curves_precision.csv` with `model, workload, threshold, metric, value, zero_div, n`; plus `anova.csv`, `tukey.csv`, `tukey_status.csv`, `kappa.json`, `analysis_meta.json`), curve selection per (model, workload), and the live-query decision (similarity + hit/miss) computed with the production `1/(1+L2)` semantics and the threshold used verbatim.
- **A thin UI entry point outside the coverage source**, following the repo's existing `scripts/` pattern, rendering three panels: threshold-performance explorer per (model, workload) with the harness `zero_div` flags surfaced and the frozen 30% hit-rate viability line; a hypothesis summary (ANOVA / Tukey status / κ) read from the bundle; and a query box that embeds the user's text with the selected model and reports similarity and the hit/miss decision at the chosen threshold.
- **In-process index for the query demo.** Verified: the vector index has no persistence (`levy/cache/vector_index.py` and `semantic_cache.py` expose no save/load) — so the app builds the index at startup by embedding the dataset's queries with the selected model. This satisfies the frozen wording ("queries tested against pre-computed configurations"); the ticket's earlier "pre-built index" phrasing was factually wrong and has been corrected.
- **Missing-bundle handling.** `results/` is gitignored, so no bundle ships in the repo: the app takes a bundle path and, when it is absent or incomplete, exits with an actionable message pointing at `scripts/reproduce.sh` — never a traceback.
- **One new dependency** (the chosen UI framework only). `pandas` and `matplotlib` already ship with the analysis pipeline, so the existing curve tables and rendering are reused rather than adding a second charting stack.
- **Docs**: run command in the README and the reproduction/architecture docs kept accurate, with the offline story stated plainly (fully offline on mock embeddings; sentence-transformers downloads on first use).

## Capabilities

### New Capabilities

- `results-dashboard`: local interactive exploration of an analysis bundle — threshold-performance curves per (model, workload), a hypothesis/κ summary, and a live query demo reporting similarity and the cache decision under the production threshold semantics.

### Modified Capabilities

_None. The dashboard reads the analysis bundle and reuses the embedding/vector-index capabilities through their existing interfaces; no shipped requirement changes._

## Impact

- **New code:** `levy/dashboard/` (loader + decision logic, unit-tested), a UI entry script, `tests/test_dashboard.py`.
- **Dependencies:** one UI framework (decision recorded in design.md) added to `environment.yml` + `pyproject.toml`.
- **Docs:** README section + pointers from `docs/REPRODUCTION.md` / `docs/ARCHITECTURE.md`; CLAUDE.md architecture map entry.
- **Coverage:** logic lives in `levy/` and is tested; the UI shell sits outside `source = ["levy"]` exactly as the `scripts/` CLIs do — no pragmas, no whole-file omits, the ≥90% branch gate stays green.
- **Not touched:** frozen documents; analysis/harness/engine behaviour; essential deliverables.
