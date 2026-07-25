# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What Levy is

Levy is a **semantic caching engine for LLM APIs**, built as the IT artefact of an
MSc Artificial Intelligence capstone project (University of Liverpool, author:
John Alejandro Mantilla Celis). It sits between an application and an LLM provider
and reuses responses for exact or semantically similar prompts, in order to measure
cost, latency, and — centrally — **false positive rates** of semantic caching across
workloads, embedding models, and similarity thresholds.

## Source of truth — READ FIRST, NEVER MODIFY

These two documents were submitted to the university. They are **FROZEN**:
do not edit, rename, move, reformat, or "fix typos" in them under any circumstance.
They are the authoritative definition of the research questions, methodology,
metrics, and deliverables. When any other file (including this one, the README,
or code) contradicts them, the frozen documents win for *research scope*;
flag the conflict instead of silently resolving it.

| Document | Role |
|---|---|
| `docs/Project_Proposal.md` | **IMMUTABLE.** Aims, objectives (O1–O4), research questions, deliverables (D1–D3), phase plan (Weeks 12–40), risks, budget. |
| `docs/Specification_and_Design_Report.md` | **IMMUTABLE.** Full specification and design: hypotheses (H0₁–H0₃), component architecture, algorithms (cache lookup, replay harness), API contract, statistical analysis plan, deliverables D1–D7. |

Everything else in the repo — all code, tests, examples, configs, and the remaining
docs — **is open to change**. The project now has new goals, schedule, and
deliverables built *on top of* the frozen baseline, so working docs and code evolve
freely as long as they don't rewrite the submitted documents.

## Key research parameters (from the frozen docs)

- **Primary question:** does embedding model selection meaningfully impact false
  positive rates in semantic caching across production LLM workloads?
- **Experimental grid:** 2 embedding models (`all-MiniLM-L6-v2` baseline vs
  `ModernBERT`) × 3 workloads (FAQ, code generation, conversational chat) ×
  5 similarity thresholds (0.70–0.90, step 0.05) = **30 configurations**.
- **Metrics:** precision, recall, **F-score with β=0.5** (precision-weighted),
  false positive rate, hit rate. Statistical analysis: two-way ANOVA + Tukey HSD;
  Cohen's kappa > 0.7 for annotation validity.
- **Success criteria:** measurable precision differences between models; hit rate
  **> 30%** for economic viability; replication within ±5%.
- **Dataset:** 900 query pairs (300 per workload) from public human-annotated
  corpora (Quora Question Pairs, Stack Overflow duplicates, ConvAI2) with the
  author's blind re-annotation. Not yet present in the repo.
- **Target stack (per spec):** FastAPI router, sentence-transformers embeddings,
  Faiss HNSW index, Anthropic SDK backend, scipy/numpy/pandas/scikit-learn,
  pytest. Licence: Apache 2.0.

## Documentation map (where knowledge lives)

| File | Status | Content |
|---|---|---|
| `docs/Project_Proposal.md` | FROZEN | Research baseline (see above). |
| `docs/Specification_and_Design_Report.md` | FROZEN | Design baseline (see above). |
| `docs/ARCHITECTURE.md` | Living, **user-facing** | The released system's architecture: component map traced to the frozen spec's named components, request flow, experiment flow, ABC+mock provider pattern. **This file (CLAUDE.md) links to it and must not duplicate it** — CLAUDE.md is the agent-facing orientation, ARCHITECTURE.md is what a third party reads. |
| `docs/REPRODUCTION.md` | Living, **user-facing** | D5 reproduction guide: Docker one-command path and the conda step-by-step path, expected outputs, the "swap in the real dataset" section (changes only `--dataset`), and the release checklist. Its commands come from `scripts/reproduce.sh`. |
| `docs/RESEARCH_OVERVIEW.md` | Historical (flagged in place) | Early research framing (CSCK508 module). Predates the proposal; its 12-week timeline and RAG workload were superseded by the frozen docs. Editable. |
| `docs/LITERATURE_REVIEW.md` | Working skeleton (flagged in place) | Paper list + research-gaps matrix. Editable, expand as needed. |
| `docs/PLANNING_HIERARCHY.md` | Working note (flagged in place) | Vision → Epic → Feature → Story → Task hierarchy used to plan work. |
| `docs/epics/EPIC-001-client-proxy-layer.md` | Historical planning (flagged in place) | Pre-implementation epic for the client/proxy layer, which shipped as `levy/api/`. Kept for provenance; `docs/ARCHITECTURE.md` and the code are authoritative. Pattern for future epics (`EPIC-00X-*.md`). |
| `README.md` | Living | User-facing install/usage docs. Keep in sync with code. **No ticket identifiers in user-facing headings** — identifiers belong in CLAUDE.md, OpenSpec, and git history. |
| `data/README.md`, `data/DATASHEET.md` | Living | What is currently in `data/` (synthetic fixtures) and the full D2 datasheet. |
| `openspec/` | Living | OpenSpec spec-driven workflow: capability specs + change proposals (see "Spec-driven workflow" below). |
| `CLAUDE.md` (this file) | Living | Orientation + ground rules for every session. |

## Code architecture (current state)

Package `levy/` — plain Python dataclasses, synchronous, provider-pluggable:

- `levy/engine.py` — `LevyEngine`, the orchestrator. Flow per `generate(prompt)`:
  exact cache → semantic cache → LLM call → store (with embedding when semantic
  cache is enabled). Records metrics at each step.
- `levy/config.py` — `LevyConfig` dataclass. Providers: `llm_provider` =
  `mock | openai | ollama`; `embedding_provider` = `mock | sentence-transformers |
  ollama`; `cache_store_type` = `memory | redis`. Loads `.env` via python-dotenv.
- `levy/models.py` — dataclasses: `LLMRequest`, `LLMResponse`, `CacheEntry`,
  `LevyResult` (`source` ∈ `llm | exact_cache | semantic_cache`), `MetricsSnapshot`.
- `levy/llm_client.py` — `LLMClient` ABC + `MockLLMClient` (0.5s sleep, reversed
  echo), `OpenAILLMClient` (raw httpx), `OllamaLLMClient`, `AnthropicLLMClient`
  (LEV-6) — synchronous wrapper around the official `anthropic` SDK
  (`messages.create`), selected via `llm_provider="anthropic"`. Retry is the
  SDK's own exponential backoff (`anthropic_max_retries`), not reimplemented;
  non-retryable errors propagate as the SDK's typed exceptions. Token usage
  (`response.usage.input_tokens`/`output_tokens`) populates `LLMResponse.token_usage`
  (sum) and `metadata` (split + model + stop_reason). A per-instance `_BudgetGuard`
  accumulates request count and estimated cost (tokens × configurable per-MTok
  prices) and raises `BudgetExceededError` before sending once the estimate
  reaches `anthropic_budget_cap_usd` (default 200.0, the frozen cap). A
  `stop_reason: "refusal"` response raises `AnthropicRefusalError` instead of
  being cached. Missing `ANTHROPIC_API_KEY` fails at construction. Fully
  offline-testable via an injectable `http_client` (`anthropic.DefaultHttpxClient`
  wrapping `httpx.MockTransport`) — no coverage pragmas needed.
- `levy/embeddings.py` — `EmbeddingClient` ABC + mock (text-seeded random,
  normalized), `SentenceTransformerClient` (accepts `trust_remote_code` for
  ModernBERT), `OllamaEmbeddingClient`.
- `levy/embedding_manager.py` — **`EmbeddingManager`** (LEV-1): resolves study-model
  aliases (`all-MiniLM-L6-v2` / `modernbert`) via a built-in registry, lazily loads
  and caches one `EmbeddingClient` per checkpoint, memoizes embeddings by
  `(model_key, sha256(text))`, applies symmetric task prefixes per model (e.g.
  `search_query: ` for ModernBERT), and exposes `embed()`, `embed_with()`,
  `get_dimension()`, `get_model_identity()`, `clear_memoization()`. The engine
  constructs one manager from `LevyConfig` and all caches go through it. Supports
  `mock`, `sentence-transformers`, and `ollama` providers.
- `levy/cache/` — `base.py` (`CacheInterface` ABC), `exact_cache.py` (SHA-256 of
  prompt as key; stores model identity in `CacheEntry.metadata`),
  `vector_index.py` (LEV-2) — `VectorIndex` ABC + `BruteForceVectorIndex` (numpy
  exact k-NN, offline default and correctness oracle) + `FaissHNSWVectorIndex`
  (`IndexHNSWFlat` wrapped in `IndexIDMap`, returns L2 distances after sqrt);
  `make_vector_index()` factory honours `vector_index_backend` config;
  `semantic_cache.py` (LEV-2) — owns a `VectorIndex` + monotonic id→`CacheEntry`
  map; retrieval uses `similarity = 1/(1+L2_distance)` per Algorithm 1 of the
  frozen S&D; all embeddings L2-normalised before indexing/querying for
  cross-model comparability; `reset()` empties index + map for per-config sweeps;
  `store.py` (`InMemoryStore`, FIFO eviction), `redis_store.py`
  (JSON-serialized entries, duck-types `InMemoryStore`; `KEYS *` + `MGET` scan for
  the semantic path — prototype-only).
- `levy/metrics.py` — `LevyMetrics`: hits by type, misses, tokens saved
  (whitespace-split approximation), latency list.
- `tests/test_levy.py` — 2 unittest tests (exact cache hit/miss, semantic
  machinery smoke test) using mock providers.
- `tests/test_anthropic_client.py` — 13 unit tests for `AnthropicLLMClient` (LEV-6):
  construction/missing-key, success (text/token_usage/metadata), retry-then-success,
  non-retryable propagation, refusal handling (incl. engine end-to-end — nothing
  cached), budget-guard halt + spend visibility, engine wiring end-to-end. Fully
  offline via `httpx.MockTransport` injected as the SDK's `http_client`.
- `tests/test_embedding_manager.py` — 20 unit tests for `EmbeddingManager`: runtime
  model switching, alias resolution, memoization, dimension/identity exposure, prefix
  handling, and default config validation. All offline (injected mock clients).
- `tests/test_vector_index.py` — 19 unit tests for `VectorIndex` + `SemanticCache`:
  add/search/reset/size, L2 normalization, zero-vector guard, similarity transform +
  threshold decisions, id→entry resolution, Faiss↔brute-force agreement (skipped
  when Faiss absent), engine end-to-end semantic cache hit/miss.
- `examples/simple_replay.py` — replays a prompt list under no-cache / exact /
  exact+semantic configs. `examples/ollama_demo.py` — end-to-end with local Ollama
  (`qwen3` LLM + `nomic-embed-text` embeddings). `examples/anthropic_smoke_check.py`
  — one-shot real-API smoke check for the Anthropic connector (billed, requires a
  real `ANTHROPIC_API_KEY`; not collected by pytest — lives outside `tests/`).
- `levy/dataset/` (LEV-3) — ground-truth dataset **platform tooling** (D2), data-agnostic:
  `schema.py` (`QueryPair` dataclass + workload constants `faq`/`code`/`chat` +
  validation; `ground_truth_label()` returns the author's blind label if set, else the
  original corpus label — the eval contract LEV-4 replays against); `io.py` (CSV/JSON
  load/save, round-trip and cross-format identical; `metadata` JSON-encoded into one CSV
  column); `sampling.py` (`CorpusSource` ABC + `QuoraQQPSource` / `StackOverflowDuplicatesSource`
  / `ConvAI2Source` adapters, each documenting its expected local raw file format — no
  network access — + `MockCorpusSource`; seeded, stratified, deterministic
  `sample_workload`/`sample_dataset`); `annotation.py` (`BlindAnnotationSession` — shows
  only `query_1`/`query_2`, never the original label; per-answer progress persistence for
  resumable 900-pair sessions; never overwrites an existing `author_label` unless
  `overwrite=True`); `kappa.py` (`cohen_kappa`/`kappa_report`, stdlib-only 2×2
  contingency, documented zero-annotated and `pe==1` edge cases). `scripts/*.py`
  (`sample_dataset.py`, `annotate_dataset.py`, `compute_kappa.py`, `export_dataset.py`)
  are thin argparse CLIs over this package, runnable fully offline.
- `data/` — `ground_truth.csv` + `ground_truth.json` currently hold **15 synthetic
  fixture pairs** (5/workload, obviously fake text, `source_corpus="synthetic-fixture"`)
  standing in for the real 900-pair dataset; `data/README.md` documents the placeholder
  status, `data/DATASHEET.md` is the full D2 datasheet skeleton (source corpora +
  licences, sampling protocol, annotation guidelines, kappa result placeholder, fallback
  corpora) with `TODO (post data-production)` markers only where the real
  sampling/annotation run is required.
- `tests/test_dataset.py` — 48 unit tests for `levy/dataset/`: schema validation, CSV/JSON
  round-trip + cross-format equality, sampling determinism/stratification, blind
  annotation (blindness, resume, no-overwrite), Cohen's kappa (perfect/chance/worked/
  degenerate cases), and CLI smoke tests against the `data/` fixtures. All offline.
- `levy/experiment/` (LEV-4) — offline replay harness per S&D Report Algorithm 2:
  `config.py` (`ExperimentConfig` + `full_grid()`, the frozen 2 models × 3 workloads ×
  5 thresholds = 30 configurations, thresholds carried verbatim on the `1/(1+L2)` scale);
  `metrics.py` (`EvaluationResult`/`DecisionRecord`, precision/recall/F0.5/FPR/hit-rate
  formulas, zero-division reported as `0.0` + flag never NaN, `check_sanity()` raising
  `ExperimentSanityError` naming the offending configuration); `replay.py` —
  `run_experiment(config, pairs)` builds a fresh `LevyEngine` per configuration (mock
  LLM, memory store), replays each `QueryPair` through the production lookup path
  (`query_1` miss-and-store, `query_2` hit/miss decision via `LevyResult.source`),
  accumulates the cache across pairs within a configuration, and increments TP/FP/TN/FN
  against `QueryPair.ground_truth_label()`; `runner.py` — `run_sweep()` shares one
  `EmbeddingManager` per model across its 15 configurations (LEV-1 memoization is not
  defeated by the sweep) and writes `results.csv` / `decisions.csv` (no timestamps or
  latency, so re-runs on identical inputs are byte-identical) plus a `run_meta.json`
  sidecar (dataset path, providers, resolved model checkpoints, grid, latency labeled
  synthetic under the mock LLM's fixed 0.5s sleep). `LevyEngine` accepts an optional
  `embedding_manager` constructor param for this sharing; default behavior unchanged.
- `scripts/run_experiments.py` — argparse CLI over `levy/experiment/runner.py`: dataset
  path (default `data/ground_truth.csv`), output directory, `--models/--workloads/
  --thresholds` grid-subset flags for smoke runs, `--embedding-provider` (default
  `mock`, fully offline against the synthetic fixture; pass `sentence-transformers` for
  a real study run once LEV-11 lands). Non-zero exit on a sanity-check failure.
- `tests/test_experiment_config.py`, `test_experiment_metrics.py`,
  `test_experiment_replay.py`, `test_experiment_runner.py` — 37 unit tests for
  `levy/experiment/`: grid enumeration/uniqueness, hand-computed metrics + zero-division
  + sanity-check violations, replay outcomes (TP/FP/TN/FN via a scripted embedding
  manager, exact-duplicate via the exact cache, cross-pair cache accumulation, fresh
  cache per `run_experiment` call), sweep determinism (byte-identical re-runs), and
  output-contract shape. All offline (mock LLM + mock/scripted embeddings).
- `levy/api/` (LEV-7) — FastAPI router exposing the engine over HTTP per the frozen
  S&D "Intended interface": `app.py` (`create_app(config, max_engines)` factory +
  module-level `app`; `POST /v1/chat/completions`, `GET /admin/cache/stats`,
  `POST /admin/cache/clear`; sync `def` endpoints run in FastAPI's threadpool —
  the async-at-boundary decision recorded in known-gap #1 above); `schemas.py`
  (Pydantic v2 request/response/error models — the only Pydantic in the repo, per
  convention); `pool.py` (`EnginePool` keyed by `(embedding_model, threshold)`,
  default cap 8, `PoolCapExceededError` beyond it; shares one `EmbeddingManager`
  per embedding_model across thresholds so a model isn't reloaded for a threshold
  change). Response body is Anthropic Messages-shaped for hit and miss alike;
  cache identity lives in `X-Cache-Status`/`X-Cache-Similarity` headers, not the
  body. `BudgetExceededError`/`AnthropicRefusalError`/`PoolCapExceededError`/any
  other exception map to structured JSON (402/502/400/500) via FastAPI exception
  handlers, not stack traces. Each chat request emits one JSON log record
  (`levy.api` logger) with `request_id`, timestamps, resolved cache config,
  prompt, cache decision, similarity, and latency — sufficient to replay a
  request sequence. Run via `uvicorn levy.api.app:app`.
- `tests/test_api_router.py` — 16 unit tests for `levy/api/`: hit/miss flows
  (exact + semantic, body-shape parity), per-request `cache_config` isolation/
  accumulation/defaults, pool-cap breach, admin stats/clear consistency,
  validation/budget/refusal/generic error mapping, structured-log record
  completeness and replayability, OpenAPI contract shape. Fully offline via
  FastAPI's `TestClient` + mock providers (Anthropic scenarios use the same
  `httpx.MockTransport` injection pattern as `test_anthropic_client.py`).
- `levy/analysis/` (LEV-8) — D3 statistical analysis pipeline, a pure consumer of
  the LEV-4 harness contract and **dataset-agnostic** (any harness output directory,
  fixture or real): `io.py` (`load_harness_outputs` over `results.csv`/`decisions.csv`/
  `run_meta.json`; required-column lists imported from `levy.experiment.runner` so the
  reader can't drift from the writer; `HarnessContractError` names the missing column
  and nothing is written on violation); `hypothesis.py` (two-way ANOVA
  `ols('fpr ~ C(model) * C(workload)')` + `anova_lm(typ=2)` — balanced design, so
  SS types coincide — reporting df/sum-sq/F/p and explicit `reject`/`retain` at
  α=0.05 for H0₁/H0₂/H0₃, plus conditional `pairwise_tukeyhsd` over 2 models /
  3 workloads / the 6 model×workload cells, always with a ran-or-skipped-and-why
  statement; residual diagnostics — design balance, Shapiro-Wilk, Levene — reported,
  never acted on. **Degenerate-response rule:** zero variance in `fpr` (what the
  synthetic fixture yields under mock embeddings) makes the F-tests undefined, so
  decisions are reported as `undefined`, never as retained nulls);
  `curves.py` (tidy threshold-vs-hit-rate / threshold-vs-precision tables per
  (model, workload) with the harness zero-division flags carried through, plus
  Agg-backend PNG+PDF figures regenerable from the tables alone, 30% viability line
  on hit-rate); `replication.py` (±5% relative tolerance with a documented 0.01
  absolute floor for near-zero references — `max(floor, rel·|ref|)` — and an
  itemized per-configuration diff table); `report.py` (bundle assembly: `anova.csv`,
  `tukey.csv`, `tukey_status.csv`, `curves_*.csv`, `kappa.json`, `figures/`,
  `analysis_meta.json`). **Kappa is consumed, not recomputed:** the section calls
  LEV-3's `levy.dataset.kappa.kappa_report` and labels fixture-derived values
  `FIXTURE ONLY`. All CSVs are timestamp-free and byte-stable; versions and the
  generation timestamp live only in `analysis_meta.json`.
- `scripts/run_analysis.py` — argparse CLI over `build_analysis_bundle`: one
  invocation emits every table and figure; non-zero exit on a contract or design
  violation, with no partial bundle written. `scripts/check_replication.py` —
  re-runs the harness over exactly the grid recorded in a reference `results.csv`
  (dataset defaults to that run's `run_meta.json`), compares precision/recall,
  exits non-zero with the diff table when out of tolerance.
- **Release packaging (LEV-9)** — `scripts/reproduce.sh` is the **single definition
  of the evaluation pipeline** (`run_experiments.py` → `run_analysis.py` →
  `check_replication.py`, `set -euo pipefail`, offline defaults: fixture dataset +
  mock embeddings; overridable via positional args or `LEVY_DATASET` /
  `LEVY_OUT_DIR` / `LEVY_EMBEDDING_PROVIDER` / `LEVY_MODELS` / `LEVY_WORKLOADS` /
  `LEVY_THRESHOLDS`). `docs/REPRODUCTION.md` shows those same commands and the
  container's `CMD` invokes the script — so a flag change breaks the script rather
  than silently staling the guide. **Do not restate the pipeline commands anywhere
  else.** `Dockerfile` builds a micromamba image from `environment.yml` (the single
  dependency source, keeping conda-forge `faiss-cpu`); `docker-compose.yml` gained a
  `pipeline` service (`docker compose run --rm pipeline`) with the pre-existing
  `redis:7-alpine` service untouched and not a dependency of the default path.
  Verified: `--network none`, no `ANTHROPIC_API_KEY`, exit 0, `results.csv`/
  `decisions.csv` byte-identical to the host conda run; cold build 5m34s, image
  16.4 GB (torch, via sentence-transformers). Docker is outside the pytest suite by
  design. `scripts/audit_release.sh` is the re-runnable release audit (LICENSE, no
  tracked secret file, 9 credential patterns over tracked files *and* over all git
  history via `git log --all --pickaxe-regex -S`, personal-data markers in `data/`,
  `.env` gitignored); pass/fail per check, non-zero exit on any finding, failure path
  verified with a planted secret. `docs/ARCHITECTURE.md` is the user-facing
  architecture doc (see the documentation map).
- `tests/test_analysis_io.py`, `test_analysis_hypothesis.py`, `test_analysis_curves.py`,
  `test_analysis_report.py`, `test_analysis_replication.py` — 75 unit tests for
  `levy/analysis/` sharing `tests/analysis_fixtures.py` (hand-crafted 30-row harness
  outputs whose ANOVA outcome is known in advance). ANOVA expectations are
  **double-sourced**: the balanced-design sums of squares are computed by hand in the
  tests and compared against statsmodels, so a disagreement fails rather than being
  trusted. Covers loader contract violations, model-effect / interaction / null /
  degenerate fixtures, Tukey ran-and-skipped paths, curve shape + flags + figure
  files, kappa provenance, byte-identical re-runs, and replication pass/fail. All
  offline.
- **Results dashboard (LEV-10, D6 — desirable, lowest priority; safe to drop)** —
  `levy/dashboard/` is the testable, framework-independent core: `bundle.py`
  (loads/validates an analysis bundle, with required files/columns imported
  from `levy.analysis` itself rather than re-declared, so a writer change
  surfaces as a load error here; typed `BundleNotFoundError`/
  `BundleContractError` naming the gap and `scripts/reproduce.sh`);
  `curves.py` (available models/workloads/pairs, per-(model, workload) curve
  slices with `zero_div`/`n` preserved); `query.py` (`build_query_index` /
  `evaluate_query` — builds a real `SemanticCache` from the dataset's queries
  and evaluates user text via the same `VectorIndex.search()` + `1/(1+L2)`
  formula `SemanticCache.get()` uses, extended only to surface the nearest
  match on a miss too; threshold is passed per call so re-evaluating never
  re-embeds the dataset). No module here imports Streamlit. `scripts/
  dashboard.py` is the thin Streamlit shell (`streamlit run scripts/
  dashboard.py -- --bundle <dir>`): threshold-performance explorer (both
  metrics, degenerate points marked, 30% viability line), a hypothesis/κ
  summary read verbatim from the bundle, and a live query box (`mock` /
  `sentence-transformers` provider choice, the latter documented as
  needing a first-run download). A missing/incomplete bundle renders the
  typed error's message and stops cleanly, never a traceback; provenance
  (fixture-only labelling, providers) is a visible banner. `tests/
  test_dashboard.py` — 17 unit tests, fully offline, headless (no Streamlit
  process): bundle load/validation (valid bundle, missing file, missing
  column), curve selection, query decision (near-duplicate hit, unrelated
  miss, threshold-flip-without-re-embedding via a counting embedding-manager
  double), and semantics parity against a direct `SemanticCache` query.

### Known gaps: current code vs frozen spec

Track these when building toward the experimental phase — they are the backlog
implied by the spec, not bugs:

1. ~~**No FastAPI router**~~ — **Resolved (LEV-7).** `levy/api/` exposes
   `POST /v1/chat/completions` (`X-Cache-Status` / `X-Cache-Similarity`
   headers, Anthropic-format body for hit and miss alike), `GET
   /admin/cache/stats`, and `POST /admin/cache/clear`. **Async decision:**
   endpoints are declared `def` (sync), so FastAPI runs them in its
   threadpool — the whole call chain (engine, caches, the LEV-6 Anthropic
   client) stays synchronous; this satisfies the frozen "asynchronous
   wrapper" intent at the HTTP boundary (concurrent request handling)
   without an `AsyncAnthropic` migration. Recorded resolution, not silent
   drift — see `openspec/changes/add-fastapi-router/design.md`.
2. ~~**No Anthropic LLM connector**~~ — **Resolved (LEV-6).** `AnthropicLLMClient`
   wraps the official `anthropic` SDK behind the existing synchronous `LLMClient`
   ABC, selected via `llm_provider="anthropic"`. **Sync-now decision:** the frozen
   S&D calls for an "asynchronous wrapper", but the whole core engine (caches,
   harness) is synchronous; this change implements the connector synchronously
   against the existing ABC and defers async to the FastAPI router (LEV-7), where
   the SDK's `AsyncAnthropic` client fits naturally — recorded as a documented
   resolution, not silent drift (see `openspec/changes/add-anthropic-connector/design.md`).
   **Model default drift:** the frozen S&D's example model string
   (`claude-3-sonnet-20240229`) is retired; the connector defaults to
   `claude-opus-4-8` instead — flagged here per the frozen-docs rule, not silently
   resolved.
3. ~~**No Faiss HNSW index**~~ — **Resolved (LEV-2).** `SemanticCache` now owns a
   `VectorIndex` (Faiss HNSW or brute-force oracle) and uses `similarity =
   1/(1+L2_distance)` per Algorithm 1. **Threshold-scale flag for LEV-4/LEV-8:**
   all embeddings are L2-normalised before indexing so the distance scale is
   identical across models. For unit vectors, `distance = sqrt(2 − 2·cosine)` and
   `similarity = 1/(1+distance)`. The frozen sweep 0.70–0.90 therefore covers a
   high-cosine band (~0.91–0.998). This is intentional and spec-mandated; do NOT
   rescale thresholds or revert to cosine. If hit-rate viability (>30%) is not
   met at this band, surface that as a research-scope finding to the supervisor.
4. ~~**No experiment harness**~~ — **Resolved (LEV-4).** `levy/experiment/` implements
   `run_experiment`/`full_grid`/30-configuration replay, TP/FP/TN/FN accounting against
   `QueryPair.ground_truth_label()`, and precision/recall/F0.5/FPR/hit-rate computation
   with zero-division-safe formulas and sanity checks. `scripts/run_experiments.py`
   drives the full grid (or a subset) fully offline via the mock LLM; results are
   validated against the committed 15-pair synthetic fixture only — a real run still
   needs LEV-11's 900-pair dataset and `sentence-transformers` providers.
5. ~~**Embedding defaults don't match the study**~~ — **Resolved (LEV-1).**
   `LevyConfig` now defaults to `sentence-transformers` / `all-MiniLM-L6-v2`;
   `EmbeddingManager` supports runtime switching to `modernbert`
   (`nomic-ai/modernbert-embed-base`) with symmetric task-prefix handling.
6. ~~**No annotated dataset**~~ — **Platform tooling resolved (LEV-3).** `levy/dataset/`
   + `scripts/` implement the schema, CSV/JSON loader (the LEV-4 contract), seeded
   stratified sampling, blind re-annotation, and Cohen's kappa. **Still open (author
   task, tracked in `openspec/changes/add-ground-truth-dataset/tasks.md` §7):** the real
   900-pair sample from Quora QQP / Stack Overflow duplicates / ConvAI2 (or an approved
   fallback corpus), the author's actual blind re-annotation of all 900 pairs, the final
   Cohen's kappa result, and replacing the synthetic fixtures currently in `data/` with
   the released dataset.
7. ~~**pytest declared but not installed**~~ — **Resolved (LEV-5).** `pytest` and
   `pytest-cov` are installed in the `levy` conda env (`environment.yml`, conda-forge)
   and mirrored in `pyproject.toml` `[dev]` extras. pytest is the canonical runner;
   `python -m unittest discover` still works but is no longer advertised as the
   default. A gated command (`--cov=levy --cov-branch --cov-fail-under=90`) enforces
   ≥90% branch coverage on `levy/`, with network-only provider internals
   (OpenAI/Ollama LLM clients, Ollama/SentenceTransformer embedding clients) excluded
   via inline `# pragma: no cover` markers. `MockLLMClient` latency is now injectable
   (`latency_seconds`, default 0.5 unchanged); tests inject 0, so the suite runs in
   ~2s instead of the previous ~81s.

## Commands

**Everything Python in this repo runs inside the `levy` conda env.** Dependencies
(numpy, httpx, sentence-transformers, redis, dotenv) are installed there and
nowhere else — a bare `python`/`pip` outside the env will fail with missing
modules. Claude Code's shell does NOT inherit the activated env, so prefix every
Python command:

```bash
# Activate first (conda run -n levy may hit shell-profile permission issues):
source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy && <command>
```

```bash
# Environment (conda, env name: levy)
conda env create -f environment.yml
conda activate levy

# Tests — pytest is canonical (installed in the conda env + pyproject [dev] extras)
python -m pytest tests/ -q                                              # fast run
python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90  # gated run (used in CI)
# unittest still works (tests are plain unittest.TestCase):
python -m unittest discover -s tests -p "test_*.py"

# Demos
python examples/simple_replay.py     # mock LLM; uses sentence-transformers if installed
python examples/ollama_demo.py       # requires `ollama serve` + qwen3 + nomic-embed-text
python examples/anthropic_smoke_check.py  # one real, billed call; requires ANTHROPIC_API_KEY in .env

# HTTP API (LEV-7) — reads .env for the configured provider's credentials
uvicorn levy.api.app:app --reload

# Whole evaluation pipeline in one command (LEV-9) — the canonical entry point.
# Offline by default (fixture dataset + mock embeddings); prefer this over
# retyping the three stages below.
scripts/reproduce.sh                          # -> results/reproduce/
scripts/reproduce.sh data/ground_truth.csv results/run-001

# Same pipeline in the container (builds from environment.yml; ~5.5 min cold, 16.4 GB):
docker compose run --rm pipeline

# Individual stages (LEV-4 sweep, LEV-8 analysis), fully offline:
python scripts/run_experiments.py --out-dir results/run-001
python scripts/run_analysis.py --results-dir results/run-001 --out-dir results/run-001/analysis
python scripts/check_replication.py --reference results/run-001/results.csv  # ±5% criterion

# Release audit (LICENSE, secrets in tree + all git history, personal data)
scripts/audit_release.sh

# Results dashboard (LEV-10, D6 — desirable): a bundle must exist first (any
# command above that writes an analysis/ dir); the `--` separator is required
# by Streamlit so --bundle/--dataset reach the script's own argparse.
streamlit run scripts/dashboard.py -- --bundle results/reproduce/analysis

# Local services (Redis 7 for cache_store_type="redis") — unchanged by the pipeline service
docker compose up -d redis
```

Secrets live in `.env` (gitignored; template in `.env.example`). Never commit
API keys.

## Spec-driven workflow (OpenSpec)

The repo uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) (CLI installed
via Homebrew at `/opt/homebrew/bin/openspec`, scaffold initialized) for planning
and tracking changes. New features should go through this flow instead of ad-hoc
edits:

- `openspec/specs/` — living capability specs (the working spec layer, built *on
  top of* the frozen university docs; they must never contradict the frozen
  research scope). Currently **8 capabilities**, one per shipped capability:
  `embedding-management`, `vector-store`, `ground-truth-dataset`,
  `experiment-harness`, `test-infrastructure`, `anthropic-connector`,
  `api-router`, `statistical-analysis`.
  **Main specs use main-spec structure** — `# <name> Specification`, a
  `Capability:` line, `## Purpose`, `## Requirements` — *never* delta headers
  (`## ADDED Requirements`) and never a `TBD` Purpose. `openspec archive` creates
  the file with a `TBD` placeholder; fill it in the same step. Getting this wrong
  fails `openspec validate --all` and has had to be repaired twice.
- `openspec/changes/` — in-flight change proposals (`proposal.md`, `design.md`,
  `tasks.md` per change); completed changes move to `openspec/changes/archive/`.
  Archived so far: `add-embedding-manager`, `add-faiss-vector-store`,
  `add-experiment-harness`, `add-test-infrastructure`, `add-anthropic-connector`,
  `add-fastapi-router`, `add-statistical-analysis`. **Still in flight:**
  `add-ground-truth-dataset` — its tooling shipped and its capability is synced
  into `openspec/specs/`, but §7 (the real 900-pair data production) is an open
  author task, so the change stays in flight; `add-release-packaging`; and
  `add-results-dashboard` (tasks complete, pending archive).
- `openspec/config.yaml` — project context injected into artifact generation.
- Slash commands (in `.claude/commands/opsx/`): `/opsx:propose` (create change +
  artifacts), `/opsx:apply` (implement tasks), `/opsx:archive` (finish + update
  specs), `/opsx:explore` (think through ideas), `/opsx:sync` (reconcile specs).
- Useful CLI: `openspec list`, `openspec status --change <name>`,
  `openspec validate --all`.

### Linear ↔ OpenSpec mapping

The engineering backlog lives in Linear (team **Levy Project**, project
**"Levy — Capstone IT Artefact"**). Each Linear issue maps 1:1 to an OpenSpec
change; the issue description carries the spec basis, scope, and acceptance
criteria that seed the change's `proposal.md`. Milestones: M1 Experiment-Ready
(2026-06-21), M2 Experiments & Analysis (2026-08-09), M3 Public Artefact
Release (2026-11-02).

| Linear | OpenSpec change | Priority | State |
|---|---|---|---|
| LEV-1 | `add-embedding-manager` | Urgent | archived |
| LEV-2 | `add-faiss-vector-store` | Urgent | archived |
| LEV-3 | `add-ground-truth-dataset` | Urgent | **in flight** — tooling shipped, §7 real-data production open |
| LEV-4 | `add-experiment-harness` | Urgent | archived |
| LEV-5 | `add-test-infrastructure` | Urgent | archived |
| LEV-6 | `add-anthropic-connector` | High | archived |
| LEV-7 | `add-fastapi-router` | High | archived |
| LEV-8 | `add-statistical-analysis` | High | archived |
| LEV-9 | `add-release-packaging` | Medium | **in flight** |
| LEV-10 | `add-results-dashboard` | Low (desirable) | **in flight** — tasks complete, not yet archived |
| LEV-11 | — (production run: real dataset + published D2/D3 outputs) | — | not started |

Critical path: LEV-1 → LEV-2 → LEV-4 → LEV-8, with LEV-3 feeding LEV-4.
When an OpenSpec change is created or archived, reference its Linear issue
and keep the issue status in sync.

## Conventions

- Python ≥ 3.10, dataclasses over Pydantic in the core package (EPIC-001 plans
  Pydantic for the API layer), synchronous code so far.
- Provider abstraction: every external dependency (LLM, embeddings, store) has an
  ABC plus a mock implementation, so tests and demos run with zero external
  services. Keep this pattern when adding Anthropic/Faiss/FastAPI.
- The mock-first design is deliberate: experiments must be reproducible offline.
- Work is planned as Epics → Features → Stories (see `docs/PLANNING_HIERARCHY.md`);
  new epics go in `docs/epics/`.
- Licence is Apache 2.0; the code and dataset will be released publicly, so keep
  the repo free of personal/sensitive data.
