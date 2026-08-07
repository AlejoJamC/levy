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
  corpora with the author's blind re-annotation. **Produced 2026-08-04.** The
  corpora actually used are Quora Question Pairs, **SODD** and **Twitter
  PIT-2015** — the frozen docs name Stack Overflow duplicates and ConvAI2; both
  substitutions are recorded as deviations in `data/DATASHEET.md` §2. Published as
  `data/ground_truth.ids.csv` (identifiers + labels, no query text — licence
  constraint); the query text is never committed and is rebuilt locally by
  `scripts/rehydrate_dataset.py`.
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
| `docs/DATA_PRODUCTION.md` | Living, **author-facing** | The D2 production runbook: the ordered acquire → pin → sample → rehydrate → annotate → kappa → commit procedure, including the two manual corpus downloads (Kaggle QQP, SODD's Drive folder) and the QQP CSV→TSV conversion. **The single place this sequence is written down** — `data/README.md` and `docs/REPRODUCTION.md` link to it rather than restating it. Procedure only; licences, protocol rationale and deviations live in `data/DATASHEET.md`. |
| `docs/RESEARCH_OVERVIEW.md` | Historical (flagged in place) | Early research framing (CSCK508 module). Predates the proposal; its 12-week timeline and RAG workload were superseded by the frozen docs. Editable. |
| `docs/LITERATURE_REVIEW.md` | Working skeleton (flagged in place) | Paper list + research-gaps matrix. Editable, expand as needed. |
| `docs/PLANNING_HIERARCHY.md` | Working note (flagged in place) | Vision → Epic → Feature → Story → Task hierarchy used to plan work. |
| `docs/epics/EPIC-001-client-proxy-layer.md` | Historical planning (flagged in place) | Pre-implementation epic for the client/proxy layer, which shipped as `levy/api/`. Kept for provenance; `docs/ARCHITECTURE.md` and the code are authoritative. Pattern for future epics (`EPIC-00X-*.md`). |
| `README.md` | Living | User-facing install/usage docs. Keep in sync with code. **No ticket identifiers in user-facing headings** — identifiers belong in CLAUDE.md, OpenSpec, and git history. |
| `data/README.md`, `data/DATASHEET.md`, `data/raw/README.md` | Living | What is committed vs. generated in `data/`, the acquire→sample→rehydrate sequence, the full D2 datasheet (corpora, licences, the three recorded frozen-doc deviations, ids-only distribution), and the per-corpus acquisition layout. `data/corpora.json` is the machine-readable provenance registry those docs point at — **read by code, so do not restate its URLs, filenames or checksums elsewhere**. |
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
- `tests/test_embedding_manager.py` — 27 unit tests for `EmbeddingManager`: runtime
  model switching, alias resolution, memoization, dimension/identity exposure, prefix
  handling, and default config validation. All offline (injected mock clients).
- `tests/test_vector_index.py` — 27 unit tests for `VectorIndex` + `SemanticCache`:
  add/search/reset/size, L2 normalization, zero-vector guard, similarity transform +
  threshold decisions, id→entry resolution, Faiss↔brute-force agreement (skipped
  when Faiss absent), engine end-to-end semantic cache hit/miss.
- `examples/simple_replay.py` — replays a prompt list under no-cache / exact /
  exact+semantic configs. `examples/ollama_demo.py` — end-to-end with local Ollama
  (`qwen3` LLM + `nomic-embed-text` embeddings). `examples/anthropic_smoke_check.py`
  — one-shot real-API smoke check for the Anthropic connector (billed, requires a
  real `ANTHROPIC_API_KEY`; not collected by pytest — lives outside `tests/`).
- `levy/dataset/` (LEV-3, extended by LEV-12) — ground-truth dataset **platform
  tooling** (D2), data-agnostic. Nothing here touches the network:
  `schema.py` (`QueryPair` dataclass + workload constants `faq`/`code`/`chat` +
  validation; `ground_truth_label()` returns the author's blind label if set, else the
  original corpus label — the eval contract LEV-4 replays against; plus
  `DistributionRecord`, the ids-and-labels record with **no query text** —
  deliberately a separate type, because `QueryPair`'s non-empty-text invariant is
  what LEV-4's replay path relies on and must not be weakened); `io.py` (CSV/JSON
  load/save, round-trip and cross-format identical; `metadata` JSON-encoded into one CSV
  column; plus `save/load_distribution_csv` + `to_distribution_records` on their own
  code path with their own `DISTRIBUTION_FIELDNAMES` — and `load_dataset` **rejects** a
  distribution file with "rehydrate first" rather than replaying absent text);
  `corpora.py` (LEV-12 — reader for `data/corpora.json`, the machine-readable
  provenance registry: URL, snapshot, licence, filenames, SHA-256, citation per
  corpus; `sha256_file`, `pin_checksums`; typed `CorpusRegistryError` naming the
  missing corpus or field); `normalize.py` (LEV-12 — stdlib `HTMLParser`
  normaliser for SODD's HTML posts, keeping code-block text; stdlib deliberately,
  because BeautifulSoup/lxml output can shift across versions and would silently
  break the rehydration round-trip); `sampling.py` (`CorpusSource` ABC +
  `QuoraQQPSource` / `SODDSource` (gzipped parquet, streamed per row-group) /
  `TwitterPIT2015Source` (tab-separated, train+dev only — the graded 0–5 test split
  is rejected, not coerced) + `MockCorpusSource`; every adapter declares its
  `label_mapping()`, `options()` and a cheap `check_fields()`; `make_source()`
  factory driven by the registry's `adapter` field; seeded, stratified,
  deterministic `sample_workload`/`sample_dataset`); `validation.py` (LEV-12 —
  pre-flight pass returning a `ValidationReport` of **every** finding rather than
  raising on the first: presence, checksum, required fields, per-class pool
  sufficiency for all three workloads at once, label domain, cross-workload corpus
  overlap. Overlap is keyed on `corpus_key`, not the display name, so three
  independent `MockCorpusSource`s are not a false positive); `annotation.py`
  (`BlindAnnotationSession` — shows only `query_1`/`query_2`, never the original
  label; per-answer progress persistence for resumable 900-pair sessions; never
  overwrites an existing `author_label` unless `overwrite=True`; **presentation
  order** is workload blocks in `DEFAULT_WORKLOAD_ORDER` = `faq,chat,code` — code
  last, longest posts — shuffled within each block under `order_seed` via
  `Random(f"{seed}:{workload}")` so a workload's order is independent of which
  others are in scope; the resolved order and seed are persisted in the progress
  file and **reused on resume**, never re-derived mid-session; `session_limit`
  ends a sitting cleanly after N labeled pairs (skips don't count); every recorded
  label is fingerprinted with its pair's `source_pair_id`, so labels for a pair
  that has since been re-sampled away are dropped and counted, not re-applied —
  progress-file v2, with the flat v1 `{pair_id: label}` map still read;
  `prune_progress()` is the **primary** defence, called by `sample_dataset.py`
  when it re-samples — the fingerprint is only the second line and cannot help a
  v1 file, which has no fingerprints to compare, and v1 is exactly what
  `data/annotation_progress.json` currently is);
  `backup.py` (`backup_file`/`backup_files` — copy to `<dir>/backups/<stem>.<UTC
  basic ISO 8601><suffix>` before any overwrite, shared timestamp across a
  multi-file write, never delete, never clobber a same-second backup, `BackupError`
  = caller writes nothing); `workload_update.py` (per-workload re-sampling of the
  live dataset: `ExcludingCorpusSource` wraps an adapter to hide already-sampled
  `source_pair_id`s **before** pre-flight so a shortfall is caught rather than
  discovered; `splice_workload` replaces one workload's rows in place, passing every
  other row through *by identity* so their `author_label`s are untouched by
  construction; `check_ids_alignment` refuses to proceed when the ids file and the
  working dataset disagree about the untouched workloads — identity only, since
  annotation legitimately updates the working dataset first); `kappa.py`
  (`cohen_kappa`/`kappa_report`, stdlib-only 2×2 contingency, documented
  zero-annotated and `pe==1` edge cases). `scripts/*.py` (`sample_dataset.py`,
  `rehydrate_dataset.py`, `annotate_dataset.py`, `compute_kappa.py`,
  `export_dataset.py`) are thin argparse CLIs over this package, runnable fully
  offline. **`scripts/fetch_corpora.py` is the sole exception** — the only
  network-touching entry point in the repository, excluded from pytest by design
  and guarded by `TestAcquisitionIsOutOfBand`.
- **The single ground truth (2026-08-06).** There is exactly one ground-truth
  dataset, at `data/ground_truth.ids.csv` + `.ids.meta.json` (published) and
  `data/ground_truth.full.{csv,json}` (working, gitignored). Never create a
  parallel, versioned or suffixed dataset — no `_v2`, `_new`, `_final`, `.bak`,
  no date in a working filename, no "candidate" beside the real one. Every
  re-sample and re-annotation writes back to those paths, and every script that
  overwrites a ground-truth or results artifact backs it up to
  `data/backups/` (or `<out-dir>/backups/`) first via `levy/dataset/backup.py`,
  aborting without writing if the backup fails. Backups are never deleted.
  `data/backups/` is gitignored in both `.gitignore` files.
- **Per-workload re-sampling (2026-08-06).** `scripts/sample_dataset.py
  --workload chat` (repeatable) resolves and reads **only** that workload's
  corpus — the others need not be on disk — and operates in place: it replaces
  that workload's rows inside the existing dataset, clears `author_label` for the
  new pairs only, carries the other workloads' rows and labels through untouched,
  and excludes every `source_pair_id` already in the dataset so the new sample is
  disjoint from what it replaced (a pool that can't cover `--n-per-workload`
  after exclusion fails naming the workload and the shortfall, writing nothing).
  It also **invalidates the annotation progress for the replaced pairs**
  (`--progress`, default `annotation_progress.json` beside `--out-csv` — derived
  from the dataset's directory so a temp-dir run cannot reach into `data/`);
  without that, the next session would re-apply the old answers to the new pairs,
  since a re-sample reuses that workload's `pair_id`s. The sidecar records, **per
  workload**, the seed and UTC timestamp of the run that produced that workload's
  rows (backfilled from the previous sidecar's top level, and flagged
  `provenance_backfilled`, for entries written before this existed); the top-level
  `seed` describes only the latest invocation. `scripts/annotate_dataset.py --workload chat` restricts the
  session to it. Procedure and guarantees: `docs/DATA_PRODUCTION.md`
  §"Re-sampling one workload after the fact".
- **Corpus acquisition + licence-safe distribution (LEV-12)** — the study's query
  text is **never committed**: Quora QQP grants no redistribution right and SODD is
  CC BY-NC-SA 4.0. `scripts/fetch_corpora.py` acquires into `data/raw/<corpus>/`
  (idempotent, checksum-verified, `.part`-then-rename so a mismatch never looks
  acquired, `--pin` to record checksums once, and a non-zero exit printing URL +
  filename + expected SHA-256 for corpora needing a human step — Quora requires
  accepting terms, SODD is a Google Drive folder). `scripts/rehydrate_dataset.py`
  rebuilds `data/ground_truth.full.{csv,json}` from `data/ground_truth.ids.csv` +
  its sidecar + `data/raw/`, **byte-identical** to what was originally sampled
  (ordering comes from the ids file, adapter options from the sidecar — nothing is
  re-derived). `scripts/sample_dataset.py` gained `--require-real` (refuses the
  mock fallback), registry-driven corpus resolution under `--raw-dir`, the
  pre-flight gate, and writes the ids file + sidecar alongside the full dataset.
  **BREAKING:** `StackOverflowDuplicatesSource` / `ConvAI2Source` and the
  `--stackoverflow-csv` / `--convai2-json` flags are gone.
- `data/` — `ground_truth.csv` + `ground_truth.json` hold **15 synthetic
  fixture pairs** (5/workload, obviously fake text, `source_corpus="synthetic-fixture"`)
  and **stay**: the offline test suite and `scripts/reproduce.sh` defaults depend on
  them, and synthetic text carries no third-party licence. `corpora.json` is the
  provenance registry; `raw/` is the acquisition target (directories tracked via
  `.gitkeep`, contents gitignored); `ground_truth.ids.csv` + `ground_truth.ids.meta.json`
  are the released D2 artifact and its run manifest; `ground_truth.full.{csv,json}`
  are the rehydrated working dataset and are gitignored. `data/README.md` documents
  what is committed vs. generated and the acquire→sample→rehydrate sequence;
  `data/DATASHEET.md` is the D2 datasheet (corpora + licences, sampling protocol,
  the three recorded deviations from the frozen docs, ids-only distribution model,
  limitations) with `TODO (post data-production)` markers only where the real
  sampling/annotation run is required.
- `tests/test_dataset.py` — 93 unit tests for `levy/dataset/`: schema validation
  (including a test that pins `QueryPair`'s non-empty-text invariant against future
  relaxation), CSV/JSON round-trip + cross-format equality, the three corpus adapters
  against committed fixtures (field/label mapping, `hard_negatives`, debatable-pair
  exclusion, graded-test-split rejection, out-of-domain labels), the `make_source`
  factory, sampling determinism/stratification, blind annotation (blindness, resume,
  no-overwrite), Cohen's kappa (perfect/chance/worked/degenerate cases), and CLI smoke
  tests against the `data/` fixtures. All offline.
- `tests/test_corpus_acquisition.py` (LEV-12) — 79 unit tests: provenance-registry
  reader (every malformed-registry path), checksum pinning, the validation report
  (two simultaneous problems both reported, pool shortfall across all three workloads
  at once, cross-workload overlap, out-of-domain label, checksum mismatch, nothing
  written on failure), the ids-only distribution format (round-trip, no text in any
  field, harness rejection), HTML normalisation (malformed markup, code blocks,
  entities), and the full sample→distribute→rehydrate round-trip asserted
  **byte-identical**. Plus `TestAcquisitionIsOutOfBand`: an AST guard that no test
  invokes `fetch_corpora.py` and no `levy/dataset/` module imports a network library.
  All offline, driven by `tests/fixtures/corpora/` — real column structure, synthetic
  content, laid out as a valid `--raw-dir` (regenerate the parquet shards with
  `python tests/fixtures/corpora/make_sodd_fixture.py`).
- `tests/test_workload_resample.py` (2026-08-06) — 55 unit tests for backups +
  per-workload re-sampling: backup naming/UTC/same-second collision/unusable
  directory, splice identity preservation and length changes, exclusion and
  post-exclusion pool counting, ids-alignment drift, and the `--workload` CLI
  end-to-end against the corpus fixtures — the untouched workloads' CSV rows
  asserted **byte-for-byte identical** (labels included), `author_label` cleared
  for the re-sampled workload only, the new sample disjoint from the old, every
  overwritten file backed up first, **nothing written when the backup fails**, no
  versioned/suffixed filename ever created, per-workload sidecar provenance, and
  rehydration still byte-identical afterwards.
- `tests/test_annotation_order.py` (2026-08-06) — 38 unit tests for presentation
  order and session control: `--workload-order` parsing (unknown/repeated rejected),
  block order, within-block shuffle determinism under `--order-seed`, workload
  scope independence, resume-in-the-recorded-order after an interrupt (including
  when a different seed is passed), `--session-limit` stopping and resuming, the
  fingerprint that stops a re-sampled workload from inheriting the old labels, v1
  progress-file compatibility, and blindness under the new ordering.
- `tests/test_results_merge.py` (2026-08-06) — 29 unit tests for the partial-run
  merge: duplicates across and within runs, incomplete grid, cell outside the grid,
  decisions merged (and a mixed-presence run rejected), dataset/provider
  disagreement, grid-order sorting, backup-before-write, **the merged 30-row set
  byte-identical to a single full sweep's `results.csv`/`decisions.csv`**, and the
  merged bundle feeding `scripts/run_analysis.py` end-to-end.
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
  `merge.py` (2026-08-06) — merges partial harness runs into one 30-row result set,
  at the level of **raw CSV rows** so a merged file is byte-identical to a single
  full run's (the harness writes pre-formatted strings; parsing to floats and
  reformatting would break that). A `config_id` present in two runs and a merged
  set that is not exactly `full_grid()` are both hard errors — the first because
  one run is stale and the merge does not get to pick a winner, the second because
  the ANOVA assumes a balanced design. Runs disagreeing on `dataset_path` /
  `embedding_provider` / `llm_provider` are refused; the merged `run_meta.json`
  records every source run under `merged_from`.
- `scripts/run_experiments.py` — argparse CLI over `levy/experiment/runner.py`: dataset
  path (default `data/ground_truth.csv`), output directory, `--models/--workloads/
  --thresholds` grid-subset flags for smoke runs, `--embedding-provider` (default
  `mock`, fully offline against the synthetic fixture; pass `sentence-transformers` for
  the real study run, which is LEV-13). Non-zero exit on a sanity-check failure.
- `scripts/merge_results.py` (2026-08-06) — argparse CLI over `levy/experiment/merge.py`:
  positional harness directories + `--out-dir` (which may be one of them, an in-place
  merge, backed up first). `--allow-partial` writes an incomplete set for diagnostics
  only. Use it when the grid ran in pieces (`--workloads chat` = 10 of 30 rows, a
  re-sampled workload re-run on its own) instead of re-running cells that were fine.
- `tests/test_experiment_config.py`, `test_experiment_metrics.py`,
  `test_experiment_replay.py`, `test_experiment_runner.py` — 32 unit tests for
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
- `levy/latency/` (LEV-14, D1's second half — the Proposal's "latency
  measurements (cache lookup overhead vs LLM call savings)", which nothing
  measured before): `timing.py` (`TimingCollector` — named segments on
  `time.perf_counter`, plus `segment()`/`mark()`/`since_ms()` helpers used at
  the instrumented call sites; knows nothing about the engine). `LevyEngine.
  generate(prompt, timing=None)` and `SemanticCache.get(request, timing=None)`
  are **opt-in and additive**: with no collector no extra clock is read, and
  `LevyResult`, `LevyMetrics` and every cache are byte-for-byte what they were.
  The total-lookup segment ends where the lookup does — a miss's LLM call and
  store are excluded, which is what keeps the segment sum under the total.
  `benchmark.py` — four-phase protocol per configuration (populate → discard
  warm-up → **warm** phase, memo seeded → **cold** phase, memo evicted
  immediately before the measured call via the new `EmbeddingManager.forget()`),
  nearest-rank p50/p95 (never interpolated — an interpolated percentile names a
  latency nobody observed). **Cold and warm embedding cost are separate reported
  figures and are never averaged**; `total_lookup` is the cold figure, the warm
  total being recoverable as `total_lookup_p50 - (embed_cold_p50 -
  embed_warm_p50)`. Probe texts are workload queries with a unique suffix, so
  every measured lookup traverses exact-miss → embed → search instead of
  short-circuiting on the exact cache. `corpus.py` — `responses.jsonl` keyed by
  `sha256(prompt)` (the same key `ExactCache` uses, so one population serves all
  ten FAQ configurations); **the prompt text is never stored**, only its hash and
  the model's own output; `CorpusLLMClient` serves it back to the replay.
  `population.py` — the billed call loop, a library function taking an injected
  client *because* the spec demands both that it be tested offline and that no
  test invoke the billed script; resume-skip, refusal-skip and a clean
  budget-guard stop that leaves valid JSONL. `report.py` — `latency.csv`,
  `latency_meta.json` (host, versions, protocol, the explicit
  reproducibility-boundary statement, the savings figure) and `llm_calls.json`
  (observed cost, never the pre-run estimate). **Every savings figure carries the
  resolved model identifier that produced it**, asserted in a test.
  `LevyEngine.__init__` gained an optional `llm_client` injection point,
  mirroring `embedding_manager`.
- `scripts/run_latency.py` — offline driver: reads the configuration list from a
  reference `results.csv` (read-only), requires an explicit `--out-dir`, serves
  `responses.jsonl` when present, writes `latency.csv` + `latency_meta.json`.
  **`scripts/populate_responses.py` is the repository's second networked entry
  point and the only one that spends money** — one real Anthropic call per unique
  prompt, `--dry-run` first, excluded from pytest by the same AST guard as
  `scripts/fetch_corpora.py` (now `TestAcquisitionIsOutOfBand.OUT_OF_BAND_SCRIPTS`,
  which also catches an import-and-call-`main()` invocation and asserts no
  `levy/latency/` module imports a network library at module scope).
- `tests/test_latency_timing.py` (10), `test_latency_benchmark.py` (15),
  `test_latency_report.py` (21), `test_latency_population.py` (8) — all offline:
  segment sums never exceeding the total on exact hit / semantic hit / miss,
  default behaviour identical without a collector, cold-vs-warm asserted by
  **counting client invocations per probe text** (not by comparing durations,
  which would pass on a fast machine even if the memo were never cleared),
  percentiles from exactly the measured repetitions, all ten FAQ rows with no
  empty percentile, the boundary statement, no savings figure without a model,
  resume-skip and budget-stop through `httpx.MockTransport`, and the reference
  result directory byte-identical after a full driver run.

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
   (`claude-3-sonnet-20240229`) is retired; the connector defaults to a current
   model instead — flagged here per the frozen-docs rule, not silently resolved.
   Since LEV-14 that default is **`claude-haiku-4-5-20251001` at $1/$5 per MTok**,
   the model the latency pilot actually calls: `anthropic_model` and the two
   price fields are one triple describing one model, and the previous
   `claude-opus-4-8` / $5 / $25 values described a model no run had ever called.
   Escalating the model means changing all three together.
3. ~~**No Faiss HNSW index**~~ — **Resolved (LEV-2).** `SemanticCache` now owns a
   `VectorIndex` (Faiss HNSW or brute-force oracle) and uses `similarity =
   1/(1+L2_distance)` per Algorithm 1. **Threshold-scale flag for LEV-4/LEV-8:**
   all embeddings are L2-normalised before indexing so the distance scale is
   identical across models. For unit vectors, `distance = sqrt(2 − 2·cosine)` and
   `similarity = 1/(1+distance)`. The frozen sweep 0.70–0.90 therefore covers a
   high-cosine band (~0.91–0.998). This is intentional and spec-mandated; do NOT
   rescale thresholds or revert to cosine.
4. ~~**No experiment harness**~~ — **Resolved (LEV-4).** `levy/experiment/` implements
   `run_experiment`/`full_grid`/30-configuration replay, TP/FP/TN/FN accounting against
   `QueryPair.ground_truth_label()`, and precision/recall/F0.5/FPR/hit-rate computation
   with zero-division-safe formulas and sanity checks. `scripts/run_experiments.py`
   drives the full grid (or a subset) fully offline via the mock LLM; results are
   validated against the committed 15-pair synthetic fixture only — the real run over
   LEV-11's 900-pair dataset with `sentence-transformers` providers is LEV-13.
5. ~~**Embedding defaults don't match the study**~~ — **Resolved (LEV-1).**
   `LevyConfig` now defaults to `sentence-transformers` / `all-MiniLM-L6-v2`;
   `EmbeddingManager` supports runtime switching to `modernbert`
   (`nomic-ai/modernbert-embed-base`) with symmetric task-prefix handling.
6. ~~**No annotated dataset**~~ — **Platform tooling resolved (LEV-3); acquisition
   and licence-safe distribution resolved (LEV-12).** `levy/dataset/` + `scripts/`
   implement the schema, CSV/JSON loader (the LEV-4 contract), the ids-only
   distribution format, the corpus provenance registry, seeded stratified sampling,
   pre-flight validation, one-command acquisition, rehydration, blind re-annotation,
   and Cohen's kappa. **Corpus deviations from the frozen docs, flagged not silently
   resolved** (rationale in `data/DATASHEET.md` §2): code workload
   "Stack Overflow duplicate questions" → **SODD** (same
   duplicate-closure source, published pre-processed release); chat workload
   **ConvAI2 → Twitter PIT-2015** (ConvAI2 ships dialogues, not pair-level human
   same-intent labels, so it cannot supply the original label the kappa criterion
   compares against); D2 released as **identifiers + labels + a rehydration script**
   rather than as query text, because QQP grants no redistribution right (the
   PAWS-QQP approach; the ±5% criterion is preserved through input checksums).
   **Data production complete (2026-08-04).** All three corpora acquired and their
   checksums pinned; 900 pairs sampled at seed 42 / `positive_ratio` 0.5 (150-150 per
   class per workload); rehydration verified byte-identical on the real 900; the
   author's blind re-annotation finished 900/900 and published in
   `data/ground_truth.ids.csv`'s `author_label` column. **Cohen's kappa = 0.5000
   (faq 0.5267, code 0.4200, chat 0.5533) — below the frozen κ > 0.7 criterion.**
   (0.3267 at first publication; `chat` was re-drawn 2026-08-06 at seed 4242 and
   `code` 2026-08-07 at seed 8484, each re-annotated blind — `data/DATASHEET.md`
   §3–§4. Those two workloads' 300 pairs are different draws from the ones first
   published; faq is unchanged at seed 42.)
   That is a research-scope finding, not a defect to code around: the
   corpora's positive classes ("closed as a duplicate", "3+ of 5 crowdworkers called
   it a paraphrase") are looser than the study's cache-substitutability label. Do
   **not** lower the threshold, re-annotate non-blind, re-sample for agreement, or
   revert `ground_truth_label()`. Breakdown and contingency options: `data/DATASHEET.md`
   §4. The synthetic fixtures in `data/ground_truth.{csv,json}` are **not** replaced —
   they stay as the permanent offline default; the real dataset lives in the gitignored
   `.full.` files, rebuilt by `scripts/rehydrate_dataset.py`. **Still open, and now
   tracked in LEV-13, not LEV-11:** the D3 production run (harness → analysis →
   replication on the real 900). LEV-11 closes on D2 alone.
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

# Corpora + real dataset (LEV-12). fetch_corpora.py is the ONLY networked entry
# point in the repo; everything else is offline. It exits non-zero printing the
# URL/filename/SHA-256 for corpora needing a human step (Quora, SODD).
python scripts/fetch_corpora.py                 # -> data/raw/ (gitignored contents)
python scripts/fetch_corpora.py --pin           # record checksums into data/corpora.json
python scripts/rehydrate_dataset.py             # ids + data/raw/ -> data/ground_truth.full.*
# Author-only, once: sample the real 900 pairs (refuses synthetic fallback)
python scripts/sample_dataset.py --require-real --n-per-workload 300 --seed 42 \
    --out-csv data/ground_truth.full.csv --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv

# Re-draw ONE workload in place (2026-08-06). Reads only that corpus; replaces
# only that workload's rows; other workloads keep their author_label. Backs up
# every file it overwrites to data/backups/ first. New pairs exclude everything
# already in the dataset. Never creates a second/versioned dataset.
python scripts/sample_dataset.py --require-real --workload chat \
    --n-per-workload 300 --seed 4242 \
    --out-csv data/ground_truth.full.csv --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv

# Annotate just that workload, 50 pairs per sitting, refreshing the published ids
# file in the same command. Blocks are presented faq,chat,code (code last) and
# shuffled within each block under --order-seed; the order is persisted so a
# resumed session keeps it.
python scripts/annotate_dataset.py \
    --dataset data/ground_truth.full.json --progress data/annotation_progress.json \
    --workload chat --session-limit 50 \
    --out-csv data/ground_truth.full.csv --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv

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

# Grid ran in pieces (e.g. `--workloads chat` = 10 of 30 rows)? Merge instead of
# re-running the cells that were fine. Fails loudly on a duplicate config_id or a
# set that isn't exactly the 30-cell grid; backs up the target first.
python scripts/merge_results.py --out-dir results/run-001 \
    results/run-faq results/run-code results/run-chat

# Release audit (LICENSE, secrets in tree + all git history, personal data,
# third-party corpus text in tracked files, data/raw/ spot-check)
scripts/audit_release.sh

# Latency (LEV-14). Offline half: reads the config list from a reference
# results.csv (read-only) and measures the lookup path per configuration.
python scripts/run_latency.py --reference results/run-003/results.csv \
    --dataset data/ground_truth.full.csv --embedding-provider sentence-transformers \
    --out-dir results/latency-faq --workload faq
# Billed half — REAL API CALLS, one per unique prompt. Dry-run first; prompts
# already in responses.jsonl are skipped, so an interrupted run resumes without
# paying twice. Set --model and BOTH price flags to the model actually used.
python scripts/populate_responses.py --dataset data/ground_truth.full.csv \
    --workload faq --out-dir results/latency-faq --dry-run

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
  research scope). Currently **11 capabilities**, one per shipped capability:
  `embedding-management`, `vector-store`, `ground-truth-dataset`,
  `experiment-harness`, `test-infrastructure`, `anthropic-connector`,
  `api-router`, `statistical-analysis`, `release-packaging`,
  `results-dashboard`, `corpus-acquisition`.
  **Main specs use main-spec structure** — `# <name> Specification`, a
  `Capability:` line, `## Purpose`, `## Requirements` — *never* delta headers
  (`## ADDED Requirements`) and never a `TBD` Purpose. `openspec archive` creates
  the file with a `TBD` placeholder; fill it in the same step. Getting this wrong
  fails `openspec validate --all` and has had to be repaired twice.
- `openspec/changes/` — in-flight change proposals (`proposal.md`, `design.md`,
  `tasks.md` per change); completed changes move to `openspec/changes/archive/`.
  Archived so far: `add-embedding-manager`, `add-faiss-vector-store`,
  `add-experiment-harness`, `add-test-infrastructure`, `add-anthropic-connector`,
  `add-fastapi-router`, `add-statistical-analysis`, `add-release-packaging`,
  `add-results-dashboard`, `add-corpus-acquisition` (2026-08-04),
  `add-ground-truth-dataset` (2026-08-05). **No changes are in flight** —
  `openspec list` reports none. The remaining D3 work (LEV-13) is a production run of
  already-shipped tooling, so it produces result artifacts rather than capability
  changes and correctly has no OpenSpec change of its own.
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
| LEV-3 | `add-ground-truth-dataset` | Urgent | archived (2026-08-05) |
| LEV-4 | `add-experiment-harness` | Urgent | archived |
| LEV-5 | `add-test-infrastructure` | Urgent | archived |
| LEV-6 | `add-anthropic-connector` | High | archived |
| LEV-7 | `add-fastapi-router` | High | archived |
| LEV-8 | `add-statistical-analysis` | High | archived |
| LEV-9 | `add-release-packaging` | Medium | archived |
| LEV-10 | `add-results-dashboard` | Low (desirable) | archived |
| LEV-11 | — (D2 data production: real dataset + published D2 artifact) | Urgent | **complete** — 900 pairs published as ids + labels; κ = 0.5000 after the chat (2026-08-06) and code (2026-08-07) re-samples, 0.3267 at first publication; below the 0.7 bar, recorded as a finding |
| LEV-12 | `add-corpus-acquisition` | High | archived (2026-08-04) |
| LEV-13 | — (D3 production run: 30 configurations + analysis + ±5% replication) | Urgent | **run complete 2026-08-07**, result of record `results/run-003/` — see the results note below |

Critical path: LEV-1 → LEV-2 → LEV-4 → LEV-8, with LEV-3 → LEV-12 → LEV-11 (D2)
→ LEV-13 (D3). **LEV-11 and LEV-13 are deliberately separate:** D2 is human-paced
annotation work, D3 is a machine run over its output, and keeping them in one issue
is what previously made neither closeable. Do not merge them back.
When an OpenSpec change is created or archived, reference its Linear issue
and keep the issue status in sync.

### D3 production run — results (2026-08-07)

Full grid over the real 900-pair dataset with `sentence-transformers` embeddings.
The `chat` cells come from a separate run over the re-sampled workload, merged in
with `scripts/merge_results.py`; faq and code are the 2026-08-06 run, unchanged.
**Result of record: `results/run-003/`** (gitignored — results ship with a release,
not the tree). Staging directories from the merges (`run-001-nochat`,
`run-002-chat`, `staging-code`, `staging-prev-minus-code`) were merge inputs, not
results: pointing the analysis, `check_replication.py` or the poster at one of them
returns a valid-looking answer covering part of the grid. Replicate and build only
against the consolidated directory. The full re-run-one-workload procedure,
including which directories are scratch, is `docs/DATA_PRODUCTION.md`
§"Re-drawing one workload".

- **H0₁ (model) retained**, p = 0.465. **H0₂ (workload) rejected**, p = 0.0188,
  Tukey ran on it. **H0₃ (interaction) retained**, p = 0.875. So: no measurable
  embedding-model effect on FPR; workload dominates.
- **Hit rate never reaches the frozen 30% viability bar.** Best cell is
  faq / all-MiniLM-L6-v2 at threshold 0.70 = 24.0%. All 30 configurations fail
  the criterion.
- **±5% replication passed**, 60/60 (configuration, metric) comparisons over all
  30 configurations, recorded in `results/run-003/replication.json`.
- Precision is high wherever anything is cached at all (0.87–1.00 on faq), which
  is the flip side of the same effect: the `1/(1+L2)` band 0.70–0.90 corresponds
  to ~0.91–0.998 cosine, so the cache almost never fires.

Both null results are findings, not defects — do not rescale the thresholds to
chase hit rate (known-gap note #3), and do not re-run to hunt for a model effect.

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
