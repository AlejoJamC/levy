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
| `docs/RESEARCH_OVERVIEW.md` | Historical/working | Early research framing (CSCK508 module). Predates the proposal; its 12-week timeline and RAG workload were superseded by the frozen docs. Editable. |
| `docs/LITERATURE_REVIEW.md` | Working skeleton | Paper list + research-gaps matrix. Editable, expand as needed. |
| `docs/PLANNING_HIERARCHY.md` | Working | Vision → Epic → Feature → Story → Task hierarchy used to plan work. |
| `docs/epics/EPIC-001-client-proxy-layer.md` | Working | Epic for the FastAPI client/proxy layer (Component 1). Pattern for future epics (`EPIC-00X-*.md`). |
| `README.md` | Living | User-facing install/usage docs. Keep in sync with code. |
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
  echo), `OpenAILLMClient` (raw httpx), `OllamaLLMClient`.
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
- `tests/test_embedding_manager.py` — 20 unit tests for `EmbeddingManager`: runtime
  model switching, alias resolution, memoization, dimension/identity exposure, prefix
  handling, and default config validation. All offline (injected mock clients).
- `tests/test_vector_index.py` — 19 unit tests for `VectorIndex` + `SemanticCache`:
  add/search/reset/size, L2 normalization, zero-vector guard, similarity transform +
  threshold decisions, id→entry resolution, Faiss↔brute-force agreement (skipped
  when Faiss absent), engine end-to-end semantic cache hit/miss.
- `examples/simple_replay.py` — replays a prompt list under no-cache / exact /
  exact+semantic configs. `examples/ollama_demo.py` — end-to-end with local Ollama
  (`llama3.2` + `mxbai-embed-large`).

### Known gaps: current code vs frozen spec

Track these when building toward the experimental phase — they are the backlog
implied by the spec, not bugs:

1. **No FastAPI router** (`/v1/chat/completions`, `/admin/cache/stats`,
   `/admin/cache/clear` with `X-Cache-Status` / `X-Cache-Similarity` headers).
   EPIC-001 covers this layer.
2. **No Anthropic LLM connector** — only mock/OpenAI/Ollama exist; the spec names
   the Anthropic API as the backend.
3. ~~**No Faiss HNSW index**~~ — **Resolved (LEV-2).** `SemanticCache` now owns a
   `VectorIndex` (Faiss HNSW or brute-force oracle) and uses `similarity =
   1/(1+L2_distance)` per Algorithm 1. **Threshold-scale flag for LEV-4/LEV-8:**
   all embeddings are L2-normalised before indexing so the distance scale is
   identical across models. For unit vectors, `distance = sqrt(2 − 2·cosine)` and
   `similarity = 1/(1+distance)`. The frozen sweep 0.70–0.90 therefore covers a
   high-cosine band (~0.91–0.998). This is intentional and spec-mandated; do NOT
   rescale thresholds or revert to cosine. If hit-rate viability (>30%) is not
   met at this band, surface that as a research-scope finding to the supervisor.
4. **No experiment harness** — `run_experiment` / 30-configuration replay,
   TP/FP/TN/FN accounting, and metric computation are not implemented.
5. ~~**Embedding defaults don't match the study**~~ — **Resolved (LEV-1).**
   `LevyConfig` now defaults to `sentence-transformers` / `all-MiniLM-L6-v2`;
   `EmbeddingManager` supports runtime switching to `modernbert`
   (`nomic-ai/modernbert-embed-base`) with symmetric task-prefix handling.
6. **No annotated dataset** in the repo yet (D2: 900 pairs, CSV/JSON + datasheet).
7. **pytest declared but not installed** in the conda env; tests currently run via
   `unittest` (see commands).

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

# Tests (pytest is in pyproject [dev] extras but NOT in the conda env yet;
# unittest always works)
python -m unittest discover -s tests -p "test_*.py"
# or, after `pip install pytest` inside the env:
python -m pytest tests/ -q

# Demos
python examples/simple_replay.py     # mock LLM; uses sentence-transformers if installed
python examples/ollama_demo.py       # requires `ollama serve` + llama3.2 + mxbai-embed-large

# Local services (Redis 7 for cache_store_type="redis")
docker-compose up -d
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
  research scope).
- `openspec/changes/` — in-flight change proposals (`proposal.md`, `design.md`,
  `tasks.md` per change); completed changes move to `openspec/changes/archive/`.
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

| Linear | OpenSpec change | Priority |
|---|---|---|
| LEV-1 | `add-embedding-manager` | Urgent |
| LEV-2 | `add-faiss-vector-store` | Urgent |
| LEV-3 | `add-ground-truth-dataset` | Urgent |
| LEV-4 | `add-experiment-harness` | Urgent |
| LEV-5 | `add-test-infrastructure` | Urgent |
| LEV-6 | `add-anthropic-connector` | High |
| LEV-7 | `add-fastapi-router` | High |
| LEV-8 | `add-statistical-analysis` | High |
| LEV-9 | `add-release-packaging` | Medium |
| LEV-10 | `add-results-dashboard` | Low (desirable) |

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
