# Levy

**Levy** is a semantic caching engine for LLM APIs, built as the IT artefact of an MSc Artificial Intelligence capstone project (University of Liverpool). It sits between your application and an LLM provider (Anthropic, or a local Ollama model) to optimize costs and latency by reusing responses for identical or semantically similar prompts.

The research behind Levy benchmarks false positive rates of semantic caching across embedding models (all-MiniLM vs ModernBERT), workloads (FAQ, code, chat), and similarity thresholds.

## Documentation

| Document | What it covers |
|---|---|
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Run the full evaluation pipeline from a fresh checkout — one Docker command, or step by step with conda. Fully offline. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component map, request flow, experiment flow, provider-abstraction patterns. |
| [data/DATASHEET.md](data/DATASHEET.md) | Dataset provenance: source corpora and licences, sampling protocol, annotation guidelines, annotator-agreement result. |
| [data/README.md](data/README.md) | What is committed in `data/` and what you generate locally. |

**Run everything in one command** (see [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for detail):

```bash
docker compose run --rm pipeline
```

## Features

- **Exact Match Caching**: Extremely fast retrieval for identical prompts.
- **Semantic Caching**: Uses vector embeddings (via `sentence-transformers`) to find and reuse answers for similar meaning queries (e.g., "What is the capital of France?" vs "Tell me France's capital").
- **Metrics**: Automatically tracks cache hit rates, latency and estimated token savings.
- **Pluggable Architecture**: Easy to swap LLM providers or Vector Stores.

## Project Structure

```
levy/
├── levy/                    # Core package
│   ├── api/                 # FastAPI router: app, Pydantic schemas, engine pool
│   ├── cache/               # Exact + semantic cache, vector index (Faiss HNSW /
│   │                        #   brute-force), InMemory/Redis stores
│   ├── llm_client.py        # LLM interaction (Mock, Ollama, Anthropic)
│   ├── embeddings.py        # EmbeddingClient ABC + Mock, SentenceTransformer, Ollama
│   ├── embedding_manager.py # EmbeddingManager: study-model registry, runtime switching,
│   │                        #   memoization, symmetric prefix handling
│   ├── engine.py            # Main orchestration engine
│   ├── config.py            # LevyConfig (providers, thresholds, store)
│   ├── metrics.py           # Hit/miss/latency/token-savings tracking
│   ├── models.py            # Data classes
│   ├── dataset/             # Ground-truth dataset platform: corpus acquisition and
│   │                        #   validation, schema, CSV/JSON I/O, seeded sampling,
│   │                        #   blind re-annotation, Cohen's kappa, duplicate prevalence
│   ├── experiment/          # Experiment harness: grid, replay, metrics, sweep runner,
│   │                        #   partial-run merge
│   ├── analysis/            # Statistical analysis: ANOVA/Tukey + effect sizes, curves,
│   │                        #   kappa section, replication check, robustness
│   │                        #   re-analysis, bundle assembly
│   ├── latency/             # Lookup-overhead and provider-latency measurement
│   └── dashboard/           # Results dashboard core (D6, desirable): bundle
│                            #   loading, curve selection, live query decision
├── scripts/                 # CLIs over the packages above: sampling, annotation,
│                            #   experiments, analysis, latency, throughput, prevalence,
│                            #   robustness, reproduce.sh, dashboard.py (Streamlit shell)
├── data/                    # Dataset: published ids + labels, corpus registry, datasheet,
│                            #   synthetic fixtures (query text is never committed)
├── release/                 # Published results: D3 grid, latency, throughput,
│                            #   prevalence, figures, provenance + checksums
├── docs/                    # Architecture, reproduction and data-production docs
├── examples/                # Demo scripts
├── tests/                   # Unit tests (offline)
├── openspec/                # Capability specs and archived change proposals
├── Dockerfile               # One-command reproduction image
├── docker-compose.yml       # `pipeline` service (+ optional Redis)
└── environment.yml          # Conda environment (single dependency source)
```

## Installation

### Using Conda (Recommended)

1. Ensure you have Conda installed.
2. Create the environment:
    ```bash
    conda env create -f environment.yml
    ```
3. Activate the environment:
    ```bash
    conda activate levy
    ```

## Running Tests

pytest is the canonical runner (declared in `environment.yml` / `pyproject.toml`
`[dev]` extras) with an enforced 90% branch-coverage gate on `levy/`:

```bash
# Fast run
python -m pytest tests/ -q

# Gated run (also used in CI): fails if branch coverage of levy/ drops below 90%
python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90
```

`python -m unittest discover -s tests -p "test_*.py"` still works (the suite is
plain `unittest.TestCase`s), but pytest is the one true command going forward.

## Usage

### Quick Start (Python)

```python
from levy import LevyEngine, LevyConfig

# Initialize with defaults (Anthropic LLM: needs ANTHROPIC_API_KEY in .env)
engine = LevyEngine()

# First call - hits the "LLM"
result1 = engine.generate("Hello world")
print(result1.source) # 'llm'

# Second call - hits the cache
result2 = engine.generate("Hello world")
print(result2.source) # 'exact_cache'
```

### Examples

| Example | Requirements | Cost |
|---|---|---|
| `examples/simple_replay.py` | `sentence-transformers` (model weights download on first run). | Free |
| `examples/ollama_demo.py` | Optional side demo of the Ollama provider; **not used in the study**. Needs a local Ollama service with the models pulled (see below). | Free (local models) |
| `examples/anthropic_smoke_check.py` | A real `ANTHROPIC_API_KEY`. | **Billed** — makes one real API call |

#### Cache-behaviour replay

```bash
python examples/simple_replay.py
```

Runs a sequence of prompts through three configurations — no cache, exact cache
only, and exact + semantic cache — and prints per-request source, latency, and a
metrics summary.

#### Ollama (optional local demo)

The Ollama provider is kept as an example of the provider abstraction. The
study itself never used it: no test, result, or release artefact involves an
Ollama model (see [Models used in the study](#models-used-in-the-study)).

1. Install and run [Ollama](https://ollama.com/).
2. Pull a generation model and an embedding model (any you like), and name
   them in `.env` — Levy's code names no model:
   ```bash
   LEVY_OLLAMA_MODEL_1=<generation model>
   LEVY_OLLAMA_EMBEDDING_MODEL_1=<embedding model>
   ```
3. Run the demo:
   ```bash
   python examples/ollama_demo.py
   ```

No automated path — not the test suite, not CI, not the container, not
`scripts/reproduce.sh` — runs any example that costs money or needs a network
service. The billed Anthropic smoke check must be invoked by hand; see
[Using Anthropic](#using-anthropic) below.

### Using Anthropic

The `anthropic` provider wraps the official `anthropic` SDK behind the same
synchronous `LLMClient` interface as Mock/Ollama:

```bash
# .env -- one model per number, the lowest is the default (format in .env.example)
ANTHROPIC_API_KEY=...
LEVY_ANTHROPIC_MODEL_1=<model id>
LEVY_ANTHROPIC_MODEL_1_INPUT_USD_PER_MTOK=<number>
LEVY_ANTHROPIC_MODEL_1_OUTPUT_USD_PER_MTOK=<number>
```

```python
config = LevyConfig(
    llm_provider="anthropic",
    # api key, model and prices come from .env (above)
    anthropic_max_retries=2,                 # SDK's own exponential backoff
    anthropic_budget_cap_usd=200.0,          # hard stop (budget cap)
)
engine = LevyEngine(config)
result = engine.generate("Hello")
```

- **Retry:** the connector doesn't reimplement backoff — it configures the
  SDK's built-in retry (connection errors, 408/409/429/5xx) via
  `anthropic_max_retries`. Non-retryable API errors propagate as the SDK's
  typed exceptions (e.g. `anthropic.BadRequestError`) rather than being
  swallowed.
- **Token accounting:** `LLMResponse.token_usage` is the real
  `input_tokens + output_tokens` from the API's usage report;
  `LLMResponse.metadata` carries the input/output split, the serving model,
  and the stop reason.
- **Budget guard:** `AnthropicLLMClient` accumulates a request count and an
  estimated cost (tokens × configured per-MTok prices) per instance. Once the
  estimate reaches `anthropic_budget_cap_usd`, further calls raise
  `BudgetExceededError` *before* sending a request. Inspect spend at any time
  via `engine.llm_client.request_count` / `.estimated_cost_usd`. This is a
  safety net, not an invoice — state is per-process and resets on restart;
  the Anthropic Console remains the authoritative billing record.
- **Refusals:** if the API returns a successful response whose `stop_reason`
  is a refusal, the client raises `AnthropicRefusalError` instead of
  returning (and thereby caching) empty content.
- **Model and prices from `.env`:** no model id or price is written in Levy's
  code; models change every few months. Each model is a numbered
  `LEVY_ANTHROPIC_MODEL_<n>` with its `_INPUT_USD_PER_MTOK` and
  `_OUTPUT_USD_PER_MTOK` (US dollars per million tokens sent and written, as a
  plain number with a dot); the lowest number is the default, and adding a
  model means adding a new number. Without them, constructing the client fails
  and names the variable.

**One-time real-API smoke check — opt-in and billed.** This makes one real API
call charged to the `ANTHROPIC_API_KEY` in your `.env`. It is deliberately
fenced off from every automated path: it lives in `examples/` rather than
`tests/`, is not named `test_*.py`, and pytest's `testpaths` is `tests` only, so
neither the test suite, nor CI, nor the container, nor `scripts/reproduce.sh`
can invoke it. Run it by hand, once, to confirm the key, model, and budget
configuration are wired correctly:

```bash
python examples/anthropic_smoke_check.py
```

### Using Redis Stack (Docker)

To use Redis for persistence:

1. Start Redis:
   ```bash
   docker-compose up -d
   ```
2. Configure `LevyConfig` to use `cache_store_type="redis"`.

## HTTP API

`levy/api/` exposes the engine over HTTP, plus admin observability and maintenance endpoints.

```bash
uvicorn levy.api.app:app --reload
```

By default the app builds its engine pool from `LevyConfig()` (reads `.env`,
so real deployments need the configured provider's credentials). Interactive
docs are auto-generated at `http://localhost:8000/docs`
(`/openapi.json` for the raw schema).

### `POST /v1/chat/completions`

Extracts the prompt from the last `messages` entry and serves it via
exact cache → semantic cache → the configured LLM provider. The response body
is in the Anthropic Messages format for both hits and misses; cache identity
lives in the response headers, not the body.

```bash
# Miss: first time seeing this prompt
curl -s -D - http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
# X-Cache-Status: MISS

# Exact hit: identical prompt, second request
curl -s -D - http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
# X-Cache-Status: HIT
# X-Cache-Similarity: 1.0

# Per-request cache_config: routes to an engine bound to this
# (embedding_model, threshold) pair; omitted fields fall back to LevyConfig defaults.
curl -s -D - http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "Tell me France'"'"'s capital"}],
        "cache_config": {"threshold": 0.80, "embedding_model": "all-MiniLM-L6-v2"}
      }'
# Semantic hit (if similar enough to a stored entry under this pair):
# X-Cache-Status: HIT
# X-Cache-Similarity: 0.87   (example)
```

### `GET /admin/cache/stats`

Aggregated hit rate, semantic-index size, and per-model cached-entry counts
across every pooled `(embedding_model, threshold)` engine instance.

```bash
curl -s http://localhost:8000/admin/cache/stats
```

### `POST /admin/cache/clear`

Empties the exact and semantic caches of every pooled engine and resets
metrics, reporting per-key entry counts cleared.

```bash
curl -s -X POST http://localhost:8000/admin/cache/clear
```

### Engine pool

The engine binds `embedding_model`/`similarity_threshold` at construction,
while the contract puts both per-request. The router resolves this with a
bounded pool (default cap: 8) keyed by `(embedding_model, threshold)`: the
first request for a pair builds an engine from the base `LevyConfig` with
those two fields overridden; later requests with the same pair reuse it (their
caches accumulate). Requesting a pair beyond the cap returns a structured
`400 pool_cap_exceeded` error naming the cap.

### Error responses

Errors are structured JSON (`{"error": ..., "detail": ..., ...}`), not stack
traces:

| Condition | Status | `error` |
|---|---|---|
| Malformed/missing `messages` | 422 | (FastAPI's standard validation body) |
| Pool cap exceeded | 400 | `pool_cap_exceeded` |
| Anthropic budget cap reached | 402 | `budget_exceeded` (includes `cap_usd`/`estimated_cost_usd`) |
| Anthropic refusal | 502 | `provider_refusal` |
| Any other provider/engine error | 500 | `provider_error` |

### Structured request logging

Every chat request emits one JSON record via stdlib logging (`levy.api`
logger): `request_id` (echoed in the response body), `arrival_ts`/
`completion_ts`, the resolved `embedding_model`/`threshold`, the `prompt`,
the cache decision `cache_source`, `similarity`, and `latency_ms`. The
records alone are enough to reconstruct and replay a request sequence with
identical cache configuration.

### Async decision (recorded, not a gap)

The design calls for an "asynchronous wrapper"; endpoints here are
declared `async`-free (`def`) so FastAPI runs them in its threadpool instead —
the whole call chain (engine, caches, the Anthropic client) is
synchronous, and blocking the event loop directly would serialize every
request. This satisfies the intent (concurrent request handling) without an
`AsyncAnthropic` migration; see `openspec/changes/archive/2026-07-25-add-fastapi-router/design.md`.

## Levy Console

A `console` section of the HTTP API, beside the endpoints above, plus a page
that shows each console request live: the path it took, where each step ran,
how long each step cost, why the cache decided what it decided, and what a hit
saved. The delivered `/v1/chat/completions` and `/admin/*` endpoints are not
changed by it, and the console never shares their engines or caches.

```bash
docker compose up -d redis
uvicorn levy.api.app:app --port 8000
```

Redis serves `cache_store=redis`; Ollama models need a running Ollama on
`localhost:11434`.

Open `http://localhost:8000/docs` and `http://localhost:8000/console` side by
side. In the docs' **console** section, `POST /console/chat` takes the prompt
and every setting as a field with its list of options:

| Setting | Options |
|---|---|
| `threshold` | number in [0, 1] (1/(1+L2) scale) |
| `embedding_provider` | `sentence-transformers`, `ollama` |
| `embedding_model` | `all-MiniLM-L6-v2`, `modernbert` (sentence-transformers); every `LEVY_OLLAMA_EMBEDDING_MODEL_<n>` (ollama) |
| `llm_provider` | `anthropic`, `ollama` |
| `llm_model` | every `LEVY_ANTHROPIC_MODEL_<n>` (anthropic) and `LEVY_OLLAMA_MODEL_<n>` (ollama) |
| `exact_cache`, `semantic_cache` | `true`, `false` |
| `vector_index_backend` | `faiss`, `brute_force` |
| `cache_store` | `memory`, `redis` |

A model must belong to its chosen provider. Providers, backends and stores are
what Levy implements (`levy/console/options.py`); the model lists come from
`.env`, and a provider with no configured model is not offered. Levy reads
`.env` again when Swagger is refreshed, so a model added there appears in the
selectors without restarting Levy. Each executed request appears on the page without a
refresh. A miss is shown waiting at the provider while its call is in flight;
the lookup steps (milliseconds) are replayed hop by hop with their real timings.

Under the request path the page shows the answer the request returned. On a
hit, the line above it sets the prompt asked beside the stored prompt the
answer was written for. A miss states the nearest stored prompt's similarity
against τ, and τ is shown with the cosine it enforces
(`d = 1/τ − 1`, `cosine = 1 − d²/2`).

| Endpoint | Purpose |
|---|---|
| `POST /console/chat` | Chat under a per-request configuration; same body and headers as `/v1/chat/completions` |
| `GET /console/traces?after=<version>` | Live trace feed the page polls |
| `POST /console/traces/clear` | Empty the page's history |
| `POST /console/cache/clear` | Empty the console's caches |
| `GET /console` | The page |

## Configuration

You can configure Levy via `LevyConfig`:

```python
config = LevyConfig(
    llm_provider="anthropic",    # default; key read from ANTHROPIC_API_KEY in .env; also "ollama"
    enable_semantic_cache=True,
    similarity_threshold=0.85,   # in 1/(1+L2) space; study sweep: 0.70–0.90
    # Embedding model for the study (default: all-MiniLM-L6-v2 baseline)
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",   # or "modernbert" for the second study model
    # Vector index backend (default: "auto" → Faiss HNSW if installed, else brute-force)
    vector_index_backend="auto",  # "auto" | "faiss" | "brute_force"
)
```

> **Faiss HNSW** is the production vector index. Install via conda to avoid Apple-Silicon segfaults:
> ```bash
> conda install -c conda-forge faiss-cpu
> ```
> If Faiss is absent the engine falls back to a brute-force numpy index automatically.

### Models used in the study

| Role | Model | Where recorded |
|---|---|---|
| Embeddings, baseline | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | `release/d3-results/run_meta.json` |
| Embeddings, comparison | `nomic-ai/modernbert-embed-base` (768-dim) | `release/d3-results/run_meta.json` |
| LLM, D3 grid (30 configurations) | `mock` — the grid replays labelled pairs, no real LLM call | `release/d3-results/run_meta.json` |
| LLM, latency measurement | `claude-haiku-4-5-20251001` (600 billed calls) | `release/latency/` |
| LLM, throughput measurement | `claude-haiku-4-5-20251001` (mock replay at its measured latency), `claude-sonnet-4-6` and `claude-opus-5` (live calls) | `release/throughput/<model-id>/` |

### Switching embedding models at runtime

The `EmbeddingManager` built into the engine resolves study-model aliases:

| Alias | Checkpoint | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` / `all-minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, study baseline |
| `modernbert` | `nomic-ai/modernbert-embed-base` | 768-dim, symmetric `search_query:` prefix applied automatically |

To switch models between experiment runs, change `embedding_model` in `LevyConfig` — no code changes required. Embeddings are memoized per `(model, text)` so replay experiments never recompute a vector.

## Ground-truth dataset tooling

`levy/dataset/` + `scripts/` provide the data-agnostic platform for D2 (900
annotated query pairs across 3 workloads: FAQ, code, chat).

Two datasets live side by side. `data/ground_truth.csv` / `.json` hold **15
synthetic fixture pairs** (5/workload, obviously fake text,
`source_corpus="synthetic-fixture"`) — the permanent offline default for the
test suite and the pipeline. The real 900 pairs are published as
`data/ground_truth.ids.csv`: identifiers and labels only, no query text, because
the source corpora grant no redistribution right. You rebuild the text locally
from your own copy of the corpora — see
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). Protocol, licences and
annotator agreement are in `data/README.md` and `data/DATASHEET.md`.

```bash
# Sample a dataset (falls back to synthetic MockCorpusSource per workload
# when a raw corpus file isn't given — fully offline):
python scripts/sample_dataset.py --n-per-workload 300 --seed 42 \
    --out-csv data/ground_truth.csv --out-json data/ground_truth.json

# Blind re-annotation session (never shows the original label; resumable):
python scripts/annotate_dataset.py \
    --dataset data/ground_truth.json --progress data/.annotation_progress.json

# Cohen's kappa (author vs. original label), overall + per-workload:
python scripts/compute_kappa.py --dataset data/ground_truth.json --strict

# Convert between CSV and JSON (identical content, round-trip safe):
python scripts/export_dataset.py --in data/ground_truth.json --out /tmp/dataset.csv
```

The author's production procedure for the real 900 pairs is in
[`docs/DATA_PRODUCTION.md`](docs/DATA_PRODUCTION.md).

`QueryPair.ground_truth_label()` (author label if annotated, else the
original corpus label) is the contract the experiment harness replays
against.

## Experiment harness

`levy/experiment/` replays annotated query pairs through the production
cache lookup path (`LevyEngine.generate()`) across the 30-configuration
grid — 2 embedding models × 3 workloads × 5 similarity thresholds
(0.70–0.90) — and accounts for TP/FP/TN/FN against each pair's ground-truth
label per the replay-harness algorithm.

```bash
# Full 30-configuration sweep on the committed synthetic fixture:
python scripts/run_experiments.py --out-dir results/smoke-run

# Smoke run: one model, one workload, two thresholds:
python scripts/run_experiments.py --out-dir results/smoke-run \
    --models all-MiniLM-L6-v2 --workloads faq --thresholds 0.70,0.90

# Real study run — the rehydrated 900-pair dataset and real embeddings:
python scripts/run_experiments.py --dataset data/ground_truth.full.csv \
    --out-dir results/study-run
```

Outputs written to `--out-dir`:
- `results.csv` — one row per configuration: confusion counts (TP/FP/TN/FN),
  precision, recall, F0.5, false positive rate, hit rate, and zero-division
  flags (0.0 instead of NaN, never silently dropped).
- `decisions.csv` — one row per replayed pair per configuration: decision
  (hit/miss), decision source (`exact_cache`/`semantic_cache`/`llm`), matched
  similarity, ground-truth label, confusion outcome.
- `run_meta.json` — dataset path, providers, resolved model checkpoints, the
  grid that was run, and latency stats (explicitly labeled synthetic under
  the mock LLM's fixed 0.5s sleep). Never merged into the two files above, so
  re-running on identical inputs reproduces `results.csv`/`decisions.csv`
  byte-identically.

`--embedding-provider` defaults to `sentence-transformers`, the study's two
models; their weights download once on first use. The committed synthetic
fixture (`data/ground_truth.csv`) validates the pipeline; research results
need the real 900-pair dataset, which is published as `data/ground_truth.ids.csv` and
reconstructed locally with `scripts/rehydrate_dataset.py` (see
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)).

## Statistical analysis

`levy/analysis/` turns a harness output directory into the D3 evidence
bundle. It is a pure consumer of the harness contract and is
**dataset-agnostic**: it analyses *any* `results.csv` produced by
`scripts/run_experiments.py`, whether that run used the committed synthetic
fixture or the real 900-pair dataset. The same command produces the
dissertation's tables once real results exist.

```bash
# One invocation -> every table and figure:
python scripts/run_analysis.py --results-dir results/study-run \
    --out-dir results/study-run/analysis

# Point the kappa section at an explicit dataset (default: the dataset_path
# recorded in the run's run_meta.json):
python scripts/run_analysis.py --results-dir results/study-run \
    --dataset data/ground_truth.csv --out-dir results/study-run/analysis
```

Outputs written to `--out-dir`:

| File | Contents |
|---|---|
| `anova.csv` | Two-way ANOVA on false positive rate, `fpr ~ C(model) * C(workload)`: df, sum of squares, F, p, and an explicit `reject`/`retain` at α=0.05 for **H0₁** (no model effect), **H0₂** (no workload effect), **H0₃** (no interaction), plus the residual row. |
| `tukey.csv` | Tukey HSD pairwise comparisons (mean difference, CI, adjusted p) for whichever effects were significant — the 6 model×workload cells when the interaction is significant. |
| `tukey_status.csv` | Whether post-hoc ran for each effect **and why** — always written, including when Tukey was skipped. |
| `curves_hit_rate.csv`, `curves_precision.csv` | Tidy threshold-vs-metric tables, 5 thresholds × 6 (model, workload) pairs, carrying the harness zero-division flags so degenerate cells stay visible. |
| `kappa.json` | Cohen's kappa **sourced from the dataset tooling** (`levy.dataset.kappa`, not reimplemented), with the 2×2 contingency, the 0.7 bar, and a provenance block that labels fixture-derived values `FIXTURE ONLY`. |
| `figures/` | `curve_hit_rate.{png,pdf}`, `curve_precision.{png,pdf}` — regenerable from the curve tables alone; the hit-rate figure carries the 30% economic-viability reference line. |
| `analysis_meta.json` | Input paths, ANOVA diagnostics (design balance, Shapiro-Wilk, Levene), and library versions. The **only** place versions and a timestamp appear. |

Every CSV is timestamp-free and byte-identical across re-runs on identical
input, matching the harness's determinism convention.

**Degenerate results are reported, not laundered.** If false positive rate
has zero variance across all configurations, the F-tests are undefined, and
the three hypotheses are reported as `undefined` rather than as retained
nulls.

### Replication check (±5%)

Success criterion: headline results replicate within ±5%.
`scripts/check_replication.py` re-runs the harness over exactly the grid
recorded in a reference `results.csv`, then compares precision and recall per
configuration:

```bash
python scripts/check_replication.py --reference results/study-run/results.csv
```

Tolerance rule (stated in every report, so the criterion stays auditable):

```
|candidate - reference| <= max(0.01, 5% * |reference|)
```

The absolute floor exists because a purely relative tolerance collapses to
demanding bit-exact equality near a reference of 0.0. Exit code is zero when
every value is within tolerance; otherwise non-zero, with a per-configuration
diff table naming the configuration, the metric, both values, and the
deviation. On one host the harness is deterministic, so a self-comparison
matches exactly; the tolerance is there for runs on another host.

The verdict is also written to `replication.json` beside the reference, so it
can be read rather than inferred from an exit code. It records which
configurations it covers, so a run over part of the grid cannot be mistaken for
one covering all of it.

## Published results

The study's results are committed under [`release/`](release/), with checksums
and a file-by-file description in [`release/PROVENANCE.md`](release/PROVENANCE.md):
the 30-configuration grid (`d3-results/`), the ANOVA/Tukey analysis and a
robustness re-analysis, lookup-latency and throughput measurements, and the
duplicate prevalence of the source corpora. Headline findings:

- **Embedding model has no measurable effect on false positive rate** (H0₁
  retained, p = 0.465); **workload does** (H0₂ rejected, p = 0.0188); no
  interaction (H0₃ retained, p = 0.875).
- **Hit rate never reaches the 30 % viability bar.** The best cell is FAQ with
  `all-MiniLM-L6-v2` at threshold 0.70, at 24.0 %.
- **±5 % replication passes** for all 30 configurations.
- **A lookup costs about 10 ms against about 3.4 s for the provider call it
  avoids** (FAQ workload, `claude-haiku-4-5-20251001`). Provider latency is
  specific to that model and does not replicate; lookup overhead does.
- **Annotator agreement is below the bar:** Cohen's κ = 0.5000 against a target
  of 0.7 (see [`data/DATASHEET.md`](data/DATASHEET.md) §4).

Robustness re-analysis of the model effect on the per-decision outcome is
produced by `scripts/run_robustness.py`; its method-by-method verdicts, including
the pair-clustered tests that reject H0₁ where the ANOVA retains it, are in
`release/d3-results/robustness/verdicts.csv`.

## Results dashboard (D6, desirable)

`levy/dashboard/` + `scripts/dashboard.py` is a local Streamlit viewer over an
analysis bundle (the output of `scripts/run_analysis.py`): threshold-vs-metric
curves per (model, workload) with degenerate points flagged, the ANOVA/Tukey/κ
summary, and a live query box that reports the cache decision for your own
text. It **never recomputes a statistic** — everything shown is read from the
bundle. D6 is the lowest-priority, desirable-only deliverable
(see `openspec/changes/archive/2026-07-26-add-results-dashboard/proposal.md`); it is not part of
`scripts/reproduce.sh` and nothing else depends on it.

```bash
# Generate a bundle first (any of the pipeline commands above), then:
streamlit run scripts/dashboard.py -- --bundle results/reproduce/analysis
```

The `--` separator is required by Streamlit so `--bundle` (and the optional
`--dataset`, used by the query panel) reach the script's own argument parser.
Curve exploration and the hypothesis summary work from the bundle alone; the
query panel additionally needs a dataset file (default: the one recorded in
the bundle's `analysis_meta.json`, falling back to `data/ground_truth.csv`).

The query panel embeds with the study's models (`sentence-transformers`);
their weights download once on first use. If the bundle directory is missing or incomplete, the app
reports exactly what's absent and the command to generate it
(`scripts/reproduce.sh`), and stops cleanly rather than showing a traceback.

## License

Apache-2.0
