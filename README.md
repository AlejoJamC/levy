# Levy

**Levy** is a semantic caching engine for LLM APIs, built as the IT artefact of an MSc Artificial Intelligence capstone project (University of Liverpool). It sits between your application and an LLM provider (Mock, OpenAI-compatible, Ollama, or Anthropic) to optimize costs and latency by reusing responses for identical or semantically similar prompts.

The research behind Levy benchmarks false positive rates of semantic caching across embedding models (all-MiniLM vs ModernBERT), workloads (FAQ, code, chat), and similarity thresholds. The authoritative project definition lives in [docs/Project_Proposal.md](docs/Project_Proposal.md) and [docs/Specification_and_Design_Report.md](docs/Specification_and_Design_Report.md) (university submissions — do not modify).

## Documentation

| Document | What it covers |
|---|---|
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Run the full evaluation pipeline from a fresh checkout — one Docker command, or step by step with conda. Fully offline. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component map traced to the frozen specification, request flow, experiment flow, provider-abstraction patterns. |
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
│   ├── cache/               # Cache logic (Exact, Semantic, InMemory/Redis stores)
│   ├── llm_client.py        # LLM interaction (Mock, OpenAI, Ollama, Anthropic)
│   ├── embeddings.py        # EmbeddingClient ABC + Mock, SentenceTransformer, Ollama
│   ├── embedding_manager.py # EmbeddingManager: study-model registry, runtime switching,
│   │                        #   memoization, symmetric prefix handling
│   ├── engine.py            # Main orchestration engine
│   ├── config.py            # LevyConfig (providers, thresholds, store)
│   ├── metrics.py           # Hit/miss/latency/token-savings tracking
│   ├── models.py            # Data classes
│   ├── dataset/             # Ground-truth dataset platform: schema, CSV/JSON
│   │                        #   I/O, seeded sampling, blind re-annotation, Cohen's kappa
│   ├── experiment/          # Experiment harness: grid, replay, metrics, sweep runner
│   ├── analysis/            # Statistical analysis: ANOVA/Tukey, curves, kappa
│   │                        #   section, replication check, bundle assembly
│   └── dashboard/           # Results dashboard core (D6, desirable): bundle
│                            #   loading, curve selection, live query decision
├── scripts/                 # CLIs over levy/dataset + levy/experiment + levy/analysis
│                            #   + dashboard.py (Streamlit UI shell)
├── data/                    # Ground-truth dataset (currently synthetic fixtures + datasheet)
├── docs/                    # Research docs (proposal & S&D report are frozen)
├── examples/                # Demo scripts
└── tests/                   # Unit tests
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

# Initialize with defaults (Mock LLM, Exact Cache only)
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
| `examples/simple_replay.py` | None for the mock path. Uses `sentence-transformers` for the semantic configuration when importable, which downloads model weights on first run (needs network); falls back to mock embeddings otherwise. | Free |
| `examples/ollama_demo.py` | A **local Ollama service** (`ollama serve`) with the `qwen3` and `nomic-embed-text` models pulled. | Free (local models) |
| `examples/anthropic_smoke_check.py` | A real `ANTHROPIC_API_KEY`. | **Billed** — makes one real API call |

#### Cache-behaviour replay

```bash
python examples/simple_replay.py
```

Runs a sequence of prompts through three configurations — no cache, exact cache
only, and exact + semantic cache — and prints per-request source, latency, and a
metrics summary.

#### Ollama (local models)

1. Install and run [Ollama](https://ollama.com/).
2. Pull required models:
   ```bash
   ollama pull qwen3
   ollama pull nomic-embed-text
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
synchronous `LLMClient` interface as Mock/OpenAI/Ollama:

```python
config = LevyConfig(
    llm_provider="anthropic",
    # anthropic_api_key defaults to the ANTHROPIC_API_KEY env var (.env)
    anthropic_model="claude-haiku-4-5-20251001",  # default; override per config
    anthropic_max_retries=2,                 # SDK's own exponential backoff
    anthropic_budget_cap_usd=200.0,          # hard stop (frozen budget cap)
    anthropic_input_price_per_mtok=1.0,      # USD / 1M input tokens
    anthropic_output_price_per_mtok=5.0,     # USD / 1M output tokens
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
- **Model default drift (documented, not silently resolved):** the frozen
  S&D Report's example model string (`claude-3-sonnet-20240229`) is retired.
  `anthropic_model` defaults to a current model instead —
  `claude-haiku-4-5-20251001`, the model the latency measurement calls. The
  model id and the two per-MTok prices are one triple describing one model:
  change them together, or the budget guard costs a model you are not running.

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

`levy/api/` exposes the engine over HTTP per the frozen S&D "Intended
interface" contract, plus admin observability/maintenance.

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

The frozen S&D calls for an "asynchronous wrapper"; endpoints here are
declared `async`-free (`def`) so FastAPI runs them in its threadpool instead —
the whole call chain (engine, caches, the Anthropic client) is
synchronous, and blocking the event loop directly would serialize every
request. This satisfies the intent (concurrent request handling) without an
`AsyncAnthropic` migration; see `openspec/changes/add-fastapi-router/design.md`.

## Configuration

You can configure Levy via `LevyConfig`:

```python
config = LevyConfig(
    llm_provider="openai",
    openai_api_key="sk-...",
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

`QueryPair.ground_truth_label()` (author label if annotated, else the
original corpus label) is the contract the experiment harness replays
against.

## Experiment harness

`levy/experiment/` replays annotated query pairs through the production
cache lookup path (`LevyEngine.generate()`) across the frozen 30-configuration
grid — 2 embedding models × 3 workloads × 5 similarity thresholds
(0.70–0.90) — and accounts for TP/FP/TN/FN against each pair's ground-truth
label per Algorithm 2 of the S&D Report.

```bash
# Full 30-configuration sweep on the committed synthetic fixture (offline, mock providers):
python scripts/run_experiments.py --out-dir results/smoke-run

# Smoke run: one model, one workload, two thresholds:
python scripts/run_experiments.py --out-dir results/smoke-run \
    --models all-MiniLM-L6-v2 --workloads faq --thresholds 0.70,0.90

# Real study run — the rehydrated 900-pair dataset and real embeddings:
python scripts/run_experiments.py --dataset data/ground_truth.full.csv \
    --embedding-provider sentence-transformers --out-dir results/run-001
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

`--embedding-provider` defaults to `mock`, so the harness runs fully offline
with zero external dependencies against the committed synthetic fixture
(`data/ground_truth.csv`). Mock embeddings are text-hashed random vectors and
don't capture semantic similarity — this validates the machinery, not
research results; a real run needs `sentence-transformers` and the real
900-pair dataset, which is published as `data/ground_truth.ids.csv` and
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
python scripts/run_analysis.py --results-dir results/run-001 \
    --out-dir results/run-001/analysis

# Point the kappa section at an explicit dataset (default: the dataset_path
# recorded in the run's run_meta.json):
python scripts/run_analysis.py --results-dir results/run-001 \
    --dataset data/ground_truth.csv --out-dir results/run-001/analysis
```

Outputs written to `--out-dir`:

| File | Contents |
|---|---|
| `anova.csv` | Two-way ANOVA on false positive rate, `fpr ~ C(model) * C(workload)`: df, sum of squares, F, p, and an explicit `reject`/`retain` at α=0.05 for **H0₁** (no model effect), **H0₂** (no workload effect), **H0₃** (no interaction), plus the residual row. |
| `tukey.csv` | Tukey HSD pairwise comparisons (mean difference, CI, adjusted p) for whichever effects were significant — the 6 model×workload cells when the interaction is significant. |
| `tukey_status.csv` | Whether post-hoc ran for each effect **and why** — always written, including when Tukey was skipped. |
| `curves_hit_rate.csv`, `curves_precision.csv` | Tidy threshold-vs-metric tables, 5 thresholds × 6 (model, workload) pairs, carrying the harness zero-division flags so degenerate cells stay visible. |
| `kappa.json` | Cohen's kappa **sourced from the dataset tooling** (`levy.dataset.kappa`, not reimplemented), with the 2×2 contingency, the 0.7 bar, and a provenance block that labels fixture-derived values `FIXTURE ONLY`. |
| `figures/` | `curve_hit_rate.{png,pdf}`, `curve_precision.{png,pdf}` — regenerable from the curve tables alone; the hit-rate figure carries the frozen 30% economic-viability reference line. |
| `analysis_meta.json` | Input paths, ANOVA diagnostics (design balance, Shapiro-Wilk, Levene), and library versions. The **only** place versions and a timestamp appear. |

Every CSV is timestamp-free and byte-identical across re-runs on identical
input, matching the harness's determinism convention.

**Degenerate results are reported, not laundered.** If false positive rate
has zero variance across all configurations — which is exactly what the
synthetic fixture produces under mock embeddings, since no semantic hits
occur — the F-tests are undefined, and the three hypotheses are reported as
`undefined` rather than as retained nulls.

### Replication check (±5%)

Frozen Success Criterion 3: headline results replicate within ±5%.
`scripts/check_replication.py` re-runs the harness over exactly the grid
recorded in a reference `results.csv`, then compares precision and recall per
configuration:

```bash
python scripts/check_replication.py --reference results/run-001/results.csv
```

Tolerance rule (stated in every report, so the criterion stays auditable):

```
|candidate - reference| <= max(0.01, 5% * |reference|)
```

The absolute floor exists because a purely relative tolerance collapses to
demanding bit-exact equality near a reference of 0.0. Exit code is zero when
every value is within tolerance; otherwise non-zero, with a per-configuration
diff table naming the configuration, the metric, both values, and the
deviation. Under mock providers the harness is byte-deterministic, so a
self-comparison matches exactly; the tolerance is there for real-provider
runs.

The verdict is also written to `replication.json` beside the reference, so it
can be read rather than inferred from an exit code. It records which
configurations it covers, so a run over part of the grid cannot be mistaken for
one covering all of it.

## Results dashboard (D6, desirable)

`levy/dashboard/` + `scripts/dashboard.py` is a local Streamlit viewer over an
analysis bundle (the output of `scripts/run_analysis.py`): threshold-vs-metric
curves per (model, workload) with degenerate points flagged, the ANOVA/Tukey/κ
summary, and a live query box that reports the cache decision for your own
text. It **never recomputes a statistic** — everything shown is read from the
bundle. D6 is the frozen plan's lowest-priority, desirable-only deliverable
(see `openspec/changes/add-results-dashboard/proposal.md`); it is not part of
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

**Offline story:** the query panel's "mock" embedding provider is fully
offline and deterministic — no network, no model download. Switching it to
"sentence-transformers" downloads model weights on first use and needs
network access. If the bundle directory is missing or incomplete, the app
reports exactly what's absent and the command to generate it
(`scripts/reproduce.sh`), and stops cleanly rather than showing a traceback.

## License

Apache-2.0
