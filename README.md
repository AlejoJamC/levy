# Levy

**Levy** is a semantic caching engine for LLM APIs, built as the IT artefact of an MSc Artificial Intelligence capstone project (University of Liverpool). It sits between your application and an LLM provider (Mock, OpenAI-compatible, Ollama, or Anthropic) to optimize costs and latency by reusing responses for identical or semantically similar prompts.

The research behind Levy benchmarks false positive rates of semantic caching across embedding models (all-MiniLM vs ModernBERT), workloads (FAQ, code, chat), and similarity thresholds. The authoritative project definition lives in [docs/Project_Proposal.md](docs/Project_Proposal.md) and [docs/Specification_and_Design_Report.md](docs/Specification_and_Design_Report.md) (university submissions — do not modify).

## Features

- **Exact Match Caching**: Extremely fast retrieval for identical prompts.
- **Semantic Caching**: Uses vector embeddings (via `sentence-transformers`) to find and reuse answers for similar meaning queries (e.g., "What is the capital of France?" vs "Tell me France's capital").
- **Metrics**: Automatically tracks cache hit rates, latency and estimated token savings.
- **Pluggable Architecture**: Easy to swap LLM providers or Vector Stores.

## Project Structure

```
levy/
├── levy/                    # Core package
│   ├── api/                 # FastAPI router (LEV-7): app, Pydantic schemas, engine pool
│   ├── cache/               # Cache logic (Exact, Semantic, InMemory/Redis stores)
│   ├── llm_client.py        # LLM interaction (Mock, OpenAI, Ollama, Anthropic)
│   ├── embeddings.py        # EmbeddingClient ABC + Mock, SentenceTransformer, Ollama
│   ├── embedding_manager.py # EmbeddingManager: study-model registry, runtime switching,
│   │                        #   memoization, symmetric prefix handling
│   ├── engine.py            # Main orchestration engine
│   ├── config.py            # LevyConfig (providers, thresholds, store)
│   ├── metrics.py           # Hit/miss/latency/token-savings tracking
│   ├── models.py            # Data classes
│   ├── dataset/              # Ground-truth dataset platform (LEV-3): schema, CSV/JSON
│   │                        #   I/O, seeded sampling, blind re-annotation, Cohen's kappa
│   └── experiment/          # Experiment harness (LEV-4): grid, replay, metrics, sweep runner
├── scripts/                 # CLIs over levy/dataset + levy/experiment
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

### Running the Experiment Script

A replay script is provided to demonstrate the cache behavior:

```bash
python examples/simple_replay.py
```

It runs a sequence of prompts through three configurations:
1. No Cache
2. Exact Cache Only
3. Semantic Cache (uses `sentence-transformers` if available)


### Running with Ollama (Local Models)

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

### Using Anthropic (LEV-6)

The `anthropic` provider wraps the official `anthropic` SDK behind the same
synchronous `LLMClient` interface as Mock/OpenAI/Ollama:

```python
config = LevyConfig(
    llm_provider="anthropic",
    # anthropic_api_key defaults to the ANTHROPIC_API_KEY env var (.env)
    anthropic_model="claude-opus-4-8",       # default; override per config
    anthropic_max_retries=2,                 # SDK's own exponential backoff
    anthropic_budget_cap_usd=200.0,          # hard stop (frozen budget cap)
    anthropic_input_price_per_mtok=5.0,      # USD / 1M input tokens
    anthropic_output_price_per_mtok=25.0,    # USD / 1M output tokens
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
  `anthropic_model` defaults to the current recommended model
  (`claude-opus-4-8`) instead.

**One-time real-API smoke check** (not part of the offline test suite — it
lives in `examples/`, is not named `test_*.py`, and makes one real, billed
API call using `ANTHROPIC_API_KEY` from `.env`):

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

## HTTP API (LEV-7)

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
| Anthropic budget cap reached (LEV-6) | 402 | `budget_exceeded` (includes `cap_usd`/`estimated_cost_usd`) |
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
the whole call chain (engine, caches, the LEV-6 Anthropic client) is
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

> **Faiss HNSW** (implemented in LEV-2) is the production vector index. Install via conda to avoid Apple-Silicon segfaults:
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

## Ground-truth dataset tooling (LEV-3)

`levy/dataset/` + `scripts/` provide the data-agnostic platform for D2 (900
annotated query pairs across 3 workloads: FAQ, code, chat). This is
**tooling only** — `data/ground_truth.csv` / `data/ground_truth.json`
currently ship 15 synthetic fixture pairs (5/workload, obviously fake text,
`source_corpus="synthetic-fixture"`), not the real dataset. See
`data/README.md` and `data/DATASHEET.md` for the full protocol and the
platform-vs-data-production split.

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
original corpus label) is the contract the experiment harness (LEV-4)
replays against.

## Experiment harness (LEV-4)

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

# Real study run (once the real 900-pair dataset from LEV-11 lands):
python scripts/run_experiments.py --dataset data/ground_truth.csv \
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
900-pair dataset (LEV-11, not yet delivered).

## License

Apache-2.0
