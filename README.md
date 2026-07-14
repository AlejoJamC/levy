# Levy

**Levy** is a semantic caching engine for LLM APIs, built as the IT artefact of an MSc Artificial Intelligence capstone project (University of Liverpool). It sits between your application and an LLM provider (Mock, OpenAI-compatible, or Ollama today; Anthropic connector planned) to optimize costs and latency by reusing responses for identical or semantically similar prompts.

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
│   ├── cache/               # Cache logic (Exact, Semantic, InMemory/Redis stores)
│   ├── llm_client.py        # LLM interaction (Mock, OpenAI, Ollama)
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

### Using Redis Stack (Docker)

To use Redis for persistence:

1. Start Redis:
   ```bash
   docker-compose up -d
   ```
2. Configure `LevyConfig` to use `cache_store_type="redis"`.

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
