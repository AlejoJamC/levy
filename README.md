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
│   └── dataset/             # Ground-truth dataset platform (LEV-3): schema, CSV/JSON
│                            #   I/O, seeded sampling, blind re-annotation, Cohen's kappa
├── scripts/                 # CLIs over levy/dataset (sample/annotate/kappa/export)
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

```bash
python -m unittest discover -s tests -p "test_*.py"
# or, if pytest is installed:
python -m pytest tests/ -q
```

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
original corpus label) is the contract the experiment harness (LEV-4) will
replay against.

## License

Apache-2.0
