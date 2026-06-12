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
├── levy/               # Core package
│   ├── cache/          # Cache logic (Exact, Semantic, InMemory/Redis stores)
│   ├── llm_client.py   # LLM interaction (Mock, OpenAI, Ollama)
│   ├── embeddings.py   # Embedding clients (Mock, sentence-transformers, Ollama)
│   ├── engine.py       # Main orchestration engine
│   ├── config.py       # LevyConfig (providers, thresholds, store)
│   ├── metrics.py      # Hit/miss/latency/token-savings tracking
│   └── models.py       # Data classes
├── docs/               # Research docs (proposal & S&D report are frozen)
├── examples/           # Demo scripts
└── tests/              # Unit tests
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
   ollama pull llama3.2
   ollama pull mxbai-embed-large
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
    similarity_threshold=0.85
)
```

## License

Apache-2.0
