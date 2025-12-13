# Levy

**Levy** is a semantic caching engine for LLM APIs, designed as a research prototype for a Computer Science Capstone project. It sits between your application and an LLM provider (like OpenAI) to optimize costs and latency by reusing responses for identical or semantically similar prompts.

## Features

- **Exact Match Caching**: Extremely fast retrieval for identical prompts.
- **Semantic Caching**: Uses vector embeddings (via `sentence-transformers`) to find and reuse answers for similar meaning queries (e.g., "What is the capital of France?" vs "Tell me France's capital").
- **Metrics**: Automatically tracks cache hit rates, latency and estimated token savings.
- **Pluggable Architecture**: Easy to swap LLM providers or Vector Stores.

## Project Structure

```
levy/
├── levy/               # Core package
│   ├── cache/          # Cache logic (Exact, Semantic, Store)
│   ├── llm_client.py   # LLM interaction (Mock, OpenAI)
│   ├── embeddings.py   # Vector embedding logic
│   ├── engine.py       # Main orchestration engine
│   └── models.py       # Data classes
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
