import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Which LLM and Ollama embedding models exist, and what Anthropic charges for
# them, changes every few months. None of it is written in the code: it is read
# from the environment (`.env`, documented in `.env.example`). LEV-23, design D14.
ANTHROPIC_MODELS_ENV = "LEVY_ANTHROPIC_MODELS"            # "model:input_usd_per_mtok:output_usd_per_mtok,..."
OLLAMA_MODELS_ENV = "LEVY_OLLAMA_MODELS"                  # "model,model,..."
OLLAMA_EMBEDDING_MODELS_ENV = "LEVY_OLLAMA_EMBEDDING_MODELS"  # "model,model,..."


def _env_list(name: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, "").split(",") if item.strip())


def _anthropic_models_from_env() -> Tuple[Tuple[str, float, float], ...]:
    """Each entry keeps a model and its two prices together: they describe one model."""
    models = []
    for item in _env_list(ANTHROPIC_MODELS_ENV):
        parts = [part.strip() for part in item.split(":")]
        try:
            if len(parts) != 3 or not parts[0]:
                raise ValueError
            models.append((parts[0], float(parts[1]), float(parts[2])))
        except ValueError:
            raise ValueError(
                f"{ANTHROPIC_MODELS_ENV}: {item!r} is not 'model:input_usd_per_mtok:output_usd_per_mtok'"
            ) from None
    return tuple(models)


def _first_ollama(name: str) -> Optional[str]:
    models = _env_list(name)
    return models[0] if models else None


def _first_anthropic(index: int):
    """Field `index` (0 model, 1 input price, 2 output price) of the default Anthropic entry."""
    models = _anthropic_models_from_env()
    return models[0][index] if models else None


@dataclass
class LevyConfig:
    """
    Configuration for the Levy Engine.
    """
    # LLM settings
    llm_provider: str = "mock"  # "mock", "ollama", "anthropic"
    mock_llm_latency_seconds: float = 0.5  # simulated delay for MockLLMClient; tests inject 0
    ollama_base_url: str = "http://localhost:11434"
    # Ollama models from LEVY_OLLAMA_MODELS; the first is the default (`model_name`).
    ollama_models: Tuple[str, ...] = field(default_factory=lambda: _env_list(OLLAMA_MODELS_ENV))
    model_name: Optional[str] = field(default_factory=lambda: _first_ollama(OLLAMA_MODELS_ENV))
    # Ollama embedding models from LEVY_OLLAMA_EMBEDDING_MODELS.
    ollama_embedding_models: Tuple[str, ...] = field(default_factory=lambda: _env_list(OLLAMA_EMBEDDING_MODELS_ENV))

    # Anthropic settings (LEV-6)
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    # Model and its two per-MTok prices are one triple describing one model (LEV-14):
    # all three come from the first entry of LEVY_ANTHROPIC_MODELS, together.
    anthropic_models: Tuple[Tuple[str, float, float], ...] = field(default_factory=_anthropic_models_from_env)
    anthropic_model: Optional[str] = field(default_factory=lambda: _first_anthropic(0))
    anthropic_max_retries: int = 2
    anthropic_budget_cap_usd: float = 200.0  # budget hard cap (~$50 expected spend)
    anthropic_input_price_per_mtok: Optional[float] = field(default_factory=lambda: _first_anthropic(1))
    anthropic_output_price_per_mtok: Optional[float] = field(default_factory=lambda: _first_anthropic(2))
    
    # Embedding settings
    embedding_provider: str = "sentence-transformers"  # "mock", "sentence-transformers", "ollama"
    embedding_model: str = "all-MiniLM-L6-v2"  # study baseline; use "modernbert" for the other study model
    
    # Storage settings
    cache_store_type: str = "memory" # "memory", "redis"
    redis_url: str = "redis://localhost:6379/0"
    
    # Cache settings
    enable_exact_cache: bool = True
    enable_semantic_cache: bool = True
    # Threshold in 1/(1+L2) similarity space (NOT cosine). Study sweep: 0.70–0.90 step 0.05.
    # For unit-norm vectors: similarity = 1/(1+sqrt(2-2*cosine)).
    # cosine≈0.95 → sim≈0.76; cosine≈0.99 → sim≈0.88; cosine≈0.90 → sim≈0.69.
    similarity_threshold: float = 0.85
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size: int = 1000  # Max number of entries in memory

    # Vector index settings (LEV-2)
    vector_index_backend: str = "auto"  # "auto" | "faiss" | "brute_force"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    def anthropic_prices(self, model: Optional[str]) -> Optional[Tuple[float, float]]:
        """(input, output) USD per MTok configured for `model`, or None when it is not configured."""
        for name, input_price, output_price in self.anthropic_models:
            if name == model:
                return input_price, output_price
        return None
