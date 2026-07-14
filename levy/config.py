import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LevyConfig:
    """
    Configuration for the Levy Engine.
    """
    # LLM settings
    llm_provider: str = "mock"  # "mock", "openai", "ollama", "anthropic"
    mock_llm_latency_seconds: float = 0.5  # simulated delay for MockLLMClient; tests inject 0
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "qwen3"  # default local model for Ollama; override per deployment

    # Anthropic settings (LEV-6)
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    # S&D Report's example model (claude-3-sonnet-20240229) is retired; defaulting to the
    # current recommended model is intentional frozen-doc drift, not silent resolution.
    anthropic_model: str = "claude-opus-4-8"
    anthropic_max_retries: int = 2
    anthropic_budget_cap_usd: float = 200.0  # frozen budget hard cap (~$50 expected spend)
    anthropic_input_price_per_mtok: float = 5.0  # USD per 1M input tokens, claude-opus-4-8
    anthropic_output_price_per_mtok: float = 25.0  # USD per 1M output tokens, claude-opus-4-8
    
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
