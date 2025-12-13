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
    llm_provider: str = "mock"  # "mock", "openai", "ollama"
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "llama3.2"  # Default change to local friendly
    
    # Embedding settings
    embedding_provider: str = "mock" # "mock", "sentence-transformers", "ollama"
    embedding_model: str = "mxbai-embed-large"
    
    # Storage settings
    cache_store_type: str = "memory" # "memory", "redis"
    redis_url: str = "redis://localhost:6379/0"
    
    # Cache settings
    enable_exact_cache: bool = True
    enable_semantic_cache: bool = True
    similarity_threshold: float = 0.85  # Cosine similarity threshold (0.0 to 1.0)
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size: int = 1000  # Max number of entries in memory
