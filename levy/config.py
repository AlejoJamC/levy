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
    llm_provider: str = "mock"  # "mock" or "openai"
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-3.5-turbo"
    
    # Embedding settings
    embedding_provider: str = "mock" # "mock" or "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Cache settings
    enable_exact_cache: bool = True
    enable_semantic_cache: bool = True
    similarity_threshold: float = 0.85  # Cosine similarity threshold (0.0 to 1.0)
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size: int = 1000  # Max number of entries in memory
