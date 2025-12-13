import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

@dataclass
class LLMRequest:
    """Standardized request object."""
    prompt: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_tokens: int = 256
    temperature: float = 0.7
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Standardized response from LLM."""
    text: str
    token_usage: int = 0
    model: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheEntry:
    """Entry stored in the cache."""
    key_hash: str  # Hash of the prompt for exact matching or ID
    prompt: str
    response_text: str
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

@dataclass
class LevyResult:
    """Final result returned to the user."""
    answer: str
    source: Literal["llm", "exact_cache", "semantic_cache"]
    latency_ms: float
    similarity_score: Optional[float] = None
    original_response: Optional[LLMResponse] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics."""
    total_requests: int
    exact_hits: int
    semantic_hits: int
    misses: int
    tokens_saved: int
    avg_latency_ms: float
