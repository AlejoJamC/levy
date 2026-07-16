"""
Pydantic v2 request/response models for the Levy API layer (LEV-7).

Pydantic is sanctioned at this boundary only (CLAUDE.md conventions); the core
package (`levy/engine.py`, `levy/models.py`, ...) stays plain dataclasses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One entry of the incoming `messages` list."""

    role: str
    content: str


class CacheConfigRequest(BaseModel):
    """Per-request override of the engine's (embedding_model, threshold) pair.

    Omitted fields fall back to the base `LevyConfig` defaults (design.md D2).
    """

    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_model: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """`POST /v1/chat/completions` request body, per the frozen S&D contract."""

    messages: List[ChatMessage] = Field(min_length=1)
    model: Optional[str] = None
    cache_config: Optional[CacheConfigRequest] = None


class ContentBlock(BaseModel):
    type: str = "text"
    text: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Anthropic Messages-API-shaped response body, identical for hits and misses.

    `request_id` is additive (not part of the Anthropic schema) so the response
    echoes the same id used in the replay-grade structured log record (design.md D8).
    """

    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[ContentBlock]
    stop_reason: Optional[str] = "end_turn"
    usage: Usage
    request_id: str


class StatsResponse(BaseModel):
    """`GET /admin/cache/stats` response: counters aggregated across every pooled engine."""

    total_requests: int
    exact_hits: int
    semantic_hits: int
    misses: int
    hit_rate: float
    tokens_saved: int
    avg_latency_ms: float
    index_size: int
    model_breakdown: Dict[str, int]


class ClearResponse(BaseModel):
    """`POST /admin/cache/clear` response: per pool-key counts of what was cleared."""

    cleared: Dict[str, Dict[str, int]]
    message: str = "Cache cleared."


class ErrorResponse(BaseModel):
    """Structured error body for pool-cap, budget-guard, refusal, and provider errors."""

    error: str
    detail: str
    cap_usd: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
