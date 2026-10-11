"""
FastAPI router exposing the Levy caching engine over HTTP (LEV-7).

Endpoints (intended interface + known-gap #1):
  POST /v1/chat/completions  -- cache-aware chat proxy, Anthropic-format body,
                                 X-Cache-Status / X-Cache-Similarity headers.
  GET  /admin/cache/stats    -- aggregated hit rate, index size, per-model breakdown.
  POST /admin/cache/clear    -- empties every pooled engine's caches + metrics.

Console section, beside the delivered endpoints above and sharing none
of their engines or caches -- see `levy/console/`:
  POST /console/chat, GET /console/traces, POST /console/traces/clear,
  POST /console/cache/clear, and the page at GET /console.

Design decision (recorded in design.md): endpoints are declared `def` (sync) so
FastAPI runs them in its threadpool -- the whole call chain (engine, caches, the
LEV-6 Anthropic client) is synchronous and blocking the event loop would
serialize all requests. This satisfies the "asynchronous wrapper" intent
at the HTTP boundary without an AsyncAnthropic migration.

Run with: uvicorn levy.api.app:app
"""

import copy
import json
import logging
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from levy import __version__
from levy.api.pool import EnginePool, PoolCapExceededError
from levy.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ClearResponse,
    ContentBlock,
    ErrorResponse,
    StatsResponse,
    Usage,
)
from levy.config import LevyConfig, current_model_catalog
from levy.console.pool import DEFAULT_CONSOLE_POOL_CAP, ConsolePool, EmbeddingManagerFactory, LLMClientFactory
from levy.console.routes import STATIC_DIR, build_console_router, publish_choices
from levy.console.tracing import DEFAULT_TRACE_BUFFER_SIZE, TraceRecorder
from levy.llm_client import AnthropicRefusalError, BudgetExceededError
from levy.models import LevyResult

logger = logging.getLogger("levy.api")

DEFAULT_POOL_CAP = 8


def _extract_prompt(messages: List[ChatMessage]) -> str:
    """The API contract carries the conversation in `messages`; the engine's
    single-turn `generate(prompt)` surface takes the most recent message."""
    return messages[-1].content


def _to_response_body(
    result: LevyResult, requested_model: Optional[str], request_id: str
) -> ChatCompletionResponse:
    if result.source == "llm" and result.original_response is not None:
        model = result.original_response.model
        input_tokens = result.original_response.metadata.get("input_tokens", 0)
        output_tokens = result.original_response.metadata.get("output_tokens", 0)
        stop_reason = result.original_response.metadata.get("stop_reason", "end_turn")
    else:
        # Cache hit: synthesize the same shape from the cached entry. Usage is
        # zeroed (that's the point of the cache); model identity comes from the
        # entry's stored embedding-model metadata, falling back to the request.
        model = result.metadata.get("canonical_name") or requested_model or "unknown"
        input_tokens = 0
        output_tokens = 0
        stop_reason = "end_turn"

    return ChatCompletionResponse(
        id=f"msg_{request_id}",
        model=model,
        content=[ContentBlock(type="text", text=result.answer)],
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        request_id=request_id,
    )


def _aggregate_stats(pool: EnginePool) -> StatsResponse:
    total_requests = exact_hits = semantic_hits = misses = tokens_saved = 0
    index_size = 0
    model_breakdown: dict = {}
    latencies: List[float] = []

    for engine in pool.all_engines():
        snap = engine.metrics.get_snapshot()
        total_requests += snap.total_requests
        exact_hits += snap.exact_hits
        semantic_hits += snap.semantic_hits
        misses += snap.misses
        tokens_saved += snap.tokens_saved
        latencies.extend(engine.metrics.latencies)

        stats = engine.get_cache_stats()
        index_size += stats["index_size"]
        for name, count in stats["model_breakdown"].items():
            model_breakdown[name] = model_breakdown.get(name, 0) + count

    hit_rate = (exact_hits + semantic_hits) / total_requests if total_requests else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return StatsResponse(
        total_requests=total_requests,
        exact_hits=exact_hits,
        semantic_hits=semantic_hits,
        misses=misses,
        hit_rate=hit_rate,
        tokens_saved=tokens_saved,
        avg_latency_ms=avg_latency,
        index_size=index_size,
        model_breakdown=model_breakdown,
    )


def create_app(
    config: Optional[LevyConfig] = None,
    max_engines: int = DEFAULT_POOL_CAP,
    console_max_engines: int = DEFAULT_CONSOLE_POOL_CAP,
    trace_buffer_size: int = DEFAULT_TRACE_BUFFER_SIZE,
    console_llm_client_factory: Optional[LLMClientFactory] = None,
    console_embedding_manager_factory: Optional[EmbeddingManagerFactory] = None,
) -> FastAPI:
    """Build a Levy API app. Tests pass a mock-provider `config` for offline runs.

    The console section gets its own engine pool and trace recorder;
    the two `console_*_factory` arguments let the offline tests give it test
    doubles, and are left unset to serve the real providers. Without an explicit
    `config`, the console reads its models from `.env` again on every schema
    load and request, so an edited `.env` needs no restart.
    """
    reload_models = config is None
    config = config or LevyConfig()
    pool = EnginePool(config, max_engines=max_engines)

    app = FastAPI(
        title="Levy Semantic Caching API",
        description=(
            "HTTP interface to the Levy semantic caching engine. "
            "`POST /v1/chat/completions` proxies through exact/semantic caches to "
            "the configured LLM provider, reporting `X-Cache-Status: HIT|MISS` and, "
            "on hits, `X-Cache-Similarity`. `GET/POST /admin/cache/*` expose "
            "observability and maintenance."
        ),
        version=__version__,
    )
    app.state.pool = pool

    @app.exception_handler(PoolCapExceededError)
    def _handle_pool_cap(request: Request, exc: PoolCapExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="pool_cap_exceeded", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(BudgetExceededError)
    def _handle_budget(request: Request, exc: BudgetExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=402,
            content=ErrorResponse(
                error="budget_exceeded",
                detail=str(exc),
                cap_usd=exc.cap_usd,
                estimated_cost_usd=exc.estimated_cost_usd,
            ).model_dump(),
        )

    @app.exception_handler(AnthropicRefusalError)
    def _handle_refusal(request: Request, exc: AnthropicRefusalError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error="provider_refusal", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(Exception)
    def _handle_generic(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="provider_error", detail=str(exc)).model_dump(),
        )

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        summary="Chat completion proxied through the cache",
        description=(
            "Extracts the prompt from the last `messages` entry, resolves an engine "
            "for the requested `cache_config` (embedding_model/threshold), and "
            "serves the response via exact cache -> semantic cache -> LLM provider. "
            "Sets `X-Cache-Status: HIT|MISS` and, on hits, `X-Cache-Similarity` "
            "(1.0 for exact-cache hits)."
        ),
    )
    def chat_completions(
        payload: ChatCompletionRequest, response: Response
    ) -> ChatCompletionResponse:
        request_id = str(uuid.uuid4())
        arrival = time.time()

        prompt = _extract_prompt(payload.messages)
        cache_config = payload.cache_config
        embedding_model = cache_config.embedding_model if cache_config else None
        threshold = cache_config.threshold if cache_config else None

        engine = pool.get(embedding_model, threshold)
        result = engine.generate(prompt)

        completion = time.time()
        latency_ms = (completion - arrival) * 1000

        body = _to_response_body(result, payload.model, request_id)

        response.headers["X-Cache-Status"] = "MISS" if result.source == "llm" else "HIT"
        if result.source != "llm":
            response.headers["X-Cache-Similarity"] = str(result.similarity_score)

        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "arrival_ts": arrival,
                    "completion_ts": completion,
                    "embedding_model": engine.config.embedding_model,
                    "threshold": engine.config.similarity_threshold,
                    "prompt": prompt,
                    "cache_source": result.source,
                    "similarity": result.similarity_score,
                    "latency_ms": latency_ms,
                }
            )
        )

        return body

    @app.get(
        "/admin/cache/stats",
        response_model=StatsResponse,
        summary="Aggregated cache statistics",
        description=(
            "Hit rate, semantic-index size, and per-model cached-entry counts, "
            "aggregated across every pooled (embedding_model, threshold) engine."
        ),
    )
    def cache_stats() -> StatsResponse:
        return _aggregate_stats(pool)

    @app.post(
        "/admin/cache/clear",
        response_model=ClearResponse,
        summary="Clear all pooled caches",
        description=(
            "Empties the exact and semantic caches of every pooled engine instance "
            "and resets metrics counters, reporting per-key entry counts cleared."
        ),
    )
    def cache_clear() -> ClearResponse:
        return ClearResponse(cleared=pool.clear_all())

    console_pool = ConsolePool(
        config,
        max_engines=console_max_engines,
        llm_client_factory=console_llm_client_factory,
        embedding_manager_factory=console_embedding_manager_factory,
    )
    recorder = TraceRecorder(max_traces=trace_buffer_size)
    app.state.console_pool = console_pool
    app.state.trace_recorder = recorder

    def console_config() -> LevyConfig:
        return config.with_models(current_model_catalog()) if reload_models else config

    app.include_router(build_console_router(console_config, console_pool, recorder, _to_response_body))
    app.mount("/console/assets", StaticFiles(directory=STATIC_DIR), name="console-assets")

    cached_openapi = app.openapi

    def openapi_with_current_choices():
        # FastAPI builds the schema once and caches it; the console's options
        # change with `.env`, so they are written into a copy on every request.
        return publish_choices(copy.deepcopy(cached_openapi()), console_config())

    app.openapi = openapi_with_current_choices

    return app


app = create_app()
