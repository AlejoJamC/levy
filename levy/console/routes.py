"""
The `/console/*` endpoints: the API's console section, beside the delivered ones.

`POST /console/chat` takes the prompt and every setting as a query parameter.
Each list-type parameter is published in the OpenAPI schema with its closed
list of options (`levy.console.options`), so Swagger renders it as a selector
and a practitioner picks providers, models, caches, backend and store instead
of guessing them in raw JSON. A request body would not do: Swagger shows a
body as an editable JSON text area.

The model lists come from `.env`, read again whenever the schema is served and
whenever a request is validated (`config_provider`), so a model added to `.env`
appears after refreshing Swagger, without restarting Levy.

The answer has the delivered chat endpoint's body and headers; the page reads
what happened from `GET /console/traces`.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from levy.api.pool import PoolCapExceededError
from levy.api.schemas import ChatCompletionResponse
from levy.config import LevyConfig
from levy.console.options import (
    CACHE_STORES,
    NO_LLM_MODELS_MESSAGE,
    SETTING_NAMES,
    VECTOR_INDEX_BACKENDS,
    ConsoleOptions,
    ConsoleSettings,
)

ConfigProvider = Callable[[], LevyConfig]
from levy.console.pool import ConsolePool, store_entry_count
from levy.console.tracing import TraceRecorder, LiveTimingCollector, llm_block, observation_dict
from levy.latency.timing import SEGMENT_LLM_CALL, SEGMENT_TOTAL_LOOKUP
from levy.llm_client import AnthropicRefusalError, BudgetExceededError
from levy.models import LevyResult, SemanticLookupObservation

STATIC_DIR = Path(__file__).resolve().parent / "static"

OUTCOMES = {"exact_cache": "exact_hit", "semantic_cache": "semantic_hit", "llm": "miss"}


def _error_kind(exc: Exception) -> str:
    """The `error` field of the structured response the existing handlers produce."""
    if isinstance(exc, PoolCapExceededError):
        return "pool_cap_exceeded"
    if isinstance(exc, BudgetExceededError):
        return "budget_exceeded"
    if isinstance(exc, AnthropicRefusalError):
        return "provider_refusal"
    return "provider_error"


def choices(config: LevyConfig) -> Tuple[ConsoleOptions, Optional[ConsoleSettings], Dict[str, Tuple[str, ...]]]:
    """The options, the defaults, and each list-type parameter's allowed values, for `config`."""
    options = ConsoleOptions.from_config(config)
    allowed = {
        "embedding_provider": options.embedding_providers,
        "embedding_model": options.all_embedding_models,
        "llm_provider": options.llm_providers,
        "llm_model": options.all_llm_models,
        "vector_index_backend": VECTOR_INDEX_BACKENDS,
        "cache_store": CACHE_STORES,
    }
    return options, options.default_settings(config), allowed


def publish_choices(schema: Dict[str, Any], config: LevyConfig) -> Dict[str, Any]:
    """
    Put each `/console/chat` parameter's current options and default into an
    OpenAPI schema, so Swagger shows them as selectors with the default
    pre-selected. An enum is written even for a single option (a one-value
    Literal would be published as `const`, which Swagger renders as a text box).
    """
    _, defaults, allowed = choices(config)
    operation = schema.get("paths", {}).get("/console/chat", {}).get("post")
    if operation is None:
        return schema
    for parameter in operation.get("parameters", []):
        name = parameter["name"]
        if name not in SETTING_NAMES:
            continue
        title = parameter["schema"].get("title", name)
        if name == "threshold":
            parameter["schema"] = {"type": "number", "minimum": 0.0, "maximum": 1.0, "title": title}
        elif name in ("exact_cache", "semantic_cache"):
            parameter["schema"] = {"type": "boolean", "title": title}
        elif allowed[name]:
            parameter["schema"] = {"type": "string", "enum": list(allowed[name]), "title": title}
        else:
            parameter["schema"] = {"type": "string", "title": title}
        if defaults is not None:
            parameter["schema"]["default"] = getattr(defaults, name)
    return schema


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _embedding_identity(engine, settings: ConsoleSettings, embedded: bool) -> Dict[str, Any]:
    """
    The embedding model behind the request. The resolved identity (with its
    dimension) is read only when the request already embedded something, so
    tracing never loads a model the request itself did not load.
    """
    if embedded:
        return {"provider": settings.embedding_provider, **engine.embedding_manager.get_model_identity().as_dict()}
    return {
        "provider": settings.embedding_provider,
        "canonical_name": settings.embedding_model,
        "checkpoint": None,
        "dimension": None,
    }


def build_console_router(
    config_provider: ConfigProvider,
    pool: ConsolePool,
    recorder: TraceRecorder,
    response_builder: Callable[[LevyResult, Optional[str], str], ChatCompletionResponse],
) -> APIRouter:
    router = APIRouter(prefix="/console", tags=["console"])

    def resolve(name: str, value: Any, defaults: ConsoleSettings, allowed: Dict[str, Tuple[str, ...]], errors: List[Dict[str, Any]]):
        if value is None:
            return getattr(defaults, name)
        if name in allowed and value not in allowed[name]:
            errors.append({
                "loc": ["query", name],
                "msg": f"{value!r} is not an option; valid: {', '.join(allowed[name])}",
                "type": "value_error",
            })
        return value

    @router.post(
        "/chat",
        response_model=ChatCompletionResponse,
        summary="Chat with a configuration chosen per request",
    )
    def console_chat(
        request: Request,
        response: Response,
        prompt: str = Query(..., min_length=1),
        threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
        embedding_provider: Optional[str] = Query(None),
        embedding_model: Optional[str] = Query(None),
        llm_provider: Optional[str] = Query(None),
        llm_model: Optional[str] = Query(None),
        exact_cache: Optional[bool] = Query(None),
        semantic_cache: Optional[bool] = Query(None),
        vector_index_backend: Optional[str] = Query(None),
        cache_store: Optional[str] = Query(None),
    ) -> ChatCompletionResponse:
        config = config_provider()
        options, defaults, allowed = choices(config)
        if defaults is None:
            raise HTTPException(status_code=503, detail=NO_LLM_MODELS_MESSAGE)
        given = {
            "threshold": threshold,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "exact_cache": exact_cache,
            "semantic_cache": semantic_cache,
            "vector_index_backend": vector_index_backend,
            "cache_store": cache_store,
        }
        errors: List[Dict[str, Any]] = []
        settings = ConsoleSettings(**{name: resolve(name, value, defaults, allowed, errors) for name, value in given.items()})
        if errors:
            raise HTTPException(status_code=422, detail=errors)
        mismatch = options.mismatched_model(settings)
        if mismatch is not None:
            name, valid = mismatch
            provider = settings.embedding_provider if name == "embedding_model" else settings.llm_provider
            raise HTTPException(
                status_code=422,
                detail=[{
                    "loc": ["query", name],
                    "msg": f"{getattr(settings, name)!r} is not a model of {provider!r}; valid: {', '.join(valid)}",
                    "type": "value_error",
                }],
            )

        arrived_at = time.time()
        t0 = time.perf_counter()
        skipped = [name for name in ("exact_cache", "semantic_cache") if not getattr(settings, name)]
        trace_id = recorder.open({
            "arrived_at": arrived_at,
            "config": settings.as_dict(),
            "config_key": settings.key,
            "config_from_request": [name for name in SETTING_NAMES if name in request.query_params],
            "skipped": skipped,
            "prompt": prompt,
        })
        observation = SemanticLookupObservation()
        collector = LiveTimingCollector(recorder, trace_id, t0, observation=observation)

        try:
            engine = pool.get(settings, config)
            result = engine.generate(prompt, timing=collector, observation=observation)
        except Exception as exc:
            recorder.finish(
                trace_id,
                outcome="error",
                total_ms=(time.perf_counter() - t0) * 1000.0,
                semantic=observation_dict(observation) if observation.reached else None,
                error={
                    "kind": _error_kind(exc),
                    "detail": str(exc),
                    "stage": recorder.current_stage(trace_id) or "router",
                },
            )
            raise

        total_ms = (time.perf_counter() - t0) * 1000.0
        outcome = OUTCOMES[result.source]
        prompt_key = (settings.key, _prompt_key(prompt))
        embedded = bool(collector.samples("embed")) or (outcome == "miss" and settings.semantic_cache)

        fields: Dict[str, Any] = {
            "outcome": outcome,
            "answer": result.answer,
            "total_ms": total_ms,
            "lookup_ms": collector.total(SEGMENT_TOTAL_LOOKUP),
            "similarity": result.similarity_score,
            "semantic": observation_dict(observation) if observation.reached else None,
            "embedding": _embedding_identity(engine, settings, embedded),
            "llm": None,
            "savings": None,
            "cache_after": {
                "exact_entries": store_entry_count(engine.store),
                "index_size": engine.semantic_cache.size(),
            },
            "error": None,
        }

        if outcome == "miss":
            llm = llm_block(settings, config, engine.llm_client, result.original_response, collector.total(SEGMENT_LLM_CALL))
            fields["llm"] = llm
            recorder.remember_origin(prompt_key, {
                "origin_id": trace_id,
                "llm_ms": llm["duration_ms"],
                "input_tokens": llm["input_tokens"],
                "output_tokens": llm["output_tokens"],
                "total_tokens": llm["total_tokens"],
                "cost_usd": llm["cost_usd"],
            })
        else:
            matched = prompt if outcome == "exact_hit" else observation.nearest_prompt
            if matched is not None:
                fields["savings"] = recorder.origin((settings.key, _prompt_key(matched)))
            fields["matched_prompt"] = matched

        recorder.finish(trace_id, **fields)

        body = response_builder(result, None, trace_id)
        response.headers["X-Cache-Status"] = "MISS" if result.source == "llm" else "HIT"
        if result.source != "llm":
            response.headers["X-Cache-Similarity"] = str(result.similarity_score)
        return body

    @router.get(
        "/traces",
        summary="Live trace feed for the console page",
    )
    def console_traces(after: int = Query(0, ge=0)) -> Dict[str, Any]:
        _, defaults, _ = choices(config_provider())
        engines = pool.items()
        exact_hits = semantic_hits = misses = 0
        exact_entries = index_size = 0
        spent = 0.0
        for _, engine in engines:
            snap = engine.metrics.get_snapshot()
            exact_hits += snap.exact_hits
            semantic_hits += snap.semantic_hits
            misses += snap.misses
            exact_entries += store_entry_count(engine.store)
            index_size += engine.semantic_cache.size()
            spent += getattr(engine.llm_client, "estimated_cost_usd", 0.0) or 0.0
        served = exact_hits + semantic_hits + misses
        return {
            "epoch": recorder.epoch,
            "version": recorder.version,
            "defaults": defaults.as_dict() if defaults is not None else None,
            "stats": {
                "served": served,
                "exact_hits": exact_hits,
                "semantic_hits": semantic_hits,
                "misses": misses,
                "hit_rate": (exact_hits + semantic_hits) / served if served else 0.0,
                "exact_entries": exact_entries,
                "index_size": index_size,
                "engines": len(engines),
            },
            "spent_usd": spent,
            "traces": recorder.since(after),
        }

    @router.post("/traces/clear", summary="Empty the console's trace history")
    def console_traces_clear() -> Dict[str, Any]:
        return {"epoch": recorder.clear(), "message": "Trace history cleared."}

    @router.post("/cache/clear", summary="Empty the console's caches")
    def console_cache_clear() -> Dict[str, Any]:
        report = pool.clear_all()
        recorder.clear_origins()
        return {"cleared": report, "message": "Console caches cleared."}

    @router.get("", include_in_schema=False)
    def console_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return router
