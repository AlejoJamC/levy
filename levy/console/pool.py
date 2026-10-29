"""
The console's engines: one `LevyEngine` per resolved `ConsoleSettings`.

A separate instance from the delivered `EnginePool`, so console traffic never
touches the delivered endpoints' caches. Embedding managers are shared per
(embedding provider, embedding model), so a model is never loaded twice.

The LLM client is built here rather than by the engine, so an Anthropic model
is billed at the prices configured for it in `.env`. Both factories
are injectable: the offline tests pass test doubles, the running console passes
nothing and gets the real providers.
"""

import dataclasses
import threading
from typing import Callable, Dict, List, Optional, Tuple

from levy.api.pool import PoolCapExceededError
from levy.cache.exact_cache import ExactCache
from levy.config import LevyConfig
from levy.console.options import ConsoleSettings
from levy.embedding_manager import EmbeddingManager
from levy.engine import LevyEngine
from levy.llm_client import AnthropicLLMClient, LLMClient, OllamaLLMClient
from levy.metrics import LevyMetrics

try:
    from levy.cache.redis_store import RedisStore
except ImportError:  # pragma: no cover -- redis is a declared dependency
    RedisStore = None

DEFAULT_CONSOLE_POOL_CAP = 16

LLMClientFactory = Callable[[ConsoleSettings, LevyConfig], LLMClient]
EmbeddingManagerFactory = Callable[[ConsoleSettings, LevyConfig], EmbeddingManager]


WARM_UP_TEXT = "levy console warm-up"

class PrefixedRedisStore:
    """
    A `RedisStore` whose keys carry one configuration's prefix.

    The delivered `RedisStore` keys entries by sha256(prompt) in one shared
    database, so two configurations (or the console and the delivered
    endpoints) would serve each other's exact entries, and it has no entry
    count for stats. This wrapper keeps the delivered store untouched and adds
    both: every key is prefixed, `count()` and `clear()` see only the prefix.
    """

    def __init__(self, inner, prefix: str):
        self.inner = inner
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str):
        return self.inner.get(self._key(key))

    def set(self, key: str, entry) -> None:
        self.inner.set(self._key(key), entry)

    def delete(self, key: str) -> None:
        self.inner.delete(self._key(key))

    def _own_keys(self) -> List:
        return list(self.inner.client.scan_iter(match=f"{self.prefix}*"))

    def count(self) -> int:
        return len(self._own_keys())

    def clear(self) -> None:
        keys = self._own_keys()
        if keys:
            self.inner.client.delete(*keys)


def store_entry_count(store) -> int:
    """Entries in an engine's exact-cache store, for either store type."""
    if isinstance(store, PrefixedRedisStore):
        return store.count()
    return len(store.entries)


def default_llm_client(settings: ConsoleSettings, base: LevyConfig) -> LLMClient:
    """The real provider client for `settings`; an Anthropic model gets its own prices."""
    if settings.llm_provider == "anthropic":
        input_price, output_price = base.anthropic_prices(settings.llm_model) or (None, None)
        return AnthropicLLMClient(
            api_key=base.anthropic_api_key,
            model=settings.llm_model,
            max_retries=base.anthropic_max_retries,
            budget_cap_usd=base.anthropic_budget_cap_usd,
            input_price_per_mtok=input_price,
            output_price_per_mtok=output_price,
        )
    return OllamaLLMClient(base_url=base.ollama_base_url, model=settings.llm_model)


def default_embedding_manager(settings: ConsoleSettings, base: LevyConfig) -> EmbeddingManager:
    return EmbeddingManager(
        model_name=settings.embedding_model,
        provider=settings.embedding_provider,
        ollama_base_url=base.ollama_base_url,
    )


class ConsolePool:
    def __init__(
        self,
        base_config: LevyConfig,
        max_engines: int = DEFAULT_CONSOLE_POOL_CAP,
        llm_client_factory: Optional[LLMClientFactory] = None,
        embedding_manager_factory: Optional[EmbeddingManagerFactory] = None,
    ):
        self.base_config = base_config
        self.max_engines = max_engines
        self._llm_client_factory = llm_client_factory or default_llm_client
        self._embedding_manager_factory = embedding_manager_factory or default_embedding_manager
        # Keyed by the settings and the model's configured prices: editing a price in
        # `.env` gives the model a fresh engine whose budget guard uses the new one.
        self._engines: Dict[Tuple[ConsoleSettings, Optional[Tuple[float, float]]], LevyEngine] = {}
        self._managers: Dict[Tuple[str, str], EmbeddingManager] = {}
        # Concurrent console requests (FastAPI threadpool) could otherwise both
        # build an engine for the same settings, or both pass the cap check.
        self._lock = threading.Lock()

    def get(self, settings: ConsoleSettings, config: Optional[LevyConfig] = None) -> LevyEngine:
        """The engine for `settings`, built from `config` (the current configuration) when new."""
        config = config or self.base_config
        key = (settings, config.anthropic_prices(settings.llm_model))
        engine = self._engines.get(key)
        if engine is not None:
            return engine
        with self._lock:
            engine = self._engines.get(key)
            if engine is not None:
                return engine
            if len(self._engines) >= self.max_engines:
                raise PoolCapExceededError(self.max_engines)
            engine = self._build(settings, config)
            self._engines[key] = engine
            return engine

    def warm_up(self, settings: ConsoleSettings, config: Optional[LevyConfig] = None) -> None:
        """Load `settings`' embedding model now, so the first request's timings exclude the model load."""
        with self._lock:
            manager = self._manager(settings, config or self.base_config)
        manager.embed(WARM_UP_TEXT)
        manager.forget(WARM_UP_TEXT)

    def _manager(self, settings: ConsoleSettings, base: LevyConfig) -> EmbeddingManager:
        manager_key = (settings.embedding_provider, settings.embedding_model)
        manager = self._managers.get(manager_key)
        if manager is None:
            manager = self._embedding_manager_factory(settings, base)
            self._managers[manager_key] = manager
        return manager

    def _build(self, settings: ConsoleSettings, base: LevyConfig) -> LevyEngine:
        manager = self._manager(settings, base)

        config = dataclasses.replace(
            base,
            llm_provider=settings.llm_provider,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            similarity_threshold=settings.threshold,
            enable_exact_cache=settings.exact_cache,
            enable_semantic_cache=settings.semantic_cache,
            vector_index_backend=settings.vector_index_backend,
            cache_store_type=settings.cache_store,
        )
        llm_client = self._llm_client_factory(settings, base)
        engine = LevyEngine(config, embedding_manager=manager, llm_client=llm_client)

        # The engine falls back to memory when Redis is unavailable; only a real
        # RedisStore is prefixed, so a fallback stays an ordinary memory store.
        if RedisStore is not None and isinstance(engine.store, RedisStore):
            engine.store = PrefixedRedisStore(engine.store, prefix=f"levy:console:{settings.key}:")
            engine.exact_cache = ExactCache(engine.store)
        return engine

    def items(self) -> List[Tuple[ConsoleSettings, LevyEngine]]:
        return [(settings, engine) for (settings, _), engine in list(self._engines.items())]

    def clear_all(self) -> Dict[str, Dict[str, int]]:
        """Empty every console engine's caches and metrics, reporting what was cleared."""
        report: Dict[str, Dict[str, int]] = {}
        for settings, engine in self.items():
            exact_count = store_entry_count(engine.store)
            semantic_count = engine.semantic_cache.size()
            engine.store.clear()
            engine.semantic_cache.clear()
            engine.metrics = LevyMetrics()
            report[settings.key] = {"exact_entries": exact_count, "semantic_entries": semantic_count}
        return report
