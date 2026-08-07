import time
import logging
from typing import Optional, Any, Dict, TYPE_CHECKING
from levy.config import LevyConfig
from levy.latency.timing import (
    SEGMENT_EXACT_LOOKUP,
    SEGMENT_TOTAL_LOOKUP,
    mark,
    segment,
    since_ms,
)
from levy.models import LLMRequest, LevyResult, LLMResponse
from levy.llm_client import LLMClient, MockLLMClient, OpenAILLMClient, OllamaLLMClient, AnthropicLLMClient
from levy.embedding_manager import EmbeddingManager
from levy.cache.store import InMemoryStore
# Load RedisStore conditionally or just import if available
try:
    from levy.cache.redis_store import RedisStore
except ImportError:
    RedisStore = None

from levy.cache.exact_cache import ExactCache
from levy.cache.semantic_cache import SemanticCache
from levy.metrics import LevyMetrics

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from levy.latency.timing import TimingCollector

logger = logging.getLogger(__name__)

class LevyEngine:
    def __init__(
        self,
        config: LevyConfig = LevyConfig(),
        embedding_manager: Optional[EmbeddingManager] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self.metrics = LevyMetrics()

        # 1. Initialize LLM Client. An injected client wins over the configured
        # provider (LEV-14: the latency replay serves responses recorded earlier
        # from a real provider, so the timed run itself makes no network call).
        if llm_client is not None:
            self.llm_client = llm_client
        elif config.llm_provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OpenAI API key required for 'openai' provider")
            self.llm_client = OpenAILLMClient(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                model=config.model_name
            )
        elif config.llm_provider == "ollama":
            self.llm_client = OllamaLLMClient(
                base_url=config.ollama_base_url,
                model=config.model_name
            )
        elif config.llm_provider == "anthropic":
            self.llm_client = AnthropicLLMClient(
                api_key=config.anthropic_api_key,
                model=config.anthropic_model,
                max_retries=config.anthropic_max_retries,
                budget_cap_usd=config.anthropic_budget_cap_usd,
                input_price_per_mtok=config.anthropic_input_price_per_mtok,
                output_price_per_mtok=config.anthropic_output_price_per_mtok,
            )
        else:
            self.llm_client = MockLLMClient(latency_seconds=config.mock_llm_latency_seconds)

        # 2. Initialize Embedding Manager (handles mock, sentence-transformers, ollama).
        # Callers (e.g. the experiment harness sweeping many configs per model) may inject
        # a shared manager so memoization survives across engine instances.
        self.embedding_manager = embedding_manager if embedding_manager is not None else EmbeddingManager.from_config(config)

        # 3. Initialize Store and Caches
        if config.cache_store_type == "redis":
            if RedisStore is None:
                logger.warning("Redis dependencies not found. Falling back to Memory.")
                self.store = InMemoryStore(max_size=self.config.cache_max_size)
            else:
                 try:
                    self.store = RedisStore(redis_url=config.redis_url, ttl=config.cache_ttl_seconds)
                 except Exception as e:
                    logger.error(f"Failed to connect to Redis: {e}. Falling back to Memory.")
                    self.store = InMemoryStore(max_size=self.config.cache_max_size)
        else:
            self.store = InMemoryStore(max_size=self.config.cache_max_size)

        self.exact_cache = ExactCache(self.store)
        self.semantic_cache = SemanticCache(
            embedding_client=self.embedding_manager,
            threshold=self.config.similarity_threshold,
            backend=self.config.vector_index_backend,
            m=self.config.hnsw_m,
            ef_construction=self.config.hnsw_ef_construction,
            ef_search=self.config.hnsw_ef_search,
        )

    def generate(self, prompt: str, timing: Optional["TimingCollector"] = None, **kwargs) -> LevyResult:
        """
        Serve `prompt` through exact cache → semantic cache → LLM.

        `timing` (LEV-14) is an opt-in measurement surface: when a collector is
        passed, the exact-cache lookup, the embedding, the index search and the
        whole lookup path are recorded into it. The returned `LevyResult`, the
        recorded metrics and the cache contents are the same either way, and
        with no collector no additional clock is read. The total-lookup segment
        ends where the lookup does — a miss's LLM call and the subsequent store
        are not lookup overhead and are excluded, which is also what keeps the
        segment sum from exceeding the total.
        """
        start_time = time.time()
        lookup_start = mark() if timing is not None else 0.0
        request = LLMRequest(prompt=prompt, extra_params=kwargs)

        # 1. Check Exact Cache
        if self.config.enable_exact_cache:
            with segment(timing, SEGMENT_EXACT_LOOKUP):
                entry = self.exact_cache.get(request)
            if entry:
                if timing is not None:
                    timing.record(SEGMENT_TOTAL_LOOKUP, since_ms(lookup_start))
                latency = (time.time() - start_time) * 1000
                self.metrics.record_hit("exact", saved_tokens=len(entry.response_text.split())) # Approx token count
                self.metrics.record_request(latency)
                logger.info(f"Exact cache hit for: {prompt[:30]}...")
                return LevyResult(
                    answer=entry.response_text,
                    source="exact_cache",
                    latency_ms=latency,
                    similarity_score=1.0,
                    metadata=entry.metadata
                )

        # 2. Check Semantic Cache
        if self.config.enable_semantic_cache:
            # Note: exact cache get doesn't compute embedding usually, 
            # but semantic needs it. Semantic cache 'get' computes it internaly if needed.
            entry = self.semantic_cache.get(request, timing=timing)
            if entry:
                if timing is not None:
                    timing.record(SEGMENT_TOTAL_LOOKUP, since_ms(lookup_start))
                latency = (time.time() - start_time) * 1000
                score = entry.metadata.get('last_similarity_score', 0.0)
                self.metrics.record_hit("semantic", saved_tokens=len(entry.response_text.split()))
                self.metrics.record_request(latency)
                logger.info(f"Semantic cache hit ({score:.4f}) for: {prompt[:30]}...")
                return LevyResult(
                    answer=entry.response_text,
                    source="semantic_cache",
                    latency_ms=latency,
                    similarity_score=score,
                    metadata=entry.metadata
                )

        # 3. LLM Call — the lookup path ends here, whatever the call costs.
        if timing is not None:
            timing.record(SEGMENT_TOTAL_LOOKUP, since_ms(lookup_start))
        logger.info(f"Cache miss. Calling LLM for: {prompt[:30]}...")
        try:
            llm_response = self.llm_client.generate(request)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e

        # 4. Store in Cache
        embedding = None
        if self.config.enable_semantic_cache:
            embedding = self.embedding_manager.embed(prompt)

        model_meta = self.embedding_manager.get_model_identity().as_dict()
        self.exact_cache.set(request, llm_response.text, embedding=embedding, metadata=model_meta)
        if self.config.enable_semantic_cache and embedding is not None:
            self.semantic_cache.set(request, llm_response.text, embedding=embedding, metadata=model_meta)

        latency = (time.time() - start_time) * 1000
        self.metrics.record_miss()
        self.metrics.record_request(latency)

        return LevyResult(
            answer=llm_response.text,
            source="llm",
            latency_ms=latency,
            original_response=llm_response
        )

    def get_metrics_summary(self) -> str:
        return str(self.metrics)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Additive accessor (LEV-7): semantic-index size + per-model cached-entry
        counts, read from CacheEntry.metadata written by the exact-cache store."""
        model_breakdown: Dict[str, int] = {}
        for entry in self.store.entries.values():
            name = entry.metadata.get("canonical_name", "unknown")
            model_breakdown[name] = model_breakdown.get(name, 0) + 1
        return {
            "index_size": self.semantic_cache.size(),
            "model_breakdown": model_breakdown,
        }
