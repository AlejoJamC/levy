import time
import logging
from typing import Optional, Any, Dict
from levy.config import LevyConfig
from levy.models import LLMRequest, LevyResult, LLMResponse
from levy.llm_client import LLMClient, MockLLMClient, OpenAILLMClient
from levy.embeddings import EmbeddingClient, MockEmbeddingClient, SentenceTransformerClient
from levy.cache.store import InMemoryStore
from levy.cache.exact_cache import ExactCache
from levy.cache.semantic_cache import SemanticCache
from levy.metrics import LevyMetrics

logger = logging.getLogger(__name__)

class LevyEngine:
    def __init__(self, config: LevyConfig = LevyConfig()):
        self.config = config
        self.metrics = LevyMetrics()
        
        # 1. Initialize LLM Client
        if config.llm_provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OpenAI API key required for 'openai' provider")
            self.llm_client = OpenAILLMClient(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                model=config.model_name
            )
        else:
            self.llm_client = MockLLMClient()

        # 2. Initialize Embedding Client
        if config.enable_semantic_cache:
            if config.embedding_provider == "sentence-transformers":
                self.embedding_client = SentenceTransformerClient(config.embedding_model)
            else:
                self.embedding_client = MockEmbeddingClient()
        else:
             self.embedding_client = MockEmbeddingClient() # Fallback

        # 3. Initialize Store and Caches
        self.store = InMemoryStore(max_size=self.config.cache_max_size)
        self.exact_cache = ExactCache(self.store)
        self.semantic_cache = SemanticCache(
            self.store, 
            self.embedding_client, 
            threshold=self.config.similarity_threshold
        )

    def generate(self, prompt: str, **kwargs) -> LevyResult:
        start_time = time.time()
        request = LLMRequest(prompt=prompt, extra_params=kwargs)
        
        # 1. Check Exact Cache
        if self.config.enable_exact_cache:
            entry = self.exact_cache.get(request)
            if entry:
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
            entry = self.semantic_cache.get(request)
            if entry:
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

        # 3. LLM Call
        logger.info(f"Cache miss. Calling LLM for: {prompt[:30]}...")
        try:
            llm_response = self.llm_client.generate(request)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e

        # 4. Store in Cache
        # We need the embedding for the semantic cache to work next time.
        # If semantic is enabled, we should compute it now to store it.
        embedding = None
        if self.config.enable_semantic_cache:
            embedding = self.embedding_client.embed(prompt)
        
        # We save to the unified store (via exact cache interface is easiest as it writes the key)
        # But we need to pass the embedding so semantic cache can find it later.
        self.exact_cache.set(request, llm_response.text, embedding=embedding)

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
