"""
Tests for LevyEngine construction branches not covered by the semantic-cache
and exact-cache end-to-end tests: provider selection (openai/ollama/redis)
and the LLM-failure path in generate(). All offline -- RedisStore's lazy
`redis.from_url()` never opens a real connection, and OpenAI/Ollama clients
are only constructed here, never asked to `.generate()` (that path makes a
real HTTP call and is pragma-excluded).
"""

import builtins
import importlib
import unittest
from unittest import mock

import levy.engine as engine_module
from levy.cache.store import InMemoryStore
from levy.config import LevyConfig
from levy.engine import LevyEngine
from levy.llm_client import LLMClient, OllamaLLMClient, OpenAILLMClient


class TestProviderSelection(unittest.TestCase):

    def test_openai_provider_without_api_key_raises(self):
        config = LevyConfig(llm_provider="openai", openai_api_key=None, embedding_provider="mock")
        with self.assertRaises(ValueError):
            LevyEngine(config)

    def test_openai_provider_with_api_key_constructs_client(self):
        config = LevyConfig(
            llm_provider="openai",
            openai_api_key="sk-test",
            embedding_provider="mock",
        )
        engine = LevyEngine(config)
        self.assertIsInstance(engine.llm_client, OpenAILLMClient)
        self.assertEqual(engine.llm_client.api_key, "sk-test")

    def test_ollama_provider_constructs_client(self):
        config = LevyConfig(llm_provider="ollama", embedding_provider="mock")
        engine = LevyEngine(config)
        self.assertIsInstance(engine.llm_client, OllamaLLMClient)


class TestRedisStoreFallback(unittest.TestCase):

    def test_falls_back_to_memory_when_redis_store_unavailable(self):
        """Simulates a levy.cache.redis_store import failure (RedisStore = None)."""
        config = LevyConfig(cache_store_type="redis", embedding_provider="mock")
        with mock.patch.object(engine_module, "RedisStore", None):
            engine = LevyEngine(config)
        self.assertIsInstance(engine.store, InMemoryStore)

    def test_constructs_redis_store_when_available(self):
        """redis.from_url() is lazy (no real connection attempt), so this
        succeeds fully offline against the redis-py client library."""
        config = LevyConfig(cache_store_type="redis", embedding_provider="mock")
        engine = LevyEngine(config)
        self.assertEqual(type(engine.store).__name__, "RedisStore")

    def test_falls_back_to_memory_when_redis_store_construction_fails(self):
        class _ExplodingRedisStore:
            def __init__(self, *args, **kwargs):
                raise ConnectionError("simulated: no redis server")

        config = LevyConfig(cache_store_type="redis", embedding_provider="mock")
        with mock.patch.object(engine_module, "RedisStore", _ExplodingRedisStore):
            engine = LevyEngine(config)
        self.assertIsInstance(engine.store, InMemoryStore)


class TestRedisStoreImportGuard(unittest.TestCase):

    def test_module_falls_back_to_none_when_redis_store_import_fails(self):
        """Simulates redis-py being absent at import time: `levy.engine` must
        still import cleanly, with its module-level `RedisStore` set to None."""
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "levy.cache.redis_store" or name.startswith("levy.cache.redis_store."):
                raise ImportError("simulated: redis-py not installed")
            return real_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", side_effect=_blocked_import):
                importlib.reload(engine_module)
            self.assertIsNone(engine_module.RedisStore)
        finally:
            importlib.reload(engine_module)  # restore normal state for subsequent tests


class TestGenerateErrorHandling(unittest.TestCase):

    def test_llm_client_exception_propagates(self):
        class _ExplodingLLMClient(LLMClient):
            def generate(self, request):
                raise RuntimeError("simulated LLM failure")

        config = LevyConfig(
            llm_provider="mock",
            embedding_provider="mock",
            enable_exact_cache=True,
            enable_semantic_cache=False,
        )
        engine = LevyEngine(config)
        engine.llm_client = _ExplodingLLMClient()
        with self.assertRaises(RuntimeError):
            engine.generate("this will miss and call the LLM")


class TestMetricsSummary(unittest.TestCase):

    def test_get_metrics_summary_returns_metrics_string(self):
        config = LevyConfig(
            llm_provider="mock",
            mock_llm_latency_seconds=0,
            embedding_provider="mock",
            enable_semantic_cache=False,
        )
        engine = LevyEngine(config)
        engine.generate("hello")
        summary = engine.get_metrics_summary()
        self.assertIn("LevyMetrics", summary)


if __name__ == "__main__":
    unittest.main()
