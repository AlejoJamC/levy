"""
Tests for levy.llm_client and levy.embeddings: abstract-base contracts,
constructor plumbing, and offline-testable branches. Network-touching
bodies (OpenAI/Ollama LLM calls, Ollama/sentence-transformers embedding
calls) are excluded from coverage via inline pragmas and are not exercised
here -- see CLAUDE.md known-gaps note and the add-test-infrastructure design.
"""

import unittest

from levy.embeddings import EmbeddingClient, OllamaEmbeddingClient
from levy.llm_client import LLMClient, MockLLMClient, OllamaLLMClient, OpenAILLMClient
from levy.models import LLMRequest


# ---------------------------------------------------------------------------
# LLMClient (abstract contract) + MockLLMClient
# ---------------------------------------------------------------------------

class _MinimalLLMClient(LLMClient):
    def generate(self, request):
        return super().generate(request)


class TestLLMClientAbstract(unittest.TestCase):

    def test_abstract_stub_body_is_inert(self):
        self.assertIsNone(_MinimalLLMClient().generate(LLMRequest(prompt="x")))


class TestMockLLMClient(unittest.TestCase):

    def test_default_latency_is_half_second(self):
        self.assertEqual(MockLLMClient().latency_seconds, 0.5)

    def test_zero_latency_does_not_sleep(self):
        client = MockLLMClient(latency_seconds=0)
        response = client.generate(LLMRequest(prompt="hello"))
        self.assertIn("olleh", response.text)  # reversed prompt

    def test_nonzero_latency_sleeps(self):
        """Exercises the `if self.latency_seconds:` True branch with a tiny delay."""
        client = MockLLMClient(latency_seconds=0.01)
        response = client.generate(LLMRequest(prompt="ab"))
        self.assertIn("ba", response.text)


# ---------------------------------------------------------------------------
# OpenAILLMClient / OllamaLLMClient constructors (attribute plumbing only --
# generate() makes a real HTTP call and is pragma-excluded)
# ---------------------------------------------------------------------------

class TestLLMClientConstructors(unittest.TestCase):

    def test_openai_client_stores_config(self):
        client = OpenAILLMClient(api_key="sk-test", base_url="https://example.test/v1", model="gpt-x")
        self.assertEqual(client.api_key, "sk-test")
        self.assertEqual(client.base_url, "https://example.test/v1")
        self.assertEqual(client.model, "gpt-x")

    def test_ollama_llm_client_stores_config(self):
        client = OllamaLLMClient(base_url="http://example.test:1", model="qwen3")
        self.assertEqual(client.base_url, "http://example.test:1")
        self.assertEqual(client.model, "qwen3")


# ---------------------------------------------------------------------------
# EmbeddingClient (abstract contract)
# ---------------------------------------------------------------------------

class _MinimalEmbeddingClient(EmbeddingClient):
    def embed(self, text):
        return super().embed(text)

    def get_dimension(self):
        return super().get_dimension()


class TestEmbeddingClientAbstract(unittest.TestCase):

    def test_abstract_stub_bodies_are_inert(self):
        client = _MinimalEmbeddingClient()
        self.assertIsNone(client.embed("x"))
        self.assertIsNone(client.get_dimension())


# ---------------------------------------------------------------------------
# OllamaEmbeddingClient (constructor + already-known-dimension branch only --
# embed()/get_dimension()-when-unknown make real HTTP calls and are pragma-excluded)
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingClient(unittest.TestCase):

    def test_constructor_stores_config(self):
        client = OllamaEmbeddingClient(base_url="http://example.test:1", model="nomic-embed-text")
        self.assertEqual(client.base_url, "http://example.test:1")
        self.assertEqual(client.model, "nomic-embed-text")
        self.assertIsNone(client._dimension)

    def test_get_dimension_returns_cached_value_without_calling_embed(self):
        client = OllamaEmbeddingClient()
        client._dimension = 42  # simulate a prior embed() call having set this
        self.assertEqual(client.get_dimension(), 42)


if __name__ == "__main__":
    unittest.main()
