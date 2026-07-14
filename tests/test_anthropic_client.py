"""
Tests for levy.llm_client.AnthropicLLMClient and its budget guard.

Fully offline: every scenario runs through an `httpx.MockTransport` injected
into the Anthropic SDK's own client construction (`http_client=`), so no
network access or ANTHROPIC_API_KEY is required. This keeps the connector
inside the coverage denominator -- see CLAUDE.md's LEV-6 design decision on
offline testability via injectable transport (no `# pragma: no cover`).
"""

import unittest

import anthropic
import httpx

from levy.config import LevyConfig
from levy.engine import LevyEngine
from levy.llm_client import (
    AnthropicLLMClient,
    AnthropicRefusalError,
    BudgetExceededError,
)
from levy.models import LLMRequest


def _message_response(
    text: str = "hello world",
    model: str = "claude-opus-4-8",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict:
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _error_response(status_code: int, error_type: str, message: str, headers=None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"type": "error", "error": {"type": error_type, "message": message}},
        headers=headers or {},
    )


def _make_client(handler, **kwargs) -> AnthropicLLMClient:
    http_client = anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))
    return AnthropicLLMClient(api_key="sk-test", http_client=http_client, **kwargs)


class TestConstruction(unittest.TestCase):

    def test_missing_api_key_raises_naming_the_variable(self):
        with self.assertRaises(ValueError) as ctx:
            AnthropicLLMClient(api_key=None)
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_missing_api_key_empty_string_raises(self):
        with self.assertRaises(ValueError):
            AnthropicLLMClient(api_key="")


class TestSuccessfulGeneration(unittest.TestCase):

    def test_success_populates_text_token_usage_and_metadata(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(
                text="hi there", model="claude-opus-4-8", input_tokens=12, output_tokens=8,
            ))

        client = _make_client(handler, max_retries=0)
        response = client.generate(LLMRequest(prompt="hello", max_tokens=64))

        self.assertEqual(response.text, "hi there")
        self.assertEqual(response.token_usage, 20)  # 12 + 8
        self.assertEqual(response.model, "claude-opus-4-8")
        self.assertEqual(response.metadata["input_tokens"], 12)
        self.assertEqual(response.metadata["output_tokens"], 8)
        self.assertEqual(response.metadata["model"], "claude-opus-4-8")
        self.assertEqual(response.metadata["stop_reason"], "end_turn")

    def test_success_records_request_count_and_estimated_cost(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(input_tokens=1_000_000, output_tokens=1_000_000))

        client = _make_client(
            handler, max_retries=0, input_price_per_mtok=5.0, output_price_per_mtok=25.0,
        )
        client.generate(LLMRequest(prompt="hello"))

        self.assertEqual(client.request_count, 1)
        self.assertAlmostEqual(client.estimated_cost_usd, 30.0)  # 1M*5/1M + 1M*25/1M


class TestRetry(unittest.TestCase):

    def test_retryable_error_then_success_returns_transparently(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return _error_response(429, "rate_limit_error", "slow down", headers={"retry-after": "0"})
            return httpx.Response(200, json=_message_response(text="recovered"))

        client = _make_client(handler, max_retries=2)
        response = client.generate(LLMRequest(prompt="hello"))

        self.assertEqual(calls["n"], 2)
        self.assertEqual(response.text, "recovered")


class TestNonRetryableErrors(unittest.TestCase):

    def test_bad_request_propagates_typed_exception(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _error_response(400, "invalid_request_error", "bad request")

        client = _make_client(handler, max_retries=0)
        with self.assertRaises(anthropic.BadRequestError):
            client.generate(LLMRequest(prompt="hello"))

    def test_bad_request_is_not_counted_toward_budget(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _error_response(400, "invalid_request_error", "bad request")

        client = _make_client(handler, max_retries=0)
        with self.assertRaises(anthropic.BadRequestError):
            client.generate(LLMRequest(prompt="hello"))
        self.assertEqual(client.request_count, 0)
        self.assertEqual(client.estimated_cost_usd, 0.0)


class TestRefusal(unittest.TestCase):

    def test_refusal_stop_reason_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(text="", stop_reason="refusal"))

        client = _make_client(handler, max_retries=0)
        with self.assertRaises(AnthropicRefusalError) as ctx:
            client.generate(LLMRequest(prompt="hello"))
        self.assertEqual(ctx.exception.stop_reason, "refusal")

    def test_refusal_is_still_counted_toward_budget(self):
        """A refusal still costs money (tokens were processed/billed), so it counts."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(
                text="", stop_reason="refusal", input_tokens=100, output_tokens=1,
            ))

        client = _make_client(handler, max_retries=0)
        with self.assertRaises(AnthropicRefusalError):
            client.generate(LLMRequest(prompt="hello"))
        self.assertEqual(client.request_count, 1)
        self.assertGreater(client.estimated_cost_usd, 0.0)

    def test_refusal_via_engine_raises_and_nothing_is_cached(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(text="", stop_reason="refusal"))

        config = LevyConfig(
            llm_provider="anthropic",
            anthropic_api_key="sk-test",
            embedding_provider="mock",
            enable_semantic_cache=False,
        )
        engine = LevyEngine(config)
        engine.llm_client = _make_client(handler, max_retries=0)

        with self.assertRaises(AnthropicRefusalError):
            engine.generate("this prompt will be refused")

        self.assertEqual(len(engine.store.entries), 0)


class TestBudgetGuard(unittest.TestCase):

    def test_cap_halts_spending_with_no_request_sent(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=_message_response())

        client = _make_client(handler, max_retries=0, budget_cap_usd=0.0)

        with self.assertRaises(BudgetExceededError) as ctx:
            client.generate(LLMRequest(prompt="hello"))

        self.assertEqual(calls["n"], 0)  # no API request sent
        self.assertEqual(ctx.exception.cap_usd, 0.0)
        self.assertEqual(ctx.exception.estimated_cost_usd, 0.0)
        self.assertIn("0.00", str(ctx.exception))

    def test_cap_reached_after_a_prior_call_halts_the_next(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(input_tokens=1_000_000, output_tokens=1_000_000))

        client = _make_client(
            handler, max_retries=0, budget_cap_usd=10.0,
            input_price_per_mtok=5.0, output_price_per_mtok=25.0,
        )
        # First call costs $30 (1M*5 + 1M*25 per-MTok), which already exceeds the $10 cap.
        client.generate(LLMRequest(prompt="hello"))
        self.assertEqual(client.request_count, 1)

        with self.assertRaises(BudgetExceededError):
            client.generate(LLMRequest(prompt="hello again"))
        self.assertEqual(client.request_count, 1)  # second call never sent


class TestEngineEndToEnd(unittest.TestCase):

    def test_generation_via_configuration_returns_anthropic_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_message_response(text="engine says hi"))

        config = LevyConfig(
            llm_provider="anthropic",
            anthropic_api_key="sk-test",
            embedding_provider="mock",
            enable_semantic_cache=False,
        )
        engine = LevyEngine(config)
        self.assertIsInstance(engine.llm_client, AnthropicLLMClient)

        engine.llm_client = _make_client(handler, max_retries=0)
        result = engine.generate("hello via anthropic")

        self.assertEqual(result.source, "llm")
        self.assertEqual(result.answer, "engine says hi")


if __name__ == "__main__":
    unittest.main()
