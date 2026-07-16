"""
Tests for levy.api (LEV-7 FastAPI router).

Fully offline: mock LLM + mock embeddings via FastAPI's in-process TestClient.
No network access, no real Anthropic/OpenAI/Ollama calls.
"""

import json
import unittest

import anthropic
import httpx
from fastapi.testclient import TestClient

from levy.api.app import create_app
from levy.config import LevyConfig
from levy.llm_client import AnthropicLLMClient


def _mock_config(**overrides) -> LevyConfig:
    defaults = dict(
        llm_provider="mock",
        embedding_provider="mock",
        mock_llm_latency_seconds=0.0,
    )
    defaults.update(overrides)
    return LevyConfig(**defaults)


def _client(config=None, max_engines=8, **client_kwargs) -> TestClient:
    app = create_app(config=config or _mock_config(), max_engines=max_engines)
    return TestClient(app, **client_kwargs)


def _chat_body(prompt: str, cache_config: dict = None) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}]}
    if cache_config is not None:
        body["cache_config"] = cache_config
    return body


class TestHitMissFlows(unittest.TestCase):

    def test_miss_returns_provider_body_and_miss_header(self):
        client = _client()
        r = client.post("/v1/chat/completions", json=_chat_body("hello world"))

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["x-cache-status"], "MISS")
        self.assertNotIn("x-cache-similarity", r.headers)
        body = r.json()
        self.assertEqual(body["content"][0]["type"], "text")
        self.assertIn("dlrow olleh", body["content"][0]["text"])
        self.assertEqual(body["model"], "mock-v1")

    def test_exact_hit_returns_hit_header_and_similarity_one(self):
        client = _client()
        prompt = "the quick brown fox"
        client.post("/v1/chat/completions", json=_chat_body(prompt))
        r = client.post("/v1/chat/completions", json=_chat_body(prompt))

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["x-cache-status"], "HIT")
        self.assertEqual(r.headers["x-cache-similarity"], "1.0")

    def test_hit_and_miss_bodies_share_the_same_shape(self):
        client = _client()
        prompt = "shape check"
        miss_body = client.post("/v1/chat/completions", json=_chat_body(prompt)).json()
        hit_body = client.post("/v1/chat/completions", json=_chat_body(prompt)).json()

        self.assertEqual(set(miss_body.keys()), set(hit_body.keys()))
        for key in ("id", "type", "role", "model", "content", "usage", "request_id"):
            self.assertIn(key, hit_body)

    def test_semantic_hit_reports_similarity(self):
        config = _mock_config(
            enable_exact_cache=False,
            enable_semantic_cache=True,
            similarity_threshold=0.0,  # accept everything (mock embeddings are random)
        )
        client = _client(config=config)

        client.post("/v1/chat/completions", json=_chat_body("first prompt"))
        r = client.post("/v1/chat/completions", json=_chat_body("a different second prompt"))

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["x-cache-status"], "HIT")
        self.assertIn("x-cache-similarity", r.headers)
        self.assertGreaterEqual(float(r.headers["x-cache-similarity"]), 0.0)


class TestPerRequestConfig(unittest.TestCase):

    def test_distinct_configs_use_distinct_cache_universes(self):
        client = _client()
        prompt = "shared prompt text"

        r1 = client.post("/v1/chat/completions", json=_chat_body(prompt, {"threshold": 0.9}))
        self.assertEqual(r1.headers["x-cache-status"], "MISS")

        # Same prompt, different threshold => different pool key => fresh cache => still MISS.
        r2 = client.post("/v1/chat/completions", json=_chat_body(prompt, {"threshold": 0.1}))
        self.assertEqual(r2.headers["x-cache-status"], "MISS")

        # Re-requesting the original threshold hits its own accumulated cache.
        r3 = client.post("/v1/chat/completions", json=_chat_body(prompt, {"threshold": 0.9}))
        self.assertEqual(r3.headers["x-cache-status"], "HIT")

    def test_same_config_accumulates_across_requests(self):
        client = _client()
        cache_config = {"threshold": 0.5}
        prompt = "accumulate me"

        r1 = client.post("/v1/chat/completions", json=_chat_body(prompt, cache_config))
        r2 = client.post("/v1/chat/completions", json=_chat_body(prompt, cache_config))

        self.assertEqual(r1.headers["x-cache-status"], "MISS")
        self.assertEqual(r2.headers["x-cache-status"], "HIT")

    def test_omitted_cache_config_fields_use_defaults(self):
        config = _mock_config(similarity_threshold=0.42)
        client = _client(config=config)

        r1 = client.post("/v1/chat/completions", json=_chat_body("default fields", {}))
        r2 = client.post("/v1/chat/completions", json=_chat_body("default fields", {}))

        self.assertEqual(r1.headers["x-cache-status"], "MISS")
        self.assertEqual(r2.headers["x-cache-status"], "HIT")

    def test_pool_cap_exceeded_returns_client_error(self):
        client = _client(max_engines=1)

        r1 = client.post("/v1/chat/completions", json=_chat_body("a", {"threshold": 0.9}))
        self.assertEqual(r1.status_code, 200)

        r2 = client.post("/v1/chat/completions", json=_chat_body("b", {"threshold": 0.1}))
        self.assertEqual(r2.status_code, 400)
        body = r2.json()
        self.assertEqual(body["error"], "pool_cap_exceeded")
        self.assertIn("1", body["detail"])


class TestAdmin(unittest.TestCase):

    def test_stats_reflect_activity(self):
        client = _client()
        client.post("/v1/chat/completions", json=_chat_body("stats prompt"))  # miss
        client.post("/v1/chat/completions", json=_chat_body("stats prompt"))  # exact hit

        stats = client.get("/admin/cache/stats").json()

        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["exact_hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)
        self.assertGreaterEqual(stats["index_size"], 1)
        self.assertIn("mock", stats["model_breakdown"])

    def test_clear_empties_caches_and_resets_counters(self):
        client = _client()
        prompt = "clear me"
        client.post("/v1/chat/completions", json=_chat_body(prompt))
        client.post("/v1/chat/completions", json=_chat_body(prompt))

        clear_body = client.post("/admin/cache/clear").json()
        self.assertTrue(any(v["exact_entries"] >= 1 for v in clear_body["cleared"].values()))

        stats = client.get("/admin/cache/stats").json()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["index_size"], 0)

        r = client.post("/v1/chat/completions", json=_chat_body(prompt))
        self.assertEqual(r.headers["x-cache-status"], "MISS")


class TestErrors(unittest.TestCase):

    def test_missing_messages_returns_422_with_field_detail(self):
        client = _client()
        r = client.post("/v1/chat/completions", json={})

        self.assertEqual(r.status_code, 422)
        detail = r.json()["detail"]
        self.assertTrue(any(err["loc"][-1] == "messages" for err in detail))

    def test_budget_guard_halt_returns_structured_402(self):
        config = _mock_config(
            llm_provider="anthropic",
            anthropic_api_key="sk-test",
            anthropic_budget_cap_usd=0.0,
        )
        app = create_app(config=config)
        client = TestClient(app)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "msg_x",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "claude-opus-4-8",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        http_client = anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))
        # Force engine construction, then inject a scripted client so the guard trips
        # deterministically with no real network access (mirrors test_anthropic_client.py).
        engine = app.state.pool.get(None, None)
        engine.llm_client = AnthropicLLMClient(
            api_key="sk-test", http_client=http_client, budget_cap_usd=0.0
        )

        r = client.post("/v1/chat/completions", json=_chat_body("too expensive"))

        self.assertEqual(r.status_code, 402)
        body = r.json()
        self.assertEqual(body["error"], "budget_exceeded")
        self.assertEqual(body["cap_usd"], 0.0)

    def test_refusal_returns_structured_502(self):
        config = _mock_config(llm_provider="anthropic", anthropic_api_key="sk-test")
        app = create_app(config=config)
        client = TestClient(app)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "msg_x",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "model": "claude-opus-4-8",
                    "stop_reason": "refusal",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            )

        http_client = anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))
        engine = app.state.pool.get(None, None)
        engine.llm_client = AnthropicLLMClient(api_key="sk-test", http_client=http_client)

        r = client.post("/v1/chat/completions", json=_chat_body("say something refused"))

        self.assertEqual(r.status_code, 502)
        body = r.json()
        self.assertEqual(body["error"], "provider_refusal")

    def test_unknown_provider_error_returns_structured_5xx(self):
        client = _client(raise_server_exceptions=False)
        engine = client.app.state.pool.get(None, None)

        class _BoomClient:
            def generate(self, request):
                raise RuntimeError("provider exploded")

        engine.llm_client = _BoomClient()

        r = client.post("/v1/chat/completions", json=_chat_body("trigger boom"))

        self.assertEqual(r.status_code, 500)
        body = r.json()
        self.assertEqual(body["error"], "provider_error")
        self.assertIn("provider exploded", body["detail"])


class TestLogging(unittest.TestCase):

    def test_record_completeness_and_replayability(self):
        config = _mock_config(similarity_threshold=0.0)
        client = _client(config=config)

        sequence = [
            ("first prompt", {"threshold": 0.9}),
            ("first prompt", {"threshold": 0.9}),  # exact hit
            ("second prompt", {"threshold": 0.1}),  # distinct config -> miss
        ]

        with self.assertLogs("levy.api", level="INFO") as cm:
            responses = [
                client.post("/v1/chat/completions", json=_chat_body(prompt, cfg))
                for prompt, cfg in sequence
            ]

        self.assertEqual(len(cm.records), len(sequence))

        required_fields = {
            "request_id",
            "arrival_ts",
            "completion_ts",
            "embedding_model",
            "threshold",
            "prompt",
            "cache_source",
            "similarity",
            "latency_ms",
        }
        replayed_prompts = []
        replayed_configs = []
        for record, response, (expected_prompt, expected_cfg) in zip(
            cm.records, responses, sequence
        ):
            payload = json.loads(record.getMessage())
            self.assertEqual(required_fields, set(payload.keys()))
            self.assertEqual(payload["prompt"], expected_prompt)
            self.assertEqual(payload["threshold"], expected_cfg["threshold"])
            self.assertEqual(payload["request_id"], response.json()["request_id"])
            replayed_prompts.append(payload["prompt"])
            replayed_configs.append(payload["threshold"])

        self.assertEqual(replayed_prompts, [p for p, _ in sequence])
        self.assertEqual(replayed_configs, [c["threshold"] for _, c in sequence])
        self.assertEqual(
            [r.json()["request_id"] for r in responses],
            [json.loads(rec.getMessage())["request_id"] for rec in cm.records],
        )


class TestOpenAPI(unittest.TestCase):

    def test_openapi_serves_and_contains_the_three_routes(self):
        client = _client()
        r = client.get("/openapi.json")

        self.assertEqual(r.status_code, 200)
        spec = r.json()
        paths = spec["paths"]
        for path in ("/v1/chat/completions", "/admin/cache/stats", "/admin/cache/clear"):
            self.assertIn(path, paths)

        chat = paths["/v1/chat/completions"]["post"]
        self.assertIn("requestBody", chat)
        self.assertIn("200", chat["responses"])

        stats = paths["/admin/cache/stats"]["get"]
        self.assertIn("200", stats["responses"])


if __name__ == "__main__":
    unittest.main()
