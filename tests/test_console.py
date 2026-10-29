"""
Tests for the console section of the API: `/console/chat` with its
closed lists, the console's own engines, the trace feed, the cache clear and
the page -- and that the delivered endpoints are left exactly as delivered.

Fully offline: the console's LLM and embedding factories are given test
doubles (mock embeddings, a zero-latency mock LLM, or the real Anthropic
client over `httpx.MockTransport`); Redis is a dict-backed fake client.
"""

import fnmatch
import re
import threading
import time
import unittest
from unittest import mock

import anthropic
import httpx
from fastapi.testclient import TestClient

from levy.api.app import create_app
from levy.cache.redis_store import RedisStore
from levy.config import LevyConfig
from levy.console.options import CACHE_STORES, VECTOR_INDEX_BACKENDS, ConsoleOptions, ConsoleSettings, resolve_backend
from levy.console.pool import (
    ConsolePool,
    PrefixedRedisStore,
    default_embedding_manager,
    default_llm_client,
    store_entry_count,
)
from levy.console.routes import STATIC_DIR
from levy.embedding_manager import EmbeddingManager
from levy.llm_client import AnthropicLLMClient, LLMClient, MockLLMClient, OllamaLLMClient
from levy.models import CacheEntry, LLMResponse


# Placeholder model ids: real ones live in .env (LEVY_*_MODELS), never in code.
ANTHROPIC = "test-anthropic-model"
OLLAMA = "test-ollama-model"
OLLAMA_EMBED = "test-embedding-model"


def _base(**overrides) -> LevyConfig:
    values = dict(
        llm_provider="mock", embedding_provider="mock", mock_llm_latency_seconds=0.0,
        anthropic_models=((ANTHROPIC, 1.0, 5.0),), anthropic_model=ANTHROPIC,
        anthropic_input_price_per_mtok=1.0, anthropic_output_price_per_mtok=5.0,
        ollama_models=(OLLAMA,), model_name=OLLAMA, ollama_embedding_models=(OLLAMA_EMBED,),
    )
    values.update(overrides)
    return LevyConfig(**values)


OPTIONS = ConsoleOptions.from_config(_base())


def default_settings(base):
    return ConsoleOptions.from_config(base).default_settings(base)


def _mock_manager(settings, base):
    return EmbeddingManager(model_name=settings.embedding_model, provider="mock")


def _client(llm_factory=None, raise_server_exceptions=True, **kwargs) -> TestClient:
    app = create_app(
        _base(),
        console_llm_client_factory=llm_factory or (lambda s, b: MockLLMClient(latency_seconds=0.0)),
        console_embedding_manager_factory=_mock_manager,
        **kwargs,
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _chat(client, prompt, **params):
    return client.post("/console/chat", params={"prompt": prompt, **params})


def _traces(client, after=0):
    return client.get("/console/traces", params={"after": after}).json()


def _anthropic_factory(input_tokens=12, output_tokens=40, budget_cap_usd=200.0):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "an answer"}],
            "model": ANTHROPIC, "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        })

    def factory(settings, base):
        http_client = anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))
        return AnthropicLLMClient(api_key="sk-test", model=settings.llm_model, max_retries=0,
                                  budget_cap_usd=budget_cap_usd, input_price_per_mtok=1.0,
                                  output_price_per_mtok=5.0, http_client=http_client)
    return factory


class FakeRedis:
    """The handful of redis-py calls RedisStore and PrefixedRedisStore make."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value.encode() if isinstance(value, str) else value

    def delete(self, *keys):
        for key in keys:
            self.data.pop(key if isinstance(key, str) else key.decode(), None)

    def scan_iter(self, match="*"):
        return [k for k in list(self.data) if fnmatch.fnmatch(k, match)]


# ---------------------------------------------------------------------------
# Options and defaults
# ---------------------------------------------------------------------------

class TestOptions(unittest.TestCase):

    def test_lists_come_from_the_configuration_and_hold_no_mock(self):
        for values in (OPTIONS.embedding_providers, OPTIONS.llm_providers,
                       OPTIONS.all_embedding_models, OPTIONS.all_llm_models):
            self.assertNotIn("mock", values)
        self.assertEqual(OPTIONS.embedding_models["sentence-transformers"], ("all-MiniLM-L6-v2", "modernbert"))
        self.assertEqual(OPTIONS.embedding_models["ollama"], (OLLAMA_EMBED,))
        self.assertEqual(OPTIONS.llm_models, {"anthropic": (ANTHROPIC,), "ollama": (OLLAMA,)})
        self.assertEqual(OPTIONS.llm_prices_per_mtok, {ANTHROPIC: (1.0, 5.0)})

    def test_unconfigured_providers_are_not_offered(self):
        only_anthropic = ConsoleOptions.from_config(_base(ollama_models=(), ollama_embedding_models=()))
        self.assertEqual(only_anthropic.llm_providers, ("anthropic",))
        self.assertEqual(only_anthropic.embedding_providers, ("sentence-transformers",))
        nothing = ConsoleOptions.from_config(_base(anthropic_models=(), ollama_models=()))
        self.assertIsNone(nothing.default_settings(_base(anthropic_models=(), ollama_models=())))

    def test_defaults_fall_back_from_a_mock_base(self):
        defaults = default_settings(_base())
        self.assertEqual(defaults.llm_provider, "anthropic")
        self.assertEqual(defaults.embedding_provider, "sentence-transformers")
        self.assertEqual(defaults.llm_model, ANTHROPIC)

    def test_defaults_follow_a_real_base(self):
        base = _base(llm_provider="ollama", embedding_provider="ollama",
                     embedding_model=OLLAMA_EMBED, vector_index_backend="brute_force",
                     cache_store_type="redis", enable_exact_cache=False)
        defaults = default_settings(base)
        self.assertEqual((defaults.llm_provider, defaults.llm_model), ("ollama", OLLAMA))
        self.assertEqual((defaults.embedding_provider, defaults.embedding_model), ("ollama", OLLAMA_EMBED))
        self.assertEqual(defaults.vector_index_backend, "brute_force")
        self.assertEqual(defaults.cache_store, "redis")
        self.assertFalse(defaults.exact_cache)

    def test_unlisted_models_fall_back_to_the_first_listed(self):
        base = _base(llm_provider="ollama", model_name="not-listed", embedding_model="not-listed")
        defaults = default_settings(base)
        self.assertEqual(defaults.llm_model, OLLAMA)
        self.assertEqual(defaults.embedding_model, "all-MiniLM-L6-v2")

    def test_resolve_backend(self):
        self.assertEqual(resolve_backend("brute_force"), "brute_force")
        self.assertIn(resolve_backend("auto"), VECTOR_INDEX_BACKENDS)

    def test_mismatched_model(self):
        base = default_settings(_base())
        self.assertIsNone(OPTIONS.mismatched_model(base))
        bad_embed = ConsoleSettings(**{**base.as_dict(), "embedding_provider": "ollama"})
        self.assertEqual(OPTIONS.mismatched_model(bad_embed)[0], "embedding_model")
        bad_llm = ConsoleSettings(**{**base.as_dict(), "llm_provider": "ollama"})
        self.assertEqual(OPTIONS.mismatched_model(bad_llm), ("llm_model", (OLLAMA,)))

    def test_default_llm_provider_is_anthropic(self):
        self.assertEqual(LevyConfig().llm_provider, "anthropic")


class TestDefaultFactories(unittest.TestCase):

    def test_anthropic_client_gets_the_model_prices(self):
        settings = default_settings(_base())
        base = _base(anthropic_api_key="sk-test")
        client = default_llm_client(settings, base)
        self.assertIsInstance(client, AnthropicLLMClient)
        self.assertEqual(client.model, settings.llm_model)
        self.assertEqual((client.budget.input_price_per_mtok, client.budget.output_price_per_mtok), (1.0, 5.0))

    def test_anthropic_without_key_fails_naming_it(self):
        with self.assertRaises(ValueError) as ctx:
            default_llm_client(default_settings(_base()), _base(anthropic_api_key=None))
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_ollama_client(self):
        ollama_settings = ConsoleSettings(**{**default_settings(_base()).as_dict(), "llm_provider": "ollama", "llm_model": OLLAMA})
        ollama_client = default_llm_client(ollama_settings, _base())
        self.assertIsInstance(ollama_client, OllamaLLMClient)
        self.assertEqual(ollama_client.model, OLLAMA)

    def test_embedding_manager_factory(self):
        manager = default_embedding_manager(default_settings(_base()), _base())
        self.assertEqual(manager._provider, "sentence-transformers")
        self.assertEqual(manager._default_model_name, "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Pool and Redis prefixing
# ---------------------------------------------------------------------------

class TestConsolePool(unittest.TestCase):

    def _pool(self, **kwargs):
        return ConsolePool(_base(), llm_client_factory=lambda s, b: MockLLMClient(latency_seconds=0.0),
                           embedding_manager_factory=_mock_manager, **kwargs)

    def test_one_engine_per_configuration_and_shared_managers(self):
        pool = self._pool()
        a = default_settings(_base())
        b = ConsoleSettings(**{**a.as_dict(), "threshold": 0.5})
        self.assertIs(pool.get(a), pool.get(a))
        self.assertIsNot(pool.get(a), pool.get(b))
        self.assertIs(pool.get(a).embedding_manager, pool.get(b).embedding_manager)

    def test_warm_up_loads_the_shared_manager_and_leaves_no_memo(self):
        pool = self._pool()
        settings = default_settings(_base())
        pool.warm_up(settings)
        manager = pool.get(settings).embedding_manager
        self.assertIs(pool._managers[(settings.embedding_provider, settings.embedding_model)], manager)
        self.assertFalse(manager.forget("levy console warm-up"))

    def test_redis_engines_get_a_prefixed_store(self):
        pool = self._pool()
        settings = ConsoleSettings(**{**default_settings(_base()).as_dict(), "cache_store": "redis"})
        engine = pool.get(settings)
        self.assertIsInstance(engine.store, PrefixedRedisStore)
        self.assertIs(engine.exact_cache.store, engine.store)
        self.assertIn(settings.key, engine.store.prefix)

    def test_prefixed_stores_never_collide(self):
        fake = FakeRedis()
        inner = RedisStore()
        inner.client = fake
        one = PrefixedRedisStore(inner, "levy:console:one:")
        two = PrefixedRedisStore(inner, "levy:console:two:")
        entry = CacheEntry(key_hash="k", prompt="p", response_text="from one")

        one.set("k", entry)

        self.assertEqual(one.get("k").response_text, "from one")
        self.assertIsNone(two.get("k"))
        self.assertEqual((one.count(), two.count()), (1, 0))
        fake.set("delivered-key", "{}")
        one.clear()
        self.assertEqual(one.count(), 0)
        self.assertIn("delivered-key", fake.data)
        two.clear()  # nothing of its own: a no-op
        one.set("k", entry)
        one.delete("k")
        self.assertEqual(store_entry_count(one), 0)

    def test_redis_engine_round_trip_and_clear(self):
        pool = self._pool()
        settings = ConsoleSettings(**{**default_settings(_base()).as_dict(), "cache_store": "redis"})
        engine = pool.get(settings)
        engine.store.inner.client = FakeRedis()
        engine.generate("a prompt")
        self.assertEqual(engine.generate("a prompt").source, "exact_cache")
        report = pool.clear_all()
        self.assertEqual(report[settings.key]["exact_entries"], 1)
        self.assertEqual(store_entry_count(engine.store), 0)


# ---------------------------------------------------------------------------
# The console section of the API
# ---------------------------------------------------------------------------

class TestConsoleSection(unittest.TestCase):

    def test_openapi_lists_the_options_under_the_console_tag(self):
        spec = _client().get("/openapi.json").json()
        operation = spec["paths"]["/console/chat"]["post"]
        self.assertEqual(operation["tags"], ["console"])
        params = {p["name"]: p["schema"] for p in operation["parameters"]}

        def enum(name):
            return params[name]["enum"]

        self.assertEqual(enum("llm_provider"), list(OPTIONS.llm_providers))
        self.assertEqual(enum("embedding_model"), list(OPTIONS.all_embedding_models))
        self.assertEqual(enum("cache_store"), ["memory", "redis"])
        self.assertEqual(enum("vector_index_backend"), ["faiss", "brute_force"])
        self.assertEqual(params["llm_provider"]["default"], "anthropic")
        for path in ("/console/traces", "/console/traces/clear", "/console/cache/clear"):
            self.assertIn(path, spec["paths"])
        self.assertNotIn("/console", spec["paths"])

    def test_a_single_option_is_still_a_selector(self):
        app = create_app(_base(anthropic_models=(), ollama_embedding_models=()),
                         console_llm_client_factory=lambda s, b: MockLLMClient(latency_seconds=0.0),
                         console_embedding_manager_factory=_mock_manager)
        client = TestClient(app)
        spec = client.get("/openapi.json").json()
        params = {p["name"]: p["schema"] for p in spec["paths"]["/console/chat"]["post"]["parameters"]}
        for name in ("llm_provider", "llm_model", "embedding_provider"):
            self.assertEqual(len(params[name]["enum"]), 1, name)
        self.assertEqual(params["exact_cache"], {"type": "boolean", "title": "Exact Cache", "default": True})
        self.assertEqual(_chat(client, "one option").status_code, 200)

    def test_models_added_to_env_appear_without_a_restart(self):
        from levy.config import ModelCatalog

        catalogs = [ModelCatalog(ollama=("local-a",))]
        with mock.patch("levy.api.app.current_model_catalog", side_effect=lambda: catalogs[-1]):
            app = create_app(console_llm_client_factory=lambda s, b: MockLLMClient(latency_seconds=0.0),
                             console_embedding_manager_factory=_mock_manager)
            client = TestClient(app)

            def llm_models():
                spec = client.get("/openapi.json").json()
                params = {p["name"]: p["schema"] for p in spec["paths"]["/console/chat"]["post"]["parameters"]}
                return params["llm_model"]["enum"]

            self.assertEqual(llm_models(), ["local-a"])
            self.assertEqual(_chat(client, "x", llm_model="local-h").status_code, 422)

            catalogs.append(ModelCatalog(ollama=("local-a", "local-h")))  # .env edited, Levy not restarted
            self.assertEqual(llm_models(), ["local-a", "local-h"])
            self.assertEqual(_chat(client, "x", llm_model="local-h").status_code, 200)

    def test_an_unlisted_value_is_rejected_with_the_options(self):
        response = _chat(_client(), "x", cache_store="disk")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["query", "cache_store"])
        self.assertIn("memory, redis", response.json()["detail"][0]["msg"])

    def test_delivered_endpoints_are_unchanged(self):
        spec = _client().get("/openapi.json").json()
        chat = spec["paths"]["/v1/chat/completions"]["post"]
        self.assertNotIn("parameters", chat)
        self.assertEqual(chat["requestBody"]["content"]["application/json"]["schema"]["$ref"],
                         "#/components/schemas/ChatCompletionRequest")
        self.assertEqual(set(spec["components"]["schemas"]["CacheConfigRequest"]["properties"]),
                         {"threshold", "embedding_model"})
        for path in ("/admin/cache/stats", "/admin/cache/clear"):
            operation = next(iter(spec["paths"][path].values()))
            self.assertNotIn("parameters", operation)
            self.assertNotIn("requestBody", operation)

    def test_console_traffic_leaves_delivered_caches_alone(self):
        client = _client()
        _chat(client, "hello there")
        _chat(client, "hello there")
        stats = client.get("/admin/cache/stats").json()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["index_size"], 0)

    def test_each_setting_is_its_own_cache_universe(self):
        client = _client()
        _chat(client, "the same prompt")
        self.assertEqual(_chat(client, "the same prompt").headers["x-cache-status"], "HIT")
        self.assertEqual(_chat(client, "the same prompt", semantic_cache="false").headers["x-cache-status"], "MISS")
        self.assertEqual(_chat(client, "the same prompt", llm_provider="ollama", llm_model=OLLAMA).headers["x-cache-status"], "MISS")

    def test_cross_provider_model_is_rejected(self):
        client = _client()
        response = _chat(client, "x", embedding_provider="ollama", embedding_model="modernbert")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["query", "embedding_model"])
        self.assertIn(OLLAMA_EMBED, response.json()["detail"][0]["msg"])
        response = _chat(client, "x", llm_provider="ollama")
        self.assertEqual(response.json()["detail"][0]["loc"], ["query", "llm_model"])
        self.assertEqual(_traces(client)["traces"], [])

    def test_no_configured_llm_model_answers_503_naming_the_variables(self):
        app = create_app(_base(anthropic_models=(), ollama_models=()),
                         console_embedding_manager_factory=_mock_manager)
        client = TestClient(app)
        response = _chat(client, "anything")
        self.assertEqual(response.status_code, 503)
        self.assertIn("LEVY_ANTHROPIC_MODEL_<n>", response.json()["detail"])
        self.assertIsNone(_traces(client)["defaults"])

    def test_mock_is_not_an_option(self):
        client = _client()
        self.assertEqual(_chat(client, "x", llm_provider="mock").status_code, 422)
        self.assertEqual(_chat(client, "x", embedding_provider="mock").status_code, 422)

    def test_response_matches_the_delivered_shape(self):
        client = _client()
        miss = _chat(client, "shape check")
        hit = _chat(client, "shape check")
        self.assertEqual(miss.headers["x-cache-status"], "MISS")
        self.assertNotIn("x-cache-similarity", miss.headers)
        self.assertEqual(hit.headers["x-cache-similarity"], "1.0")
        self.assertEqual(set(miss.json()), set(hit.json()))
        self.assertEqual(miss.json()["request_id"], _traces(client)["traces"][0]["id"])


class TestTraceFeed(unittest.TestCase):

    def test_miss_exact_and_semantic_traces(self):
        client = _client()
        _chat(client, "what is the refund window", threshold=0.0)
        _chat(client, "what is the refund window", threshold=0.0)
        semantic = _chat(client, "how long do I have to ask for a refund", threshold=0.0)
        miss, exact, sem = _traces(client)["traces"]

        self.assertEqual(miss["outcome"], "miss")
        self.assertEqual([s["name"] for s in miss["spans"]], ["exact_lookup", "llm_call", "store"])
        self.assertEqual(miss["semantic"]["index_size"], 0)
        self.assertEqual(miss["config_from_request"], ["threshold"])
        self.assertEqual(miss["cache_after"], {"exact_entries": 1, "index_size": 1})
        self.assertEqual(miss["llm"]["provider"], "anthropic")
        self.assertEqual(miss["embedding"]["canonical_name"], "mock")

        self.assertEqual(exact["outcome"], "exact_hit")
        self.assertIsNone(exact["semantic"])
        self.assertEqual(exact["savings"]["origin_id"], miss["id"])
        self.assertEqual(exact["matched_prompt"], "what is the refund window")
        self.assertIsNone(exact["embedding"]["dimension"])

        self.assertEqual(sem["outcome"], "semantic_hit")
        self.assertEqual(sem["semantic"]["nearest_prompt"], "what is the refund window")
        self.assertAlmostEqual(sem["semantic"]["similarity"], float(semantic.headers["x-cache-similarity"]))
        self.assertEqual(sem["savings"]["origin_id"], miss["id"])
        self.assertEqual([s["name"] for s in sem["spans"]], ["exact_lookup", "embed", "index_search"])

    def test_switched_off_caches_are_skipped(self):
        client = _client()
        _chat(client, "no semantic here", semantic_cache="false", exact_cache="false")
        trace = _traces(client)["traces"][0]
        self.assertEqual(trace["skipped"], ["exact_cache", "semantic_cache"])
        self.assertIsNone(trace["semantic"])
        self.assertNotIn("embed", [s["name"] for s in trace["spans"]])
        self.assertNotIn("exact_lookup", [s["name"] for s in trace["spans"]])

    def test_provider_location(self):
        client = _client()
        _chat(client, "to ollama", llm_provider="ollama", llm_model=OLLAMA)
        _chat(client, "to anthropic")
        ollama, anth = _traces(client)["traces"]
        self.assertEqual((ollama["llm"]["location"], ollama["llm"]["endpoint"]), ("local", "http://localhost:11434"))
        self.assertEqual(anth["llm"]["location"], "remote")

    def test_anthropic_tokens_cost_and_spend(self):
        client = _client(llm_factory=_anthropic_factory(input_tokens=12, output_tokens=40))
        _chat(client, "billed prompt")
        _chat(client, "billed prompt")
        feed = _traces(client)
        miss, hit = feed["traces"]
        expected = 12 / 1e6 * 1.0 + 40 / 1e6 * 5.0
        self.assertEqual((miss["llm"]["input_tokens"], miss["llm"]["output_tokens"]), (12, 40))
        self.assertEqual(miss["llm"]["stop_reason"], "end_turn")
        self.assertEqual(miss["llm"]["endpoint"], "https://api.anthropic.com")
        self.assertAlmostEqual(miss["llm"]["cost_usd"], expected)
        self.assertAlmostEqual(hit["savings"]["cost_usd"], expected)
        self.assertAlmostEqual(feed["spent_usd"], expected)
        self.assertEqual(feed["stats"]["served"], 2)
        self.assertEqual(feed["stats"]["hit_rate"], 0.5)

    def test_cursor_clear_and_epoch(self):
        client = _client()
        _chat(client, "one")
        first = _traces(client)
        _chat(client, "two")
        second = _traces(client, after=first["version"])
        self.assertEqual([t["prompt"] for t in second["traces"]], ["two"])

        cleared = client.post("/console/traces/clear").json()
        after = _traces(client)
        self.assertEqual(after["traces"], [])
        self.assertEqual(after["epoch"], cleared["epoch"])
        self.assertNotEqual(after["epoch"], first["epoch"])

    def test_buffer_is_bounded(self):
        client = _client(trace_buffer_size=2)
        for prompt in ("a", "b", "c"):
            _chat(client, prompt)
        self.assertEqual([t["prompt"] for t in _traces(client)["traces"]], ["b", "c"])

    def test_cache_clear_empties_console_caches_and_origins(self):
        client = _client()
        _chat(client, "cached prompt")
        report = client.post("/console/cache/clear").json()
        self.assertEqual(list(report["cleared"].values())[0]["exact_entries"], 1)
        self.assertEqual(_chat(client, "cached prompt").headers["x-cache-status"], "MISS")
        third = _chat(client, "cached prompt")
        traces = _traces(client)["traces"]
        self.assertEqual(traces[2]["savings"]["origin_id"], traces[1]["id"])
        self.assertEqual(third.headers["x-cache-status"], "HIT")


class TestErrorsAndInFlight(unittest.TestCase):

    def test_budget_guard_is_traced(self):
        client = _client(llm_factory=_anthropic_factory(budget_cap_usd=0.0))
        response = _chat(client, "over budget")
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["error"], "budget_exceeded")
        trace = _traces(client)["traces"][0]
        self.assertEqual(trace["outcome"], "error")
        self.assertEqual(trace["error"]["kind"], "budget_exceeded")
        self.assertEqual(trace["error"]["stage"], "llm_call")
        self.assertEqual(trace["semantic"]["index_size"], 0)

    def test_pool_cap_is_traced(self):
        client = _client(console_max_engines=1)
        _chat(client, "first")
        response = _chat(client, "second", threshold=0.5)
        self.assertEqual(response.status_code, 400)
        trace = _traces(client)["traces"][-1]
        self.assertEqual(trace["error"]["kind"], "pool_cap_exceeded")
        self.assertEqual(trace["error"]["stage"], "router")
        self.assertIsNone(trace["semantic"])

    def test_refusal_and_provider_errors_are_traced(self):
        class Refusing(LLMClient):
            def generate(self, request):
                from levy.llm_client import AnthropicRefusalError
                raise AnthropicRefusalError("refusal")

        class Broken(LLMClient):
            def generate(self, request):
                raise RuntimeError("connection reset")

        client = _client(llm_factory=lambda s, b: Refusing())
        self.assertEqual(_chat(client, "refused").status_code, 502)
        self.assertEqual(_traces(client)["traces"][0]["error"]["kind"], "provider_refusal")

        client = _client(llm_factory=lambda s, b: Broken(), raise_server_exceptions=False)
        self.assertEqual(_chat(client, "broken").status_code, 500)
        trace = _traces(client)["traces"][0]
        self.assertEqual(trace["error"]["kind"], "provider_error")
        self.assertIn("connection reset", trace["error"]["detail"])

    def test_pending_provider_call_is_visible(self):
        class Blocking(LLMClient):
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def generate(self, request):
                self.started.set()
                self.release.wait(5)
                return LLMResponse(text="late answer", token_usage=2, model="m")

        blocking = Blocking()
        client = _client(llm_factory=lambda s, b: blocking)
        worker = threading.Thread(target=_chat, args=(client, "slow prompt"))
        worker.start()
        try:
            self.assertTrue(blocking.started.wait(5))
            trace = _traces(client)["traces"][0]
            self.assertEqual(trace["status"], "in_flight")
            self.assertEqual(trace["current_stage"], "llm_call")
            self.assertEqual(trace["semantic"]["index_size"], 0)
        finally:
            blocking.release.set()
            worker.join(5)
        deadline = time.time() + 5
        while _traces(client)["traces"][0]["status"] != "done" and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(_traces(client)["traces"][0]["answer"], "late answer")


class TestConsolePage(unittest.TestCase):

    def test_page_and_assets_are_served(self):
        client = _client()
        page = client.get("/console")
        self.assertEqual(page.status_code, 200)
        self.assertIn("<title>Levy Console</title>", page.text)
        for asset in ("console.js", "console.css"):
            self.assertEqual(client.get(f"/console/assets/{asset}").status_code, 200)

    def test_docs_hide_parameter_notes(self):
        page = _client().get("/docs")
        self.assertEqual(page.status_code, 200)
        self.assertIn(".parameter__enum, .parameter__default, .parameter__extension { display: none; }", page.text)

    def test_assets_reference_no_external_origin(self):
        pattern = re.compile(r"""(src|href)\s*=\s*["']https?://|url\(\s*["']?https?://|fetch\(\s*["']https?://|@import\s+["']?https?://""")
        for path in STATIC_DIR.iterdir():
            self.assertIsNone(pattern.search(path.read_text()), f"external reference in {path.name}")


if __name__ == "__main__":
    unittest.main()
