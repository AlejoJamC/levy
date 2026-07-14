"""
Tests for levy.cache.base, exact_cache, store, and redis_store.

All offline: RedisStore is exercised against a small in-memory fake client
(no real Redis server), matching the duck-typed interface it implements.
"""

import time
import unittest

from levy.cache.base import CacheInterface
from levy.cache.exact_cache import ExactCache
from levy.cache.redis_store import RedisStore
from levy.cache.store import InMemoryStore
from levy.models import CacheEntry, LLMRequest


# ---------------------------------------------------------------------------
# CacheInterface (abstract contract)
# ---------------------------------------------------------------------------

class _MinimalCache(CacheInterface):
    """Delegates straight to the ABC stubs so their bodies execute."""

    def get(self, request):
        return super().get(request)

    def set(self, request, response_text, embedding=None):
        return super().set(request, response_text, embedding=embedding)

    def clear(self):
        return super().clear()


class TestCacheInterface(unittest.TestCase):

    def test_abstract_stub_bodies_are_inert(self):
        cache = _MinimalCache()
        self.assertIsNone(cache.get(LLMRequest(prompt="x")))
        self.assertIsNone(cache.set(LLMRequest(prompt="x"), "resp"))
        self.assertIsNone(cache.clear())


# ---------------------------------------------------------------------------
# ExactCache
# ---------------------------------------------------------------------------

class TestExactCache(unittest.TestCase):

    def test_expired_entry_is_deleted_and_treated_as_miss(self):
        store = InMemoryStore()
        cache = ExactCache(store)
        request = LLMRequest(prompt="hello")
        cache.set(request, "world")

        key = cache._get_key("hello")
        store.entries[key].expires_at = time.time() - 1  # force expiry

        self.assertIsNone(cache.get(request))
        self.assertNotIn(key, store.entries)

    def test_clear_is_a_no_op(self):
        """ExactCache.clear() is intentionally a no-op (store may be shared)."""
        cache = ExactCache(InMemoryStore())
        self.assertIsNone(cache.clear())


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------

class TestInMemoryStore(unittest.TestCase):

    def test_fifo_eviction_when_max_size_reached(self):
        store = InMemoryStore(max_size=1)
        store.set("a", CacheEntry(key_hash="a", prompt="a", response_text="A"))
        store.set("b", CacheEntry(key_hash="b", prompt="b", response_text="B"))
        self.assertNotIn("a", store.entries)
        self.assertIn("b", store.entries)
        self.assertEqual(len(store.entries), 1)

    def test_delete_removes_entry_with_embedding_from_vector_index(self):
        store = InMemoryStore()
        entry = CacheEntry(key_hash="a", prompt="a", response_text="A", embedding=[0.1, 0.2])
        store.set("a", entry)
        self.assertEqual(store.get_all_with_embeddings(), [entry])

        store.delete("a")
        self.assertNotIn("a", store.entries)
        self.assertEqual(store.get_all_with_embeddings(), [])

    def test_clear_empties_both_entries_and_vector_index(self):
        store = InMemoryStore()
        store.set("a", CacheEntry(key_hash="a", prompt="a", response_text="A", embedding=[0.1]))
        store.clear()
        self.assertEqual(store.entries, {})
        self.assertEqual(store.get_all_with_embeddings(), [])

    def test_delete_missing_key_is_a_no_op(self):
        store = InMemoryStore()
        store.delete("does-not-exist")  # must not raise
        self.assertEqual(store.entries, {})


# ---------------------------------------------------------------------------
# RedisStore (duck-typed against a fake in-memory redis client)
# ---------------------------------------------------------------------------

class _FakeRedisClient:
    """Implements just enough of the redis-py surface for RedisStore."""

    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value, ex=None):
        self._data[key] = value

    def delete(self, key):
        self._data.pop(key, None)

    def keys(self, pattern="*"):
        return list(self._data.keys())

    def mget(self, keys):
        return [self._data.get(k) for k in keys]

    def flushdb(self):
        self._data.clear()


def _redis_store_with_fake_client():
    store = RedisStore.__new__(RedisStore)  # bypass __init__'s real redis.from_url()
    store.client = _FakeRedisClient()
    store.ttl = 3600
    return store


class TestRedisStore(unittest.TestCase):

    def test_set_then_get_roundtrips_entry(self):
        store = _redis_store_with_fake_client()
        entry = CacheEntry(key_hash="k1", prompt="hello", response_text="world")
        store.set("k1", entry)
        restored = store.get("k1")
        self.assertEqual(restored.prompt, "hello")
        self.assertEqual(restored.response_text, "world")

    def test_get_missing_key_returns_none(self):
        store = _redis_store_with_fake_client()
        self.assertIsNone(store.get("missing"))

    def test_get_malformed_json_returns_none(self):
        store = _redis_store_with_fake_client()
        store.client.set("bad", "not valid json {{{")
        self.assertIsNone(store.get("bad"))

    def test_delete_removes_key(self):
        store = _redis_store_with_fake_client()
        entry = CacheEntry(key_hash="k1", prompt="hello", response_text="world")
        store.set("k1", entry)
        store.delete("k1")
        self.assertIsNone(store.get("k1"))

    def test_get_all_with_embeddings_returns_stored_entries(self):
        store = _redis_store_with_fake_client()
        entry = CacheEntry(key_hash="k1", prompt="hello", response_text="world", embedding=[0.1, 0.2])
        store.set("k1", entry)
        entries = store.get_all_with_embeddings()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].prompt, "hello")

    def test_get_all_with_embeddings_empty_store(self):
        store = _redis_store_with_fake_client()
        self.assertEqual(store.get_all_with_embeddings(), [])

    def test_get_all_with_embeddings_skips_malformed_entries(self):
        store = _redis_store_with_fake_client()
        store.client.set("bad", "not valid json {{{")
        self.assertEqual(store.get_all_with_embeddings(), [])

    def test_get_all_with_embeddings_skips_falsy_values(self):
        """mget can return None for a key that expired between KEYS and MGET."""
        store = _redis_store_with_fake_client()
        entry = CacheEntry(key_hash="k1", prompt="hello", response_text="world")
        store.set("k1", entry)
        store.client._data["expired-key"] = None  # simulates a raced-out key
        store.client.keys = lambda pattern="*": ["k1", "expired-key"]
        entries = store.get_all_with_embeddings()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].prompt, "hello")

    def test_clear_flushes_db(self):
        store = _redis_store_with_fake_client()
        store.set("k1", CacheEntry(key_hash="k1", prompt="hello", response_text="world"))
        store.clear()
        self.assertIsNone(store.get("k1"))


if __name__ == "__main__":
    unittest.main()
