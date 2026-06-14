"""
Tests for EmbeddingManager (LEV-1).

All tests run offline with mock clients injected into the manager's client cache,
so no model downloads or network access are required.
"""

import unittest
from levy.config import LevyConfig
from levy.embedding_manager import EmbeddingManager, KNOWN_MODEL_NAMES, _resolve
from levy.embeddings import MockEmbeddingClient


class _RecordingMock(MockEmbeddingClient):
    """MockEmbeddingClient that records every text it receives."""

    def __init__(self, dimension=384):
        super().__init__(dimension=dimension)
        self.received_texts = []
        self.call_count = 0

    def embed(self, text):
        self.received_texts.append(text)
        self.call_count += 1
        return super().embed(text)


def _manager_with_injected_mock(model_name, mock_checkpoint, dimension=384):
    """Return a manager with a RecordingMock pre-injected for the given checkpoint."""
    manager = EmbeddingManager(model_name, provider="sentence-transformers")
    mock = _RecordingMock(dimension=dimension)
    manager._clients[mock_checkpoint] = mock
    return manager, mock


# ---------------------------------------------------------------------------
# 5.1  Runtime selection and switching
# ---------------------------------------------------------------------------

class TestRuntimeSelection(unittest.TestCase):

    def test_manager_with_model_a_then_model_b(self):
        """embed_with() can use two different study models in one process."""
        manager = EmbeddingManager("all-MiniLM-L6-v2", provider="sentence-transformers")
        mock_a = _RecordingMock(dimension=384)
        mock_b = _RecordingMock(dimension=768)
        manager._clients["sentence-transformers/all-MiniLM-L6-v2"] = mock_a
        manager._clients["nomic-ai/modernbert-embed-base"] = mock_b

        vec_a = manager.embed_with("all-MiniLM-L6-v2", "hello")
        vec_b = manager.embed_with("modernbert", "hello")

        self.assertEqual(len(vec_a), 384)
        self.assertEqual(len(vec_b), 768)

    def test_unknown_model_raises_with_known_list(self):
        """Unknown model name raises ValueError naming the model and listing known models."""
        manager = EmbeddingManager("all-MiniLM-L6-v2", provider="sentence-transformers")
        with self.assertRaises(ValueError) as ctx:
            manager.embed_with("does-not-exist", "text")
        msg = str(ctx.exception)
        self.assertIn("does-not-exist", msg)
        for name in KNOWN_MODEL_NAMES:
            self.assertIn(name, msg)

    def test_alias_all_minilm_resolves(self):
        """Short alias 'all-minilm' resolves without error."""
        manager = EmbeddingManager("all-minilm", provider="sentence-transformers")
        mock = _RecordingMock()
        manager._clients["sentence-transformers/all-MiniLM-L6-v2"] = mock
        manager.embed("test")
        self.assertEqual(mock.call_count, 1)


# ---------------------------------------------------------------------------
# 5.2  Alias resolution exposes resolved checkpoint ids
# ---------------------------------------------------------------------------

class TestAliasResolution(unittest.TestCase):

    def test_modernbert_alias_resolves_to_nomic_checkpoint(self):
        spec = _resolve("modernbert")
        self.assertEqual(spec.checkpoint, "nomic-ai/modernbert-embed-base")
        self.assertEqual(spec.canonical_name, "modernbert")

    def test_all_minilm_alias_resolves(self):
        spec = _resolve("all-minilm")
        self.assertEqual(spec.checkpoint, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(spec.canonical_name, "all-MiniLM-L6-v2")

    def test_full_name_alias_resolves(self):
        spec = _resolve("all-MiniLM-L6-v2")
        self.assertEqual(spec.checkpoint, "sentence-transformers/all-MiniLM-L6-v2")

    def test_get_model_identity_exposes_checkpoint(self):
        manager, _ = _manager_with_injected_mock(
            "modernbert", "nomic-ai/modernbert-embed-base", dimension=768
        )
        identity = manager.get_model_identity()
        self.assertEqual(identity.checkpoint, "nomic-ai/modernbert-embed-base")
        self.assertEqual(identity.canonical_name, "modernbert")


# ---------------------------------------------------------------------------
# 5.3  Memoization
# ---------------------------------------------------------------------------

class TestMemoization(unittest.TestCase):

    def test_same_text_computed_once(self):
        """Repeated embed() calls for the same text use the memoized vector."""
        manager, mock = _manager_with_injected_mock(
            "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"
        )
        v1 = manager.embed("hello")
        v2 = manager.embed("hello")
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(v1, v2)

    def test_different_texts_each_computed(self):
        manager, mock = _manager_with_injected_mock(
            "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"
        )
        manager.embed("hello")
        manager.embed("world")
        self.assertEqual(mock.call_count, 2)

    def test_same_text_different_models_are_independent(self):
        """Same text embedded under two models produces two independent cache entries."""
        manager = EmbeddingManager("all-MiniLM-L6-v2", provider="sentence-transformers")
        mock_a = _RecordingMock(dimension=384)
        mock_b = _RecordingMock(dimension=768)
        manager._clients["sentence-transformers/all-MiniLM-L6-v2"] = mock_a
        manager._clients["nomic-ai/modernbert-embed-base"] = mock_b

        manager.embed_with("all-MiniLM-L6-v2", "hello")
        manager.embed_with("modernbert", "hello")

        self.assertEqual(mock_a.call_count, 1)
        self.assertEqual(mock_b.call_count, 1)
        self.assertEqual(len(manager._memo), 2)

    def test_clear_memoization_forces_recompute(self):
        manager, mock = _manager_with_injected_mock(
            "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"
        )
        manager.embed("hello")
        self.assertEqual(mock.call_count, 1)
        manager.clear_memoization()
        manager.embed("hello")
        self.assertEqual(mock.call_count, 2)


# ---------------------------------------------------------------------------
# 5.4  Dimension and identity exposure
# ---------------------------------------------------------------------------

class TestDimensionAndIdentity(unittest.TestCase):

    def test_get_dimension_returns_positive_integer(self):
        manager, _ = _manager_with_injected_mock(
            "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", dimension=384
        )
        dim = manager.get_dimension()
        self.assertIsInstance(dim, int)
        self.assertGreater(dim, 0)
        self.assertEqual(dim, 384)

    def test_get_model_identity_as_dict(self):
        manager, _ = _manager_with_injected_mock(
            "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", dimension=384
        )
        d = manager.get_model_identity().as_dict()
        self.assertIn("canonical_name", d)
        self.assertIn("checkpoint", d)
        self.assertIn("dimension", d)
        self.assertEqual(d["dimension"], 384)

    def test_mock_provider_identity(self):
        manager = EmbeddingManager("all-MiniLM-L6-v2", provider="mock")
        identity = manager.get_model_identity()
        self.assertEqual(identity.canonical_name, "mock")
        self.assertIsInstance(identity.dimension, int)


# ---------------------------------------------------------------------------
# 5.5  Symmetric task-prefix handling
# ---------------------------------------------------------------------------

class TestPrefixHandling(unittest.TestCase):

    def test_modernbert_receives_prefix_on_embed(self):
        """ModernBERT model receives 'search_query: ' prefix for every text."""
        manager, mock = _manager_with_injected_mock(
            "modernbert", "nomic-ai/modernbert-embed-base", dimension=768
        )
        manager.embed("hello world")
        self.assertEqual(mock.received_texts[0], "search_query: hello world")

    def test_modernbert_prefix_same_on_store_and_lookup(self):
        """Store and lookup embeddings are identical because the same prefix is applied."""
        manager, mock = _manager_with_injected_mock(
            "modernbert", "nomic-ai/modernbert-embed-base", dimension=768
        )
        vec_store = manager.embed("what is Python?")
        vec_lookup = manager.embed("what is Python?")
        self.assertEqual(vec_store, vec_lookup)
        self.assertEqual(mock.call_count, 1)  # memoized after first call

    def test_minilm_receives_raw_text(self):
        """all-MiniLM-L6-v2 passes raw text without any prefix."""
        manager, mock = _manager_with_injected_mock(
            "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", dimension=384
        )
        manager.embed("hello world")
        self.assertEqual(mock.received_texts[0], "hello world")


# ---------------------------------------------------------------------------
# 5.6  Default configuration matches the study baseline
# ---------------------------------------------------------------------------

class TestDefaultConfig(unittest.TestCase):

    def test_default_embedding_provider_is_sentence_transformers(self):
        config = LevyConfig()
        self.assertEqual(config.embedding_provider, "sentence-transformers")

    def test_default_embedding_model_is_all_minilm(self):
        config = LevyConfig()
        self.assertEqual(config.embedding_model, "all-MiniLM-L6-v2")

    def test_manager_from_default_config_uses_correct_model(self):
        config = LevyConfig()
        manager = EmbeddingManager.from_config(config)
        mock = _RecordingMock(dimension=384)
        manager._clients["sentence-transformers/all-MiniLM-L6-v2"] = mock
        manager.embed("test")
        # Verify text was passed raw (no prefix for all-MiniLM)
        self.assertEqual(mock.received_texts[0], "test")


if __name__ == "__main__":
    unittest.main()
