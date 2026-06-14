"""
EmbeddingManager: single entry-point for embedding generation in the Levy study.

Responsibilities:
- Registry mapping study-model aliases to concrete checkpoints (D4).
- Lazy construction and caching of one EmbeddingClient per checkpoint (D3).
- In-memory memoization keyed by (model_key, sha256(text)) (D5).
- Symmetric task-prefix handling per model so callers never see prefixes (D2).
- Mock-provider bypass for offline operation.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from levy.embeddings import EmbeddingClient, MockEmbeddingClient, SentenceTransformerClient, OllamaEmbeddingClient


@dataclass(frozen=True)
class _ModelSpec:
    canonical_name: str
    provider: str
    checkpoint: str
    prefix: str
    trust_remote_code: bool = False


# Single authoritative registry of study models (design.md D4).
# Keys are all accepted aliases; value is the resolved spec.
_REGISTRY: Dict[str, _ModelSpec] = {
    "all-minilm": _ModelSpec(
        canonical_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
        checkpoint="sentence-transformers/all-MiniLM-L6-v2",
        prefix="",
    ),
    "all-MiniLM-L6-v2": _ModelSpec(
        canonical_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
        checkpoint="sentence-transformers/all-MiniLM-L6-v2",
        prefix="",
    ),
    "modernbert": _ModelSpec(
        canonical_name="modernbert",
        provider="sentence-transformers",
        checkpoint="nomic-ai/modernbert-embed-base",
        prefix="search_query: ",
        trust_remote_code=True,
    ),
}

KNOWN_MODEL_NAMES = sorted(set(spec.canonical_name for spec in _REGISTRY.values()))


def _resolve(model_name: str) -> _ModelSpec:
    spec = _REGISTRY.get(model_name)
    if spec is None:
        raise ValueError(
            f"Unknown embedding model {model_name!r}. "
            f"Known models: {KNOWN_MODEL_NAMES}"
        )
    return spec


def _memo_key(model_key: str, text: str) -> Tuple[str, str]:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (model_key, text_hash)


class ModelIdentity:
    """Carries the resolved model information for downstream consumers."""

    def __init__(self, canonical_name: str, checkpoint: str, dimension: int) -> None:
        self.canonical_name = canonical_name
        self.checkpoint = checkpoint
        self.dimension = dimension

    def as_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "checkpoint": self.checkpoint,
            "dimension": self.dimension,
        }


class EmbeddingManager:
    """
    Single entry-point for embedding generation.

    Usage (study models):
        manager = EmbeddingManager.from_config(config)
        vector = manager.embed("some text")        # uses config.embedding_model
        vector = manager.embed_with("modernbert", "some text")  # runtime switch

    Usage (mock / offline):
        config.embedding_provider = "mock"
        manager = EmbeddingManager.from_config(config)
        # All embed calls return deterministic random vectors; no network or disk I/O.

    Usage (Ollama demo path):
        config.embedding_provider = "ollama"
        manager = EmbeddingManager.from_config(config)
        # Delegates to OllamaEmbeddingClient; embed_with() ignores model_name.
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "sentence-transformers",
        mock_dimension: int = 384,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self._default_model_name = model_name
        self._provider = provider
        self._mock_dimension = mock_dimension
        self._ollama_base_url = ollama_base_url

        # Lazy-loaded clients keyed by checkpoint string (or "mock" / "ollama").
        self._clients: Dict[str, EmbeddingClient] = {}
        # Memoization cache: (model_key, sha256(text)) → vector
        self._memo: Dict[Tuple[str, str], List[float]] = {}

    @classmethod
    def from_config(cls, config) -> "EmbeddingManager":
        """Construct from a LevyConfig instance."""
        return cls(
            model_name=config.embedding_model,
            provider=config.embedding_provider,
            ollama_base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed text using the configured default model."""
        return self.embed_with(self._default_model_name, text)

    def embed_with(self, model_name: str, text: str) -> List[float]:
        """Embed text using a specific study-model alias (runtime switching)."""
        if self._provider == "mock":
            client = self._get_mock_client()
            key = _memo_key("mock", text)
            if key not in self._memo:
                self._memo[key] = client.embed(text)
            return self._memo[key]

        if self._provider == "ollama":
            client = self._get_ollama_client()
            key = _memo_key("ollama:" + model_name, text)
            if key not in self._memo:
                self._memo[key] = client.embed(text)
            return self._memo[key]

        spec = _resolve(model_name)
        prefixed = spec.prefix + text
        memo_k = _memo_key(spec.checkpoint, prefixed)
        if memo_k not in self._memo:
            client = self._get_st_client(spec)
            self._memo[memo_k] = client.embed(prefixed)
        return self._memo[memo_k]

    def get_dimension(self, model_name: Optional[str] = None) -> int:
        """Return the embedding dimension for the given (or default) model."""
        name = model_name or self._default_model_name
        if self._provider == "mock":
            return self._get_mock_client().get_dimension()
        if self._provider == "ollama":
            return self._get_ollama_client().get_dimension()
        spec = _resolve(name)
        return self._get_st_client(spec).get_dimension()

    def get_model_identity(self, model_name: Optional[str] = None) -> "ModelIdentity":
        """Return canonical name, resolved checkpoint id, and dimension."""
        name = model_name or self._default_model_name
        if self._provider == "mock":
            return ModelIdentity(
                canonical_name="mock",
                checkpoint="mock",
                dimension=self._get_mock_client().get_dimension(),
            )
        if self._provider == "ollama":
            client = self._get_ollama_client()
            return ModelIdentity(
                canonical_name="ollama",
                checkpoint=name,
                dimension=client.get_dimension(),
            )
        spec = _resolve(name)
        return ModelIdentity(
            canonical_name=spec.canonical_name,
            checkpoint=spec.checkpoint,
            dimension=self._get_st_client(spec).get_dimension(),
        )

    def clear_memoization(self) -> None:
        """Evict all cached embeddings (for tests or per-configuration resets)."""
        self._memo.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mock_client(self) -> MockEmbeddingClient:
        if "mock" not in self._clients:
            self._clients["mock"] = MockEmbeddingClient(dimension=self._mock_dimension)
        return self._clients["mock"]  # type: ignore[return-value]

    def _get_ollama_client(self) -> OllamaEmbeddingClient:
        if "ollama" not in self._clients:
            self._clients["ollama"] = OllamaEmbeddingClient(
                base_url=self._ollama_base_url,
                model=self._default_model_name,
            )
        return self._clients["ollama"]  # type: ignore[return-value]

    def _get_st_client(self, spec: _ModelSpec) -> SentenceTransformerClient:
        if spec.checkpoint not in self._clients:
            self._clients[spec.checkpoint] = SentenceTransformerClient(
                model_name=spec.checkpoint,
                trust_remote_code=spec.trust_remote_code,
            )
        return self._clients[spec.checkpoint]  # type: ignore[return-value]
