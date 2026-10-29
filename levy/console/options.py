"""
The closed lists a console request chooses its configuration from.

Two kinds of list, kept apart on purpose:

- **What Levy implements** -- provider types, vector index backends, cache
  stores, and the study's two embedding models. These are Levy's own
  capabilities and live here, in code.
- **Which models exist and what they cost** -- Anthropic models with their
  per-MTok prices, Ollama LLM models, Ollama embedding models. These change
  every few months and are never written in the code: they come from the
  configuration, i.e. from the numbered `LEVY_ANTHROPIC_MODEL_<n>`,
  `LEVY_OLLAMA_MODEL_<n>` and `LEVY_OLLAMA_EMBEDDING_MODEL_<n>` variables in `.env`.

A provider with no configured model is not offered. No list contains a mock:
the mock clients are test doubles, not something a practitioner can pick.
"""

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from levy.config import LevyConfig
from levy.embedding_manager import KNOWN_MODEL_NAMES

VECTOR_INDEX_BACKENDS: Tuple[str, ...] = ("faiss", "brute_force")
CACHE_STORES: Tuple[str, ...] = ("memory", "redis")

SETTING_NAMES: Tuple[str, ...] = (
    "threshold",
    "embedding_provider",
    "embedding_model",
    "llm_provider",
    "llm_model",
    "exact_cache",
    "semantic_cache",
    "vector_index_backend",
    "cache_store",
)

NO_LLM_MODELS_MESSAGE = (
    "No LLM model is configured: set LEVY_ANTHROPIC_MODEL_<n> (with its two _USD_PER_MTOK prices) "
    "and/or LEVY_OLLAMA_MODEL_<n> in .env"
)


@dataclass(frozen=True)
class ConsoleSettings:
    """One console request's resolved configuration; also its engine's pool key."""

    threshold: float
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    exact_cache: bool
    semantic_cache: bool
    vector_index_backend: str
    cache_store: str

    @property
    def key(self) -> str:
        """Short stable id: names this configuration's cache universe and Redis prefix."""
        raw = "|".join(str(getattr(self, name)) for name in SETTING_NAMES)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _unique(values) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class ConsoleOptions:
    """The lists one console offers, read from the configuration it was built with."""

    embedding_models: Dict[str, Tuple[str, ...]]
    llm_models: Dict[str, Tuple[str, ...]]
    llm_prices_per_mtok: Dict[str, Tuple[float, float]]

    @classmethod
    def from_config(cls, config: LevyConfig) -> "ConsoleOptions":
        embedding_models = {"sentence-transformers": tuple(KNOWN_MODEL_NAMES)}
        if config.ollama_embedding_models:
            embedding_models["ollama"] = tuple(config.ollama_embedding_models)
        llm_models: Dict[str, Tuple[str, ...]] = {}
        if config.anthropic_models:
            llm_models["anthropic"] = tuple(name for name, _, _ in config.anthropic_models)
        if config.ollama_models:
            llm_models["ollama"] = tuple(config.ollama_models)
        prices = {name: (input_price, output_price) for name, input_price, output_price in config.anthropic_models}
        return cls(embedding_models=embedding_models, llm_models=llm_models, llm_prices_per_mtok=prices)

    @property
    def embedding_providers(self) -> Tuple[str, ...]:
        return tuple(self.embedding_models)

    @property
    def llm_providers(self) -> Tuple[str, ...]:
        return tuple(self.llm_models)

    @property
    def all_embedding_models(self) -> Tuple[str, ...]:
        return _unique(m for ms in self.embedding_models.values() for m in ms)

    @property
    def all_llm_models(self) -> Tuple[str, ...]:
        return _unique(m for ms in self.llm_models.values() for m in ms)

    def mismatched_model(self, settings: ConsoleSettings) -> Optional[Tuple[str, Tuple[str, ...]]]:
        """
        The (parameter, valid models) pair for a model that does not belong to its
        chosen provider, or None when both pairs are consistent. Swagger cannot
        narrow one selector by another, so each model selector lists every model.
        """
        if settings.embedding_model not in self.embedding_models.get(settings.embedding_provider, ()):
            return "embedding_model", self.embedding_models.get(settings.embedding_provider, ())
        if settings.llm_model not in self.llm_models.get(settings.llm_provider, ()):
            return "llm_model", self.llm_models.get(settings.llm_provider, ())
        return None

    def default_settings(self, base: LevyConfig) -> Optional[ConsoleSettings]:
        """
        The console's defaults, read from the base configuration. A base value the
        lists do not offer (a test double's `mock` provider, an unconfigured
        model) falls back to the first configured option. None when no LLM model
        is configured at all.
        """
        if not self.llm_models:
            return None
        embedding_provider = base.embedding_provider if base.embedding_provider in self.embedding_models else self.embedding_providers[0]
        embedding_models = self.embedding_models[embedding_provider]
        embedding_model = base.embedding_model if base.embedding_model in embedding_models else embedding_models[0]

        llm_provider = base.llm_provider if base.llm_provider in self.llm_models else self.llm_providers[0]
        configured_model = base.anthropic_model if llm_provider == "anthropic" else base.model_name
        llm_models = self.llm_models[llm_provider]
        llm_model = configured_model if configured_model in llm_models else llm_models[0]

        return ConsoleSettings(
            threshold=base.similarity_threshold,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            exact_cache=base.enable_exact_cache,
            semantic_cache=base.enable_semantic_cache,
            vector_index_backend=resolve_backend(base.vector_index_backend),
            cache_store=base.cache_store_type if base.cache_store_type in CACHE_STORES else CACHE_STORES[0],
        )


def resolve_backend(configured: str) -> str:
    """The backend `make_vector_index` actually builds for `configured`."""
    if configured in VECTOR_INDEX_BACKENDS:
        return configured
    try:
        import faiss  # noqa: F401
    except ImportError:
        return "brute_force"
    return "faiss"
