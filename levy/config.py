import os
import re
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Tuple
from dotenv import dotenv_values, find_dotenv, load_dotenv

# Variables already set in the process before `.env` is read: those win over `.env`.
_PROCESS_ENV = set(os.environ)
load_dotenv()
_DOTENV_PATH = find_dotenv()
_LOADED_FROM_DOTENV = {key for key in (dotenv_values(_DOTENV_PATH) if _DOTENV_PATH else {}) if key not in _PROCESS_ENV}

# Models and prices come from `.env` (format in `.env.example`), never from code.
_ANTHROPIC_MODEL = re.compile(r"LEVY_ANTHROPIC_MODEL_(\d+)")
_ANTHROPIC_PRICE = re.compile(r"LEVY_ANTHROPIC_MODEL_(\d+)_(INPUT|OUTPUT)_USD_PER_MTOK")
_OLLAMA_MODEL = re.compile(r"LEVY_OLLAMA_MODEL_(\d+)")
_OLLAMA_EMBEDDING_MODEL = re.compile(r"LEVY_OLLAMA_EMBEDDING_MODEL_(\d+)")
_USD_AMOUNT = re.compile(r"\d+(\.\d+)?")


@dataclass(frozen=True)
class ModelCatalog:
    """The models `.env` names. Anthropic entries are (model, input USD/MTok, output USD/MTok)."""

    anthropic: Tuple[Tuple[str, float, float], ...] = ()
    ollama: Tuple[str, ...] = ()
    ollama_embedding: Tuple[str, ...] = ()


def read_model_catalog(environ: Mapping[str, Optional[str]]) -> ModelCatalog:
    """Read the numbered model variables from `environ`; a wrong value fails naming its variable."""
    anthropic: Dict[int, str] = {}
    prices: Dict[Tuple[int, str], float] = {}
    ollama: Dict[int, str] = {}
    embedding: Dict[int, str] = {}
    for key, raw in environ.items():
        value = (raw or "").strip()
        price = _ANTHROPIC_PRICE.fullmatch(key)
        if price:
            if not value:  # a template line left blank is not configured
                continue
            if not _USD_AMOUNT.fullmatch(value):
                raise ValueError(
                    f"{key}={raw!r}: expected US dollars per million tokens as a plain number "
                    f"with a dot for decimals, e.g. 1.25 (no $, no currency, no comma)"
                )
            prices[(int(price.group(1)), price.group(2))] = float(value)
            continue
        for pattern, target in ((_ANTHROPIC_MODEL, anthropic), (_OLLAMA_MODEL, ollama), (_OLLAMA_EMBEDDING_MODEL, embedding)):
            match = pattern.fullmatch(key)
            if match and value:
                target[int(match.group(1))] = value

    for number, side in sorted(prices):
        if number not in anthropic:
            raise ValueError(
                f"LEVY_ANTHROPIC_MODEL_{number}_{side}_USD_PER_MTOK is set but LEVY_ANTHROPIC_MODEL_{number} is not"
            )
    entries = []
    for number in sorted(anthropic):
        missing = [
            f"LEVY_ANTHROPIC_MODEL_{number}_{side}_USD_PER_MTOK"
            for side in ("INPUT", "OUTPUT")
            if (number, side) not in prices
        ]
        if missing:
            raise ValueError(f"LEVY_ANTHROPIC_MODEL_{number}={anthropic[number]!r} needs {' and '.join(missing)}")
        entries.append((anthropic[number], prices[(number, "INPUT")], prices[(number, "OUTPUT")]))
    return ModelCatalog(
        anthropic=tuple(entries),
        ollama=tuple(ollama[n] for n in sorted(ollama)),
        ollama_embedding=tuple(embedding[n] for n in sorted(embedding)),
    )


def model_environment() -> Dict[str, Optional[str]]:
    """
    `.env` read again now, plus the process environment, so an edited `.env` is
    seen without restarting Levy. A variable set in the process (not loaded from
    `.env` at start-up) wins over `.env`, as usual.
    """
    environ: Dict[str, Optional[str]] = dict(dotenv_values(_DOTENV_PATH)) if _DOTENV_PATH else {}
    environ.update({key: value for key, value in os.environ.items() if key not in _LOADED_FROM_DOTENV})
    return environ


def current_model_catalog() -> ModelCatalog:
    return read_model_catalog(model_environment())


def _default_ollama() -> Optional[str]:
    models = current_model_catalog().ollama
    return models[0] if models else None


def _default_anthropic(index: int):
    """Field `index` (0 model, 1 input price, 2 output price) of the default Anthropic entry."""
    models = current_model_catalog().anthropic
    return models[0][index] if models else None


@dataclass
class LevyConfig:
    """
    Configuration for the Levy Engine.
    """
    # LLM settings
    # "anthropic", "ollama"; "mock" is a test double for the offline suite only
    llm_provider: str = "anthropic"
    mock_llm_latency_seconds: float = 0.5  # simulated delay for MockLLMClient; tests inject 0
    ollama_base_url: str = "http://localhost:11434"
    # Ollama models from LEVY_OLLAMA_MODEL_<n>; the lowest number is the default (`model_name`).
    ollama_models: Tuple[str, ...] = field(default_factory=lambda: current_model_catalog().ollama)
    model_name: Optional[str] = field(default_factory=_default_ollama)
    # Ollama embedding models from LEVY_OLLAMA_EMBEDDING_MODEL_<n>.
    ollama_embedding_models: Tuple[str, ...] = field(default_factory=lambda: current_model_catalog().ollama_embedding)

    # Anthropic settings (LEV-6)
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    # A model and its two per-MTok prices are one triple describing one model (LEV-14):
    # all three come from the lowest-numbered LEVY_ANTHROPIC_MODEL_<n>, together.
    anthropic_models: Tuple[Tuple[str, float, float], ...] = field(default_factory=lambda: current_model_catalog().anthropic)
    anthropic_model: Optional[str] = field(default_factory=lambda: _default_anthropic(0))
    anthropic_max_retries: int = 2
    anthropic_budget_cap_usd: float = 200.0  # budget hard cap (~$50 expected spend)
    anthropic_input_price_per_mtok: Optional[float] = field(default_factory=lambda: _default_anthropic(1))
    anthropic_output_price_per_mtok: Optional[float] = field(default_factory=lambda: _default_anthropic(2))
    
    # Embedding settings
    embedding_provider: str = "sentence-transformers"  # "sentence-transformers", "ollama"; "mock" is a test double
    embedding_model: str = "all-MiniLM-L6-v2"  # study baseline; use "modernbert" for the other study model
    
    # Storage settings
    cache_store_type: str = "memory" # "memory", "redis"
    redis_url: str = "redis://localhost:6379/0"
    
    # Cache settings
    enable_exact_cache: bool = True
    enable_semantic_cache: bool = True
    # Threshold in 1/(1+L2) similarity space (NOT cosine). Study sweep: 0.70–0.90 step 0.05.
    # For unit-norm vectors: similarity = 1/(1+sqrt(2-2*cosine)).
    # cosine≈0.95 → sim≈0.76; cosine≈0.99 → sim≈0.88; cosine≈0.90 → sim≈0.69.
    similarity_threshold: float = 0.85
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size: int = 1000  # Max number of entries in memory

    # Vector index settings (LEV-2)
    vector_index_backend: str = "auto"  # "auto" | "faiss" | "brute_force"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    def anthropic_prices(self, model: Optional[str]) -> Optional[Tuple[float, float]]:
        """(input, output) USD per MTok configured for `model`, or None when it is not configured."""
        for name, input_price, output_price in self.anthropic_models:
            if name == model:
                return input_price, output_price
        return None

    def with_models(self, catalog: ModelCatalog) -> "LevyConfig":
        """This configuration with its models and prices replaced by `catalog` (lowest number = default)."""
        anthropic = catalog.anthropic[0] if catalog.anthropic else (None, None, None)
        return replace(
            self,
            anthropic_models=catalog.anthropic,
            anthropic_model=anthropic[0],
            anthropic_input_price_per_mtok=anthropic[1],
            anthropic_output_price_per_mtok=anthropic[2],
            ollama_models=catalog.ollama,
            model_name=catalog.ollama[0] if catalog.ollama else None,
            ollama_embedding_models=catalog.ollama_embedding,
        )
