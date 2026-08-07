"""
Offline benchmark of the cache lookup path (LEV-14 / 3.1, 3.2).

Measures what a lookup costs, decomposed by segment, against a cache populated
by a full replay of the workload — the steady state a lookup actually meets,
not an empty index.

**Measurement protocol, per configuration:**

1. *Populate.* Replay every pair of the workload through `LevyEngine.generate()`
   untimed, so the index holds the same entries the D3 run left it holding.
2. *Warm up.* `warmup` further lookups, untimed and discarded. The first calls
   into an interpreter pay for imports, first-touch allocation and (for a real
   encoder) lazily built kernels; including them would report start-up cost as
   lookup cost.
3. *Warm phase.* `repetitions` timed lookups whose embedding is memoized before
   the measured call. Yields `embed_warm`.
4. *Cold phase.* `repetitions` timed lookups whose memo entry is evicted
   immediately before the measured call, so the encoder actually runs. Yields
   `embed_cold` and `total_lookup`.

**Probe texts** are workload queries carrying a unique numeric suffix. The
suffix is what makes each measured lookup traverse the whole path: a repeated
text would be found by the exact cache and return before reaching the embedder
or the index, measuring a dictionary lookup instead of a cache lookup. It costs
a token or two of encoder input and changes no branch that is taken.

A probe is by construction a miss, so each measured lookup is stored like any
other miss and the index grows by one per measured lookup — `2 x repetitions`
entries on top of a fully populated workload. That is small, monotone, and what
a live cache does anyway; it is not suppressed, because suppressing it would
mean measuring a lookup path that differs from the one in production.

**`total_lookup` is the cold figure.** The two regimes are not averaged
anywhere else, and they must not be silently averaged here. The cold total is
the cost a lookup pays for a query this process has not embedded before, which
is the realistic case; the warm total is recoverable from the same row as
`total_lookup_p50 - (embed_cold_p50 - embed_warm_p50)`.

`index_search` and `exact_lookup` are pooled across both phases: neither
depends on the memo, so splitting them would report the same quantity twice.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from levy.config import LevyConfig
from levy.dataset.schema import QueryPair
from levy.embedding_manager import EmbeddingManager
from levy.engine import LevyEngine
from levy.experiment.config import ExperimentConfig
from levy.latency.timing import (
    SEGMENT_EMBED,
    SEGMENT_EXACT_LOOKUP,
    SEGMENT_INDEX_SEARCH,
    SEGMENT_TOTAL_LOOKUP,
    TimingCollector,
)
from levy.llm_client import LLMClient

DEFAULT_WARMUP = 5
DEFAULT_REPETITIONS = 30
DEFAULT_PROBE_SEED = 42

# Reported segment names. `embed` is split by regime; the rest keep the engine's
# segment names so a row can be traced back to the call site that produced it.
SEGMENT_EMBED_COLD = "embed_cold"
SEGMENT_EMBED_WARM = "embed_warm"

REPORTED_SEGMENTS = (
    SEGMENT_EMBED_COLD,
    SEGMENT_EMBED_WARM,
    SEGMENT_INDEX_SEARCH,
    SEGMENT_EXACT_LOOKUP,
    SEGMENT_TOTAL_LOOKUP,
)


class BenchmarkError(Exception):
    """Raised when a configuration cannot be benchmarked as specified."""


@dataclass(frozen=True)
class SegmentStats:
    """Percentiles of one segment, over exactly the measured repetitions."""

    n: int
    p50_ms: float
    p95_ms: float


@dataclass
class ConfigurationLatency:
    """One benchmarked configuration: its identity and its segment percentiles."""

    config: ExperimentConfig
    n_lookups: int
    segments: Dict[str, SegmentStats] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Percentiles
# ----------------------------------------------------------------------


def percentile(values: List[float], q: float) -> float:
    """
    Nearest-rank percentile: the smallest observed value at or above rank q.

    Deliberately not interpolated. An interpolated p95 of 30 samples reports a
    number that was never measured, which is the wrong thing to publish as a
    latency figure; nearest-rank always names a real observation.
    """
    if not values:
        raise BenchmarkError(f"cannot take the p{q:g} of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def require_samples(config_id: str, samples_by_segment: Dict[str, List[float]]) -> None:
    """
    Refuse to report a segment the lookup never reached.

    An empty segment means the measured path skipped it — an exact hit that
    never embedded, say. Reporting it as a zero would publish "the index search
    took no time" for a search that did not happen.
    """
    for name, samples in samples_by_segment.items():
        if not samples:
            raise BenchmarkError(
                f"configuration {config_id}: segment {name!r} recorded no sample; "
                "the lookup path did not reach it"
            )


def _stats(values: List[float]) -> SegmentStats:
    return SegmentStats(n=len(values), p50_ms=percentile(values, 50), p95_ms=percentile(values, 95))


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------


def probe_texts(pairs: List[QueryPair], count: int, seed: int = DEFAULT_PROBE_SEED) -> List[str]:
    """
    `count` unique probe texts drawn deterministically from the workload.

    Uniqueness is required (see the module docstring); determinism under `seed`
    is what makes a re-run of the benchmark measure the same lookups.
    """
    if not pairs:
        raise BenchmarkError("no pairs for this workload; nothing to probe")
    rng = random.Random(seed)
    bases = [pair.query_2 for pair in pairs]
    return [f"{bases[rng.randrange(len(bases))]} #probe{index}" for index in range(count)]


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------


def _build_engine(
    config: ExperimentConfig,
    embedding_manager: EmbeddingManager,
    llm_client: Optional[LLMClient],
) -> LevyEngine:
    """A fresh engine for one configuration, matching the harness's own setup."""
    engine_config = LevyConfig(
        llm_provider="mock",
        mock_llm_latency_seconds=0,
        embedding_provider="mock",  # overridden by the injected manager
        embedding_model=config.model,
        enable_exact_cache=True,
        enable_semantic_cache=True,
        similarity_threshold=config.threshold,
        cache_store_type="memory",
        # The populated FAQ cache holds 2 entries per pair; the default 1000-entry
        # cap would evict during population and measure a smaller index than the
        # D3 run used.
        cache_max_size=1_000_000,
    )
    return LevyEngine(engine_config, embedding_manager=embedding_manager, llm_client=llm_client)


def benchmark_configuration(
    config: ExperimentConfig,
    pairs: List[QueryPair],
    embedding_manager: EmbeddingManager,
    llm_client: Optional[LLMClient] = None,
    warmup: int = DEFAULT_WARMUP,
    repetitions: int = DEFAULT_REPETITIONS,
    probe_seed: int = DEFAULT_PROBE_SEED,
) -> ConfigurationLatency:
    """Run the four-phase protocol above for one configuration."""
    if repetitions < 1:
        raise BenchmarkError(f"repetitions must be >= 1, got {repetitions}")
    if warmup < 0:
        raise BenchmarkError(f"warmup must be >= 0, got {warmup}")

    workload_pairs = [pair for pair in pairs if pair.workload == config.workload]
    engine = _build_engine(config, embedding_manager, llm_client)

    # 1. Populate.
    for pair in workload_pairs:
        engine.generate(pair.query_1)
        engine.generate(pair.query_2)

    probes = probe_texts(workload_pairs, warmup + 2 * repetitions, seed=probe_seed)
    warmup_probes = probes[:warmup]
    warm_probes = probes[warmup : warmup + repetitions]
    cold_probes = probes[warmup + repetitions :]

    # 2. Warm up — timed by nothing, discarded by construction.
    for text in warmup_probes:
        engine.generate(text)

    embed_warm: List[float] = []
    embed_cold: List[float] = []
    index_search: List[float] = []
    exact_lookup: List[float] = []
    total_cold: List[float] = []

    # 3. Warm phase: memo seeded immediately before the measured call.
    for text in warm_probes:
        embedding_manager.embed(text)
        collector = TimingCollector()
        engine.generate(text, timing=collector)
        embed_warm.extend(collector.samples(SEGMENT_EMBED))
        index_search.extend(collector.samples(SEGMENT_INDEX_SEARCH))
        exact_lookup.extend(collector.samples(SEGMENT_EXACT_LOOKUP))

    # 4. Cold phase: memo seeded, then evicted immediately before the measured
    #    call — so the entry demonstrably existed and was demonstrably cleared.
    for text in cold_probes:
        embedding_manager.embed(text)
        embedding_manager.forget(text)
        collector = TimingCollector()
        engine.generate(text, timing=collector)
        embed_cold.extend(collector.samples(SEGMENT_EMBED))
        index_search.extend(collector.samples(SEGMENT_INDEX_SEARCH))
        exact_lookup.extend(collector.samples(SEGMENT_EXACT_LOOKUP))
        total_cold.extend(collector.samples(SEGMENT_TOTAL_LOOKUP))

    require_samples(
        config.config_id,
        {
            SEGMENT_EMBED_WARM: embed_warm,
            SEGMENT_EMBED_COLD: embed_cold,
            SEGMENT_INDEX_SEARCH: index_search,
            SEGMENT_EXACT_LOOKUP: exact_lookup,
            SEGMENT_TOTAL_LOOKUP: total_cold,
        },
    )

    return ConfigurationLatency(
        config=config,
        n_lookups=2 * repetitions,
        segments={
            SEGMENT_EMBED_COLD: _stats(embed_cold),
            SEGMENT_EMBED_WARM: _stats(embed_warm),
            SEGMENT_INDEX_SEARCH: _stats(index_search),
            SEGMENT_EXACT_LOOKUP: _stats(exact_lookup),
            SEGMENT_TOTAL_LOOKUP: _stats(total_cold),
        },
    )


def benchmark_configurations(
    configs: List[ExperimentConfig],
    pairs: List[QueryPair],
    embedding_provider: str = "mock",
    llm_client: Optional[LLMClient] = None,
    warmup: int = DEFAULT_WARMUP,
    repetitions: int = DEFAULT_REPETITIONS,
    probe_seed: int = DEFAULT_PROBE_SEED,
) -> tuple:
    """
    Benchmark every configuration, sharing one `EmbeddingManager` per model.

    Sharing matters twice over: it is what the sweep runner does (so the
    measured setup matches the measured system), and a per-configuration
    manager would reload the encoder five times per model.

    Returns `(results, model_identities)`, the identities keyed by model alias
    for the metadata sidecar.
    """
    managers: Dict[str, EmbeddingManager] = {}
    results: List[ConfigurationLatency] = []

    for config in configs:
        manager = managers.get(config.model)
        if manager is None:
            manager = EmbeddingManager(model_name=config.model, provider=embedding_provider)
            managers[config.model] = manager
        results.append(
            benchmark_configuration(
                config,
                pairs,
                embedding_manager=manager,
                llm_client=llm_client,
                warmup=warmup,
                repetitions=repetitions,
                probe_seed=probe_seed,
            )
        )

    identities = {model: manager.get_model_identity().as_dict() for model, manager in managers.items()}
    return results, identities
