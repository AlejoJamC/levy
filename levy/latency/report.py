"""
Latency artefact writers (LEV-14 / 4.1-4.3).

Three files, written to a dedicated directory outside the deterministic D3
result set:

- `latency.csv`       — one row per configuration, p50/p95 per segment.
- `latency_meta.json` — host, versions, protocol parameters, the reproducibility
                        boundary, and the headline savings figure.
- `llm_calls.json`    — the real-provider call summary and its observed cost.

Nothing here writes to `results.csv` or `decisions.csv`, and no latency value
takes part in the +/-5 % replication criterion.
"""

import csv
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from levy.latency.benchmark import (
    ConfigurationLatency,
    SEGMENT_EMBED_COLD,
    SEGMENT_EMBED_WARM,
    percentile,
)
from levy.latency.corpus import ResponseRecord
from levy.latency.timing import SEGMENT_EXACT_LOOKUP, SEGMENT_INDEX_SEARCH, SEGMENT_TOTAL_LOOKUP

PathLike = Union[str, Path]

# Column order of latency.csv. The five segments each contribute a p50 and a p95;
# a single-sample figure is never reported (see benchmark.py).
SEGMENT_COLUMN_ORDER = (
    SEGMENT_EMBED_COLD,
    SEGMENT_EMBED_WARM,
    SEGMENT_INDEX_SEARCH,
    SEGMENT_EXACT_LOOKUP,
    SEGMENT_TOTAL_LOOKUP,
)

LATENCY_FIELDNAMES = ["config_id", "model", "workload", "threshold", "n_lookups"] + [
    f"{segment}_{stat}_ms" for segment in SEGMENT_COLUMN_ORDER for stat in ("p50", "p95")
]

# Stated verbatim in latency_meta.json. The one thing a reader of these numbers
# must not get wrong is which of the two measurements they can reproduce.
REPRODUCIBILITY_BOUNDARY = (
    "Lookup-overhead measurement (latency.csv) is reproducible: it runs entirely offline "
    "and a third party re-running it on their own hardware should obtain comparable "
    "figures, subject to the host specification recorded here. "
    "Provider-latency measurement (llm_calls.json) is NOT reproducible: it is specific to "
    "the model, provider, region, network path and moment recorded with it, and re-running "
    "it will not reproduce these values. Every provider-derived figure carries the resolved "
    "model identifier that produced it and must not be read as model-independent."
)


def _library_versions() -> Dict[str, str]:
    """Versions of the libraries whose speed the measurement actually depends on."""
    versions: Dict[str, str] = {}
    for name in ("numpy", "faiss", "torch", "sentence_transformers", "anthropic"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover -- optional at measurement time
            versions[name] = "not installed"
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def host_specification() -> Dict[str, object]:
    """Everything about this machine a reader needs to judge comparability."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "library_versions": _library_versions(),
    }


# ----------------------------------------------------------------------
# latency.csv
# ----------------------------------------------------------------------


def write_latency_csv(results: List[ConfigurationLatency], path: PathLike) -> None:
    """One row per configuration. Written with the csv module's own quoting rules."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LATENCY_FIELDNAMES)
        writer.writeheader()
        for result in results:
            row = {
                "config_id": result.config.config_id,
                "model": result.config.model,
                "workload": result.config.workload,
                "threshold": f"{result.config.threshold:.2f}",
                "n_lookups": result.n_lookups,
            }
            for segment in SEGMENT_COLUMN_ORDER:
                stats = result.segments[segment]
                row[f"{segment}_p50_ms"] = f"{stats.p50_ms:.4f}"
                row[f"{segment}_p95_ms"] = f"{stats.p95_ms:.4f}"
            writer.writerow(row)


# ----------------------------------------------------------------------
# llm_calls.json
# ----------------------------------------------------------------------


def summarise_calls(
    records: Iterable[ResponseRecord],
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> dict:
    """
    Summarise a real-provider run: latency distribution, tokens, observed cost.

    Cost is computed from the token counts the provider actually reported, at
    the prices configured for the model actually used — an observation, not the
    pre-run estimate.
    """
    records = list(records)
    if not records:
        return {
            "n_calls": 0,
            "note": "no provider calls recorded",
        }

    latencies = [record.latency_ms for record in records]
    input_tokens = sum(record.input_tokens for record in records)
    output_tokens = sum(record.output_tokens for record in records)
    models = sorted({record.model for record in records})
    timestamps = sorted(record.timestamp_utc for record in records)

    total_cost = (
        input_tokens / 1_000_000 * input_price_per_mtok
        + output_tokens / 1_000_000 * output_price_per_mtok
    )

    return {
        "n_calls": len(records),
        "model": models[0] if len(models) == 1 else models,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
        },
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "input_mean_per_call": input_tokens / len(records),
            "output_mean_per_call": output_tokens / len(records),
        },
        "prices_per_mtok_usd": {
            "input": input_price_per_mtok,
            "output": output_price_per_mtok,
        },
        "observed_cost_usd": {
            "total": total_cost,
            "per_call": total_cost / len(records),
        },
        "window_utc": {"first_call": timestamps[0], "last_call": timestamps[-1]},
    }


def write_llm_calls(summary: dict, path: PathLike, extra: Optional[dict] = None) -> None:
    """Write the provider-call summary, merging any `extra` sections (e.g. a repriced estimate)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def price_at(summary: dict, input_price_per_mtok: float, output_price_per_mtok: float, model_label: str) -> dict:
    """
    Reprice a recorded run at another model's prices (LEV-14 / 7.4).

    Token totals come from the pilot's own calls, so this is a costed
    projection of the same work, not a fresh estimate — and it is labelled with
    the model whose prices were applied, never with the model that ran.
    """
    tokens = summary["tokens"]
    total = (
        tokens["input_total"] / 1_000_000 * input_price_per_mtok
        + tokens["output_total"] / 1_000_000 * output_price_per_mtok
    )
    return {
        "model_priced": model_label,
        "basis": "token totals observed in this run, priced at the named model's rates",
        "prices_per_mtok_usd": {"input": input_price_per_mtok, "output": output_price_per_mtok},
        "projected_cost_usd": {"total": total, "per_call": total / summary["n_calls"]},
    }


# ----------------------------------------------------------------------
# latency_meta.json
# ----------------------------------------------------------------------


def savings_figure(
    results: List[ConfigurationLatency],
    call_summary: Optional[dict],
) -> dict:
    """
    Median lookup overhead added vs median provider latency avoided (LEV-14 / 8.1).

    The overhead is the median across configurations of each configuration's
    p50 total lookup — one number for "what the cache costs to consult". The
    avoided figure is the provider run's p50 end-to-end latency. Both are
    reported alongside the model identifier that produced the second, because
    a savings figure without its model is a claim about caching in general,
    which this measurement does not support.
    """
    overhead_p50 = statistics.median(
        result.segments[SEGMENT_TOTAL_LOOKUP].p50_ms for result in results
    ) if results else None

    if not call_summary or not call_summary.get("n_calls"):
        return {
            "lookup_overhead_p50_ms": overhead_p50,
            "provider_latency_p50_ms": None,
            "model": None,
            "note": (
                "No provider calls recorded, so no savings figure is derived. "
                "Lookup overhead alone is not a savings claim."
            ),
        }

    avoided = call_summary["latency_ms"]["p50"]
    model = call_summary["model"]
    return {
        "lookup_overhead_p50_ms": overhead_p50,
        "provider_latency_p50_ms": avoided,
        "net_saved_per_hit_ms": avoided - overhead_p50,
        "overhead_as_fraction_of_provider_latency": overhead_p50 / avoided,
        "model": model,
        "note": (
            f"Latency avoided per cache hit, measured against {model}. Provider latency is "
            "model-, provider-, region- and time-dependent; this figure belongs to that model "
            "and does not generalise. A larger model would avoid more, not less."
        ),
    }


def write_latency_meta(
    path: PathLike,
    results: List[ConfigurationLatency],
    dataset_path: PathLike,
    embedding_provider: str,
    model_identities: Dict[str, dict],
    warmup: int,
    repetitions: int,
    probe_seed: int,
    reference_results: Optional[PathLike] = None,
    call_summary: Optional[dict] = None,
    response_corpus: Optional[dict] = None,
) -> None:
    """Write the metadata sidecar, including the reproducibility boundary and the savings figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "reference_results": None if reference_results is None else str(reference_results),
        "embedding_provider": embedding_provider,
        "model_identities": model_identities,
        "n_configurations": len(results),
        "protocol": {
            "warmup_iterations": warmup,
            "measured_repetitions": repetitions,
            "probe_seed": probe_seed,
            "percentile_method": "nearest-rank (no interpolation)",
            "clock": "time.perf_counter (monotonic)",
            "note": (
                "Warm-up iterations are discarded and take no part in any reported "
                "percentile; each reported percentile is computed from exactly the "
                "measured repetitions of its segment."
            ),
        },
        "host": host_specification(),
        "reproducibility_boundary": REPRODUCIBILITY_BOUNDARY,
        "savings": savings_figure(results, call_summary),
        "response_corpus": response_corpus,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
        fh.write("\n")
