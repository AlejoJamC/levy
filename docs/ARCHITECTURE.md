# Levy — Architecture

This document describes the architecture of the released artefact for someone
reading or extending the public repository. It is traceable to the frozen
specification: [`docs/Specification_and_Design_Report.md`](Specification_and_Design_Report.md)
names five component responsibilities and three headline components, and every
one of them maps to a module below.

For running the system, see [REPRODUCTION.md](REPRODUCTION.md). For usage and
configuration, see the [README](../README.md).

---

## 1. What the system is

Levy is a semantic caching engine that sits between an application and an LLM
provider. For each prompt it tries, in order:

1. **exact cache** — SHA-256 of the prompt as key;
2. **semantic cache** — nearest neighbour in a vector index, accepted when
   similarity clears a threshold;
3. **the LLM provider** — on a miss; the response (and its embedding, when the
   semantic cache is enabled) is then stored.

Around that engine sits the research apparatus: a ground-truth dataset platform,
an offline replay harness over the frozen experimental grid, and a statistical
analysis pipeline that turns harness output into the evidence bundle.

---

## 2. Component map (spec → code)

### 2.1 The three headline components

The frozen specification's deliverable D1 names three components. All three
ship:

| Frozen spec component | Module | Notes |
|---|---|---|
| **FastAPI router** | [`levy/api/`](../levy/api/) | `app.py` (endpoints + exception handlers), `schemas.py` (Pydantic v2 request/response models), `pool.py` (`EnginePool`). Exposes `POST /v1/chat/completions`, `GET /admin/cache/stats`, `POST /admin/cache/clear`. |
| **Embedding manager + Faiss HNSW index** | [`levy/embedding_manager.py`](../levy/embedding_manager.py), [`levy/cache/vector_index.py`](../levy/cache/vector_index.py) | Study-model registry and memoization; `IndexHNSWFlat` wrapped in `IndexIDMap`, with a brute-force numpy index as correctness oracle and offline fallback. |
| **Anthropic connector** | [`levy/llm_client.py`](../levy/llm_client.py) (`AnthropicLLMClient`) | Wraps the official `anthropic` SDK: SDK-native retry, real token accounting, per-instance budget guard, refusal handling. |

### 2.2 The five component responsibilities

| Frozen spec responsibility | Module | Implementation |
|---|---|---|
| FastAPI Router | [`levy/api/app.py`](../levy/api/app.py) | The three documented endpoints; errors map to structured JSON, never stack traces. |
| Cache Orchestrator | [`levy/engine.py`](../levy/engine.py) (`LevyEngine`) | Owns the lookup flow, the hit/miss decision, and metric recording. The router adds one structured JSON log record per request, sufficient to replay a request sequence. |
| Embedding Manager | [`levy/embedding_manager.py`](../levy/embedding_manager.py) | Resolves study-model aliases through a registry, lazily loads one client per checkpoint, memoizes by `(model_key, sha256(text))`, applies each model's symmetric task prefix. |
| Vector Store (Faiss) | [`levy/cache/vector_index.py`](../levy/cache/vector_index.py), [`levy/cache/semantic_cache.py`](../levy/cache/semantic_cache.py) | HNSW index over L2 distance, plus a monotonic id → `CacheEntry` map — the spec's "separate metadata dictionary". |
| LLM Connector | [`levy/llm_client.py`](../levy/llm_client.py) | `LLMClient` ABC with mock, OpenAI-compatible, Ollama, and Anthropic implementations. |

**Recorded deviation — synchronous connector.** The frozen spec describes the
LLM connector as an "asynchronous wrapper". The engine, caches, and harness are
synchronous throughout, so the connector implements the synchronous `LLMClient`
ABC and concurrency is handled at the HTTP boundary instead: the router's
endpoints are declared `def`, so FastAPI runs them in its threadpool and
requests are served concurrently without an `AsyncAnthropic` migration. This is
a documented resolution, not silent drift — the rationale is in
`openspec/changes/add-anthropic-connector/design.md` and
`openspec/changes/add-fastapi-router/design.md`.

### 2.3 Research layers

These support the empirical work rather than the serving path:

| Layer | Module | Role |
|---|---|---|
| Dataset platform | [`levy/dataset/`](../levy/dataset/) | `schema.py` (`QueryPair`, workload constants, `ground_truth_label()`), `io.py` (CSV/JSON, round-trip identical), `sampling.py` (corpus adapters + seeded stratified sampling), `annotation.py` (blind, resumable re-annotation), `kappa.py` (Cohen's kappa). |
| Replay harness | [`levy/experiment/`](../levy/experiment/) | `config.py` (the frozen grid), `replay.py` (replay through the *production* lookup path), `metrics.py` (precision, recall, F₀.₅, FPR, hit rate + sanity checks), `runner.py` (sweep + deterministic output files). |
| Analysis pipeline | [`levy/analysis/`](../levy/analysis/) | `io.py` (harness-contract reader), `hypothesis.py` (two-way ANOVA + conditional Tukey HSD), `curves.py` (threshold-selection tables and figures), `replication.py` (±5% criterion), `report.py` (bundle assembly). |
| Results dashboard (D6, desirable) | [`levy/dashboard/`](../levy/dashboard/), [`scripts/dashboard.py`](../scripts/dashboard.py) | `bundle.py` (loads/validates an analysis bundle, columns sourced from `levy.analysis` itself), `curves.py` (selection helpers), `query.py` (live query decision via a real `SemanticCache`, same `1/(1+L2)` formula). The Streamlit shell in `scripts/` is a thin viewer with no logic of its own; it never recomputes a statistic. Lowest-priority deliverable — safe to drop, not on the `reproduce.sh` path. |
| CLIs | [`scripts/`](../scripts/) | Thin argparse wrappers: dataset sampling/annotation/kappa/export, experiment sweep, analysis bundle, replication check, plus `reproduce.sh` (whole pipeline), `audit_release.sh` (release audit), and `dashboard.py` (Streamlit UI shell, D6). |

---

## 3. Request flow

A single `POST /v1/chat/completions`:

```
[Client]
   │  prompt (+ optional per-request cache_config)
   ▼
[FastAPI router — levy/api/app.py]
   │  resolves (embedding_model, threshold) → EnginePool
   ▼
[Cache orchestrator — LevyEngine.generate()]
   │
   ├─▶ [Exact cache]  SHA-256(prompt) hit? ──▶ return  (X-Cache-Status: HIT, similarity 1.0)
   │
   ├─▶ [Embedding manager] embed(prompt), memoized, L2-normalised
   │        │
   │        ▼
   │   [Vector index] k-NN search → L2 distance
   │        │
   │        ▼
   │   similarity = 1 / (1 + distance)
   │   similarity ≥ threshold? ──▶ return cached response  (X-Cache-Status: HIT)
   │
   └─▶ [LLM connector] provider call (Anthropic / OpenAI-compatible / Ollama / mock)
            │
            ▼
       store response in the exact cache, and its embedding in the vector index
            │
            ▼
        return  (X-Cache-Status: MISS)
```

The response body is Anthropic Messages-shaped for hits and misses alike; cache
identity lives in the `X-Cache-Status` / `X-Cache-Similarity` headers, so a
client cannot accidentally depend on hit-vs-miss body differences.

### 3.1 The similarity scale

Per Algorithm 1 of the frozen specification, similarity is derived from L2
distance as `similarity = 1 / (1 + distance)`. All embeddings are L2-normalised
before indexing and querying, so the distance scale is identical across
embedding models and results are comparable between them.

For unit vectors, `distance = √(2 − 2·cosine)`. The frozen threshold sweep
0.70–0.90 therefore corresponds to a high-cosine band of roughly 0.91–0.998.
This is intentional and spec-mandated; the thresholds are carried verbatim and
never rescaled.

### 3.2 The engine pool

`LevyEngine` binds its embedding model and threshold at construction, while the
API contract puts both in the *request*. `levy/api/pool.py` reconciles the two
with a bounded pool keyed by `(embedding_model, threshold)`: the first request
for a pair builds an engine with those two fields overridden; later requests
with that pair reuse it, so their caches accumulate. One `EmbeddingManager` is
shared per embedding model across thresholds, so changing a threshold never
reloads a model. Requesting a pair beyond the cap (default 8) returns a
structured `400 pool_cap_exceeded`.

---

## 4. Experiment flow

The harness deliberately replays through the same `LevyEngine.generate()` path
that serves HTTP traffic — the measured cache behaviour is the shipped cache
behaviour, not a reimplementation.

```
[Ground-truth dataset]  900 QueryPairs (300 per workload)
   │  levy.dataset.io
   ▼
[Grid]  2 embedding models × 3 workloads × 5 thresholds = 30 configurations
   │  levy.experiment.config.full_grid()
   ▼
[For each configuration]  fresh LevyEngine (mock LLM, memory store)
   │    query_1 → miss and store
   │    query_2 → hit/miss decision via LevyResult.source
   │    compare against QueryPair.ground_truth_label() → TP / FP / TN / FN
   ▼
[results.csv, decisions.csv, run_meta.json]
   │  levy.analysis.io
   ▼
[Analysis bundle]  ANOVA + Tukey, threshold curves + figures, kappa, metadata
   │
   ▼
[Replication check]  re-run the recorded grid, compare precision/recall at ±5%
```

Determinism is a design property: `results.csv` and `decisions.csv` contain no
timestamps and no latency, so re-running on identical inputs reproduces them
byte-identically. Everything time-varying (versions, generation timestamp,
latency statistics) lives in the `run_meta.json` / `analysis_meta.json`
sidecars.

---

## 5. Cross-cutting patterns

### 5.1 ABC + mock provider for every external dependency

Every external dependency is reached through an abstract base class that has a
mock implementation:

| Dependency | ABC | Implementations |
|---|---|---|
| LLM | `LLMClient` | `MockLLMClient`, `OpenAILLMClient`, `OllamaLLMClient`, `AnthropicLLMClient` |
| Embeddings | `EmbeddingClient` | mock (text-seeded, normalised), `SentenceTransformerClient`, `OllamaEmbeddingClient` |
| Vector index | `VectorIndex` | `BruteForceVectorIndex` (exact k-NN oracle), `FaissHNSWVectorIndex` |
| Cache store | `CacheInterface` | `InMemoryStore` (FIFO eviction), `RedisStore` |
| Corpus | `CorpusSource` | `QuoraQQPSource`, `StackOverflowDuplicatesSource`, `ConvAI2Source`, `MockCorpusSource` |

This is not incidental — it is what makes the experiments reproducible offline.
The whole test suite and the default pipeline run with **zero external
services**: no API key, no model download, no network. Keep this pattern when
adding a provider.

### 5.2 Configuration in one place

[`levy/config.py`](../levy/config.py) holds a single `LevyConfig` dataclass:
provider selection (`llm_provider`, `embedding_provider`, `cache_store_type`),
the embedding model and similarity threshold, the vector index backend, and the
Anthropic settings (model, retries, budget cap, per-MTok prices). It loads
`.env` via python-dotenv. Secrets never appear in code or in tracked files —
only `.env.example` is committed, and `scripts/audit_release.sh` enforces that.

### 5.3 Dataclasses in the core, Pydantic at the edge

The core package uses plain dataclasses. Pydantic appears only in
`levy/api/schemas.py`, where request validation is the actual requirement. This
keeps the engine importable and testable without a validation framework in the
hot path.

### 5.4 Degenerate results are reported, not laundered

Zero-division in a metric is reported as `0.0` plus an explicit flag, never as
NaN and never silently dropped. Where a statistical test is undefined — for
instance zero variance in false positive rate, which is exactly what the
committed fixture yields under mock embeddings — the hypothesis decision is
reported as `undefined` rather than as a retained null. Residual diagnostics
(design balance, Shapiro-Wilk, Levene) are reported and never acted on
automatically.

---

## 6. Where to start reading

| If you want to… | Start at |
|---|---|
| Understand the cache decision | [`levy/engine.py`](../levy/engine.py) |
| Understand the similarity maths | [`levy/cache/semantic_cache.py`](../levy/cache/semantic_cache.py) |
| Add an LLM or embedding provider | [`levy/llm_client.py`](../levy/llm_client.py), [`levy/embeddings.py`](../levy/embeddings.py) |
| Serve the cache over HTTP | [`levy/api/app.py`](../levy/api/app.py) |
| Reproduce the experiments | [REPRODUCTION.md](REPRODUCTION.md), [`scripts/reproduce.sh`](../scripts/reproduce.sh) |
| Explore results interactively (D6) | [`levy/dashboard/`](../levy/dashboard/), [`scripts/dashboard.py`](../scripts/dashboard.py) |
| Understand the dataset and its provenance | [`data/DATASHEET.md`](../data/DATASHEET.md), [`levy/dataset/`](../levy/dataset/) |
| Check the research definition | [`docs/Project_Proposal.md`](Project_Proposal.md), [`docs/Specification_and_Design_Report.md`](Specification_and_Design_Report.md) (both frozen) |
