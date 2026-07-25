# Design: add-experiment-harness

## Context

All prerequisites are merged on `main`: `EmbeddingManager` with runtime model switching and `(model, sha256(text))` memoization (LEV-1), `SemanticCache` over a `VectorIndex` with `similarity = 1/(1+L2)` and `reset()` (LEV-2), and the `levy/dataset` platform with the `QueryPair` schema, CSV/JSON loader, and a synthetic 15-pair fixture at `data/ground_truth.csv|json` (LEV-3). The frozen S&D Report defines the harness precisely: Algorithm 2 (`run_experiment`) plus the "Experimental procedure" (fresh cache per configuration; per pair submit `query1` — miss by construction, stored — then `query2` — hit/miss decision vs threshold; increment TP/FP/TN/FN; compute metrics after all pairs). The harness must run fully offline (mock providers) and be deterministic; LEV-8's statistical analysis consumes its output; the replication success criterion is ±5% on headline precision/recall.

## Goals / Non-Goals

**Goals:**
- Implement Algorithm 2 faithfully: one accumulating cache per configuration, sequential pair replay in dataset order, confusion accounting against `original_label`.
- Sweep the full frozen grid: 2 models × 3 workloads × 5 thresholds (0.70–0.90 step 0.05) = 30 configurations, one command, fully offline.
- Deterministic, machine-readable outputs: per-configuration results CSV + per-pair decision log (LEV-8's input contract).
- Statistical sanity checks per the frozen QA plan.

**Non-Goals:**
- No ANOVA/Tukey/curves — that is LEV-8, consuming this harness's CSV.
- No Anthropic connector (LEV-6); responses never affect hit/miss decisions, so the mock LLM suffices. The harness must not need changes when LEV-6 lands.
- No real 900-pair dataset (LEV-11); the harness is data-agnostic and runs on whatever the `levy/dataset` loader provides.
- No threshold rescaling, no alternative similarity metrics, no FastAPI surface.

## Decisions

1. **Replay through `LevyEngine.generate()` (production path), not raw `SemanticCache`.** Algorithm 2's `cache.get()` encapsulates the cache behavior; the engine's flow (exact cache → semantic cache → mock LLM → store) is the production lookup of Algorithm 1. A hit is `LevyResult.source ∈ {exact_cache, semantic_cache}`; the decision log records which path decided. *Alternative rejected:* driving `SemanticCache` directly — diverges from the production decision (would miss exact-duplicate pairs the exact cache legitimately catches) and re-implements store logic the engine already owns.
2. **Fresh engine per configuration; accumulating cache within a configuration.** Exactly per the frozen procedure: initialise an empty cache per configuration (`SemanticCache.reset()` + new exact cache via a new engine instance); never reset between pairs, so `query_2` is compared against the whole index accumulated so far — including other pairs' stored queries — as in production replay.
3. **One `EmbeddingManager` per model, shared across its 15 configurations.** LEV-1's memoization by `(model_key, sha256(text))` exists precisely so a sweep does not recompute embeddings per threshold. Requires allowing the engine to accept an injected manager (small, backward-compatible constructor addition; default behavior unchanged).
4. **Metric definitions (fixed formulas, zero-division defined):** precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, `F0.5 = 1.25·P·R/(0.25·P+R)`, FPR `FP/(FP+TN)`, hit rate `(TP+FP)/N`. Any 0/0 case is reported as `0.0` with a flag column rather than NaN, so the CSV stays machine-readable for scipy/pandas downstream.
5. **Determinism guarantee covers decisions and metrics, not wall-clock.** `results.csv` and `decisions.csv` contain no timestamps or latency values; re-running on the same inputs must reproduce them byte-identically (mock embeddings are text-seeded; sentence-transformers models are deterministic on CPU; pair order = dataset file order). Latency (per-lookup `perf_counter` stats) goes to a sidecar `run_meta.json` together with run parameters (dataset path, model checkpoints, grid), explicitly excluded from the determinism check. Mock-LLM latency (fixed 0.5 s sleep) is synthetic and labeled as such in the sidecar.
6. **Thresholds are used exactly as configured on the `1/(1+L2)` similarity scale.** The frozen sweep 0.70–0.90 maps to a high-cosine band for unit vectors; CLAUDE.md records this as intentional and spec-mandated. The harness passes thresholds through untouched; if hit-rate viability (>30%) fails at this band on real data, that is a research finding for the supervisor, not a code change.
7. **Package layout:** `levy/experiment/` with `config.py` (`ExperimentConfig`, grid enumeration), `replay.py` (`run_experiment(config, pairs) -> EvaluationResult` per Algorithm 2), `metrics.py` (`EvaluationResult`, formula implementations, sanity checks), `runner.py` (sweep orchestration, output writers), plus `scripts/run_experiments.py` (argparse CLI: dataset path, output dir, optional grid subset for smoke runs). Names `ExperimentConfig`/`run_experiment`/`EvaluationResult` follow the frozen pseudocode.

## Risks / Trade-offs

- [Accumulated-cache cross-pair hits: `query_2` may match a *different* pair's stored query, making FP/TP attribution depend on dataset order] → This is what the frozen procedure prescribes ("comparing query2's embedding against the index"); the decision log records the matched entry and similarity so any such hit is auditable. Order is fixed by the dataset file, keeping it deterministic.
- [Mock-provider results are not research results] → The fixture run validates machinery only; `run_meta.json` records providers/dataset so LEV-8 can refuse non-real inputs. Real runs need LEV-11's data (and optionally sentence-transformers providers), no harness change.
- [Engine constructor change for manager injection could ripple] → Keep it optional-with-default; existing tests must pass unchanged.
- [Latency numbers on the mock LLM are meaningless for savings claims] → Labeled synthetic in the sidecar; real latency measurement becomes meaningful only with a real connector (LEV-6), which the sidecar format already accommodates.

## Migration Plan

Pure addition: new package + script + tests; one backward-compatible engine constructor parameter. No data migration, no rollback complexity — reverting the change removes the harness only.

## Open Questions

- None blocking. Whether real experiment runs (LEV-8 era) also populate caches via the Anthropic connector is deferred to LEV-6 and does not affect the harness contract (responses don't influence decisions).
