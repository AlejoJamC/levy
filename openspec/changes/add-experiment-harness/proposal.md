# Proposal: add-experiment-harness

**Linear:** LEV-4 | **Maps to:** Objectives O1–O3, Deliverable D3 | **Spec basis:** S&D Report Algorithm 2 (`run_experiment`) + "Experimental procedure".

## Why

The study's raw results — precision, recall, F0.5, false positive rate, and hit rate across the frozen 30-configuration grid — cannot be produced today: the engine, embedding manager, vector store (LEV-1/LEV-2), and dataset platform (LEV-3) all exist, but nothing replays annotated query pairs through the cache and accounts for TP/FP/TN/FN against the human labels. This harness is the last critical-path artefact before the experiments themselves (LEV-8 consumes its output).

## What Changes

- New `levy/experiment/` package implementing the offline replay protocol of Algorithm 2: per configuration, reset to a fresh empty cache; for each `QueryPair`, submit `query_1` (miss by construction, stored), then `query_2` (hit/miss decision); compare the decision against `original_label` and increment TP / FP / TN / FN.
- `ExperimentConfig` (embedding model, workload, similarity threshold) and a sweep runner over the full frozen grid: 2 models (`all-MiniLM-L6-v2`, `modernbert`) × 3 workloads (faq, code, chat) × 5 thresholds (0.70–0.90, step 0.05) = 30 configurations. Thresholds apply on the `1/(1+L2)` similarity scale exactly as stored — no rescaling (frozen-spec mandate, see CLAUDE.md known-gaps note).
- Per-configuration metrics: precision, recall, F0.5 (β=0.5, precision-weighted), FP rate `FP/(FP+TN)`, hit rate, plus cache-lookup latency accounting.
- Machine-readable outputs: one CSV row per configuration and a per-pair decision log (audit trail) — the input contract for LEV-8's statistical analysis.
- One-command entry point (`scripts/run_experiments.py`) that runs all 30 configurations end-to-end, fully offline and deterministic with the mock LLM; dataset input via the `levy/dataset` loader (`data/ground_truth.csv|json` — synthetic fixture until LEV-11 delivers the real 900 pairs).
- Statistical sanity checks per the frozen QA plan: rates within [0,1], confusion counts sum to the pair count per configuration.

## Capabilities

### New Capabilities

- `experiment-harness`: offline replay of annotated query pairs across the 30-configuration grid, confusion-matrix accounting against ground-truth labels, per-configuration metric computation, deterministic machine-readable outputs (results CSV + per-pair decision log).

### Modified Capabilities

_None. The harness consumes `embedding-management`, `vector-store`, and the dataset capability (in-flight `add-ground-truth-dataset` delta) through their existing requirements; no spec-level behavior of existing capabilities changes._

## Impact

- **New code:** `levy/experiment/` (config, replay, metrics, sweep, output writers), `scripts/run_experiments.py`, `tests/test_experiment_*.py`.
- **Consumed unchanged:** `LevyEngine`/`SemanticCache.reset()` (LEV-2), `EmbeddingManager` (LEV-1), `levy/dataset` loader + `QueryPair` schema (LEV-3), mock LLM/embedding providers.
- **Dependencies:** none added — stdlib `csv`/`json` + numpy, per repo convention.
- **Downstream:** LEV-8 (statistical analysis) consumes the results CSV; optional LEV-6 Anthropic connector can later populate caches realistically without changing the harness (responses do not affect hit/miss decisions).
- **Data note:** runs against the synthetic 15-pair fixture until LEV-11 lands the real dataset; the harness is data-agnostic by design.
