## Why

The approved Project Proposal defines economic viability with two measurements: hit rate **and** "latency measurements (cache lookup overhead vs LLM call savings)". Hit rate is delivered; latency is measured nowhere. Every timing the repository can produce today comes from `MockLLMClient`, which sleeps a fixed 0.5 s, and `LevyEngine` times only the whole `generate()` path with a single clock read, so the cache's own cost cannot be separated from the response it avoids. No run has ever exercised the Anthropic connector against corpus data.

This change adds the missing measurement: a decomposed, offline measurement of what the cache costs per lookup, and a real-provider measurement of what a cache hit avoids, run over the FAQ workload — the only workload where the cache measurably operates (24.0 % best hit rate, against 2.3 % chat and 2.0 % code).

## What Changes

- **New timing instrumentation in the engine.** Per-request segment timings — embedding generation, vector index search, exact-cache lookup, total lookup path — exposed additively. `LevyResult.latency_ms`, the `LevyMetrics` contract and default engine behaviour are unchanged.
- **New offline benchmark path.** Repeated, warm-up-preceded sampling of the lookup path over the FAQ pairs, reporting embedding cost **cold** (memoisation cleared) and **warm** (memoised) as separate figures, plus p50/p95 per segment.
- **New real-response population path.** A networked, billed entry point that calls the Anthropic API once per unique FAQ prompt and records wall-clock latency, input/output token counts, resolved model identifier and UTC timestamp alongside the response. Excluded from the offline test suite in the same manner as `scripts/fetch_corpora.py`.
- **New artefact directory `results/latency-faq/`**, holding `latency.csv`, `llm_calls.json`, `latency_meta.json` and a gitignored `responses.jsonl`. Written outside the deterministic result set.
- **No change to the D3 result contract.** `results.csv` and `decisions.csv` keep excluding latency and timestamps, so harness re-runs stay byte-identical and the ±5 % replication criterion is untouched. Latency values never enter that criterion.
- **`results/run-003/` is read-only for this change.** No file under it is created, modified, renamed or deleted.

## Capabilities

### New Capabilities

- `latency-measurement`: decomposed measurement of cache lookup overhead and of real-provider call latency, their artefact contract, their reproducibility boundary (overhead reproducible, provider latency not), and their isolation from the deterministic D3 result set.

### Modified Capabilities

None. `experiment-harness` keeps its existing requirement that `results.csv` and `decisions.csv` carry no latency or timestamps — this change depends on that requirement rather than altering it. `anthropic-connector` is used as specified, with no requirement change.

## Impact

- **Code:** `levy/engine.py` (segment timing hooks), `levy/cache/semantic_cache.py` and `levy/cache/vector_index.py` (search timing surface), `levy/embedding_manager.py` (cold/warm measurement support), a new `levy/latency/` package, new scripts for the offline benchmark and the networked population run.
- **Configuration:** `anthropic_model` and the two per-MTok price fields in `levy/config.py` are set per run; the stale `claude-opus-4-8` default and its prices are not assumed to describe the model actually used.
- **Artefacts:** new `results/latency-faq/` tree; `.gitignore` entry for the response corpus.
- **Documentation:** methodology and the FAQ-only deviation recorded in `data/DATASHEET.md` §2; `docs/REPRODUCTION.md` gains the offline benchmark and marks the networked run as out-of-band.
- **Cost:** one billed run of at most ~600 calls, piloted on `claude-haiku-4-5-20251001`, bounded by the existing `_BudgetGuard`.
- **Tests:** offline coverage for the instrumentation and the artefact writers, keeping the `--cov-fail-under=90` gate green; the networked path stays out of the suite.
- **Tracking:** LEV-14.
