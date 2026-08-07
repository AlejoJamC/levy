## Context

The approved Project Proposal commits to assessing economic viability through hit rate **and** "latency measurements (cache lookup overhead vs LLM call savings)". Hit rate is delivered in `results/run-003/`. Latency has never been measured.

Three properties of the current code shape this design:

1. **`LevyEngine.generate()` times only the whole path** with a single clock read at entry (`levy/engine.py:82`) and one at each exit (`:89`, `:107`, `:138`). The cost of embedding, of the index search and of the store lookup cannot be separated from each other.
2. **`EmbeddingManager` memoises by `(model_key, sha256(text))`** (`levy/embedding_manager.py:121`, `:158-162`). During a threshold sweep the same texts are re-embedded across configurations, so timings taken naively during a sweep measure a dictionary lookup, not an encoder.
3. **`results.csv` and `decisions.csv` deliberately exclude latency and timestamps** (`levy/experiment/runner.py:99`, `:130`) so re-runs are byte-identical. That property is what `scripts/check_replication.py` and the ±5 % criterion rest on.

The measurement also needs a real provider call, which the repository has never made against corpus data: every run to date used `MockLLMClient` and its fixed 0.5 s sleep (`levy/llm_client.py:15-21`).

## Goals / Non-Goals

**Goals:**

- Report what the cache costs per lookup, decomposed by segment rather than as one wall-clock number.
- Report what a cache hit avoids, measured against a real provider.
- Keep the lookup-overhead measurement reproducible by a third party on their own hardware.
- Leave the D3 result set and the replication criterion provably untouched.

**Non-Goals:**

- Re-running or altering the 30-configuration grid.
- Real-model population of `chat` or `code`.
- Making the provider-latency measurement reproducible — it cannot be, and the design records that rather than pretending otherwise.
- Feeding any latency value into the ±5 % replication criterion.
- Optimising the lookup path. This change measures; it does not tune.

## Decisions

### Timing is an opt-in collector, not a field on `LevyResult`

A `TimingCollector` is passed into the engine when measuring and is absent otherwise. Segment boundaries push durations into it; when it is absent the code path is unchanged.

*Alternative considered:* add `segment_timings` to `LevyResult`. Rejected — `LevyResult` is consumed by the API layer, the harness and the dashboard, and every consumer would have to grow a branch for a field that is empty in all normal operation.

*Alternative considered:* wrap the clients in timing decorators from the outside. Rejected — the exact-cache and semantic-cache boundaries inside `generate()` are not observable from outside the engine, so the decomposition would be incomplete in exactly the place the measurement is about.

### Cold and warm embedding cost are separate reported figures

A cold sample clears the memoisation entry for its text immediately before the measured call; a warm sample measures the memoised path. Both are reported; neither is averaged into the other.

Memoisation is a real property of the artefact, not a measurement artefact, so suppressing it would understate the cache's steady-state advantage and reporting only it would hide the first-request cost. Both numbers are true and answer different questions.

### The provider run is a separate, out-of-band entry point

Population lives in its own script, is never invoked by a test, and no module the offline suite imports pulls in a network library at module scope. This mirrors the treatment of `scripts/fetch_corpora.py`, which is already the repository's single networked entry point and is guarded by an AST test.

*Alternative considered:* a `--real-llm` flag on `scripts/run_experiments.py`. Rejected — it puts a billed network call one typo away from the deterministic harness, which is the one thing this change must not endanger.

### The response corpus is built once, keyed by prompt hash

`ExactCache` keys on `sha256(prompt)` (`levy/cache/exact_cache.py:13`), so a single corpus of at most ~600 real responses (300 pairs × 2 queries) serves all ten FAQ configurations. Re-population is skipped for any key already present, which also makes an interrupted run resumable without double spend.

### Artefacts go to a dedicated directory, outside the deterministic set

`results/latency-faq/` holds `latency.csv`, `llm_calls.json`, `latency_meta.json` and the gitignored `responses.jsonl`. Nothing is added to `results.csv` or `decisions.csv`.

*Alternative considered:* extend `results.csv` with latency columns. Rejected — it would break byte-identical re-runs and therefore the replication check, trading the criterion that passes for a number that does not need to live there.

### The response corpus is not committed

It contains model output generated over Quora QQP and other corpus text, under the same licence reasoning that keeps `data/ground_truth.full.*` out of the tree.

### Model selection is staged, and every figure names its model

Stage 1 pilots on `claude-haiku-4-5-20251001` to convert the cost estimate into an observed figure and to exercise the connector, budget guard, retry path and corpus writer against the live API at volume. Stage 2 prices a Sonnet-class repeat from the pilot's actual token totals. Stage 3, escalating to `claude-opus-5`, is optional and gated on nothing but judgement.

Provider latency is model-dependent, so measurement B belongs to whichever model produced it; a Haiku figure is a lower bound on what a larger model would have avoided. Measurement A never touches the provider and is unaffected by the choice. Prices come from the price list for the model actually used, not from the `claude-opus-4-8` values currently sitting in `levy/config.py:28-29`.

### FAQ only

The frozen documents impose no per-workload requirement on provider calls: the experimental procedure and the harness pseudo-code contain no LLM call, and the budget line is an estimate. FAQ is the only workload where the cache measurably operates — 24.0 % best hit rate against 2.3 % (chat) and 2.0 % (code) — so measuring savings elsewhere would characterise a path taken fewer than once in forty lookups.

## Risks / Trade-offs

- **An accidental write destroys `results/run-003/`** → `results/` is fully gitignored with zero tracked files, so there is no git recovery. A SHA-256 manifest of the whole directory is recorded and a copy placed outside the `results/` tree *before* any work starts, and every hash is re-verified afterwards. Every script involved requires an explicit `--out-dir`; none has a default that can resolve to `run-003`.
- **Wall-clock noise on a laptop** → warm-up iterations are discarded, repetitions are configurable, and p50/p95 are reported instead of a mean. The host specification and library versions are recorded so a reader can judge comparability.
- **A Haiku-derived savings figure gets read as a general result** → every reported figure carries its resolved model identifier, and the metadata sidecar states the reproducibility boundary explicitly.
- **The billed run overruns** → the existing `_BudgetGuard` halts before sending once the estimate reaches the cap, and the hash-keyed corpus makes a resumed run skip everything already paid for.
- **Instrumentation slows the normal path** → the collector is absent outside measurement, so the production path takes no additional clock reads.
- **Coverage gate regression** → the new offline modules are unit-tested; the networked entry point is excluded from the suite the same way `scripts/fetch_corpora.py` is.

## Migration Plan

Additive throughout. No existing artefact changes format, no configuration default changes meaning, and no consumer of `LevyResult`, `results.csv` or `decisions.csv` needs updating. Reverting is deletion of the new package, scripts and output directory.

## Open Questions

- Whether Stage 2 (Sonnet-class repeat) happens at all is decided from the pilot's observed token totals, not in advance.
- The per-MTok prices for `claude-opus-5` are set from the current price list at the time of a Stage 3 run; no figure is assumed here.
