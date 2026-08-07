## 1. Guard the existing result set

- [x] 1.1 Write a SHA-256 manifest of every file under `results/run-003/` (including `analysis/`) to `results/latency-faq/run-003.manifest.sha256`
- [x] 1.2 Copy `results/run-003/` to a location outside the `results/` tree and record that path in the manifest file's header comment
- [x] 1.3 Add `results/latency-faq/responses.jsonl` to `.gitignore` and confirm `git check-ignore` reports it as ignored

## 2. Timing instrumentation

- [x] 2.1 Add a `TimingCollector` to a new `levy/latency/` package: named-segment accumulation, monotonic clock, no dependency on the engine
- [x] 2.2 Add an optional `timing` parameter to `LevyEngine.generate()` that records the embedding, index-search, exact-lookup and total-lookup segments; leave the signature default and all return values unchanged
- [x] 2.3 Surface search duration from `SemanticCache.get()` / `VectorIndex.search()` into the collector without altering their return types
- [x] 2.4 Unit-test that segment sums never exceed the recorded total, on an exact hit, a semantic hit and a miss
- [x] 2.5 Unit-test that with no collector passed, `LevyResult` fields and `LevyMetrics` counters are identical to the pre-change values

## 3. Offline benchmark

- [x] 3.1 Implement `levy/latency/benchmark.py`: warm-up iterations discarded, configurable repetitions, p50/p95 per segment
- [x] 3.2 Implement cold sampling by clearing the `EmbeddingManager` memo entry for the text immediately before the measured call; implement warm sampling against the memoised path
- [x] 3.3 Unit-test that a cold sample invokes the underlying embedding client and a warm sample does not, using a call-counting client double
- [x] 3.4 Unit-test that reported p50/p95 are computed from exactly the measured repetitions and exclude warm-up iterations

## 4. Artefact writers

- [x] 4.1 Implement the `latency.csv` writer: one row per configuration with `config_id`, `model`, `threshold`, `n_lookups`, and p50/p95 for embed-cold, embed-warm, index-search, exact-lookup and total-lookup
- [x] 4.2 Implement the `latency_meta.json` writer: host specification, OS, Python and library versions, embedding provider, seeds, warm-up and repetition counts, and the explicit statement that overhead replicates and provider latency does not
- [x] 4.3 Implement the `llm_calls.json` writer: call count, p50/p95/min/max latency, input/output token totals, resolved model identifier, UTC window, observed total cost and observed cost per call
- [x] 4.4 Unit-test that a completed offline run writes `latency.csv` and `latency_meta.json` with all ten FAQ rows and no empty percentile field
- [x] 4.5 Unit-test that the metadata sidecar contains the reproducibility-boundary statement

## 5. Networked population entry point

- [x] 5.1 Implement `scripts/populate_responses.py`: one Anthropic call per unique prompt, recording latency, input/output tokens, resolved model identifier and UTC timestamp per call
- [x] 5.2 Write the response corpus as JSONL keyed by `sha256(prompt)`, skipping any key already present so an interrupted run resumes without double spend
- [x] 5.3 Propagate `BudgetExceededError` as a clean stop that leaves the partial corpus valid and readable
- [x] 5.4 Extend the existing out-of-band AST guard so it also asserts that no test invokes `scripts/populate_responses.py` and that no `levy/latency/` module imports a network library at module scope
- [x] 5.5 Unit-test the corpus writer, resume-skip behaviour and budget-stop path offline via `httpx.MockTransport`, using the injection pattern from `tests/test_anthropic_client.py`

## 6. Timed replay driver

- [x] 6.1 Implement `scripts/run_latency.py`: reads the FAQ configuration list from a reference `results.csv`, requires an explicit `--out-dir`, replays the ten FAQ configurations with instrumentation active
- [x] 6.2 Load responses from the corpus when present so the replay makes no provider calls
- [x] 6.3 Unit-test the driver end to end offline with mock providers, asserting `latency.csv` and `latency_meta.json` are written and the reference directory is unmodified

## 7. Pilot run

- [x] 7.1 Set `anthropic_model` to `claude-haiku-4-5-20251001` and the two per-MTok price fields to that model's published prices
- [x] 7.2 Run `scripts/populate_responses.py` over the FAQ workload and confirm `results/latency-faq/llm_calls.json` records the full call set with a non-zero observed cost
- [x] 7.3 Run `scripts/run_latency.py --out-dir results/latency-faq` and confirm `latency.csv` contains all ten FAQ configurations
- [x] 7.4 Compute the Sonnet-class cost of an equivalent run from the pilot's recorded token totals and record it in `llm_calls.json`

## 8. Headline figure

- [x] 8.1 Derive median lookup overhead added versus median provider latency avoided on FAQ, and write it into `latency_meta.json` with the resolved model identifier attached
- [x] 8.2 Assert in a test that any savings figure written to an artefact is accompanied by a model identifier in the same artefact

## 9. Documentation

- [x] 9.1 Record the deviation in `data/DATASHEET.md` §2: real-response population limited to FAQ, with the hit-rate rationale, noting `chat` and `code` remain mock-populated
- [x] 9.2 Add the offline benchmark command to `docs/REPRODUCTION.md` and mark `scripts/populate_responses.py` as out-of-band and billed, alongside the existing treatment of `scripts/fetch_corpora.py`
- [x] 9.3 Add `levy/latency/` and the two new scripts to the code-architecture section of `CLAUDE.md`

## 10. Verification

- [x] 10.1 Re-verify every hash in `results/latency-faq/run-003.manifest.sha256` and confirm zero differences
- [x] 10.2 Confirm `results/` contains no file written by this change outside `results/latency-faq/`
- [x] 10.3 Run `scripts/check_replication.py --reference results/run-003/results.csv` and confirm 60/60 still passes
- [x] 10.4 Run `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` and confirm it passes
- [x] 10.5 Run `openspec validate --all` and confirm it passes
