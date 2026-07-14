# Tasks: add-experiment-harness

## 1. Engine support

- [x] 1.1 Add optional `EmbeddingManager` injection to `LevyEngine` (backward-compatible constructor parameter, default constructs from `LevyConfig` as today); all existing tests pass unchanged.

## 2. Experiment package core

- [x] 2.1 Create `levy/experiment/config.py`: `ExperimentConfig` dataclass (embedding model, workload, threshold) and `full_grid()` enumerating the frozen 30 configurations (2 models × 3 workloads × 5 thresholds 0.70–0.90 step 0.05), thresholds carried verbatim.
- [x] 2.2 Create `levy/experiment/metrics.py`: `EvaluationResult` dataclass (config identity, TP/FP/TN/FN, precision, recall, F0.5, FPR, hit rate, zero-division flags) + formula implementations with the defined zero-denominator behavior (0.0 + flag, never NaN) + sanity checks (counts sum to N, rates within [0,1]) that raise naming the offending configuration.
- [x] 2.3 Create `levy/experiment/replay.py`: `run_experiment(config, pairs) -> EvaluationResult` per Algorithm 2 — fresh engine per configuration (mock LLM, memory store, configured model + threshold), replay pairs in dataset order (`query_1` submit/store, `query_2` submit/decide via `LevyResult.source`), accumulate cache within the configuration, increment TP/FP/TN/FN, collect per-pair decision records (pair id, decision, source, similarity, label, outcome).
- [x] 2.4 Create `levy/experiment/runner.py`: sweep orchestration over the grid sharing one `EmbeddingManager` per model across its 15 configurations; writers for `results.csv` (one row per configuration, no timestamps/latency), `decisions.csv` (per-pair audit log), and `run_meta.json` sidecar (dataset path, providers, model checkpoints, grid, latency stats labeled synthetic under the mock LLM).

## 3. CLI

- [x] 3.1 Create `scripts/run_experiments.py` (argparse): dataset path (default `data/ground_truth.csv`), output directory, optional grid subset flags (`--models/--workloads/--thresholds`) for smoke runs; loads via the `levy/dataset` loader; runs fully offline; non-zero exit on sanity-check failure.
- [x] 3.2 Run the CLI end-to-end on the committed synthetic fixture with mock providers; confirm 30 result rows + decision log + sidecar are produced.

## 4. Tests (offline, unittest style)

- [x] 4.1 Grid tests: `full_grid()` yields exactly 30 distinct configs with the frozen values; thresholds unmodified.
- [x] 4.2 Metrics tests: hand-computed case TP=8/FP=2/TN=7/FN=3 → precision 0.8, recall 8/11, F0.5 ≈0.7843, FPR 2/9, hit rate 0.5; zero-denominator cases → 0.0 + flag; sanity-check violations raise.
- [x] 4.3 Replay tests on synthetic pairs with known expected outcomes: constructed near-identical pair → TP at low threshold; constructed unrelated pair → TN; forced FP/FN cases; exact-duplicate `query_2` decided by exact cache and logged with its source; cache accumulation across pairs within a config; fresh cache between configs.
- [x] 4.4 Determinism test: run the sweep twice on the fixture → `results.csv` and `decisions.csv` byte-identical; confirm neither contains timestamps/latency.
- [x] 4.5 Output-contract tests: results CSV has one row per configuration with all required columns; decision log has one row per pair per configuration, resolvable by pair id.
- [x] 4.6 Full suite green: `python -m unittest discover -s tests -p "test_*.py"` (all pre-existing tests still pass, incl. engine injection change).

## 5. Docs & sync

- [x] 5.1 Update README.md (experiment harness section: command, outputs, offline note) and CLAUDE.md (architecture map: `levy/experiment/`, `scripts/run_experiments.py`; mark known-gap #4 resolved).
- [x] 5.2 `openspec validate --all` passes; keep Linear LEV-4 in sync (reference this change, tick acceptance criteria as delivered).
