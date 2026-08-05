# Proposal: add-ground-truth-dataset

**Linear issue:** [LEV-3](https://linear.app/levy-project/issue/LEV-3) — Ground Truth Dataset: 900 annotated query pairs across 3 workloads (D2)
**Maps to:** Objective O1/O2, Deliverable D2 (frozen baseline: `docs/Specification_and_Design_Report.md` §A "Data required" + §C "D2: Annotated Ground Truth Dataset"; `docs/Project_Proposal.md` Phase 1 "Literature & Data" + Risk 1 fallback corpora)
**Feeds:** LEV-4 `add-experiment-harness` — the harness replays `query_1`/`query_2` per pair against the cache and scores hit/miss against the ground-truth label (Algorithm 2, S&D Report). LEV-4 cannot start against a stable dataset shape without this change.

## Why

D2 requires 900 human-annotated query pairs (300 per workload: FAQ, code,
chat), each carrying the original corpus label *and* the author's blind
re-annotation, released in CSV and JSON with a datasheet, with Cohen's kappa
(target > 0.7) computed over the full set. None of that exists in the repo
yet, and — critically — LEV-4 needs a **stable schema and loader contract**
to build the experiment harness against, independent of whether the real
900-pair sampling and annotation run has happened yet.

## Explicit scope split: platform tooling vs. data production

This change delivers the **data-agnostic platform** only. It does **not**
deliver the real dataset. Two categories of work exist under D2/LEV-3, and
this proposal implements only the first:

| Category | In this change? | What it is |
|---|---|---|
| **Platform tooling** | Yes — implemented here | `QueryPair` schema + validation; CSV/JSON loader (the LEV-4 contract); seeded stratified sampling pipeline with `CorpusSource` adapters for Quora QQP / Stack Overflow duplicates / ConvAI2-derived pairs (reading local raw files, never downloading); a resumable blind re-annotation CLI/session; a Cohen's kappa calculator with per-workload breakdown; CLIs wrapping all of the above; 15 synthetic fixture pairs (5/workload) in `data/` so the pipeline and LEV-4's future tests have something concrete and offline to run against; a full datasheet skeleton with everything fillable now filled in. |
| **Data production** | No — remains an open author task, tracked in `tasks.md` §7 | Actually downloading Quora QQP / Stack Overflow duplicates / ConvAI2 (or a fallback: MS MARCO, CodeSearchNet, DailyDialog); running the sampler against them to draw the real 900 pairs; the author performing the full blind re-annotation session over all 900; computing and recording the final Cohen's kappa; filling in the datasheet's `TODO (post data-production)` placeholders; replacing the synthetic fixtures in `data/` with the real released dataset. |

Everything in the second column requires the author's manual, blind
judgment and real corpus downloads — it is explicitly out of scope for an
agent-executed change and is not attempted here.

## What Changes

- New `levy/dataset/` package: `schema.py` (`QueryPair`, workload constants,
  validation), `io.py` (CSV/JSON load/save — the LEV-4 contract, round-trip
  safe), `sampling.py` (`CorpusSource` ABC + `QuoraQQPSource` /
  `StackOverflowDuplicatesSource` / `ConvAI2Source` / `MockCorpusSource`
  adapters + seeded stratified `sample_workload`/`sample_dataset`),
  `annotation.py` (`BlindAnnotationSession` — resumable, blind, no
  unintentional overwrite), `kappa.py` (`cohen_kappa`/`kappa_report`, stdlib
  + numpy only, documented edge-case behavior).
- New `scripts/` CLIs: `sample_dataset.py`, `annotate_dataset.py`,
  `compute_kappa.py`, `export_dataset.py` — thin argparse wrappers, all
  runnable fully offline against the fixtures.
- New `data/ground_truth.csv` + `data/ground_truth.json`: 15 synthetic
  fixture pairs (5/workload), obviously fake text, `source_corpus =
  "synthetic-fixture"`, marked in `metadata.provenance`. New `data/README.md`
  (placeholder disclosure) and `data/DATASHEET.md` (full D2 datasheet
  skeleton with `TODO (post data-production)` markers only where the real
  sampling/annotation run is required).
- New `tests/test_dataset.py`: schema validation, CSV/JSON round-trip,
  sampling determinism/stratification, blind-annotation flow (blindness,
  resume, no-overwrite), Cohen's kappa (perfect/chance/worked/degenerate
  cases), CLI smoke tests against the fixtures — all offline.
- Docs: `README.md` gains a dataset-tooling section; `CLAUDE.md` gains
  `levy/dataset` + `scripts/` + `data/` to the architecture map and flips
  known-gap item 6 from "no tooling" to "tooling done, real data pending".

## Capabilities

### New Capabilities

- `ground-truth-dataset`: the `QueryPair` schema and its validation rules;
  CSV/JSON persistence with round-trip and cross-format content equality;
  seeded, stratified sampling from a `CorpusSource`; blind, resumable,
  non-destructive re-annotation; Cohen's kappa computation with documented
  edge-case behavior (empty set, partially annotated, degenerate
  all-one-class).

### Modified Capabilities

<!-- none: this is the first spec covering dataset tooling -->

## Impact

- **Code:** new `levy/dataset/` package (5 modules), new `scripts/*.py` (4
  CLIs), new `tests/test_dataset.py`. No changes to existing `levy/`
  modules (engine, cache, embeddings) — this change is additive and has no
  runtime dependency on the rest of the package (LEV-4 is the consumer, not
  yet built).
- **Dependencies:** none added. Uses only stdlib (`csv`, `json`, `random`,
  `argparse`, `dataclasses`, `pathlib`) plus `numpy`, all already present in
  the `levy` conda env. No pandas/scipy/scikit-learn.
- **Data:** new `data/ground_truth.csv`, `data/ground_truth.json`,
  `data/README.md`, `data/DATASHEET.md` — all synthetic/template content,
  explicitly not the real dataset (no real corpus text is committed).
- **Tests:** new, fully offline `tests/test_dataset.py`. Existing 89 tests
  (`test_levy.py`, `test_embedding_manager.py`, `test_vector_index.py`) are
  untouched and must stay green.
- **Downstream:** LEV-4 (`add-experiment-harness`) loads datasets via
  `levy.dataset.io.load_dataset()` and evaluates via
  `QueryPair.ground_truth_label()`; LEV-8 (`add-statistical-analysis`)
  consumes the same schema for per-workload breakdowns. Neither is
  implemented by this change.
- **Not in scope:** downloading or committing any real corpus data; the
  actual 900-pair sample; the author's real blind re-annotation session;
  the final Cohen's kappa result; the experiment harness itself (LEV-4).
