# Tasks: add-ground-truth-dataset

Implementation context for the executing agent: read `proposal.md`,
`design.md`, and `specs/ground-truth-dataset/spec.md` in this directory
first. Run every Python command inside the conda env:
`source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy`.
Tracking issue: Linear LEV-3.

Sections 1-6 are **platform tooling** — implemented and tested in this
change. Section 7 is **data production** — requires the author's manual
work (real corpus downloads, blind judgment, a real 900-pair run) and is
explicitly NOT performed by this change; those checkboxes are left unticked
as the open work this change unblocks but does not do.

## 1. Schema

- [x] 1.1 Create `levy/dataset/schema.py`: workload constants (`faq`,
  `code`, `chat`) and `WORKLOADS` tuple; `QueryPair` dataclass with
  `pair_id`, `workload`, `source_corpus`, `source_pair_id`, `query_1`,
  `query_2`, `original_label`, `author_label`, `metadata` (design.md D1)
- [x] 1.2 Implement `validate_query_pair` / `__post_init__` validation
  (non-empty ids/queries, workload in `WORKLOADS`, labels in `{0,1,None}`
  as appropriate) raising `QueryPairValidationError`
- [x] 1.3 Implement `ground_truth_label()` (author label if set, else
  original label) and `to_dict()`/`from_dict()` (design.md D1)

## 2. CSV/JSON persistence (LEV-4 contract)

- [x] 2.1 Create `levy/dataset/io.py`: `FIELDNAMES`, `save_csv`/`load_csv`,
  `save_json`/`load_json`, `save_dataset`/`load_dataset` (extension
  dispatch) (design.md D2)
- [x] 2.2 `metadata` JSON-encoded into one CSV column; `author_label` empty
  string <-> `None` <-> JSON `null` (design.md D2)
- [x] 2.3 Clear `DatasetValidationError` on missing columns (CSV) or
  non-list JSON, naming the file and (for CSV) the row number

## 3. Sampling pipeline

- [x] 3.1 Create `levy/dataset/sampling.py`: `RawCandidatePair`,
  `CorpusSource` ABC (`iter_candidates()`, no network access) (design.md D3)
- [x] 3.2 Implement `QuoraQQPSource` (TSV: `id, qid1, qid2, question1,
  question2, is_duplicate`), `StackOverflowDuplicatesSource` (CSV:
  `pair_id, question1_id, question1, question2_id, question2,
  is_duplicate`), `ConvAI2Source` (JSON: `pair_id, utterance_1,
  utterance_2, same_intent`) — each documents its expected raw format and
  reads only local files (design.md D3)
- [x] 3.3 Implement `MockCorpusSource` (deterministic, synthetic, balanced,
  no file I/O) for tests and CLI fallback
- [x] 3.4 Implement `sample_workload` (sort-then-sample determinism,
  stratified positive/negative draw, raises `CorpusSourceError` on
  shortfall) and `sample_dataset` (all 3 workloads) (design.md D4)

## 4. Blind re-annotation

- [x] 4.1 Create `levy/dataset/annotation.py`: `BlindAnnotationSession`
  (shows only `query_1`/`query_2`; records `author_label`; injected
  `input_fn`/`output_fn` for testability) (design.md D5)
- [x] 4.2 Implement per-answer progress persistence to a JSON progress file
  and progress-file merge on session construction (resumable) (design.md D5)
- [x] 4.3 Implement no-overwrite-by-default + explicit `overwrite=True`
  escape hatch; `AnnotationSummary` return value; `q`/`s` quit/skip handling

## 5. Cohen's kappa

- [x] 5.1 Create `levy/dataset/kappa.py`: 2x2 contingency, `po`/`pe`/`kappa`
  computation, stdlib-only arithmetic (design.md D6)
- [x] 5.2 Document and implement the zero-annotated-pairs and `pe == 1`
  edge cases (design.md D6)
- [x] 5.3 Implement `kappa_report` (overall + per-workload breakdown)

## 6. CLIs, fixtures, tests, docs (platform completion)

- [x] 6.1 `scripts/sample_dataset.py`: argparse CLI, falls back to
  `MockCorpusSource` per workload when a raw file path is omitted, writes
  CSV+JSON
- [x] 6.2 `scripts/annotate_dataset.py`: argparse CLI over
  `BlindAnnotationSession`, real `input()`/`print()` by default (injectable
  for tests), saves merged CSV+JSON
- [x] 6.3 `scripts/compute_kappa.py`: argparse CLI, overall + per-workload
  report, `--strict`/`--threshold` gate that only fires once the dataset is
  fully annotated
- [x] 6.4 `scripts/export_dataset.py`: CSV<->JSON conversion CLI
- [x] 6.5 Generate `data/ground_truth.csv` + `data/ground_truth.json`: 15
  synthetic pairs (5/workload) built through the real schema/io code,
  `source_corpus="synthetic-fixture"`, `metadata.provenance=
  "synthetic-fixture"`, two seeded author/original disagreements for a
  non-trivial kappa demo (design.md D7)
- [x] 6.6 Write `data/README.md` (placeholder disclosure, do-not-treat-as-
  research-data warning) and `data/DATASHEET.md` (full D2 datasheet
  skeleton, `TODO (post data-production)` markers only where the real
  sampling/annotation run is required)
- [x] 6.7 Write `tests/test_dataset.py`: schema validation; CSV/JSON
  round-trip + cross-format equality; sampling determinism + stratification
  + insufficient-candidates error; blind annotation blindness + resume +
  no-overwrite + skip/quit; kappa perfect/chance/worked-example/degenerate/
  unannotated-excluded/per-workload cases; CLI smoke tests (subprocess)
  against `data/ground_truth.json` — all offline
- [x] 6.8 Run the full suite green:
  `python -m unittest discover -s tests -p "test_*.py"` (89 tests: 41
  pre-existing + 48 new, all passing)
- [x] 6.9 Manually run each CLI once against the fixtures to confirm
  offline end-to-end behavior (`sample_dataset.py`, `annotate_dataset.py`,
  `compute_kappa.py` incl. `--strict`, `export_dataset.py`)
- [x] 6.10 Run `openspec validate --all` and fix any errors
- [x] 6.11 Update `README.md` (dataset tooling section + commands) and
  `CLAUDE.md` (architecture map: `levy/dataset`, `scripts/`, `data/`;
  known-gaps item 6 updated to reflect tooling-done/data-pending)

## 7. Data production (author task — NOT performed by this change)

- [ ] 7.1 Obtain the real Quora Question Pairs, Stack Overflow duplicate
  questions, and ConvAI2 (or approved fallback: MS MARCO / CodeSearchNet /
  DailyDialog) raw corpus files locally, respecting each corpus's licence
- [ ] 7.2 If needed, perform the ConvAI2 utterance-pairing derivation step
  (raw ConvAI2 has no native duplicate-intent pair labels) documented in
  `levy/dataset/sampling.ConvAI2Source`
- [ ] 7.3 Run `scripts/sample_dataset.py` against the real corpus files with
  the seed, `n_per_workload=300`, and `positive_ratio` to be used for the
  released dataset; record those parameters in `data/DATASHEET.md` §3
- [ ] 7.4 Run `scripts/annotate_dataset.py` to completion: the author's full
  blind re-annotation of all 900 pairs (original labels never shown)
- [ ] 7.5 Run `scripts/compute_kappa.py --strict` over the completed 900-pair
  set; record the overall and per-workload kappa results in
  `data/DATASHEET.md` §4 (target: overall kappa > 0.7 per the frozen S&D
  Report)
- [ ] 7.6 Replace the synthetic fixtures with the real dataset at
  `data/ground_truth.csv` / `data/ground_truth.json` (same filenames, same
  schema — no downstream code change needed)
- [ ] 7.7 Fill in the remaining `TODO (post data-production)` markers in
  `data/DATASHEET.md` (final counts, any fallback corpus actually used,
  limitations discovered during real sampling/annotation)
- [ ] 7.8 Update Linear LEV-3: tick data-production acceptance criteria,
  set status once the real dataset is released
