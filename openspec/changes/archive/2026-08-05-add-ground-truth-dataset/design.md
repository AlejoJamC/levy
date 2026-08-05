# Design: add-ground-truth-dataset

## Context

The frozen S&D Report defines D2: "900 query pairs (300 per workload)
sampled from publicly available human-annotated corpora. Each pair retains
the original human label and the author's blind re-annotation. Cohen's
kappa between the author and the original labels is computed over the full
set with a target of kappa > 0.7. Released publicly in CSV and JSON formats
with a datasheet..." Algorithm 2 (`run_experiment`) shows exactly how the
harness will consume this dataset: `query_1` always misses (populates the
cache), `query_2`'s hit/miss decision is compared against
`pair.is_duplicate` (the ground-truth label) to accumulate TP/FP/TN/FN.

Nothing in the current codebase (`levy/`) touches this. LEV-1 and LEV-2 are
done (embedding manager, Faiss vector store); LEV-4 (experiment harness) is
next on the critical path and needs a dataset to replay. The real 900-pair
dataset requires downloading three separate public corpora and the author
personally performing a blind re-annotation of every pair — none of that is
executable by an agent in this change. What *is* buildable now, and what
unblocks LEV-4 regardless of when the real data lands, is the platform: a
frozen schema, a loader LEV-4 can code against today, a sampling pipeline
the author can point at real corpora later, and annotation/kappa tooling
the author can run interactively.

Constraints: Python >= 3.10, dataclasses (no Pydantic), synchronous code,
stdlib argparse for CLIs, no pandas/scipy/scikit-learn (not in the `levy`
conda env), ABC + mock/offline pattern for every external dependency
(established by `EmbeddingClient`, `VectorIndex`), no personal/sensitive
data, Apache-2.0.

## Goals / Non-Goals

**Goals:**
- A `QueryPair` schema stable enough for LEV-4 to build against today.
- A CSV/JSON loader with validated round-trip equality (the LEV-4 contract).
- A sampling pipeline that is deterministic (seeded) and stratified, with
  adapters documented against each target corpus's published raw format,
  without downloading anything.
- A blind, resumable, non-destructive re-annotation flow usable from a
  terminal.
- A Cohen's kappa implementation with clearly documented edge-case behavior.
- Small, obviously-synthetic fixture data so every piece of tooling (and
  later, LEV-4) has something to run against offline.

**Non-Goals:**
- Downloading, licensing-clearing, or committing any real corpus data.
- Running the real 900-pair sample or the real blind annotation.
- The experiment harness itself (LEV-4) — this change only guarantees the
  loader contract it will use.
- A GUI or web annotation tool — a synchronous stdin/stdout CLI loop is
  sufficient and matches the "keep it simple, mock-first" convention.
- Persisting/versioning multiple dataset revisions — one canonical
  `data/ground_truth.{csv,json}` pair, replaced in place when real data
  lands (mirrors how `docs/*.md` frozen docs are handled: the *files* stay
  put, only their content changes at a well-defined milestone).

## Decisions

### D1. `QueryPair` is a flat dataclass; `ground_truth_label()` picks the eval label

Fields: `pair_id`, `workload`, `source_corpus`, `source_pair_id`, `query_1`,
`query_2`, `original_label` (int 0/1), `author_label` (`Optional[int]`),
`metadata` (dict). `source_corpus` + `source_pair_id` give traceability back
to the origin corpus for audit, independent of the (small, sequential)
`pair_id`.

`ground_truth_label()` returns `author_label` if set, else
`original_label`. This is deliberate: LEV-4's Algorithm 2 compares against
"the original human label" today (frozen pseudocode names
`pair.is_duplicate`), but D2's whole point is that the *author's* blind
label is the one that should ultimately be trusted for evaluation once
annotation is complete. Exposing one method that degrades gracefully means
LEV-4 can be written once and its behavior gets more accurate as annotation
progresses, without a schema or API change.

- **Alternative considered:** always use `original_label` for evaluation,
  treat `author_label` as a kappa-only side channel. Rejected: the frozen
  report's own framing ("the author's blind re-annotations for validation")
  implies the author label is the more trustworthy ground truth once
  present; LEV-4 should not have to special-case this.

### D2. CSV/JSON round-trip: `metadata` JSON-encoded into one CSV column

CSV has no native nested-object type. Rather than flattening `metadata`
into ad-hoc extra columns (fragile — set of keys varies), the whole dict is
JSON-encoded into a single `metadata` CSV column, decoded back to a dict on
load. `author_label` is empty string in CSV / `null` in JSON for
unannotated pairs; both decode to Python `None`. `FIELDNAMES` is the single
source of truth for column order and is asserted present on load
(`DatasetValidationError` naming any missing column) — this is the exact
mechanism that keeps CSV and JSON "identical content" as D2 requires.

- **Alternative considered:** separate `.metadata.json` sidecar file
  keyed by `pair_id`. Rejected: two files to keep in sync is worse than one
  JSON-encoded column, and breaks "a CSV row is a complete record."

### D3. `CorpusSource` ABC, one adapter per corpus, `MockCorpusSource` for tests

Mirrors the `EmbeddingClient`/`VectorIndex` provider-abstraction convention:
an ABC (`iter_candidates() -> Iterator[RawCandidatePair]`) plus adapters.
`QuoraQQPSource` (TSV: `id, qid1, qid2, question1, question2,
is_duplicate` — the published Kaggle/GLUE QQP shape), `StackOverflow
DuplicatesSource` (CSV: `pair_id, question1_id, question1, question2_id,
question2, is_duplicate` — the commonly published Stack Overflow duplicate-
questions export shape; documented as adjustable if the author's concrete
download differs), `ConvAI2Source` (JSON: pre-derived `{pair_id,
utterance_1, utterance_2, same_intent}` records — ConvAI2 ships dialogues,
not duplicate-intent pairs, so a derivation step by the author precedes this
adapter; documented as such). None of these adapters fetch anything over the
network; they only parse a local file path passed in by the caller.
`MockCorpusSource` generates a deterministic, fully synthetic, balanced
candidate pool with no file I/O, for tests and CLI fallback.

- **Alternative considered:** a single generic "pairs CSV" adapter used for
  all three corpora, requiring the author to pre-normalize every corpus
  into one shape before sampling. Rejected: pushes format-translation work
  onto the author with no tooling support; documenting the *actual*
  published shape per corpus (even if approximate for Stack
  Overflow/ConvAI2, which lack one universal canonical export) is more
  useful and the adapter boundary is exactly where format differences
  should be absorbed.

### D4. Sampling determinism: sort-then-sample, not iteration order

`sample_workload` splits candidates into positive/negative pools, **sorts
each pool by `source_pair_id`** before sampling, then draws
`round(n * positive_ratio)` / remainder via `random.Random(seed).sample(...)`
and shuffles the concatenation with the same RNG. Sorting first removes any
dependency on file-read order or dict/set iteration order — determinism
must survive re-running the sampler against the same file on a different
machine or Python version, not just twice in the same process.

- **Alternative considered:** `random.seed(seed)` (global RNG). Rejected:
  global RNG state is a footgun for concurrent/repeated use within one
  process (e.g. sampling three workloads in one script run); a scoped
  `random.Random(seed)` instance avoids cross-workload interference.

### D5. Blind annotation: progress file is the resume mechanism, not the dataset file

`BlindAnnotationSession` writes `{pair_id: author_label}` to a small JSON
progress file after **every single answer** (not batched), and merges it
into the in-memory `QueryPair` list on construction. This means:
(a) a 900-pair session can be safely interrupted at any point — at most the
answer-in-progress is lost, never anything already recorded; (b) the
*dataset* file itself is only rewritten when the caller explicitly asks
(`scripts/annotate_dataset.py --out-csv/--out-json`, defaulting to
`--dataset`'s sibling paths), keeping "the dataset on disk" and "in-progress
annotation state" as separate concerns with separate save cadences.
Existing `author_label`s (from the dataset file or a prior progress file)
are never overwritten unless `overwrite=True` is passed explicitly — this
is a hard safety rule, not a default-off convenience flag, because
re-running a blind session over already-answered pairs (even
accidentally) must not silently corrupt a completed annotation.

- **Alternative considered:** rely solely on periodically re-saving the
  full dataset file as the resume mechanism. Rejected: re-serializing 900
  `QueryPair` records (with all their fields) after every keystroke is
  unnecessary I/O and risk surface compared to appending one small
  `{pair_id: label}` entry to a progress file.

### D6. Cohen's kappa: stdlib arithmetic over the 2x2 contingency, explicit degenerate rule

`cohen_kappa` builds the 2x2 confusion (`tp`/`fp`/`fn`/`tn` named for
compactness, not because either annotator is a "positive class"; see
`kappa.py` docstring) over pairs with `author_label is not None`, then:
`po = (tp+tn)/n`; `pe = p_original_1 * p_author_1 + (1-p_original_1) *
(1-p_author_1)`; `kappa = (po - pe) / (1 - pe)`.

Two documented edge cases:
- **Zero annotated pairs:** `kappa=None`, `n_annotated=0` — nothing to
  compute; callers (e.g. `compute_kappa.py --strict`) must not treat this
  as a threshold failure, only as "not ready yet."
- **`pe == 1` (both annotators' label-1 marginals are exactly 0 or exactly
  1):** guarded to avoid a division by zero, returning `kappa=1.0` if
  `po` is also 1.0, else `0.0`. Note (proved in review, recorded here so a
  future reader doesn't "fix" the branch): for a genuine 2x2 contingency
  built from real counts, `pe == 1` is only reachable when *both* marginals
  are simultaneously 0 or simultaneously 1, and in every such case `po == 1`
  follows automatically from the same counts — so the `else 0.0` arm is a
  defensive guard against floating-point edge cases, not a reachable
  "disagreement under a degenerate table" scenario in practice.

- **Alternative considered:** raise on `pe == 1` and force the caller to
  handle it. Rejected: kappa must always return *something* for a
  fully-agreed, single-class annotated subset (common early in an
  annotation session, e.g. the first 10 pairs answered are all "yes") —
  raising would make `compute_kappa.py` unusable mid-session.

### D7. Fixture data: 15 synthetic pairs, computed via the real schema/io code, not hand-written JSON

`data/ground_truth.csv`/`.json` are generated by constructing `QueryPair`
objects through the actual `levy.dataset.schema`/`io` code (so they are
schema-valid by construction, not just visually plausible), with
deliberately fabricated text (`"fixture-account"`, `"fixture-lang"`,
`"fixture-work"`, etc.) and `source_corpus = "synthetic-fixture"` /
`metadata.provenance = "synthetic-fixture"` on every row. Two pairs are
seeded with an author/original disagreement so `compute_kappa.py` against
the fixtures demonstrates a non-trivial (not 0, not 1) kappa end to end.

- **Alternative considered:** leave `author_label` unset on all fixture
  pairs (mirroring the real dataset's pre-annotation state). Rejected: that
  would make `scripts/compute_kappa.py` un-demoable against the committed
  fixtures (it would report "no annotated pairs" for every CLI smoke test
  and for anyone cloning the repo); fully labeling the *fixtures* (which
  are fake anyway) is a better offline demo of the whole pipeline than
  fidelity to the real dataset's temporal state.

## Risks / Trade-offs

- [Real corpora's actual published column names differ from what
  `StackOverflowDuplicatesSource`/`ConvAI2Source` assume] → both are
  documented as adjustable in their docstrings; the author renames columns
  or adjusts the `required` set when pointing the sampler at a concrete
  download, rather than the adapter silently guessing a mapping.
- [`positive_ratio` stratification target may not be achievable if a real
  corpus's class balance is too skewed for 300/workload] → `sample_workload`
  raises `CorpusSourceError` naming the shortfall rather than silently
  under-sampling; the author decides whether to fall back to another corpus
  (Risk 1 fallbacks: MS MARCO, CodeSearchNet, DailyDialog) or adjust
  `positive_ratio`.
- [Fixture data mistaken for real research data downstream] → `source_corpus
  = "synthetic-fixture"`, `metadata.provenance`, `data/README.md`'s explicit
  warning, and this proposal's scope table all flag it; LEV-4's own tests
  should assert they are not silently treating fixture output as
  dissertation-grade results.
- [Blind annotation progress file and dataset file drift apart if a session
  crashes between "recorded in progress" and "merged back into the dataset
  file"] → by design (D5) this is safe: re-running
  `scripts/annotate_dataset.py` with the same `--progress` path re-merges
  the progress file into a freshly loaded dataset before continuing, so no
  answered pair is lost even if the dataset file was never re-saved after
  the crash.

## Migration Plan

Pure addition; nothing existing changes. No rollback complexity — this
change touches no file outside `levy/dataset/`, `scripts/`, `data/`,
`tests/test_dataset.py`, and doc updates to `README.md`/`CLAUDE.md`.

## Open Questions

- Exact raw file layout for the real Stack Overflow duplicates and ConvAI2-
  derived downloads is not pinned to one canonical dataset release (unlike
  QQP, which has one well-known shape). This is intentionally left to the
  author to confirm against whatever concrete file they download, with the
  adapters documenting the expected shape and failing loudly (naming
  missing columns) rather than guessing — not a blocking question for this
  change, since no real download is being performed here.
