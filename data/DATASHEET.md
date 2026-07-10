# Datasheet: Levy Ground-Truth Dataset (D2)

Follows the spirit of Gebru et al. (2021), "Datasheets for Datasets,"
adapted to this capstone's scope. This datasheet describes the **intended**
900-pair dataset defined by the frozen `docs/Specification_and_Design_Report.md`
(§A "Data required", §C Deliverable D2). Sections describing the sampling
protocol, label definitions, and licences are filled in now, because they are
fixed by the frozen research design. Sections that depend on actually running
the sampling and annotation (final counts, final kappa, final file hashes)
are marked `TODO (post data-production)` — see `data/README.md` for what
currently ships in `data/ground_truth.{csv,json}` instead (15 synthetic
fixture pairs, not real data).

## 1. Motivation

**For what purpose was the dataset created?**

To empirically test whether embedding-model choice (`all-MiniLM-L6-v2` vs.
ModernBERT) meaningfully affects false-positive rates in semantic caching for
LLM APIs, across three workload types (FAQ, code generation, conversational
chat). The dataset supplies ground-truth duplicate/non-duplicate labels so
that a cache's hit/miss decision on `query_2` (after `query_1` has populated
the cache) can be scored as TP/FP/TN/FN per Algorithm 2 of the S&D Report.

**Who created it?**

John Alejandro Mantilla Celis, MSc Artificial Intelligence, University of
Liverpool, as part of the Levy capstone project. Query text is drawn from
pre-existing public corpora (see §2); the author does not author new query
text, only samples and re-annotates it.

## 2. Composition

**What do the instances represent?**

Each instance is a `QueryPair` (see `levy/dataset/schema.py`): two natural-
language queries (`query_1`, `query_2`) from the same workload, an
`original_label` (1 = duplicate/same-intent, 0 = not, per the source
corpus's original human annotators), and an `author_label` (the same binary
judgment from the author's independent, blind re-annotation).

**How many instances, and of what workload?**

900 total: 300 FAQ, 300 code, 300 chat. Each workload's 300 pairs are
stratified — target 50/50 duplicate/non-duplicate — from its source corpus
(see `positive_ratio` in `levy/dataset/sampling.sample_workload`; the actual
ratio used for the released dataset will be recorded here:
`TODO (post data-production): record actual positive_ratio and per-workload counts`).

**Source corpora (primary):**

| Workload | Corpus | Licence | Notes |
|---|---|---|---|
| FAQ | Quora Question Pairs (QQP) | Non-commercial research use, publicly released by Quora / distributed via Kaggle & GLUE | Binary duplicate-question label from Quora's original moderation process |
| Code | Stack Overflow duplicate questions | CC BY-SA 4.0 (Stack Exchange Data Dump terms) | Binary duplicate label from Stack Overflow's community duplicate-closure process |
| Chat | ConvAI2 (PersonaChat) — derived same-intent pairs | Data released for the ConvAI2 NeurIPS 2018 competition, research use | ConvAI2 ships dialogues, not duplicate-intent pairs; a derivation step (pairing utterances, labeling same-intent vs. different-intent) is required before sampling — see `levy/dataset/sampling.ConvAI2Source` docstring |

**Fallback corpora** (per `docs/Project_Proposal.md` Risk 1 — "primary
corpus unavailable or insufficient in size/quality"): if a primary corpus
cannot be used or does not yield enough qualifying pairs for a workload, the
author substitutes:

| Workload | Fallback corpus |
|---|---|
| FAQ | MS MARCO (QA pairs) |
| Code | CodeSearchNet |
| Chat | DailyDialog |

Any fallback substitution actually used will be recorded here:
`TODO (post data-production): note any fallback corpus actually used, and why`.

**Does the dataset contain personal or sensitive data?**

No. Source corpora are public NLP benchmark releases already stripped of
direct identifiers by their original publishers; no additional personal data
is collected, and query text is not linked to any individual's real-world
identity as part of this project.

## 3. Collection / sampling process

**How was the data sampled?**

`levy/dataset/sampling.py` implements seeded, stratified sampling:

1. A `CorpusSource` adapter reads a local raw corpus file (never downloaded
   by this code — the raw file must already exist on disk) and yields
   `RawCandidatePair` records with the corpus's original label.
2. Candidates are split into positive/negative pools, each sorted by
   `source_pair_id` for determinism, then sampled via `random.Random(seed)`
   to hit the target `n` and `positive_ratio` for that workload.
3. The chosen pairs are shuffled (same `random.Random(seed)`) and assigned
   sequential `pair_id`s (`<workload>-0000`, `<workload>-0001`, ...).

**Seed:** `42` (default; see `scripts/sample_dataset.py --seed`). Same seed
+ same raw corpus file + same `n` + same `positive_ratio` reproduces an
identical sample — this is unit-tested (`tests/test_dataset.py`,
`test_same_seed_is_deterministic`).

**Sample size:** 300 pairs per workload (900 total), per D2.

**Traceability:** every `QueryPair` retains `source_corpus` (which corpus)
and `source_pair_id` (the pair's id within that corpus), so any sampled pair
can be traced back to its origin for audit.

`TODO (post data-production): record the exact seed, positive_ratio, raw
corpus file versions/checksums, and sampling date actually used for the
released 900-pair dataset.`

## 4. Preprocessing / labeling — blind re-annotation

**Label definitions:**

- `original_label`: 1 if the source corpus's original human annotators
  judged `query_1` and `query_2` to be duplicates / the same
  question-intent; 0 otherwise.
- `author_label`: 1 if the author, during a **blind** re-annotation session
  (via `levy/dataset/annotation.BlindAnnotationSession` /
  `scripts/annotate_dataset.py`), independently judged `query_1` and
  `query_2` to express the same question-intent; 0 otherwise. `None` until
  annotated.

**Blind annotation protocol:**

- The annotator (the author) is shown only `query_1` and `query_2` — never
  `original_label`, `source_corpus`, or `source_pair_id` (which could hint
  at provenance/label) — implemented in
  `BlindAnnotationSession.run()`.
- Guidance given to the annotator: judge whether a user who asked `query_1`
  and got an answer would find that answer acceptable for `query_2` as well
  — i.e. "same question-intent", not "textually similar." Two questions
  with very different wording but the same intent are a match (1); two
  questions with similar wording but materially different intent (e.g.
  different constraints, different sub-topic) are not (0).
- The session is resumable: progress (`author_label`s recorded so far) is
  written to a progress JSON file after every single answer, so interrupting
  a 900-pair session loses no completed work.
- Existing `author_label`s are never overwritten by a later session unless
  `--overwrite` is passed explicitly.

**Cohen's kappa:**

Computed by `levy/dataset/kappa.cohen_kappa` / `scripts/compute_kappa.py`
over the full 900-pair set, comparing `original_label` vs. `author_label`.
Success threshold (frozen S&D Report): **kappa > 0.7**.

`TODO (post data-production): record the final overall kappa, the per-
workload kappa breakdown, and the confusion matrix, once the author has
completed the blind re-annotation of all 900 pairs. Run:`
```bash
python scripts/compute_kappa.py --dataset data/ground_truth.json --strict
```

## 5. Uses

**Intended use:** input to the Levy experiment harness (LEV-4): for each
pair, `query_1` is submitted to an empty cache (always a miss, populates the
cache), then `query_2` is submitted and the cache's hit/miss decision is
compared against `QueryPair.ground_truth_label()` (author label if present,
else original label) to accumulate TP/FP/TN/FN and derive precision, recall,
F0.5, false-positive rate, and hit rate per the 30-configuration grid.

**Should not be used for:** training or fine-tuning embedding/LLM models
(it is an evaluation set, deliberately small and workload-stratified, not
a training corpus); any purpose requiring the underlying corpora's licences
to be waived (redistribution must respect each source corpus's licence,
§2).

## 6. Distribution

Released publicly alongside the Levy code repository under Apache-2.0 (code
licence — see root `LICENSE`; the query text itself remains subject to its
originating corpus's licence, §2), in both CSV and JSON, with identical
content (`levy/dataset/io.py` guarantees round-trip equality — see
`tests/test_dataset.py::TestCsvJsonRoundTrip`).

## 7. Maintenance

Maintained by the author as part of the capstone repository. Once the real
900-pair dataset is committed, `data/ground_truth.csv` /
`data/ground_truth.json` are treated as a frozen research artifact (like
`docs/Project_Proposal.md`) — subsequent corrections should be additive
(e.g. a documented erratum) rather than silent edits, to preserve
reproducibility of any published results.

## 8. Known limitations

- **Single annotator.** Only one blind re-annotation is performed (by the
  author); inter-annotator agreement is measured against the corpus's
  original label, not against a second independent human annotator. This
  bounds how the kappa result should be interpreted (agreement with the
  original corpus's annotation process, not a full inter-annotator-agreement
  study).
- **Workload-corpus mapping is a design choice, not a guarantee of
  representativeness.** Quora QQP (FAQ), Stack Overflow duplicates (code),
  and ConvAI2-derived pairs (chat) are proxies for "FAQ", "code generation",
  and "conversational chat" LLM-API workloads respectively; they are not
  drawn from actual LLM-API traffic logs.
- **ConvAI2 requires a derivation step.** Unlike QQP and Stack Overflow,
  ConvAI2 does not natively ship duplicate-intent pair labels; the chat
  workload's pairs depend on an author-performed pairing/labeling step
  before `ConvAI2Source` can run (documented in
  `levy/dataset/sampling.py`), which is itself a light annotation act
  distinct from the blind re-annotation in §4.
- **Small fixture data ships in its place today.** See `data/README.md` —
  `data/ground_truth.{csv,json}` currently contain only 15 synthetic
  placeholder pairs, not the real 900.

`TODO (post data-production): add any limitations discovered while actually
sampling/annotating (e.g. corpora language distribution, average query
length per workload, prevalence of near-duplicate-but-not-duplicate pairs).`
