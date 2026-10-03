# Datasheet: Levy Ground-Truth Dataset (D2)

Follows the spirit of Gebru et al. (2021), "Datasheets for Datasets", adapted to
this project's scope. It describes the released 900-pair dataset: where the
pairs come from, how they were sampled and labelled, how well the labels agree
with their source corpora, and what is and is not distributed. `data/README.md`
lists the files.

| | |
|---|---|
| Pairs | 900 — 300 faq, 300 code, 300 chat; `positive_ratio` 0.5, 150/150 per class per workload |
| Corpora | Quora Question Pairs, SODD, Twitter PIT-2015; checksums pinned in `corpora.json` |
| Blind re-annotation | 900 / 900 |
| **Cohen's κ (overall)** | **0.5000** — below the κ > 0.7 criterion (faq 0.5267, code 0.4200, chat 0.5533); a finding, see §4 |
| Distribution | Identifiers and labels only; query text is rebuilt locally (§6) |

## 1. Motivation

**For what purpose was the dataset created?**

To test empirically whether embedding-model choice (`all-MiniLM-L6-v2` vs.
ModernBERT) meaningfully affects false-positive rates in semantic caching for
LLM APIs, across three workload types (FAQ, code generation, conversational
chat). The dataset supplies ground-truth duplicate / non-duplicate labels so
that a cache's hit/miss decision on `query_2` (after `query_1` has populated the
cache) can be scored as TP/FP/TN/FN.

**Who created it?**

John Alejandro Mantilla Celis, MSc Artificial Intelligence, University of
Liverpool, as part of the Levy capstone project. Query text comes from
pre-existing public corpora (§2); the author does not write query text, only
samples and re-annotates it.

## 2. Composition

**What do the instances represent?**

Each instance is a `QueryPair` (`levy/dataset/schema.py`): two natural-language
queries (`query_1`, `query_2`) from the same workload, an `original_label`
(1 = duplicate / same intent, per the source corpus's original human
annotators) and an `author_label` (the same binary judgement from the author's
independent, blind re-annotation).

**How many instances, and of what workload?**

900 in total, 300 per workload, stratified 50/50 on `original_label`.

| Workload | Corpus | n | `original_label == 1` | `author_label == 1` | Median query length (chars) |
|---|---|---:|---:|---:|---:|
| faq | quora-qqp | 300 | 150 | 163 | 51 |
| code | sodd | 300 | 150 | 67 | 600 |
| chat | twitter-pit2015 | 300 | 150 | 143 | 41 |
| **total** | | **900** | **450** | **373** | |

The sample is balanced by construction on `original_label`, not on
`author_label`; that asymmetry is the κ result of §4.

**Source corpora**

Canonical URLs, snapshot identifiers, expected filenames, pinned SHA-256
checksums and citations are recorded machine-readably in
[`corpora.json`](corpora.json), which the acquisition, validation and
rehydration code all read. The table below is the human-readable summary, not a
second source of truth.

| Workload | Corpus | Licence | Positive label | Notes |
|---|---|---|---|---|
| FAQ | Quora Question Pairs (QQP) | Quora Terms of Service — non-commercial research use, **no redistribution grant** | `is_duplicate == 1` | Binary duplicate-question label from Quora's moderation process. Acquisition requires accepting terms on the hosting platform. |
| Code | SODD — Stack Overflow Duplicity Dataset, released with MQDD (Pasek et al., RANLP 2023) | CC BY-NC-SA 4.0 | `label == 0` (`duplicates`) | Stack Overflow duplicate closures, derived from the archive.org SO dump of June 2020. Native classes: 0 duplicates, 1 similar (fulltext), 2 similar (tags), 3 different, 4 accepted answer. Negatives are class 3 by default; classes 1/2 are available as adversarially hard negatives behind an explicit option. Posts are HTML and are normalised deterministically (`levy/dataset/normalize.py`). |
| Chat | Twitter PIT-2015 (SemEval-2015 Task 1, Xu et al.) | SemEval-2015 shared-task release, research use | 3–5 of 5 crowdsourced yes-votes | Binary paraphrase judgement crowdsourced via Amazon Mechanical Turk. Only the train and dev splits are used: the test split carries a single expert 0–5 grade instead of a vote count, and the adapter rejects it rather than coercing one scale onto the other. Pairs at 2-of-5 votes are "debatable" by the task's own guidance and are excluded. |

**Corpus choices**

1. **Code: SODD rather than a raw Stack Overflow dump.** The same underlying
   source (Stack Overflow's community duplicate-closure process), taken from a
   published, pre-processed release rather than a fresh Stack Exchange dump.
2. **Chat: Twitter PIT-2015 rather than ConvAI2.** ConvAI2 ships multi-turn
   dialogues, not pair-level human same-intent labels, so it cannot supply the
   *original human label* that the Cohen's kappa comparison (§4) needs; deriving
   one would mean the author annotating both sides, which is not an independent
   comparison. PIT-2015 supplies a genuine crowdsourced pair label.
3. **Release format: identifiers plus labels plus a rehydration script, not
   query text.** Quora Question Pairs grants no redistribution right and SODD is
   non-commercial share-alike, so publishing the text from an Apache-2.0
   repository is not available. The same approach is used for PAWS-QQP. §6
   describes the mechanism; replication is preserved through checksums of the
   raw inputs.
4. **Real-model responses: FAQ only.** The latency measurement ("cache lookup
   overhead vs LLM call savings") needs real provider responses to measure what
   a hit avoids. They were obtained for the FAQ workload only; chat and code
   remain mock-populated. FAQ is the only workload where the cache measurably
   operates — its best cell reaches a 24.0 % hit rate, against 2.3 % (chat) and
   2.0 % (code) — so on the other two a real-response run would characterise a
   path taken fewer than once in forty lookups, at full price. The measured
   saving per hit is therefore a **FAQ figure produced by one named model**,
   recorded with that model identifier in `release/latency/llm_calls.json` and
   `latency_meta.json`; provider latency does not replicate, while the
   lookup-overhead half is offline and does. No pair, label or identifier in the
   dataset is affected, and the response corpus is not part of it.

No fallback corpus was used. (MS MARCO, CodeSearchNet and DailyDialog were the
candidate substitutes had a primary corpus proved unusable.)

**Does the dataset contain personal or sensitive data?**

No. The source corpora are public NLP benchmark releases already stripped of
direct identifiers by their publishers; no additional personal data is
collected, and query text is not linked to any individual's identity.

## 3. Collection / sampling process

`levy/dataset/sampling.py` implements seeded, stratified sampling:

0. `scripts/fetch_corpora.py` acquires the raw corpora into `data/raw/` and
   verifies each file against the checksum pinned in `corpora.json`. It is the
   only step that touches the network; everything after it is offline.
1. A `CorpusSource` adapter reads a local raw corpus file and yields candidate
   pairs, mapping the corpus's native label onto the binary study label via the
   mapping the adapter declares. Values outside the declared domain are
   reported, never coerced; in-domain values in neither class (a debatable PIT
   pair, a SODD accepted-answer row) are excluded.
2. `levy/dataset/validation.py` runs one pre-flight pass over all three
   workloads, reporting every problem at once — file presence, checksum
   agreement, required columns, positive/negative pool sufficiency, label
   domains, and absence of corpus overlap between workloads. Sampling does not
   start, and nothing is written, unless that pass is clean.
3. Candidates are split into positive/negative pools, each sorted by
   `source_pair_id` for determinism, then sampled via `random.Random(seed)` to
   hit the target `n` and `positive_ratio`.
4. The chosen pairs are shuffled (same generator) and assigned sequential
   `pair_id`s (`<workload>-0000`, `<workload>-0001`, …).

Adapter options that affect content — SODD's `hard_negatives` (default off), the
shards and splits read per corpus, and the HTML normalisation rule — are
explicit, defaulted, and recorded in the run manifest.

**Production runs refuse synthetic data.** `scripts/sample_dataset.py
--require-real` turns the offline `MockCorpusSource` fallback into a hard error
naming the missing corpus, so the released dataset cannot silently contain pairs
whose `source_corpus` is `mock`.

Same seed + same raw corpus file + same `n` + same `positive_ratio` reproduces
an identical sample; this is unit-tested (`tests/test_dataset.py`,
`test_same_seed_is_deterministic`).

**Traceability.** Every `QueryPair` retains `source_corpus` and
`source_pair_id`, so any sampled pair can be traced back to its origin.

**Run manifest.** `data/ground_truth.ids.meta.json`, written by the sampling run
and published with the dataset, records the seed, `positive_ratio`, every
content-affecting adapter option, the corpus snapshots, the SHA-256 of each raw
input file and tool versions. It contains no query text, so a third party can
prove they hold byte-identical inputs without anyone redistributing corpus text.
Rehydrating from the published `ground_truth.ids.csv` reproduces every field of
all 900 pairs exactly; that property is what the identifiers-only release rests
on (§6).

### Per-workload sampling

Each workload is drawn from its own corpus, with its own candidate pool and its
own seeded stream, so there is no shared quantity for a single seed to
coordinate. Each workload's seed and sampling date are recorded separately in
the manifest. A workload may be re-sampled on its own; the other two, and their
`author_label`s, are unaffected. Corpus snapshots are checksum-pinned, so the
sampling date does not change the pool a workload is drawn from. Procedure:
[`../docs/DATA_PRODUCTION.md`](../docs/DATA_PRODUCTION.md).

| Workload | Corpus | Seed | Drawn (UTC) | n |
|---|---|---:|---|---:|
| faq | quora-qqp | 42 | 2026-08-04 | 300 |
| code | sodd | 8484 | 2026-08-07 11:00 | 300 |
| chat | twitter-pit2015 | 4242 | 2026-08-06 21:19 | 300 |

**Re-draws.** The dataset was first published with all three workloads at seed
42, with overall κ = 0.3267 (faq 0.5267, code 0.2267, chat 0.2267). `chat`
(2026-08-06) and then `code` (2026-08-07) were re-drawn at the seeds above and
re-annotated blind; the new pairs are disjoint from the ones they replaced. The
published dataset reflects the current draws, so those two workloads' 300 pairs
differ from the first publication; faq is unchanged.

## 4. Labelling — blind re-annotation

**Label definitions**

- `original_label`: 1 if the source corpus's original human annotators judged
  `query_1` and `query_2` to be duplicates / the same question intent; 0
  otherwise.
- `author_label`: 1 if the author, in a **blind** re-annotation session
  (`levy/dataset/annotation.BlindAnnotationSession` /
  `scripts/annotate_dataset.py`), independently judged the two queries to
  express the same question intent; 0 otherwise. `None` until annotated.

**Blind annotation protocol**

- The annotator is shown only `query_1` and `query_2` — never `original_label`,
  `source_corpus` or `source_pair_id`.
- Guidance: judge whether a user who asked `query_1` and got an answer would
  find that answer acceptable for `query_2` as well — "same question intent",
  not "textually similar". Very different wording with the same intent is a
  match (1); similar wording with materially different intent (different
  constraints, different sub-topic) is not (0).
- Presentation order is workload blocks `faq, chat, code` (code last, the
  longest posts), shuffled within each block under a recorded order seed. The
  resolved order is persisted and reused on resume.
- The session is resumable: progress is written after every single answer, so
  an interrupted session loses no completed work. A sitting can be ended cleanly
  after N labelled pairs (`--session-limit`).
- Existing `author_label`s are never overwritten unless `--overwrite` is passed.

**Cohen's kappa**

Computed by `levy/dataset/kappa.cohen_kappa` / `scripts/compute_kappa.py` over
the full 900 pairs, comparing `original_label` and `author_label`:

```bash
python scripts/compute_kappa.py --dataset data/ground_truth.full.json --strict
```

| Scope | n | κ | Observed agreement | Expected agreement |
|---|---:|---:|---:|---:|
| **overall** | 900 | **0.5000** | 0.7500 | 0.5 |
| faq (quora-qqp) | 300 | 0.5267 | 0.7633 | 0.5 |
| code (sodd) | 300 | **0.4200** | 0.7100 | 0.5 |
| chat (twitter-pit2015) | 300 | **0.5533** | 0.7767 | 0.5 |

Overall confusion (`original` × `author`): TP 299, FP 74, FN 151, TN 376.

| Workload | corpus dup / author not | corpus not / author dup |
|---|---:|---:|
| faq | 29 | 42 |
| code | 85 of 150 | 2 |
| chat | 37 | 30 |

κ = 0.5000 does not meet the κ > 0.7 criterion. The disagreement is
concentrated in `code`: SODD's positive class means "closed as a duplicate on
Stack Overflow", a judgement about whether one thread's answers resolve
another's, while the protocol asks the narrower question of whether the *same
cached answer* would satisfy the second query. So κ measures construct
alignment between each corpus's label and the study's, not annotator
reliability.

This is reported as a finding. `ground_truth_label()` returns the author's blind
label where set; the κ threshold was not lowered, nothing was re-annotated
non-blind, and the labels were not changed to raise agreement. Alternatives
considered and not taken: substituting fallback corpora for `code` / `chat`
(costs a second 900-pair annotation pass and need not raise κ), and restricting
the SODD positive class (changes the sampling protocol above).

## 5. Uses

**Intended use:** input to the Levy experiment harness. For each pair,
`query_1` is submitted to an empty cache (always a miss, populates the cache),
then `query_2` is submitted and the cache's hit/miss decision is compared
against `QueryPair.ground_truth_label()` (author label if present, else
original label) to accumulate TP/FP/TN/FN and derive precision, recall, F0.5,
false-positive rate and hit rate per configuration.

**Should not be used for:** training or fine-tuning embedding or LLM models (it
is a small, workload-stratified evaluation set, not a training corpus); or any
purpose requiring the underlying corpora's licences to be waived —
redistribution must respect each source corpus's licence (§2).

## 6. Distribution — identifiers and labels, not text

The dataset is released with the Levy code under Apache-2.0 (see root
`LICENSE`). The **query text is not redistributed**: Quora Question Pairs grants
no redistribution right and SODD is CC BY-NC-SA 4.0, both incompatible with an
Apache-2.0 public repository. Treatment is uniform across all three corpora
regardless of each licence's individual terms.

| File | Contents | Tracked |
|---|---|---|
| `ground_truth.ids.csv` | `pair_id`, `workload`, `source_corpus`, `source_pair_id`, `original_label`, `author_label`, `metadata` — **no query text** | yes |
| `ground_truth.ids.meta.json` | the run manifest of §3 | yes |
| `corpora.json` | per-corpus URL, snapshot, licence, filenames, checksums, citation | yes |
| `ground_truth.full.{csv,json}` | the reconstructed working dataset, query text included | **no** — gitignored |
| `ground_truth.{csv,json}` | 15 synthetic fixture pairs (see `README.md`) | yes |

To obtain the working dataset, a reader acquires the corpora themselves and
rehydrates:

```bash
python scripts/fetch_corpora.py
python scripts/rehydrate_dataset.py
```

The reconstruction is **lossless**: a dataset sampled, reduced to identifiers
and rehydrated is byte-identical to the dataset originally sampled. Ordering is
taken from the identifiers file and the adapter options from the manifest, so
nothing is re-derived and nothing can drift. This is what preserves the ±5 %
replication criterion under a distribution model that carries no text.

The two formats have distinct code paths in `levy/dataset/io.py`, and the
identifiers file is deliberately **not** a harness input: pointing
`load_dataset` at it fails with an instruction to rehydrate first, rather than
replaying pairs with absent text.

`scripts/audit_release.sh` enforces the property rather than trusting it: it
fails if any tracked file carries query text attributed to a third-party corpus,
scans all of git history including stashes, and spot-checks tracked files
against strings sampled from a populated `data/raw/`.

## 7. Maintenance

Maintained by the author as part of the capstone repository. The released
artefact is `data/ground_truth.ids.csv` plus `data/ground_truth.ids.meta.json`;
corrections are additive (a documented erratum) rather than silent edits, to
preserve reproducibility of published results. `data/ground_truth.{csv,json}`
remain the 15 synthetic fixtures and are never replaced by the real dataset,
which would put licensed text into a public repository.

## 8. Known limitations

- **Single annotator.** Only one blind re-annotation was performed, by the
  author. Agreement is measured against the corpus's original label, not against
  a second independent human annotator, so κ reflects agreement with the
  original corpus's annotation process, not a full inter-annotator study.
- **The corpora disagree with the study's own label definition, unevenly.**
  κ = 0.5000 overall (§4), and in `code` the author rejected 85 of 150
  corpus-labelled duplicates. The three workloads do not carry equally
  trustworthy ground truth, and the `code` results need the caveat most (κ
  0.4200, against faq 0.5267 and chat 0.5533).
- **Workload–corpus mapping is a proxy.** Quora QQP (FAQ), SODD (code) and
  Twitter PIT-2015 (chat) stand in for FAQ, code-generation and conversational
  LLM-API workloads; they are not drawn from actual LLM-API traffic.
- **The chat corpus is Twitter, not assistant dialogue.** Tweets on trending
  topics are short, informal and topical in ways conversational LLM traffic is
  not. PIT-2015 pairs share heavy lexical overlap across negatives while
  differing in intent — a hard negative pool by accident rather than by design.
  Read chat results as "short informal paraphrase".
- **Query length varies by more than an order of magnitude.** Median length is
  51 characters (faq), 41 (chat) and 600 (code), with the longest `code` post at
  18,938 characters. Both embedding models truncate at their own token limits,
  so `code` is systematically more truncated, which confounds workload with
  truncation and plausibly inflates false positives for `code` independently of
  the embedding model.
- **SODD text is normalised, and the normalisation is part of the artefact.**
  Posts are HTML containing code; they are reduced to plain text by a fixed
  stdlib rule recorded in the run manifest. The rule keeps code-block content
  but is a simplification of the original posts; any imperfection in it is
  reproducible rather than divergent.
- **Language distribution is not verified.** All three corpora are
  predominantly English but none is language-filtered, and no language
  detection was run over the sample.
- **The released dataset requires the reader to acquire the corpora.** Because
  no query text is redistributed (§6), reproducing the working dataset depends
  on the upstream corpora remaining available in the pinned snapshot. A checksum
  mismatch is reported loudly, but an upstream that disappears cannot be
  recovered from this repository.
