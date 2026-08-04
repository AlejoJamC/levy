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

---

> ### Update 2026-08-04 — data production has run; the kappa criterion is NOT met
>
> `TODO (post data-production)` markers below are kept in place, each answered by
> a dated note beside it.
>
> | | |
> |---|---|
> | Corpora acquired | all three, checksums pinned in `corpora.json` |
> | Sample | 900 pairs, seed 42, `positive_ratio` 0.5, 150/150 per class per workload |
> | Rehydration round-trip | verified byte-identical on the real 900 |
> | Blind re-annotation | 900 / 900 |
> | **Cohen's kappa (overall)** | **κ = 0.3267 — below the frozen κ > 0.7 threshold** |
>
> A research-scope finding, escalated per `docs/Project_Proposal.md` Risk 1, not
> coded around. Breakdown and contingency options: §4.
>
> Superseded by this run and struck through in place: §7's expectation that the
> real dataset replaces `data/ground_truth.{csv,json}`, and §8's last bullet.

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

> **Update 2026-08-04 — answered.** `positive_ratio = 0.5`, achieved exactly.
> Counts as sampled and as re-annotated:
>
> | Workload | Corpus | n | `original_label == 1` | `author_label == 1` | Median query length (chars) |
> |---|---|---:|---:|---:|---:|
> | faq | quora-qqp | 300 | 150 | 163 | 51 |
> | code | sodd | 300 | 150 | 34 | 600 |
> | chat | twitter-pit2015 | 300 | 150 | 68 | 41 |
> | **total** | | **900** | **450** | **265** | |
>
> The sample is balanced by construction on `original_label`. It is *not*
> balanced on `author_label`, and that asymmetry is the kappa result of §4: the
> author judged far fewer pairs to be same-intent than the source corpora did,
> overwhelmingly in the `code` workload.

**Source corpora (primary):**

Canonical URLs, snapshot identifiers, expected filenames, pinned SHA-256
checksums and citations are recorded machine-readably in
[`corpora.json`](corpora.json), which the acquisition, validation and
rehydration code all read. The table below is the human-readable summary, not a
second source of truth.

| Workload | Corpus | Licence | Positive label | Notes |
|---|---|---|---|---|
| FAQ | Quora Question Pairs (QQP) | Quora Terms of Service — non-commercial research use, **no redistribution grant** | `is_duplicate == 1` | Binary duplicate-question label from Quora's original moderation process. Acquisition requires accepting terms on the hosting platform. |
| Code | SODD — Stack Overflow Duplicity Dataset, released with MQDD (Pasek et al., RANLP 2023) | CC BY-NC-SA 4.0 | `label == 0` (`duplicates`) | Stack Overflow duplicate closures, derived from the archive.org SO dump of June 2020. Native classes: 0 duplicates, 1 similar (fulltext), 2 similar (tags), 3 different, 4 accepted answer. Negatives are class 3 by default; classes 1/2 are available as adversarially hard negatives behind an explicit option. Posts are HTML and are normalised deterministically (`levy/dataset/normalize.py`). |
| Chat | Twitter PIT-2015 (SemEval-2015 Task 1, Xu et al.) | SemEval-2015 shared-task release, research use | 3–5 of 5 crowdsourced yes-votes | Binary paraphrase judgment crowdsourced via Amazon Mechanical Turk. Only the train and dev splits are used: the test split carries a single expert 0–5 grade instead of a vote count, and the adapter rejects it rather than coercing one scale onto the other. Pairs at 2-of-5 votes are "debatable" by the task's own guidance and are excluded. |

**Deviations from the frozen documents.** Recorded here as decisions with their
rationale, per the project rule that a conflict with `docs/Project_Proposal.md`
or `docs/Specification_and_Design_Report.md` is surfaced rather than silently
resolved. Supervisor sign-off is tracked separately (LEV-11).

1. **Code workload corpus: "Stack Overflow duplicate questions" → SODD.** The
   same underlying source (Stack Overflow's community duplicate-closure
   process), taken from a published, pre-processed release rather than from a
   fresh Stack Exchange dump, given the current dump access situation. The
   label semantics the frozen design relies on are unchanged.
2. **Chat workload corpus: ConvAI2 → Twitter PIT-2015.** ConvAI2 ships
   multi-turn dialogues, not pair-level human same-intent labels. It therefore
   cannot supply the *original human label* that the Cohen's kappa criterion
   (§4) compares the author's blind re-annotation against — deriving those
   labels would mean the author annotating both sides, which is not an
   independent comparison. PIT-2015 supplies a genuine crowdsourced pair label.
3. **D2 release format: query text → identifiers plus labels plus a
   rehydration script.** The frozen design calls for the dataset to be released
   in CSV and JSON carrying the pairs. Quora Question Pairs grants no
   redistribution right and SODD is non-commercial share-alike, so publishing
   the text from an Apache-2.0 repository is not available. This is the same
   approach Google takes for PAWS-QQP. §6 describes the mechanism; the ±5%
   replication criterion is preserved through checksums of the raw inputs.

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

0. `scripts/fetch_corpora.py` acquires the raw corpora into `data/raw/` and
   verifies each file against the checksum pinned in `corpora.json`. It is the
   only step that touches the network; everything after it is offline.
1. A `CorpusSource` adapter reads a local raw corpus file and yields
   `RawCandidatePair` records, mapping the corpus's native label onto the
   binary study label via the mapping the adapter declares. Values outside the
   declared domain are reported, never coerced; in-domain values in neither
   class (a debatable PIT pair, a SODD accepted-answer row) are excluded.
2. `levy/dataset/validation.py` runs one pre-flight pass over all three
   workloads, reporting every problem at once — file presence, checksum
   agreement, required columns, positive/negative pool sufficiency, label
   domains, and absence of corpus overlap between workloads. Sampling does not
   start, and nothing is written, unless that pass is clean.
3. Candidates are split into positive/negative pools, each sorted by
   `source_pair_id` for determinism, then sampled via `random.Random(seed)`
   to hit the target `n` and `positive_ratio` for that workload.
4. The chosen pairs are shuffled (same `random.Random(seed)`) and assigned
   sequential `pair_id`s (`<workload>-0000`, `<workload>-0001`, ...).

**Adapter options that affect content** are explicit, defaulted, and recorded
in the run manifest rather than left implicit: SODD's `hard_negatives` (default
off), the shards and splits read per corpus, and the HTML normalisation rule.

**Production runs refuse synthetic data.** `scripts/sample_dataset.py
--require-real` turns the offline `MockCorpusSource` fallback into a hard error
naming the missing corpus, so a released dataset cannot silently contain pairs
whose `source_corpus` is `mock`.

**Seed:** `42` (default; see `scripts/sample_dataset.py --seed`). Same seed
+ same raw corpus file + same `n` + same `positive_ratio` reproduces an
identical sample — this is unit-tested (`tests/test_dataset.py`,
`test_same_seed_is_deterministic`).

**Sample size:** 300 pairs per workload (900 total), per D2.

**Traceability:** every `QueryPair` retains `source_corpus` (which corpus)
and `source_pair_id` (the pair's id within that corpus), so any sampled pair
can be traced back to its origin for audit.

**Run manifest.** `data/ground_truth.ids.meta.json`, written by the sampling
run and published alongside the dataset, records the seed, `positive_ratio`,
every content-affecting adapter option, the corpus snapshots and the SHA-256 of
each raw input file, plus tool versions. It contains no query text, so a third
party can prove they hold byte-identical inputs without anyone redistributing
corpus text.

`TODO (post data-production): record the exact seed, positive_ratio, raw
corpus file versions/checksums, and sampling date actually used for the
released 900-pair dataset.`

> **Update 2026-08-04 — answered.** Seed `42`, `positive_ratio` 0.5,
> `n_per_workload` 300, sampled 2026-08-04. Input checksums, corpus snapshots,
> adapter options and tool versions live in `ground_truth.ids.meta.json` and
> `corpora.json` — not copied here, so the two cannot diverge.
>
> Round-trip verified on the real data: rehydrating from the published
> `ground_truth.ids.csv` reproduced every field of all 900 pairs exactly. That is
> the property the identifiers-only release model rests on (§6).

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

> **Update 2026-08-04 — answered. The criterion is NOT met.**
>
> The command above is superseded: ~~`--dataset data/ground_truth.json`~~ points
> at the 15 synthetic fixture pairs. The real dataset is the rehydrated working
> file, which is gitignored and never committed (§6):
>
> ```bash
> python scripts/compute_kappa.py --dataset data/ground_truth.full.json --strict
> ```
>
> Result over the full 900, `original_label` vs `author_label`:
>
> | Scope | n | κ | Observed agreement | Expected agreement |
> |---|---:|---:|---:|---:|
> | **overall** | 900 | **0.3267** | 0.6633 | 0.5 |
> | faq (quora-qqp) | 300 | 0.5267 | 0.7633 | 0.5 |
> | code (sodd) | 300 | 0.2267 | 0.6133 | 0.5 |
> | chat (twitter-pit2015) | 300 | 0.2267 | 0.6133 | 0.5 |
>
> Overall confusion (`original` × `author`): TP 206, FP 59, FN 244, TN 391.
>
> The disagreement is one-directional — the author judged fewer pairs same-intent
> than the corpora did, in every workload:
>
> | Workload | corpus dup / author not | corpus not / author dup |
> |---|---:|---:|
> | faq | 29 | 42 |
> | code | **116 of 150** | **0** |
> | chat | 99 | 17 |
>
> `code` is the finding. SODD's positive class means "closed as a duplicate on
> Stack Overflow" — a judgment about whether one thread's answers resolve
> another's. The protocol asks the narrower question of whether the *same cached
> answer* would satisfy the second query. So κ measures construct alignment
> between each corpus's label and the study's, not annotator reliability.
>
> **Contingency (supervisor's call), in order of least disruption to the frozen
> design:**
>
> 1. Report κ as a finding and proceed with `author_label` as ground truth —
>    already what `ground_truth_label()` returns. Recommended: the primary
>    question (does model choice affect FPR) is unaffected by which of two
>    defensible label sets is used, provided the choice is declared.
> 2. Invoke Proposal Risk 1 substitution for `code`/`chat` (CodeSearchNet,
>    DailyDialog). Costs a second 900-pair annotation pass and need not raise κ.
> 3. Restrict the SODD positive class — changes the sampling protocol above.
>
> Not done, deliberately: lowering the threshold, re-annotating non-blind,
> re-sampling until κ clears, or reverting `ground_truth_label()`.
>
> Option 1 is the author's recommendation, on the grounds that the κ gap is
> informative about the corpora rather than about the annotation, and that §8's
> first limitation already bounds how κ may be read.

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

## 6. Distribution — identifiers and labels, not text

The dataset is released alongside the Levy code repository under Apache-2.0
(the code licence — see root `LICENSE`). The **query text is not
redistributed**: Quora Question Pairs grants no redistribution right and SODD
is CC BY-NC-SA 4.0, and both terms are incompatible with an Apache-2.0 public
repository. Treatment is uniform across all three corpora regardless of each
licence's individual terms.

What is published:

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

The reconstruction is **lossless**: a dataset sampled, reduced to identifiers,
and rehydrated is byte-identical to the dataset originally sampled. Ordering is
taken from the identifiers file and the adapter options from the manifest, so
nothing is re-derived and nothing can drift. This is what preserves the ±5%
replication criterion under a distribution model that carries no text; it is
the same approach Google uses for PAWS-QQP.

The two formats have distinct code paths in `levy/dataset/io.py`, and the
identifiers file is deliberately **not** a harness input — pointing
`load_dataset` at it fails with an instruction to rehydrate first, rather than
replaying pairs with absent text. Full-dataset CSV/JSON round-trip equality is
unchanged (`tests/test_dataset.py::TestCsvJsonRoundTrip`).

`scripts/audit_release.sh` enforces the property rather than trusting it: it
fails if any tracked file carries query text attributed to a third-party
corpus, and separately spot-checks tracked files against strings sampled from a
populated `data/raw/`.

## 7. Maintenance

Maintained by the author as part of the capstone repository. ~~Once the real
900-pair dataset is committed, `data/ground_truth.csv` /
`data/ground_truth.json` are treated as a frozen research artifact (like
`docs/Project_Proposal.md`)~~ — subsequent corrections should be additive
(e.g. a documented erratum) rather than silent edits, to preserve
reproducibility of any published results.

> **Update 2026-08-04 — the struck sentence is void; do not act on it.** It
> predates deviation 3 in §2 and the distribution model in §6. The real dataset
> is never committed and `data/ground_truth.{csv,json}` are never replaced by it
> — that would put QQP and CC BY-NC-SA SODD text into a public Apache-2.0 repo.
>
> The frozen artifact is `data/ground_truth.ids.csv` plus
> `data/ground_truth.ids.meta.json`; corrections additive only.
> `data/ground_truth.{csv,json}` remain the 15 synthetic fixtures — not research
> data, not frozen.

## 8. Known limitations

- **Single annotator.** Only one blind re-annotation is performed (by the
  author); inter-annotator agreement is measured against the corpus's
  original label, not against a second independent human annotator. This
  bounds how the kappa result should be interpreted (agreement with the
  original corpus's annotation process, not a full inter-annotator-agreement
  study).
- **Workload-corpus mapping is a design choice, not a guarantee of
  representativeness.** Quora QQP (FAQ), SODD (code), and Twitter PIT-2015
  (chat) are proxies for "FAQ", "code generation", and "conversational chat"
  LLM-API workloads respectively; they are not drawn from actual LLM-API
  traffic logs.
- **The chat corpus is Twitter, not assistant dialogue.** PIT-2015 supplies a
  real crowdsourced paraphrase label, which ConvAI2 could not (§2, deviation
  2), but tweets on trending topics are short, informal and topical in ways
  conversational LLM traffic is not. Read the chat workload's results as
  "short informal paraphrase", not as "chat assistant traffic".
- **SODD text is normalised, and the normalisation is part of the artifact.**
  SODD posts are HTML containing code; they are reduced to plain text by a
  fixed stdlib rule recorded in the run manifest. The rule keeps code-block
  content, but it is a simplification of the original posts, and any
  imperfection in it is reproducible rather than divergent.
- **The released dataset requires the reader to acquire the corpora.** Because
  no query text is redistributed (§6), reproducing the working dataset depends
  on the upstream corpora remaining available in the pinned snapshot. A
  checksum mismatch is reported loudly, but an upstream that disappears cannot
  be recovered from this repository.
- ~~**Small fixture data ships in its place today.** See `data/README.md` —
  `data/ground_truth.{csv,json}` currently contain only 15 synthetic
  placeholder pairs, not the real 900.~~
  **Update 2026-08-04:** struck because "placeholder" describes a temporary state
  that no longer exists. The fixtures are permanent — the offline default — and
  the real 900 ship beside them, not instead of them (§7). Still true: those two
  files hold 15 synthetic pairs.

`TODO (post data-production): add any limitations discovered while actually
sampling/annotating (e.g. corpora language distribution, average query
length per workload, prevalence of near-duplicate-but-not-duplicate pairs).`

> **Update 2026-08-04 — answered. Limitations observed during the real run:**
>
> - **The corpora disagree with the study's own label definition, unevenly.**
>   The headline limitation, quantified in §4: κ = 0.3267 overall, and in the
>   `code` workload the author rejected 116 of 150 corpus-labelled duplicates.
>   Any result broken down by workload must be read with this in view — the three
>   workloads do not carry equally trustworthy ground truth.
> - **Query length varies by more than an order of magnitude across workloads.**
>   Median `query` length is 51 characters (faq), 41 (chat) and 600 (code), with
>   the longest `code` post at 18,938 characters. Both study embedding models
>   truncate at their own token limits, so the `code` workload is systematically
>   more truncated than the other two. Some `code` differences may therefore be
>   invisible to the encoder — a confound between workload and truncation that
>   the frozen design does not control for, and that plausibly inflates false
>   positives for `code` independently of embedding-model choice.
> - **`chat` is short, informal and topic-clustered.** PIT-2015 pairs are drawn
>   from trending-topic tweets, so many negatives share heavy lexical overlap
>   while differing in intent — an adversarially hard negative pool by accident
>   rather than by design. This is the near-duplicate-but-not-duplicate
>   prevalence the original marker asked about, and it is concentrated in `chat`.
> - **Language distribution is not verified.** All three corpora are
>   predominantly English but none is language-filtered, and no language
>   detection was run over the sample. Non-English pairs may be present at low
>   frequency and are not excluded or counted.
> - **`.gitignore` alone did not protect the licensed text.** During this run a
>   `git stash --all` placed a hand-made backup of the annotated 900 (full query
>   text) into `refs/stash`, where the then-current `*.csv` deny-by-default rule
>   could not reach it, because stashing bypasses ignore rules. It was caught by
>   `scripts/audit_release.sh` check 4 — which scans `git log --all`, including
>   `refs/stash` — and cleared before any push; no remote ref ever contained it.
>   The rules were then extended to `.json`, `.parquet` and `.data` under `data/`
>   (previously only `.csv`/`.tsv` were covered, leaving the JSON half of the same
>   dataset protected by filename alone). Recorded here because the mitigation
>   that worked was the audit, not the ignore file, and the release process should
>   continue to treat the audit as the gate.
