# Levy — D2 Data Production Runbook

**Audience:** the author, once. **Outcome:** the real 900-pair ground-truth
dataset (D2), annotated, with its published identifiers-only artifact.

This is the only document that carries the end-to-end data-production
procedure. It covers exactly one subject — producing D2 — and nothing else:

- It is **not** the reproduction guide. A third party reproducing the study
  runs steps 1–3 and 5 only, and finds them in
  [`REPRODUCTION.md`](REPRODUCTION.md) §"Step 2b".
- It is **not** the evaluation pipeline. Running the experiments and the
  analysis over the finished dataset is `scripts/reproduce.sh`, documented in
  [`REPRODUCTION.md`](REPRODUCTION.md).
- Corpus licences, the sampling protocol, the three recorded deviations from
  the frozen documents, and the dataset's known limitations are in
  [`../data/DATASHEET.md`](../data/DATASHEET.md). This file is procedure only.

Work through the steps in order. ~~Steps 1–5 can be re-run freely; they are
idempotent and deterministic.~~ Step 6 is the long one — 900 judgments — and is
resumable.

> **Update 2026-08-04 — the struck claim is only true before step 6.**
> Steps 1–4 are re-runnable at any time. **Step 5 is not, once step 6 has
> started:** `scripts/rehydrate_dataset.py` writes
> `data/ground_truth.full.{csv,json}` from `ground_truth.ids.csv`, whose
> `author_label` column is empty until step 8 — so re-running step 5 after
> annotating overwrites the working dataset with an unlabelled copy. The labels
> are recoverable (`data/annotation_progress.json` holds all 900 and step 6
> re-applies them on the next run), but the working file is clobbered in the
> meantime. To verify the round-trip after step 6, write somewhere else:
>
> ```bash
> python scripts/rehydrate_dataset.py --out-csv /tmp/rt.csv --out-json /tmp/rt.json
> ```
>
> **Update 2026-08-06 — the clobber is now recoverable, but still avoid it.**
> `scripts/rehydrate_dataset.py` copies an existing
> `data/ground_truth.full.{csv,json}` to `data/backups/ground_truth.full.<UTC>.csv`
> before overwriting it, and aborts without writing if that copy cannot be made.
> Every script that touches a ground-truth artifact does the same
> (`levy/dataset/backup.py`); backups are never deleted and never overwritten.
> That is a safety net, not a reason to re-run step 5 on top of annotated work —
> write somewhere else as above.

---

## Run status — Update 2026-08-04

| Step | State |
|---|---|
| 1–3 acquire, verify, pin | done — all three corpora present, checksums pinned |
| 4 sample 900 | done — seed 42, ratio 0.5, 300/workload |
| 5 round-trip on real data | done — every field of all 900 matched exactly |
| 6 blind re-annotation | done — 900 / 900 |
| 7 Cohen's kappa | done — **κ = 0.3267, below the frozen κ > 0.7 bar**; see [`../data/DATASHEET.md`](../data/DATASHEET.md) §4 for the breakdown and the contingency options. Do not adjust the threshold or re-annotate non-blind. |
| 8 refresh the ids file with your labels | done — verified 900 / 900 rows of `data/ground_truth.ids.csv` carry `author_label`, so the published D2 artifact ships the re-annotation |
| 9 audit, then commit | done — audit passes 8/8 |

---

## Before you start

| | |
|---|---|
| Time | ~30 min for steps 1–5; step 6 is several sittings |
| Needs network | Step 1 and step 2 only |
| Needs an account | Kaggle, for the Quora corpus (step 2A) |
| Disk | a few GB under `data/raw/` |

Every command below assumes the conda environment is active. Claude Code's
shell does not inherit it, so activate it once per terminal session:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy
```

Run everything from the repository root.

---

## Step 1 — Automatic acquisition

```bash
python scripts/fetch_corpora.py
```

Downloads the `chat` corpus (Twitter PIT-2015) into
`data/raw/twitter-pit2015/`, extracting `train.data` and `dev.data` from the
shared-task release archive.

**This command exits with status 1, and that is the expected outcome.** Two of
the three corpora cannot be fetched without a human step, so the run ends by
printing, for each of them, the URL to open, the filenames to save, where to
save them, and the expected SHA-256. Step 2 is that human step.

---

## Step 2 — Your manual downloads

### 2A — Quora Question Pairs (`faq` workload)

Open <https://www.kaggle.com/competitions/quora-question-pairs/data>, sign in,
accept the competition rules, and download `train.csv.zip`. Unzip it anywhere.

The adapter expects the tab-separated form of this file. Kaggle ships CSV with
the right columns (`id, qid1, qid2, question1, question2, is_duplicate`), so
convert it into place — adjust the path at the end to wherever you unzipped:

```bash
python -c "
import csv, sys
src, dst = sys.argv[1], 'data/raw/quora-qqp/train.tsv'
with open(src, newline='', encoding='utf-8') as fin, open(dst, 'w', newline='', encoding='utf-8') as fout:
    w = csv.writer(fout, delimiter='\t')
    for row in csv.reader(fin):
        w.writerow(row)
print('wrote', dst)
" ~/Downloads/train.csv
```

The conversion quotes any field containing a tab or newline, and the adapter
reads it back through the same `csv` module, so question text survives intact.

### 2B — SODD, the Stack Overflow Duplicity Dataset (`code` workload)

Open <https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3>
and download these two files into `data/raw/sodd/`:

```text
SODD_train.parquet.gzip
SODD_dev.parquet.gzip
```

Nothing to convert — the adapter reads gzipped parquet directly. Do not
decompress them.

### Where the files must end up

```text
data/raw/
├── quora-qqp/        train.tsv
├── sodd/             SODD_train.parquet.gzip, SODD_dev.parquet.gzip
└── twitter-pit2015/  train.data, dev.data          (step 1 put these here)
```

Filenames must match exactly; they come from `data/corpora.json`, which the
tooling reads. Contents of `data/raw/` are gitignored and must never be
committed — see [`../data/raw/README.md`](../data/raw/README.md).

---

## Step 3 — Verify, then pin the checksums

```bash
python scripts/fetch_corpora.py
```

Must now exit 0, reporting every file as `present`. If a file is still
`manual`, it is not where the tool expects it — re-check the filename against
the tree above.

```bash
python scripts/fetch_corpora.py --pin
```

Records the SHA-256 of each file you actually hold into `data/corpora.json`.
Do this once, now, before sampling. From this point on a checksum mismatch is
a hard failure, which is what turns an upstream re-release into a loud error
instead of a silent divergence in your results.

`data/corpora.json` is now modified — that change gets committed at step 9.

---

## Step 4 — Sample the 900 pairs

```bash
python scripts/sample_dataset.py --require-real \
    --n-per-workload 300 --seed 42 --positive-ratio 0.5 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv
```

**This produces one dataset of 900 rows, not three files of 300.** The
per-workload split lives in the `workload` column — 300 `faq`, 300 `code`, 300
`chat`, each stratified 150 duplicate / 150 non-duplicate. Every downstream
tool reads the single file.

`--require-real` refuses to substitute synthetic data: if a corpus is missing
the run fails naming it, rather than quietly emitting pairs whose
`source_corpus` is `mock`.

Four files are written:

| File | What | Committed |
|---|---|---|
| `data/ground_truth.full.csv` / `.json` | the working dataset, query text included | no — gitignored |
| `data/ground_truth.ids.csv` | identifiers and labels, no text — the published artifact | yes |
| `data/ground_truth.ids.meta.json` | seed, ratio, adapter options, corpus snapshots, input checksums | yes |

Before sampling, a pre-flight pass validates all three workloads together and
prints their pool sizes. If anything is wrong it reports **every** problem at
once and writes nothing — fix them all, then re-run. Common findings:

| Finding | Meaning |
|---|---|
| `[class-pool] … needs 150 positive pairs, only N available` | that corpus cannot fill the stratum; see the frozen Proposal's Risk 1 fallback path before substituting anything |
| `[checksum] … does not match the pinned` | the file changed since you pinned it |
| `[required-fields] … missing columns` | wrong file, or the QQP conversion in 2A did not run |
| `[label-domain] … outside the declared domain` | an unexpected label value; reported, never coerced |

---

## Step 5 — Confirm the round-trip on the real data

The published artifact carries no query text, so the whole release depends on
rehydration reconstructing the dataset exactly. Prove it on the real 900 —
fixtures already prove it in CI, but this is the artifact you are shipping:

```bash
cp data/ground_truth.full.csv /tmp/sampled-900.csv
python scripts/rehydrate_dataset.py
cmp /tmp/sampled-900.csv data/ground_truth.full.csv && echo "byte-identical"
```

If `cmp` reports a difference, stop and investigate before annotating — the
identifiers file cannot reproduce what you sampled.

---

## Step 6 — Blind re-annotation of all 900 pairs

```bash
python scripts/annotate_dataset.py \
    --dataset data/ground_truth.full.json \
    --progress data/annotation_progress.json \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json
```

You are shown only `query_1` and `query_2` — never the original corpus label,
the corpus name, or the pair id. That blindness is what makes the Cohen's
kappa in step 7 a meaningful agreement measure rather than a restatement of
the corpus's own labels.

**The judgment to make:** would someone who asked `query_1`, and received a
good answer, be satisfied by that same answer for `query_2`? That is
same-question-intent, not textual similarity. Different wording with the same
intent is a match (`1`); similar wording with materially different intent —
different constraints, a different sub-topic — is not (`0`).

Progress is written to the progress file after **every single answer**, so you
can stop at any point with `Ctrl-C` and resume by re-running the identical
command. Already-annotated pairs are never re-asked, and an existing
`author_label` is never overwritten unless you pass `--overwrite`.

> **Update 2026-08-06 — presentation order and session control.**
> ~~Pairs are presented in dataset order.~~ Pairs are now presented **workload
> block by workload block in `faq,chat,code` order, shuffled within each block**
> under `--order-seed` (default `0`). Code goes last because its pairs are Stack
> Overflow posts, by far the longest to read. The shuffle matters for the same
> reason blindness does: in file order a long run of near-duplicates lets
> position stand in for content, and the pair after forty of them is judged
> against the run rather than on its own.
>
> The resolved order is written into the progress file, so an interrupted session
> resumes in the order it started in rather than re-deriving one — including if
> you pass a different `--order-seed` on the way back, which the session tells you
> it is ignoring.
>
> Four flags, all optional:
>
> | Flag | Use |
> |---|---|
> | `--workload chat` | restrict the session to one workload (repeatable) — this is what makes a single-workload re-annotation ask only about that workload |
> | `--workload-order code,faq,chat` | override the block order; an unknown or repeated workload is rejected |
> | `--order-seed 7` | change the within-block shuffle; recorded in the progress file, never left to vary silently |
> | `--session-limit 50` | end the sitting cleanly after 50 labeled pairs, so a long run splits into short sessions without `Ctrl-C`. Skips do not consume the limit |
>
> Add `--out-ids data/ground_truth.ids.csv` to refresh the published artifact in
> the same command — that is step 8, which no longer needs doing by hand.
>
> A 50-pair sitting over a freshly re-sampled chat workload:
>
> ```bash
> python scripts/annotate_dataset.py \
>     --dataset data/ground_truth.full.json \
>     --progress data/annotation_progress.json \
>     --workload chat --session-limit 50 \
>     --out-csv data/ground_truth.full.csv \
>     --out-json data/ground_truth.full.json \
>     --out-ids data/ground_truth.ids.csv
> ```

---

## Step 7 — Cohen's kappa

```bash
python scripts/compute_kappa.py --dataset data/ground_truth.full.json --strict
```

Reports the overall kappa plus a per-workload breakdown, comparing your
`author_label` against the corpus's `original_label`. `--strict` exits non-zero
if the fully annotated dataset falls below the frozen success threshold of
**κ > 0.7**. It does not gate while pairs remain unannotated, so running it
mid-way is safe and informative.

Record the result in [`../data/DATASHEET.md`](../data/DATASHEET.md) §4, where a
`TODO (post data-production)` marker is waiting for it.

---

## Step 8 — Refresh the published identifiers with your labels

`data/ground_truth.ids.csv` was written at step 4, **before** annotation, so
its `author_label` column is empty. Regenerate it from the annotated dataset,
or the published artifact ships without your labels.

> **Update 2026-08-06 — this is now a flag on step 6, not a separate step.**
> Pass `--out-ids data/ground_truth.ids.csv` to `scripts/annotate_dataset.py` and
> the identifiers file is refreshed from the annotated dataset every time the
> session ends, backed up first like every other ground-truth write. The
> hand-rolled one-liner below still works and still prints the same count, but a
> step whose whole purpose is "do not forget to do this" is better as a flag than
> as a paragraph. If you run a session without `--out-ids`, the script says so.
>
> ~~Regenerate it with the snippet below.~~

```bash
python -c "
from levy.dataset.io import load_json, to_distribution_records, save_distribution_csv
pairs = load_json('data/ground_truth.full.json')
save_distribution_csv(to_distribution_records(pairs), 'data/ground_truth.ids.csv')
print(sum(p.author_label is not None for p in pairs), 'of', len(pairs), 'annotated')
"
```

It should print `900 of 900 annotated`. Anything less means step 6 is
unfinished.

> **Update 2026-08-04 — done.** Verified against the tracked file: 900 / 900 rows
> of `data/ground_truth.ids.csv` carry `author_label`, so the published artifact
> ships the blind re-annotation and a replicator's `ground_truth_label()` uses it
> rather than falling back to `original_label`.
>
> Two properties of this step worth keeping on the record, because they are what
> make it safe to re-run:
>
> - It is licence-safe. `save_distribution_csv` writes only the
>   `DistributionRecord` fields, which have no query text by construction.
> - The sidecar `ground_truth.ids.meta.json` does **not** need regenerating — it
>   records sampling inputs, not labels.
>
> Re-run `scripts/audit_release.sh` afterwards, and commit
> `data/ground_truth.ids.csv` on its own.

---

## Step 9 — Audit, then commit

```bash
scripts/audit_release.sh
```

Must exit 0. Check 6 confirms no tracked file carries query text attributed to
a third-party corpus; check 7 now runs for real, sampling strings out of your
populated `data/raw/` and looking for them in tracked files.

```bash
git status --short
```

Confirm the output shows **nothing** under `data/raw/` and **no** `.full.`
file. Then commit exactly these three:

```bash
git add data/corpora.json data/ground_truth.ids.csv data/ground_truth.ids.meta.json
git commit
```

### What must never be committed

| | Why |
|---|---|
| anything under `data/raw/` | Quora grants no redistribution right; SODD is CC BY-NC-SA 4.0 |
| `data/ground_truth.full.csv` / `.json` | carries that same corpus text |
| ~~`data/annotation_progress.json`~~ | ~~working file, superseded by the labels in the ids file~~ — **Update 2026-08-04: it IS committed, deliberately.** No query text, so no licence risk; it is a second version-controlled copy of the 900 annotations. See the archived `add-corpus-acquisition` task 7.5. |

The first two are gitignored and the audit enforces the outcome, but check
`git status` anyway — a `git add -f` would defeat both.

> **Update 2026-08-04 — `git add -f` is not the only way past `.gitignore`.
> `git stash` is.** `git stash -u` stashes untracked files and `git stash --all`
> stashes **ignored** files too, straight into `refs/stash`, with no ignore rule
> consulted. In this run a `git stash --all` put
> `data/annotated-900.backup.csv` — the annotated 900 *with query text* — into
> the object database. The `*.csv` rule was correct and simply not consulted.
>
> **`scripts/audit_release.sh` check 4** caught it, because it scans
> `git log --all`, which includes `refs/stash`. No remote ref ever contained the
> blob (`git push` and `git push --all` do not push `refs/stash`; **`git push
> --mirror` does** — never mirror-push this repository). Cleared; audit passes 8/8.
>
> Practical rules that follow:
>
> - Treat the audit, not `.gitignore`, as the release gate. Run it before every
>   push of a new ref, not only before a release.
> - Prefer `git stash push -- <paths>` over `git stash --all` in this repository.
>   Note that everything after `--` is a pathspec, so `-m "msg"` must come
>   *before* it — `git stash push -m "msg" -- <paths>`.
> - After dropping a stash that held corpus text, expire it for real; until then
>   it survives as a dangling object that any copy of `.git` carries:
>
>   ```bash
>   git reflog expire --expire-unreachable=now --all && git gc --prune=now
>   ```
>
> - `git stash list` should be empty, or hold only stashes you have checked, before
>   any release.

---

## After this runbook

The dataset is now the harness input. Point the pipeline at it — one argument
changes, nothing else:

```bash
LEVY_EMBEDDING_PROVIDER=sentence-transformers \
    scripts/reproduce.sh data/ground_truth.full.csv results/run-001
```

Then fill in the remaining `TODO (post data-production)` markers in
`data/DATASHEET.md`: actual positive ratio and per-workload counts, any
fallback corpus used, the sampling date, and the kappa breakdown from step 7.

---

## Re-sampling one workload after the fact

Added 2026-08-06. Use this when one workload's sample turns out to be the
problem — a corpus whose positive class does not carry the study's notion of
cache-substitutability, say — and the other two are fine. It replaces that
workload's 300 rows inside the live dataset and leaves the other 600 rows, with
their `author_label`s, exactly as they are.

**There is one ground truth, at its canonical paths, always.** This procedure
rewrites those paths. It does not produce `ground_truth_v2.csv`, a dated working
copy, or a candidate file living beside the real one; if you find yourself
wanting one, back up (the tooling already did) and keep working in place.

```bash
# 1. Re-draw the workload. Reads only that workload's corpus, so the other
#    two need not even be on disk.
python scripts/sample_dataset.py --require-real --workload chat \
    --n-per-workload 300 --seed 4242 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv

# 2. Annotate the 300 new pairs — and nothing else.
python scripts/annotate_dataset.py \
    --dataset data/ground_truth.full.json \
    --progress data/annotation_progress.json \
    --workload chat --session-limit 50 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv

# 3. Re-check kappa over the whole 900, and re-run the audit before committing.
python scripts/compute_kappa.py --dataset data/ground_truth.full.json
scripts/audit_release.sh
```

What step 1 guarantees, and refuses to do without:

| | |
|---|---|
| Disjointness | new pairs are drawn from the candidate pool **minus every `source_pair_id` already in the dataset**, so a re-sample cannot re-draw what it replaces — at the same seed or any other |
| Shortfall | if the pool cannot cover `--n-per-workload` after that exclusion, the run fails naming the workload and the shortfall, and writes nothing. Do not lower `--n-per-workload`: that changes the frozen design |
| Other workloads | passed through untouched, `author_label` included — the test suite asserts the untouched rows are byte-for-byte identical |
| Labels | cleared for the re-sampled workload **only**, which is what makes step 2 present exactly those 300 pairs |
| Stale answers | a re-sample reuses that workload's `pair_id`s, so step 1 **removes those pairs' entries from `data/annotation_progress.json`** (backed up first, other workloads' answers untouched) and reports the count. Without that, step 2 would re-apply the old answers to the new pairs — the file predates per-label fingerprints, so it carries nothing for the session itself to check against. Override the path with `--progress`; the default is derived from `--out-csv`'s directory, never a fixed repo path |
| Backups | `data/ground_truth.full.{csv,json}`, the ids file and the sidecar are copied to `data/backups/<name>.<UTC>.<ext>` before anything is written; a backup that cannot be made aborts the run |
| Provenance | the sidecar records **per workload** the seed and UTC timestamp of the run that produced *its* rows, so a dataset whose workloads were sampled at different times is still reproducible. The top-level `seed` describes the latest invocation only. Entries from a sidecar written before this existed are backfilled from its top level and flagged `provenance_backfilled`, so a filled-in seed is never mistaken for a first-hand per-workload record |
| Drift | if the ids file and the working dataset disagree about the workloads *not* being re-sampled, the run refuses and tells you to rehydrate — rather than publishing whichever file it happened to read |

Then the D3 run. If you re-run only the affected workload's 10 configurations,
merge them back with `scripts/merge_results.py` — see
[`D3_PRODUCTION_RUN.md`](D3_PRODUCTION_RUN.md) §"Step 4b".

Record the re-sample in `data/DATASHEET.md`: which workload, the new seed, the
date, and **why**. A dataset whose workloads were drawn on different days is a
fact about the study, not an implementation detail.

---

## Troubleshooting

**Step 1 exits 1 and I have not done anything wrong.** That is correct
behaviour — it is telling you which corpora need step 2. Re-run it after step 2
and it exits 0.

**A file is reported `manual` after I downloaded it.** The filename does not
match `data/corpora.json`. Compare against the tree in step 2; a browser that
appended ` (1)` or stripped `.gzip` is the usual cause.

**Checksum mismatch after pinning.** The file changed, or you pinned a
different copy. Re-acquire it; if upstream genuinely re-released, that is a
research-scope event — record the new snapshot in `data/corpora.json` and note
it in the datasheet rather than silently re-pinning.

**Pre-flight reports a pool shortfall.** The corpus cannot fill a stratum at
300 pairs. Do not lower `--n-per-workload` to make it pass — that changes the
frozen design. The Proposal's Risk 1 contingency covers corpus substitution.

**Rehydration says a `source_pair_id` was not found.** The corpus on disk is
not the snapshot the dataset was sampled from. Compare checksums against
`data/corpora.json`.

## Related documentation

- [`../data/DATASHEET.md`](../data/DATASHEET.md) — corpora, licences, protocol, deviations, limitations
- [`../data/README.md`](../data/README.md) — what is committed versus generated in `data/`
- [`../data/raw/README.md`](../data/raw/README.md) — per-corpus acquisition layout
- [`REPRODUCTION.md`](REPRODUCTION.md) — reproducing the study from the published artifact
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the dataset feeds the harness
