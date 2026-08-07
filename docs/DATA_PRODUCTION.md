# Levy — Production Runbook

**Audience:** the author. **Outcome:** the 900-pair ground-truth dataset (D2) and
the 30-configuration evaluation results (D3).

This is the only document carrying the end-to-end production procedure: acquire
the corpora, sample, annotate, run the grid, analyse, replicate. Work through it
in order.

What it is not:

- **Not the reproduction guide.** A third party reproducing the study runs the
  acquire and rehydrate steps only, and finds them in
  [`REPRODUCTION.md`](REPRODUCTION.md).
- **Not the pipeline definition.** `scripts/reproduce.sh` is the single
  definition of the evaluation pipeline; Part 2 below wraps it.
- **Not the datasheet.** Corpus licences, the sampling protocol, the recorded
  deviations from the frozen documents and the dataset's limitations are in
  [`../data/DATASHEET.md`](../data/DATASHEET.md).

---

## Current state

| | |
|---|---|
| Corpora | all three acquired, checksums pinned in `data/corpora.json` |
| Dataset | 900 pairs, 300 per workload, 900/900 blind-annotated |
| Cohen's κ | 0.5000 (faq 0.5267, code 0.4200, chat 0.5533) — below the frozen κ > 0.7 bar; a finding, see [`../data/DATASHEET.md`](../data/DATASHEET.md) §4 |
| D3 result of record | `results/run-003/` — H0₁ retained, H0₂ rejected, H0₃ retained; best hit rate 24.0%, so all 30 configurations miss the 30% bar; ±5% replication passed 60/60 |

---

## Before you start

| | |
|---|---|
| Time | ~30 min for the acquire/sample steps; annotation is several sittings; the grid run is ~5 h |
| Needs network | acquisition only |
| Needs an account | Kaggle, for the Quora corpus |
| Disk | a few GB under `data/raw/`, plus model checkpoints |

Activate the environment once per terminal session, and run everything from the
repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy
```

---

# Part 1 — The dataset (D2)

## Step 1 — Automatic acquisition

```bash
python scripts/fetch_corpora.py
```

Downloads the `chat` corpus (Twitter PIT-2015) into `data/raw/twitter-pit2015/`,
extracting `train.data` and `dev.data` from the shared-task release archive.

**This command exits with status 1, and that is the expected outcome.** Two of
the three corpora cannot be fetched without a human step, so the run ends by
printing, for each of them, the URL to open, the filenames to save, where to save
them, and the expected SHA-256. Step 2 is that human step.

## Step 2 — Your manual downloads

### 2A — Quora Question Pairs (`faq` workload)

Open <https://www.kaggle.com/competitions/quora-question-pairs/data>, sign in,
accept the competition rules, and download `train.csv.zip`. Unzip it anywhere.

The adapter expects the tab-separated form. Kaggle ships CSV with the right
columns (`id, qid1, qid2, question1, question2, is_duplicate`), so convert it
into place — adjust the path at the end to wherever you unzipped:

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

## Step 3 — Verify, then pin the checksums

```bash
python scripts/fetch_corpora.py
```

Must now exit 0, reporting every file as `present`. If a file is still `manual`,
it is not where the tool expects it — re-check the filename against the tree
above.

```bash
python scripts/fetch_corpora.py --pin
```

Records the SHA-256 of each file you actually hold into `data/corpora.json`. Do
this once, before sampling. From this point a checksum mismatch is a hard
failure, which is what turns an upstream re-release into a loud error instead of
a silent divergence in your results.

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
`chat`, each stratified 150 duplicate / 150 non-duplicate. Every downstream tool
reads the single file.

`--require-real` refuses to substitute synthetic data: if a corpus is missing the
run fails naming it, rather than quietly emitting pairs whose `source_corpus` is
`mock`.

Four files are written:

| File | What | Committed |
|---|---|---|
| `data/ground_truth.full.csv` / `.json` | the working dataset, query text included | no — gitignored |
| `data/ground_truth.ids.csv` | identifiers and labels, no text — the published artifact | yes |
| `data/ground_truth.ids.meta.json` | per-workload seed and date, ratio, adapter options, corpus snapshots, input checksums | yes |

Before sampling, a pre-flight pass validates every workload in scope together and
prints their pool sizes. If anything is wrong it reports **every** problem at once
and writes nothing — fix them all, then re-run. Common findings:

| Finding | Meaning |
|---|---|
| `[class-pool] … needs 150 positive pairs, only N available` | that corpus cannot fill the stratum; see the frozen Proposal's Risk 1 fallback path before substituting anything |
| `[checksum] … does not match the pinned` | the file changed since you pinned it |
| `[required-fields] … missing columns` | wrong file, or the QQP conversion in 2A did not run |
| `[label-domain] … outside the declared domain` | an unexpected label value; reported, never coerced |

## Step 5 — Confirm the round-trip

The published artifact carries no query text, so the whole release depends on
rehydration reconstructing the dataset exactly. Prove it on the real 900 —
fixtures already prove it in CI, but this is the artifact you are shipping:

```bash
python scripts/rehydrate_dataset.py --out-csv /tmp/rt.csv --out-json /tmp/rt.json
cmp /tmp/rt.csv data/ground_truth.full.csv && echo "byte-identical"
```

If `cmp` reports a difference, stop and investigate before annotating — the
identifiers file cannot reproduce what you sampled.

Write to `/tmp` rather than in place. Rehydration rebuilds the working dataset
from `ground_truth.ids.csv`, so running it in place after annotating replaces
your labels with whatever the ids file holds. It backs the working dataset up to
`data/backups/` first and aborts if it cannot, so the labels are recoverable —
but recovering is work you do not need to do.

## Step 6 — Blind re-annotation

```bash
python scripts/annotate_dataset.py \
    --dataset data/ground_truth.full.json \
    --progress data/annotation_progress.json \
    --session-limit 50 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv
```

You are shown only `query_1` and `query_2` — never the original corpus label, the
corpus name, or the source pair id. That blindness is what makes the Cohen's
kappa in step 7 a meaningful agreement measure rather than a restatement of the
corpus's own labels.

**The judgment to make:** would someone who asked `query_1`, and received a good
answer, be satisfied by that same answer for `query_2`? That is
same-question-intent, not textual similarity. Different wording with the same
intent is a match (`1`); similar wording with materially different intent —
different constraints, a different sub-topic — is not (`0`).

Pairs are presented workload block by workload block in `faq,chat,code` order,
shuffled within each block. Code goes last because its pairs are Stack Overflow
posts, by far the longest to read. The shuffle matters for the same reason
blindness does: in file order a long run of near-duplicates lets position stand in
for content, and the pair after forty of them is judged against the run rather
than on its own.

Progress is written after **every single answer**, so you can stop at any point
and resume by re-running the identical command. Already-annotated pairs are never
re-asked. The resolved presentation order is stored in the progress file, so a
resumed session continues in the order it started in.

Optional flags:

| Flag | Use |
|---|---|
| `--session-limit 50` | end the sitting cleanly after 50 labelled pairs. Skips do not consume the limit |
| `--workload chat` | restrict the session to one workload (repeatable) |
| `--workload-order code,faq,chat` | override the block order; an unknown or repeated workload is rejected |
| `--order-seed 7` | change the within-block shuffle; recorded in the progress file |
| `--overwrite` | re-ask pairs that already carry an `author_label` |

`--out-ids` refreshes the published identifiers file from your labels every time
the session ends, backed up first. Without it the published artifact ships
without your annotations, and the script says so.

## Step 7 — Cohen's kappa

```bash
python scripts/compute_kappa.py --dataset data/ground_truth.full.json --strict
```

Reports the overall kappa plus a per-workload breakdown, comparing your
`author_label` against the corpus's `original_label`. `--strict` exits non-zero
if the fully annotated dataset falls below the frozen success threshold of
**κ > 0.7**. It does not gate while pairs remain unannotated, so running it
mid-way is safe and informative.

Record the result in [`../data/DATASHEET.md`](../data/DATASHEET.md) §4.

## Step 8 — Audit, then commit

```bash
scripts/audit_release.sh
```

Must exit 0. Check 6 confirms no tracked file carries query text attributed to a
third-party corpus; check 7 samples strings out of your populated `data/raw/` and
looks for them in tracked files.

```bash
git status --short
```

Confirm the output shows **nothing** under `data/raw/` and **no** `.full.` file.
Then commit the published artifacts:

```bash
git add data/corpora.json data/ground_truth.ids.csv data/ground_truth.ids.meta.json data/annotation_progress.json
```

### What must never be committed

| | Why |
|---|---|
| anything under `data/raw/` | Quora grants no redistribution right; SODD is CC BY-NC-SA 4.0 |
| `data/ground_truth.full.csv` / `.json` | carries that same corpus text |
| `data/backups/` | timestamped copies of the above |

`data/annotation_progress.json` **is** committed, deliberately: it carries no
query text, and it is a second version-controlled copy of the 900 annotations.

The gitignore rules are a staging filter only. `git add -f`, `git stash -u` and
`git stash --all` all bypass them — `git stash --all` stashes *ignored* files
straight into `refs/stash` with no ignore rule consulted, and that has already put
900 rows of corpus text into the object database once. `scripts/audit_release.sh`
check 4 is the backstop: it scans `git log --all`, which includes `refs/stash`.

Practical rules:

- Treat the audit, not `.gitignore`, as the release gate. Run it before every push
  of a new ref.
- Prefer `git stash push -- <paths>` over `git stash --all`. Everything after `--`
  is a pathspec, so `-m "msg"` must come before it.
- `git push --mirror` pushes `refs/stash` — never mirror-push this repository.
- After dropping a stash that held corpus text, expire it for real; until then it
  survives as a dangling object in any copy of `.git`:

  ```bash
  git reflog expire --expire-unreachable=now --all && git gc --prune=now
  ```

---

# Part 2 — The evaluation run (D3)

30 configurations: 2 embedding models × 3 workloads × 5 thresholds. Sweep →
analysis bundle → ±5% replication.

## Embedding models

`--embedding-provider sentence-transformers`, loading both study checkpoints from
HuggingFace, as the frozen documents mandate. Registry wiring in
`levy/embedding_manager.py`:

| Alias | Resolved checkpoint | Prefix | `trust_remote_code` |
|---|---|---|---|
| `all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | *(none)* | `False` |
| `modernbert` | `nomic-ai/modernbert-embed-base` | `search_query: ` | `True` |

Both checkpoints are cached locally (`~/.cache/huggingface/hub/`), so the run
needs no network unless the cache is cleared. On a fresh machine the first run
downloads ~90 MB + ~572 MB.

## Step 9 — Pre-flight the dataset

```bash
python -c "
from collections import Counter
from levy.dataset.io import load_dataset
p = load_dataset('data/ground_truth.full.csv')
print('pairs:', len(p))
print('workloads:', dict(Counter(x.workload for x in p)))
"
```

Expect `pairs: 900` and `{'faq': 300, 'code': 300, 'chat': 300}`. **Stop if it
differs.**

The label distribution the harness will see (`ground_truth_label()` →
`author_label`) is not balanced, because your labels are stricter than the
corpora's. Check it, because it bounds what recall on each workload can support:

```bash
python -c "
from collections import Counter
from levy.dataset.io import load_dataset
p = load_dataset('data/ground_truth.full.csv')
for w in ('faq','code','chat'):
    c = Counter(x.ground_truth_label() for x in p if x.workload==w)
    print(f'{w:5} {c[1]:3} pos / {c[0]:3} neg')
"
```

## Step 10 — Smoke run (do not skip)

One model × one workload × one threshold, on the real dataset with the real
provider. This exercises ModernBERT's `trust_remote_code` path *before* you commit
to a five-hour run. Run ModernBERT first — it is the larger download and the only
arm with a remote-code dependency.

```bash
python scripts/run_experiments.py --dataset data/ground_truth.full.csv \
    --embedding-provider sentence-transformers \
    --models modernbert --workloads faq --thresholds 0.80 \
    --out-dir results/smoke-modernbert
```

```bash
python scripts/run_experiments.py --dataset data/ground_truth.full.csv \
    --embedding-provider sentence-transformers \
    --models all-MiniLM-L6-v2 --workloads faq --thresholds 0.80 \
    --out-dir results/smoke-minilm
```

Each smoke config is ~5 minutes, almost all of it the synthetic LLM delay. Check
both wrote a one-row `results.csv`, and that `run_meta.json` records the
**resolved checkpoints**, not the aliases:

```bash
python -c "
import json
for d in ('results/smoke-modernbert','results/smoke-minilm'):
    print(d, json.load(open(d+'/run_meta.json'))['model_identities'])
"
```

Expect `nomic-ai/modernbert-embed-base` and
`sentence-transformers/all-MiniLM-L6-v2`. If a checkpoint shows as an alias or as
`mock`, **stop** — the run would be unattributable.

## Step 11 — Full run (~5 hours)

The three pipeline stages in one command. Run it detached, because it outlives a
terminal session. Everything is redirected to the log, including the completion
marker, so nothing holds the terminal:

```bash
nohup sh -c 'LEVY_EMBEDDING_PROVIDER=sentence-transformers scripts/reproduce.sh data/ground_truth.full.csv results/run-001; echo "== RUN DONE (exit $?) =="' > results/run-001.log 2>&1 < /dev/null & disown
```

Check on it:

```bash
grep "RUN DONE" results/run-001.log
```

Empty means still running. The script is `set -euo pipefail`, so any stage failing
aborts the rest. A non-zero exit on stage `[3/3]` means the replication criterion
failed — that is a **result**, not a crash; keep the outputs and record it.

## Step 12 — Verify the bundle

```bash
ls -la results/run-001/ results/run-001/analysis/
```

| Path | Contents |
|---|---|
| `results.csv` | 30 rows; `config_id, model, workload, threshold, n, tp, fp, tn, fn, precision, recall, f0_5, fpr, hit_rate, *_zero_div` |
| `decisions.csv` | per-pair decisions |
| `run_meta.json` | dataset path, providers, resolved checkpoints, grid |
| `replication.json` | ±5% verdict, tolerance rule, per-comparison table, and which configurations it covers |
| `analysis/anova.csv` | H0₁/H0₂/H0₃ — df, sum-sq, F, p, reject/retain at α=0.05 |
| `analysis/tukey.csv`, `tukey_status.csv` | post-hoc results, and per effect whether it ran and why |
| `analysis/curves_hit_rate.csv`, `curves_precision.csv` | threshold curves per (model, workload) |
| `analysis/kappa.json` | κ, consumed from `levy.dataset.kappa` — not recomputed |
| `analysis/figures/` | `curve_hit_rate.{png,pdf}`, `curve_precision.{png,pdf}` |
| `analysis/analysis_meta.json` | input paths, library versions, diagnostics |

Row count must be 30:

```bash
python -c "
import csv; r=list(csv.DictReader(open('results/run-001/results.csv')))
print('rows:', len(r), '| unique configs:', len({x['config_id'] for x in r}))
"
```

## Step 13 — Read the numbers

```bash
python -c "
import csv
r=list(csv.DictReader(open('results/run-001/results.csv')))
print(f\"{'model':<20}{'workload':<8}{'thr':<7}{'hit_rate':<10}{'precision':<11}{'fpr':<8}\")
for x in sorted(r, key=lambda x:(x['model'],x['workload'],float(x['threshold']))):
    v = float(x['hit_rate'])
    print(f\"{x['model']:<20}{x['workload']:<8}{x['threshold']:<7}{v:<10.4f}{float(x['precision']):<11.4f}{float(x['fpr']):<8.4f}\" + ('  <30%' if v < 0.30 else ''))
"
```

What to record:

1. **H0₁, H0₂, H0₃** — each an explicit reject or retain at α = 0.05, from
   `anova.csv`. If the analysis reports `undefined`, that means zero variance in
   `fpr`; report it as undefined, **never as a retained null**.
2. **Tukey** — ran or skipped, with the reason, from `tukey_status.csv`.
3. **Hit rate against the 30% viability bar**, per workload and threshold. Note
   the threshold band: embeddings are L2-normalised and similarity is `1/(1+L2)`,
   so the frozen 0.70–0.90 sweep covers a high-cosine band (~0.91–0.998). That is
   spec-mandated.
4. **Replication** — pass/fail and coverage, from `replication.json`.

`results/` is gitignored; published outputs are attached to a release, not
committed.

## Runtime — and why it is mostly sleep

Measured on the real 900 pairs with mock embeddings and zero latency, so this
isolates call counts from embedding cost:

| Workload | Mock-LLM calls per configuration | Compute |
|---|---|---|
| faq | 600 | 0.24 s |
| code | 599 | 0.20 s |
| chat | 574 | 0.18 s |

`run_sweep`'s `llm_latency_seconds` defaults to 0.5 s and
`scripts/run_experiments.py` exposes no flag to override it:

```
17,730 calls x 0.5 s ≈ 2.5 h   (sweep)
             + the same again   (replication re-runs the grid)
                    ≈ 5 h total
```

Real embedding cost adds only minutes: `EmbeddingManager` memoizes by
`(model_key, sha256(text))` and `run_sweep` shares one manager per model across
its 15 configurations, so each model embeds the ~1,800 unique texts once. Real
embeddings produce more semantic hits than the mock and therefore fewer LLM calls,
so 5 h is an upper bound.

Adding a latency flag would cut the run to ~10 minutes, but it is a code change to
shipped tooling and belongs in its own issue, not mid-run.

---

# Re-drawing one workload

Use this when one workload's sample turns out to be the problem and the other two
are fine. It replaces that workload's 300 rows inside the live dataset and leaves
the other 600, with their `author_label`s, exactly as they are.

**There is one ground truth, at its canonical paths, always.** This procedure
rewrites those paths. It does not produce `ground_truth_v2.csv`, a dated working
copy, or a candidate file beside the real one.

Likewise for results: exactly one directory is the result of record — the merged,
30-row one. The staging directories below exist only to feed the merge and are
deleted at the end. Pointing the analysis, the replication check or the poster at
a staging directory returns a valid-looking answer covering part of the grid.

Set the three variables once, then work down:

```bash
W=code                      # the workload being re-drawn
PREV=results/run-002        # current result of record
NEXT=results/run-003        # the one this procedure produces
```

**1. Re-draw the workload.** Reads only that workload's corpus, so the other two
need not be on disk.

```bash
python scripts/sample_dataset.py --require-real --workload "$W" \
    --n-per-workload 300 --seed 8484 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv
```

**2. Annotate the 300 new pairs — and nothing else.** Repeat until it reports
`remaining=0`.

```bash
python scripts/annotate_dataset.py \
    --dataset data/ground_truth.full.json \
    --progress data/annotation_progress.json \
    --workload "$W" --session-limit 50 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv
```

**3. Re-check kappa over the whole 900.**

```bash
python scripts/compute_kappa.py --dataset data/ground_truth.full.json
```

**4. Re-run that workload's 10 configurations into a staging directory.**

```bash
nohup sh -c "python scripts/run_experiments.py --dataset data/ground_truth.full.csv --workloads $W --embedding-provider sentence-transformers --out-dir results/staging-$W; echo '== SWEEP DONE (exit \$?) =='" > results/staging-$W.log 2>&1 < /dev/null & disown
```

**5. Drop that workload's now-stale rows from a copy of `PREV`.** The merge
refuses duplicate `config_id`s by design — one of the two runs is stale and the
merge does not get to pick a winner — so the stale rows go before merging, not
during.

```bash
python - <<PY
import csv, shutil
from pathlib import Path
prev, staged, w = Path("$PREV"), Path("results/staging-prev-minus-$W"), "$W"
staged.mkdir(parents=True, exist_ok=True)
for name in ("results.csv", "decisions.csv"):
    rows = list(csv.DictReader((prev / name).open()))
    kept = [r for r in rows if r["workload"] != w]
    with (staged / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(kept)
    print(name, len(rows), "->", len(kept))
shutil.copy2(prev / "run_meta.json", staged / "run_meta.json")
PY
```

**6. Merge the two staging directories into the new result of record.**

```bash
python scripts/merge_results.py --out-dir "$NEXT" \
    "results/staging-prev-minus-$W" "results/staging-$W"
```

**7. Rebuild the analysis bundle.**

```bash
python scripts/run_analysis.py --results-dir "$NEXT" --out-dir "$NEXT/analysis" \
    --dataset data/ground_truth.full.csv
```

**8. Replicate against the consolidated set** — never a staging directory. All 30
configurations, one verdict file, ~3 hours.

```bash
nohup sh -c 'python scripts/check_replication.py --reference '"$NEXT"'/results.csv --dataset data/ground_truth.full.csv --embedding-provider sentence-transformers; echo "== REPLICATION DONE (exit $?) =="' > results/replication.log 2>&1 < /dev/null & disown
```

**9. Rebuild the poster, delete the staging directories, run the audit.**

```bash
python docs/poster/build_poster.py --results-dir "$NEXT"
rm -rf "results/staging-$W" "results/staging-prev-minus-$W"
scripts/audit_release.sh
```

`PREV` stays on disk as history. Record the new result of record in this file's
"Current state" table, and the re-draw in
[`../data/DATASHEET.md`](../data/DATASHEET.md) §3 — workload, seed, date, reason.

What step 1 guarantees, and refuses to proceed without:

| | |
|---|---|
| Disjointness | new pairs are drawn from the candidate pool **minus every `source_pair_id` already in the dataset**, so a re-draw cannot re-draw what it replaces — at the same seed or any other |
| Shortfall | if the pool cannot cover `--n-per-workload` after that exclusion, the run fails naming the workload and the shortfall, and writes nothing. Do not lower `--n-per-workload`: that changes the frozen design |
| Other workloads | passed through untouched, `author_label` included; the test suite asserts those rows are byte-for-byte identical |
| Labels | cleared for the re-drawn workload only, which is what makes step 2 present exactly those 300 pairs |
| Stale answers | the new pairs reuse that workload's `pair_id`s, so step 1 removes their entries from the progress file (backed up first, other workloads untouched) and reports the count. Otherwise step 2 would re-apply the old answers to the new pairs |
| Backups | the working dataset, the ids file, the sidecar and the progress file are copied to `data/backups/` before anything is written; a backup that cannot be made aborts the run |
| Provenance | the sidecar records each workload's own seed and UTC timestamp, so every workload stays independently reproducible |
| Drift | if the ids file and the working dataset disagree about the workloads *not* being re-drawn, the run refuses and tells you to rehydrate |

**Known rough edge.** Step 5 is a hand-rolled filter — the only step here with no
tool behind it and no test covering it. A `--replace-workload` flag on
`merge_results.py` would remove it, and with it the `staging-prev-minus-*`
directory. Not implemented.

---

## Do not

Per the frozen documents — these are findings to report, never things to code
around:

- **Do not rescale the thresholds** to chase hit rate. The 0.70–0.90 band on the
  `1/(1+L2)` scale is spec-mandated.
- **Do not lower the κ > 0.7 bar**, re-annotate non-blind, re-draw for agreement,
  or change `ground_truth_label()`.
- **Do not substitute either embedding model.** The pair is the independent
  variable of the primary research question, O2, H0₁ and Success Criterion 1.
- **Do not edit the two frozen documents** for any reason.
- **Do not replace `data/ground_truth.{csv,json}`** — the synthetic fixtures are
  the permanent offline default for the test suite and `reproduce.sh`.

---

## Troubleshooting

**Acquisition exits 1 and I have not done anything wrong.** Correct behaviour —
it is telling you which corpora need the manual step. Re-run it afterwards and it
exits 0.

**A file is reported `manual` after I downloaded it.** The filename does not match
`data/corpora.json`. A browser that appended ` (1)` or stripped `.gzip` is the
usual cause.

**Checksum mismatch after pinning.** The file changed, or you pinned a different
copy. Re-acquire it; if upstream genuinely re-released, that is a research-scope
event — record the new snapshot in `data/corpora.json` and note it in the
datasheet rather than silently re-pinning.

**Pre-flight reports a pool shortfall.** The corpus cannot fill a stratum at 300
pairs. Do not lower `--n-per-workload`. The Proposal's Risk 1 contingency covers
corpus substitution.

**Rehydration says a `source_pair_id` was not found.** The corpus on disk is not
the snapshot the dataset was sampled from. Compare checksums against
`data/corpora.json`.

**A long run seems stuck with no output.** Both the sweep and the replication
check print nothing between start and finish. Confirm with `pgrep -fl
run_experiments.py` or `pgrep -fl check_replication.py`; the completion marker in
the log is the only progress signal.

---

## Related documentation

- [`../data/DATASHEET.md`](../data/DATASHEET.md) — corpora, licences, protocol, deviations, limitations
- [`../data/README.md`](../data/README.md) — what is committed versus generated in `data/`
- [`../data/raw/README.md`](../data/raw/README.md) — per-corpus acquisition layout
- [`REPRODUCTION.md`](REPRODUCTION.md) — reproducing the study from the published artifact
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the dataset feeds the harness
