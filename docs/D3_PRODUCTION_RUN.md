# D3 Production Run — Runbook (LEV-13)

**Status:** Living, author-facing. Sibling of [`DATA_PRODUCTION.md`](DATA_PRODUCTION.md),
which is the D2 runbook. That one produced the dataset; this one runs the shipped
tooling over it and produces the D3 result artifacts.

**Scope:** the ordered procedure for LEV-13 — 30-configuration sweep → statistical
analysis bundle → ±5% replication check.

This document **does not restate the evaluation pipeline**. `scripts/reproduce.sh`
is its single definition (see `CLAUDE.md`, LEV-9). Everything below either wraps
that script or happens before/after it.

---

## 0. Answer up front

**The code is ready. No code changes are required to run D3.**

Every component LEV-13 consumes is shipped and archived (LEV-4 harness, LEV-8
analysis, LEV-9 packaging), the real dataset is on disk, and the model registry
resolves both study checkpoints with the correct prefixes and flags. The run is
an execution task, not a development task.

---

## 1. Embedding provider and models

`--embedding-provider sentence-transformers`, loading both study checkpoints from
HuggingFace. This is what the frozen documents mandate
(`docs/Specification_and_Design_Report.md:170`, `docs/Project_Proposal.md:94`) and
what the code has been wired for since LEV-1, so it carries **no deviation entry** —
unlike the two corpus substitutions in `data/DATASHEET.md` §2.

Registry wiring, verified in `levy/embedding_manager.py`:

| Alias | Resolved checkpoint | Prefix | `trust_remote_code` |
|---|---|---|---|
| `all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | *(none)* | `False` |
| `modernbert` | `nomic-ai/modernbert-embed-base` | `search_query: ` | `True` |

---

## 2. Preconditions — verified 2026-08-05

| Check | State | Re-verify with |
|---|---|---|
| conda env `levy` | complete (faiss 1.10.0, sentence-transformers 5.5.1, torch 2.9.1, statsmodels 0.14.6) | `conda activate levy && python -c "import faiss, sentence_transformers, statsmodels"` |
| Real dataset | `data/ground_truth.full.csv`, **900 pairs, 300 per workload** | step 2 below |
| Raw corpora | `quora-qqp` 56M · `sodd` 1.8G · `twitter-pit2015` 7.5M | `du -sh data/raw/*/` |
| Grid | `full_grid()` = **30** configurations | `python -c "from levy.experiment.config import full_grid; print(len(full_grid()))"` |
| Gitignore | `results/` and `data/ground_truth.full.*` ignored | `grep -n 'results/' .gitignore` |

Label distribution the harness will see (`ground_truth_label()` → `author_label`):

```
faq   163 pos / 137 neg
code   34 pos / 266 neg      <-- recall on code rests on 34 positives
chat   68 pos / 232 neg
```

This is the κ = 0.3267 asymmetry propagating into D3. It is **a finding, not a defect
to correct** — it bounds what the code workload can support, and is not something to
code around.

---

## 3. Step-by-step

Every Python command assumes the env is active. Claude Code's shell does not
inherit it, so prefix accordingly:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate levy
```

### Step 1 — Rehydrate the dataset (skip if already present)

`data/ground_truth.full.csv` is gitignored and rebuilt from the committed ids file
plus `data/raw/`. It is currently present and dated 2026-08-04; only re-run this if
it is missing or `data/raw/` changed.

```bash
python scripts/rehydrate_dataset.py
```

### Step 2 — Pre-flight the dataset

Confirms the file loads through the same path the harness uses, at the expected shape.

```bash
python -c "
from collections import Counter
from levy.dataset.io import load_dataset
p = load_dataset('data/ground_truth.full.csv')
print('pairs:', len(p))
print('workloads:', dict(Counter(x.workload for x in p)))
"
```

Expect `pairs: 900` and `{'faq': 300, 'code': 300, 'chat': 300}`. **Stop if it differs.**

### Step 3 — Smoke run (do not skip)

One model × one workload × one threshold, on the **real** dataset with the **real**
provider. This forces the first-run model downloads and exercises ModernBERT's
`trust_remote_code` path *before* committing to a ~5 hour run. Run ModernBERT first —
it is the larger download and the only arm with a remote-code dependency.

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

**Both checkpoints are already cached on this machine** (downloaded 2026-06-13):
`~/.cache/huggingface/hub/models--nomic-ai--modernbert-embed-base` (572M, revision
`d556a88e`) and `models--sentence-transformers--all-MiniLM-L6-v2` (87M, revision
`1110a243`). Both snapshots are complete — `model.safetensors`, `modules.json`,
`1_Pooling/config.json`, tokenizer files — so **the run needs no network at all**
unless the cache is cleared. On a fresh machine this step downloads ~90MB + ~572MB.

Each smoke config is ~5 minutes, almost all of it the synthetic LLM delay described
in §4.

Check both wrote a `results.csv` with one row, and that `run_meta.json` records the
**resolved checkpoints** — not the aliases:

```bash
python -c "
import json
for d in ('results/smoke-modernbert','results/smoke-minilm'):
    print(d, json.load(open(d+'/run_meta.json'))['model_identities'])
"
```

Expect `nomic-ai/modernbert-embed-base` and `sentence-transformers/all-MiniLM-L6-v2`.
If a checkpoint shows as an alias or as `mock`, **stop** — the run would be
unattributable.

### Step 4 — Full production run (~5 hours)

The three pipeline stages in one command. Run it detached with a log, because it
outlives a terminal session.

```bash
nohup env LEVY_EMBEDDING_PROVIDER=sentence-transformers \
    scripts/reproduce.sh data/ground_truth.full.csv results/run-001 \
    > results/run-001.log 2>&1 &
```

Monitor:

```bash
tail -f results/run-001.log
```

The script is `set -euo pipefail`, so any stage failing aborts the rest. A non-zero
exit on stage `[3/3]` means the replication criterion failed — that is a **result**,
not a crash; keep the outputs and record it.

### Step 5 — Verify the bundle

```bash
ls -la results/run-001/ results/run-001/analysis/ results/run-001/analysis/figures/
```

Expected inventory:

| Path | Contents |
|---|---|
| `results/run-001/results.csv` | 30 rows; columns `config_id, model, workload, threshold, n, tp, fp, tn, fn, precision, recall, f0_5, fpr, hit_rate, precision_zero_div, recall_zero_div, fpr_zero_div` |
| `results/run-001/decisions.csv` | per-pair decisions: `config_id, model, workload, threshold, pair_id, decision, source, similarity, label` |
| `results/run-001/run_meta.json` | dataset path, providers, resolved checkpoints, grid |
| `analysis/anova.csv` | H0₁/H0₂/H0₃ — df, sum-sq, F, p, reject/retain at α=0.05 |
| `analysis/tukey.csv` | pairwise post-hoc (headers only if skipped) |
| `analysis/tukey_status.csv` | per-effect ran/skipped **and why** |
| `analysis/curves_hit_rate.csv`, `curves_precision.csv` | threshold curves per (model, workload) |
| `analysis/kappa.json` | κ, consumed from `levy.dataset.kappa` — **not** recomputed |
| `analysis/figures/` | `curve_hit_rate.{png,pdf}`, `curve_precision.{png,pdf}` |
| `analysis/analysis_meta.json` | input paths, library versions, diagnostics, timestamp |

Row-count check:

```bash
python -c "
import csv; r=list(csv.DictReader(open('results/run-001/results.csv')))
print('rows:', len(r), '| unique configs:', len({(x['model'],x['workload'],x['threshold']) for x in r}))
"
```

Both must be **30**.

### Step 6 — Extract the numbers LEV-13 asks for

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

Record, per LEV-13's acceptance criteria:

1. **H0₁, H0₂, H0₃** — each an explicit reject or retain at α = 0.05, read from
   `anova.csv`. If the analysis reports `undefined`, that means zero variance in
   `fpr`; report it as undefined, **never as a retained null**.
2. **Tukey** — ran or skipped, with the reason, from `tukey_status.csv`.
3. **Hit rate vs the 30% viability bar** — per workload and threshold, from the
   table above. Note the threshold band: embeddings are L2-normalised and
   similarity is `1/(1+L2)`, so the frozen 0.70–0.90 sweep covers a high-cosine
   band (~0.91–0.998). That is spec-mandated.
4. **Replication** — pass/fail from stage `[3/3]` in the log, with the diff table if
   it failed.

### Step 7 — Close out LEV-13

`results/` is gitignored — published outputs are attached to a release, not committed.

---

## 4. Expected runtime — and why it is mostly sleep

Measured on the real 900 pairs (mock embeddings, zero latency, so this isolates
call counts from embedding cost):

| Workload | Mock-LLM calls per configuration | Compute |
|---|---|---|
| faq | 600 | 0.24 s |
| code | 599 | 0.20 s |
| chat | 574 | 0.18 s |

`run_sweep`'s `llm_latency_seconds` defaults to **0.5 s** (`levy/experiment/runner.py:61`)
and **`scripts/run_experiments.py` exposes no flag to override it**. So:

```
17,730 calls x 0.5 s ≈ 2.5 h   (stage [1/3] sweep)
             + the same again   (stage [3/3] replication re-runs the grid)
                    ≈ 5 h total
```

Real embedding cost adds only minutes: `EmbeddingManager` memoizes by
`(model_key, sha256(text))` and `run_sweep` shares one manager per model across its
15 configurations, so each model embeds the ~1,800 unique texts once.

Real embeddings will produce **more** semantic hits than the mock, hence **fewer**
LLM calls, so 5 h is an **upper bound**.

---

## 5. Known gotchas

- **The 5 hours is almost entirely `time.sleep`.** Do not interpret it as compute
  load. Note the asymmetry: `scripts/check_replication.py:58` *does* expose
  `--llm-latency-seconds`, and `scripts/reproduce.sh` never passes it, while
  `run_experiments.py` has no equivalent at all. Adding a flag would make the run
  ~10 minutes — but it is a code change to shipped, archived tooling, so it belongs
  in its own issue, **not** mid-production-run.
- **No network needed.** Both checkpoints are already cached (§3, step 3), and the
  Anthropic backend is never used by this pipeline — the harness replays through
  the mock LLM by design.
- **A failed stage `[3/3]` is a result.** Keep the outputs and record the diff.

---

## 6. Do NOT

Per `CLAUDE.md` and the frozen documents — these are findings to report, never
things to code around:

- **Do not rescale the thresholds** to chase hit rate. The 0.70–0.90 band on the
  `1/(1+L2)` scale is spec-mandated.
- **Do not lower the κ > 0.7 bar**, re-annotate non-blind, re-sample for agreement,
  or change `ground_truth_label()`.
- **Do not substitute either embedding model.** The pair is the independent
  variable of the primary research question, O2, H0₁ and Success Criterion 1.
- **Do not edit the two frozen documents** for any reason.
- **Do not replace `data/ground_truth.{csv,json}`** — the synthetic fixtures are
  the permanent offline default for the test suite and `reproduce.sh`.
