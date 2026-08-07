# Levy — Reproduction Guide

This guide runs Levy's full evaluation pipeline from a fresh checkout. Every
command can be executed verbatim.

**The default path is fully offline.** It uses the dataset committed in `data/`
and mock providers, so it needs **no API key, no model download, and no network
access**. There are two routes:

- [Route A — Docker](#route-a--docker-one-command): one command, nothing to
  install but Docker.
- [Route B — conda](#route-b--conda-local-environment): the development
  environment, step by step.

Both routes run the same pipeline, because both invoke the same script
([`scripts/reproduce.sh`](../scripts/reproduce.sh)). That script is the single
definition of the pipeline: the commands below are the commands it runs. If a
CLI flag ever changes, the script fails loudly rather than leaving this guide
quietly stale.

> **What the committed dataset is.** `data/ground_truth.csv` currently holds
> **15 synthetic fixture pairs** (5 per workload, obviously fabricated text,
> `source_corpus="synthetic-fixture"`), not the real 900-pair research dataset.
> It exists so the pipeline is runnable end-to-end today. **No metric produced
> from it is a research result** — under mock embeddings no semantic hits occur,
> so false positive rate has zero variance and the hypothesis tests are
> correctly reported as `undefined`. See [`data/README.md`](../data/README.md)
> and [`data/DATASHEET.md`](../data/DATASHEET.md). Running against the real
> dataset changes exactly one argument — see
> [Swapping in the real dataset](#swapping-in-the-real-dataset).

---

## Route A — Docker (one command)

Requires Docker only. The image environment is built from `environment.yml`,
the repository's single dependency specification, so the container and the local
conda environment cannot drift apart.

```bash
docker compose run --rm pipeline
```

That builds the image on first use and runs the whole pipeline — sweep,
analysis, replication check — writing outputs to `./results/reproduce/` on the
host.

Equivalent without compose:

```bash
docker build -t levy:latest .
```

```bash
docker run --rm -v "$PWD/results:/opt/levy/results" levy:latest
```

**Expect a cold build of roughly 5–6 minutes and a ~16 GB image** (measured on
Apple Silicon with a warm network). The size comes from `environment.yml` pulling
`sentence-transformers`, hence torch — this is a research artefact optimised for
environment fidelity, not for distribution size. Rebuilds after a source-only
change are fast: the conda solve is cached until `environment.yml` itself changes.

**The default path needs no network and no API key.** Confirm it yourself — this
completes with exit code 0 and produces `results.csv` / `decisions.csv`
byte-identical to the conda route:

```bash
docker run --rm --network none -v "$PWD/results:/opt/levy/results" levy:latest
```

**Redis is unaffected.** The previously documented service still works exactly
as before and the pipeline does not depend on it (the engine defaults to the
in-memory store):

```bash
docker compose up -d redis
```

**Opt-in overrides** (each one has a cost, so none is the default):

```bash
# Real embeddings — downloads model weights on first use, needs network:
docker compose run --rm -e LEVY_EMBEDDING_PROVIDER=sentence-transformers pipeline
```

```bash
# A different dataset (mounted into the image's data/ directory):
docker compose run --rm -e LEVY_DATASET=data/ground_truth.json pipeline
```

`ANTHROPIC_API_KEY` is passed through to the container if set, but **the
pipeline never calls a real LLM** — the harness replays through the mock LLM by
design. The key matters only if you use the image to serve the HTTP API.

> The Docker route is outside the pytest suite, which covers the Python the
> container invokes rather than the image build itself.

---

## Route B — conda (local environment)

### Step 1 — Create the environment

```bash
conda env create -f environment.yml
```

```bash
conda activate levy
```

`faiss-cpu` comes from conda-forge deliberately: the pip wheel segfaults on
Apple Silicon. If Faiss is absent for any reason the engine falls back to an
exact brute-force numpy index automatically, and results are unchanged.

Verify the environment with the gated test suite (offline, mock providers only —
the same command CI runs):

```bash
python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90
```

### Step 2 — The dataset

The repository ships a runnable dataset at `data/ground_truth.csv`, so this step
is **optional**. To regenerate a fixture dataset yourself (fully offline — with
no raw corpus file it falls back to a synthetic `MockCorpusSource` per
workload):

```bash
python scripts/sample_dataset.py --n-per-workload 5 --seed 42 \
    --out-csv data/ground_truth.csv --out-json data/ground_truth.json
```

Annotation validity, as reported in the analysis bundle, comes from the dataset
tooling and can be inspected directly:

```bash
python scripts/compute_kappa.py --dataset data/ground_truth.json
```

Full dataset-production protocol — corpora, licences, sampling, blind
re-annotation, the κ > 0.7 bar — is in [`data/DATASHEET.md`](../data/DATASHEET.md).

#### Step 2b — The real dataset: acquire, then rehydrate

Skip this unless you want the real 900-pair dataset rather than the fixture.
(Producing that dataset in the first place — sampling, blind annotation,
kappa — is the author's one-time procedure, in
[`DATA_PRODUCTION.md`](DATA_PRODUCTION.md). This section is the reader's
half: acquire and rehydrate.)

The study's query text comes from Quora Question Pairs, SODD and Twitter
PIT-2015. None of it is in this repository: Quora grants no redistribution
right and SODD is CC BY-NC-SA, so what is published is
`data/ground_truth.ids.csv` — which pairs were sampled and how they are
labeled, with no text — plus the sidecar recording the seed, adapter options
and input checksums. You supply the corpora; two commands put the text back.

```bash
python scripts/fetch_corpora.py
```

It acquires what it can into `data/raw/`, verifies every file against the
checksums in [`data/corpora.json`](../data/corpora.json), and exits non-zero
printing the exact URL, filename and expected checksum for any corpus that
needs a human step (Quora and SODD both do — one requires accepting terms, the
other is a Google Drive folder). Re-run it once those are in place; it skips
what already verifies. See [`data/raw/README.md`](../data/raw/README.md) for
the per-corpus layout.

```bash
python scripts/rehydrate_dataset.py
```

This writes `data/ground_truth.full.{csv,json}` — gitignored, because it
carries the corpus text. The reconstruction is lossless: it is byte-identical
to the dataset originally sampled, which is what preserves the ±5% replication
criterion without anyone redistributing corpus text. Point the pipeline at it
with `LEVY_DATASET=data/ground_truth.full.csv` (see "Swapping in the real
dataset" below).

### Step 3 — Run the whole pipeline

One command, all three stages:

```bash
scripts/reproduce.sh
```

Defaults: dataset `data/ground_truth.csv`, output directory
`results/reproduce`, embedding provider `mock`, the full frozen grid. Override
positionally or by environment variable:

```bash
scripts/reproduce.sh data/ground_truth.csv results/run-001
```

| Variable | Default | Meaning |
|---|---|---|
| `LEVY_DATASET` | `data/ground_truth.csv` | dataset file (`.csv` or `.json`) |
| `LEVY_OUT_DIR` | `results/reproduce` | output directory |
| `LEVY_EMBEDDING_PROVIDER` | `mock` | `mock`, `sentence-transformers`, `ollama` |
| `LEVY_MODELS` | full grid | comma-separated model subset |
| `LEVY_WORKLOADS` | `faq,code,chat` | comma-separated workload subset |
| `LEVY_THRESHOLDS` | `0.70,…,0.90` | comma-separated threshold subset |

### Step 3a–3c — The same pipeline, stage by stage

These are exactly the commands `scripts/reproduce.sh` runs. Use them when you
want to inspect or rerun one stage.

**3a. Experiment sweep** — the frozen grid: 2 embedding models × 3 workloads ×
5 similarity thresholds = **30 configurations**.

```bash
python scripts/run_experiments.py \
    --dataset data/ground_truth.csv \
    --out-dir results/reproduce \
    --embedding-provider mock
```

Writes to `results/reproduce/`:

| File | Contents |
|---|---|
| `results.csv` | one row per configuration: TP/FP/TN/FN, precision, recall, F₀.₅, false positive rate, hit rate, zero-division flags |
| `decisions.csv` | one row per replayed pair per configuration: decision, decision source, matched similarity, ground-truth label, confusion outcome |
| `run_meta.json` | dataset path, providers, resolved model checkpoints, the grid that ran, latency statistics (labelled synthetic under the mock LLM) |

`results.csv` and `decisions.csv` carry no timestamps and no latency, so a
re-run on identical inputs reproduces them **byte-identically**. Everything
time-varying lives in `run_meta.json`.

A smaller subset, for a quick check:

```bash
python scripts/run_experiments.py --out-dir results/smoke \
    --models all-MiniLM-L6-v2 --workloads faq --thresholds 0.70,0.90
```

**3b. Analysis bundle** — one invocation emits every table and figure.

```bash
python scripts/run_analysis.py \
    --results-dir results/reproduce \
    --out-dir results/reproduce/analysis \
    --dataset data/ground_truth.csv
```

Writes to `results/reproduce/analysis/`:

| File | Contents |
|---|---|
| `anova.csv` | two-way ANOVA on false positive rate, `fpr ~ C(model) * C(workload)`: df, sum of squares, F, p, and an explicit `reject`/`retain` at α=0.05 for H0₁ (no model effect), H0₂ (no workload effect), H0₃ (no interaction) |
| `tukey.csv` | Tukey HSD pairwise comparisons for whichever effects were significant |
| `tukey_status.csv` | whether post-hoc ran for each effect **and why** — always written, including when it was skipped |
| `curves_hit_rate.csv`, `curves_precision.csv` | tidy threshold-vs-metric tables, 5 thresholds × 6 (model, workload) pairs, carrying the harness zero-division flags |
| `kappa.json` | Cohen's kappa from the dataset tooling (not reimplemented), with the 2×2 contingency, the 0.7 bar, and a provenance block labelling fixture-derived values `FIXTURE ONLY` |
| `figures/` | `curve_hit_rate.{png,pdf}`, `curve_precision.{png,pdf}` — regenerable from the tables alone; the hit-rate figure carries the 30% economic-viability line |
| `analysis_meta.json` | input paths, ANOVA diagnostics (design balance, Shapiro-Wilk, Levene), library versions — the only place a timestamp appears |

Exit is non-zero on a harness-contract or design violation, and no partial
bundle is written.

**3c. Replication check (±5%)** — frozen Success Criterion 3.

```bash
python scripts/check_replication.py \
    --reference results/reproduce/results.csv \
    --dataset data/ground_truth.csv
```

Re-runs the harness over exactly the grid recorded in the reference
`results.csv` and compares precision and recall per configuration against:

```
|candidate - reference| <= max(0.01, 5% * |reference|)
```

The absolute floor exists because a purely relative tolerance collapses into
demanding bit-exact equality near a reference of 0.0. Exit code is zero when
every value is within tolerance; otherwise non-zero, with a per-configuration
diff table naming the configuration, the metric, both values, and the deviation.
Under mock providers the harness is byte-deterministic, so a self-comparison
matches exactly — the tolerance is there for real-provider runs.

The verdict is also written to `replication.json` beside the reference
`results.csv`, so it can be read rather than inferred from an exit code:

| Field | Contents |
|---|---|
| `passed` | whether every comparison was within tolerance |
| `rule`, `relative_tolerance`, `absolute_floor` | the tolerance actually applied |
| `n_comparisons`, `n_out_of_tolerance` | how many values were compared, and how many failed |
| `config_ids` | **which configurations this verdict covers** — a run over a subset of the grid produces a valid verdict for that subset only |
| `comparisons` | per (configuration, metric): reference, candidate, deviation, tolerance, within/outside |

Use `--out-json` to write it elsewhere, or `--no-json` to skip it.

### Expected output

The pipeline prints three stage banners and ends with:

```
[1/3] Experiment sweep -> results/reproduce
[run_experiments] wrote 30 configuration result(s) to results/reproduce

[2/3] Statistical analysis -> results/reproduce/analysis
[run_analysis] analysed 30 configuration(s) from results/reproduce
  H0_1 (model): p=nan -> undefined
  H0_2 (workload): p=nan -> undefined
  H0_3 (model:workload): p=nan -> undefined
  Tukey HSD was not run: fpr has zero variance across all configurations, ...
  kappa=0.722... (threshold 0.7, FIXTURE ONLY)
[run_analysis] wrote the analysis bundle to results/reproduce/analysis

[3/3] Replication check against results/reproduce/results.csv
RESULT: replication PASSED
```

The `undefined` hypothesis decisions are the **correct** result for the fixture
dataset under mock embeddings, not a failure: mock embeddings are text-hashed
random vectors, so no semantic hits occur, false positive rate is constant at
zero across all 30 configurations, and the ANOVA F-tests are genuinely
undefined. The pipeline reports that rather than laundering it into three
retained nulls.

---

## Swapping in the real dataset

The pipeline is dataset-agnostic. Moving from the committed fixture to the real
900-pair dataset changes **exactly one argument** — the dataset path. No
different steps, no separate guide, no code change. (Produce that file first
with Step 2b above: acquire the corpora, then rehydrate.)

```bash
scripts/reproduce.sh data/ground_truth.full.csv results/run-001
```

or, stage by stage, the same `--dataset` value in each of the three commands
above. In Docker:

```bash
docker compose run --rm -e LEVY_DATASET=data/ground_truth.full.csv pipeline
```

For an actual research run, also switch to real embeddings — this downloads
model weights on first use and needs network access:

```bash
LEVY_EMBEDDING_PROVIDER=sentence-transformers \
    scripts/reproduce.sh data/ground_truth.full.csv results/run-001
```

The output structure is identical; only the numbers change. With real
embeddings the false positive rate varies across configurations, so the ANOVA
F-tests become defined and H0₁–H0₃ get real `reject`/`retain` decisions, with
Tukey HSD following up any significant effect.

### Expected κ on the real dataset — 0.5000, below the 0.7 bar

Worth stating plainly, so a result that looks like a mistake is not mistaken for
one. Running the kappa tool on the real dataset gives:

```bash
python scripts/compute_kappa.py --dataset data/ground_truth.full.json --strict
```

```text
overall: kappa=0.5000  (faq 0.5267, code 0.4200, chat 0.5533)
```

`--strict` **exits non-zero**, because the frozen success criterion is κ > 0.7.
That is the real, recorded outcome, not a setup error on your side — the earlier
`0.722 (FIXTURE ONLY)` figure in [Expected output](#expected-output) comes from
the 15 synthetic fixture pairs and is not comparable.

The shortfall is a property of the corpora rather than of the annotation: their
positive classes ("closed as a duplicate on Stack Overflow", "3 or more of 5
crowdworkers called it a paraphrase") are looser than the study's question of
whether one cached answer would serve both queries. The full breakdown,
confusion matrix and the contingency options are in
[`data/DATASHEET.md`](../data/DATASHEET.md) §4.

Consequence for reading D3: every precision, false-positive rate and ANOVA
p-value is computed against `author_label`, since that is what
`QueryPair.ground_truth_label()` returns. Those numbers are valid, but they are
relative to that label set — evaluating against `original_label` instead would
give materially different figures on the 225 pairs where the two disagree.

---

## Latency: what a lookup costs, and what a hit avoids

The Project Proposal assesses economic viability on hit rate **and** latency —
"cache lookup overhead vs LLM call savings". The two halves are measured
separately because only one of them replicates.

### The reproducible half — lookup overhead (offline)

Runs offline with mock providers and makes no provider call. It reads the
configuration list from an existing `results.csv` (read-only; nothing is
written to that directory) and measures each configuration's lookup path:

```bash
python scripts/run_latency.py --reference results/reproduce/results.csv --out-dir results/latency-faq
```

Writes `latency.csv` — per configuration, p50/p95 for embedding **cold** and
**warm**, index search, exact-cache lookup and the total lookup path — and
`latency_meta.json`, which records the host specification, library versions,
warm-up and repetition counts, and the reproducibility boundary. Add
`--embedding-provider sentence-transformers` and `--dataset
data/ground_truth.full.csv` to measure the real encoders on the real dataset.

Cold and warm embedding figures are reported separately and are never averaged:
cold is what a query this process has not seen costs, warm is what the
memoised path costs, and both are true of the artefact.

### The non-reproducible half — provider latency (billed, out-of-band)

`scripts/populate_responses.py` is **the second networked entry point in this
repository and the only one that spends money**, alongside
`scripts/fetch_corpora.py`. It calls a real Anthropic model once per unique
prompt of a workload and records the response, its latency and its token
counts. No test invokes it — an AST guard in `tests/test_corpus_acquisition.py`
enforces that — and reproducing the results above does not require running it.

Re-running it will **not** reproduce the published provider figures: they are
specific to the model, provider, region, network path and moment recorded with
them. Every figure derived from them carries its resolved model identifier.

```bash
# Estimate first: prints the call count and sends nothing.
python scripts/populate_responses.py --dataset data/ground_truth.full.csv \
    --workload faq --out-dir results/latency-faq --dry-run
```

Prompts already in the corpus are skipped, so an interrupted run resumes
without paying twice, and the budget guard halts before sending once the
estimated spend reaches its cap. The response corpus itself
(`results/latency-faq/responses.jsonl`) is gitignored: it is model output
generated over licensed corpus text.

## Exploring results interactively (D6, desirable)

Once a bundle exists (Step 3b above, or `results/reproduce/analysis` from the
default pipeline run), an optional local dashboard renders it interactively:

```bash
streamlit run scripts/dashboard.py -- --bundle results/reproduce/analysis
```

Threshold-vs-metric curves, the ANOVA/Tukey/κ summary, and a live query box
are all read from the bundle — nothing is recomputed. This is the frozen
plan's lowest-priority, desirable-only deliverable (D6): it is not part of
`scripts/reproduce.sh` and not required to reproduce any result above. See the
[README](../README.md#results-dashboard-d6-desirable) for the offline story
and the `--dataset` option.

## Serving the cache over HTTP

Not part of the evaluation pipeline, but part of the artefact:

```bash
uvicorn levy.api.app:app --reload
```

Interactive docs at `http://localhost:8000/docs`. Endpoints, the per-request
`cache_config`, the engine pool, and the error contract are documented in the
[README](../README.md#http-api). By default the app builds its engine pool from
`LevyConfig()`, which reads `.env`, so a real deployment needs the configured
provider's credentials.

---

## Release checklist

Everything below is a command, so each claim stays re-runnable rather than being
a one-time assertion.

### Licence / secrets / personal-data audit

```bash
scripts/audit_release.sh
```

| # | Check |
|---|---|
|1|`LICENSE` present, non-empty, and the declared Apache 2.0 licence |
|2|No `.env`, key, or credential file tracked by git (only `.env.example`) |
|3|No secret-shaped string in any tracked file (9 credential patterns) |
|4|No commit on any branch ever introduced a secret-shaped string (`git log --all -S`, pickaxe regex) |
|5|No email or phone-number markers in tracked `data/` files |
|6|No tracked file carries query text attributed to a third-party corpus |
|7|No tracked file contains a string sampled from a populated `data/raw/` |
|8|`.env` is gitignored |

Checks 6 and 7 are the licence gate. Quora Question Pairs grants no
redistribution right and SODD is CC BY-NC-SA 4.0, so the query text must never
reach the remote — check 6 verifies that by attribution (`source_corpus`), and
check 7 does not trust that column at all: it samples real strings out of your
populated `data/raw/` and looks for them in tracked files. Check 7 skips cleanly
when `data/raw/` is empty, which is the state of a clean clone.

Note also what check 4 covers, because it is easy to under-read: it scans
`git log --all`, which includes `refs/stash`. Ignore rules do not apply to
`git stash -u` or `git stash --all`, so the audit is the only thing that sees
corpus text hidden in a stash.

`scripts/audit_release.sh` prints a pass/fail line per check and exits non-zero
on any finding. All eight must pass. The failure path works too: planting a
credential-shaped string in a tracked file makes check 3 fail and the script
exit 1.

### Test suite

```bash
python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90
```

Fully offline — mock providers only, no API key, no network. This is the same
command CI runs (`.github/workflows/tests.yml`).

### Pipeline

```bash
scripts/reproduce.sh
```

Non-zero exit from any stage fails the whole script (`set -euo pipefail`).

### Frozen documents

`docs/Project_Proposal.md` and `docs/Specification_and_Design_Report.md` are the
university submissions and are never modified. Confirm at any time:

```bash
git log --oneline -- docs/Project_Proposal.md docs/Specification_and_Design_Report.md
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: numpy` (or any dependency) | The `levy` conda environment is not active. `conda activate levy`. |
| `scripts/reproduce.sh: Permission denied` | `chmod +x scripts/reproduce.sh`, or run it as `bash scripts/reproduce.sh`. |
| Segfault importing faiss | The pip `faiss-cpu` wheel on Apple Silicon. Install from conda-forge (`environment.yml` already does). |
| Hypothesis decisions are `undefined` | Expected with the fixture dataset under mock embeddings — see [Expected output](#expected-output). |
| Docker build is slow the first time | One-time conda solve plus torch. Subsequent builds reuse the cached environment layer. |
| A model download starts unexpectedly | `LEVY_EMBEDDING_PROVIDER` is not `mock`. The default path never downloads anything. |

## Related documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — component map, request flow, provider abstractions
- [README.md](../README.md) — installation, configuration, HTTP API reference
- [data/DATASHEET.md](../data/DATASHEET.md) — dataset provenance, sampling, annotation protocol
- [docs/Project_Proposal.md](Project_Proposal.md), [docs/Specification_and_Design_Report.md](Specification_and_Design_Report.md) — the frozen research and design baseline
