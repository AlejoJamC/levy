# data/

This directory holds the ground-truth dataset for the Levy capstone study
(Deliverable D2 — 900 annotated query pairs across 3 workloads).

## Current contents are placeholders

`ground_truth.csv` and `ground_truth.json` in this directory contain **15
synthetic fixture pairs (5 per workload)**, not real data. Every row carries
`source_corpus = "synthetic-fixture"` and `metadata.provenance =
"synthetic-fixture"`. All query text is obviously fabricated (e.g.
"fixture-account", "fixture-lang", "fixture-work") — none of it comes from
Quora Question Pairs, SODD, Twitter PIT-2015, or any other real corpus.

These files exist so that:

- the schema (`levy/dataset/schema.py`), CSV/JSON loader
  (`levy/dataset/io.py`), sampling pipeline (`levy/dataset/sampling.py`),
  blind annotation tool (`levy/dataset/annotation.py`), and Cohen's kappa
  calculator (`levy/dataset/kappa.py`) all have something concrete to run
  against in tests and offline CLI smoke tests;
- LEV-4 (the experiment harness) can be developed and unit-tested against a
  dataset with the exact shape the real dataset will have, before the real
  dataset exists.

**Downstream code (in particular LEV-4's `run_experiment`) MUST NOT treat
this fixture data as research data.** No metric computed against these 15
pairs is meaningful for the dissertation; it only exercises code paths.

## What is committed, and what is generated

| File | What it is | Tracked |
|---|---|---|
| `ground_truth.csv` / `.json` | the 15 synthetic fixture pairs above; the offline test suite and `scripts/reproduce.sh` defaults depend on them | yes |
| `corpora.json` | corpus provenance registry: URL, snapshot, licence, filenames, SHA-256, citation per corpus. Read by code, not just documentation | yes |
| `ground_truth.ids.csv` | **the released D2 artifact** — identifiers and labels for the real 900 pairs, no query text | yes |
| `ground_truth.ids.meta.json` | sampling sidecar: seed, positive ratio, adapter options, corpus snapshots, input checksums, tool versions | yes |
| `raw/` | acquired third-party corpora — directories tracked, contents never committed (see `raw/README.md`) | dirs only |
| `ground_truth.full.csv` / `.json` | the rehydrated working dataset, query text included | no — gitignored |

The real query text is **not in this repository and never will be**: Quora
Question Pairs grants no redistribution right and SODD is CC BY-NC-SA 4.0. What
is published is which pairs were sampled and how they are labeled; you supply
the corpora. `scripts/audit_release.sh` enforces this rather than trusting it.

## Producing (or reproducing) the real dataset

```bash
python scripts/fetch_corpora.py        # 1. acquire corpora into data/raw/
python scripts/sample_dataset.py \     # 2. sample — author, once
    --require-real --n-per-workload 300 --seed 42 \
    --out-csv data/ground_truth.full.csv \
    --out-json data/ground_truth.full.json \
    --out-ids data/ground_truth.ids.csv
python scripts/rehydrate_dataset.py    # 3. rebuild the text from data/raw/
```

Step 1 exits non-zero with the exact URL, filename and expected checksum for
each corpus needing a human step. Step 2 refuses to substitute synthetic data
(`--require-real`) and runs a pre-flight validation pass that reports every
problem across all three workloads at once, writing nothing if any of them
fails. A reader reproducing the dataset runs steps 1 and 3 only — step 3 alone
reconstructs a byte-identical copy of what step 2 produced.

Then, for the author's annotation pass:

4. `scripts/annotate_dataset.py` in blind mode (original labels are never
   shown during annotation), against the rehydrated dataset.
5. `scripts/compute_kappa.py` to confirm Cohen's kappa > 0.7 over the 900
   pairs.

See `data/DATASHEET.md` for the full protocol, corpus licences, the three
recorded deviations from the frozen documents, and known limitations. The
experiment harness reads the rehydrated dataset through the existing
`--dataset` flag; the fixture defaults are untouched, so the offline pipeline
keeps working unchanged.
