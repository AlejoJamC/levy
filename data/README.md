# data/

This directory holds the ground-truth dataset for the Levy capstone study
(Deliverable D2 — 900 annotated query pairs across 3 workloads).

## ~~Current contents are placeholders~~ The fixtures are permanent, not placeholders

> **Update 2026-08-04.** The heading above is struck through, not removed: it
> was written before the real dataset existed, and "placeholder" implied the
> fixtures would eventually be replaced. They will not be. The real 900 pairs
> ship **beside** them as `ground_truth.ids.csv` plus the rehydration script —
> never in place of them — because committing the query text would breach the
> Quora and SODD licences (see `DATASHEET.md` §2 deviation 3, §6, §7).
> `ground_truth.{csv,json}` keep their permanent role as the offline default for
> the test suite and `scripts/reproduce.sh`. Everything the section below says
> about *what those files contain* remains accurate.
>
> Status of the real dataset as of 2026-08-04: sampled (seed 42), fully
> re-annotated (900/900), and the `author_label` column of
> `ground_truth.ids.csv` is populated for all 900 rows — so the published
> artifact carries the blind re-annotation, and a replicator's
> `ground_truth_label()` uses it rather than falling back to `original_label`.
> Cohen's κ = 0.3267, **below the frozen κ > 0.7 criterion**; that is escalated
> as a research finding, not worked around — see `DATASHEET.md` §4.

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

The sequence is acquire → sample → rehydrate → annotate → kappa. The
step-by-step procedure, with every command and the manual download steps, is
**[`docs/DATA_PRODUCTION.md`](../docs/DATA_PRODUCTION.md)** — the single place
it is written down, so it cannot drift here.

In short: `scripts/fetch_corpora.py` populates `data/raw/` (exiting non-zero
with the exact URL, filename and expected checksum for each corpus needing a
human step); `scripts/sample_dataset.py --require-real` draws the 900 pairs
behind a pre-flight validation gate that reports every problem across all
three workloads at once and writes nothing if any of them fails; and
`scripts/rehydrate_dataset.py` rebuilds the query text from `data/raw/`.

**A reader reproducing the study runs only the fetch and the rehydrate** —
rehydration alone reconstructs a byte-identical copy of the dataset the author
sampled. See [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).

See `data/DATASHEET.md` for corpus licences, the sampling protocol, the three
recorded deviations from the frozen documents, and known limitations. The
experiment harness reads the rehydrated dataset through the existing
`--dataset` flag; the fixture defaults are untouched, so the offline pipeline
keeps working unchanged.
