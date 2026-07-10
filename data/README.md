# data/

This directory holds the ground-truth dataset for the Levy capstone study
(Deliverable D2 — 900 annotated query pairs across 3 workloads).

## Current contents are placeholders

`ground_truth.csv` and `ground_truth.json` in this directory contain **15
synthetic fixture pairs (5 per workload)**, not real data. Every row carries
`source_corpus = "synthetic-fixture"` and `metadata.provenance =
"synthetic-fixture"`. All query text is obviously fabricated (e.g.
"fixture-account", "fixture-lang", "fixture-work") — none of it comes from
Quora Question Pairs, Stack Overflow, ConvAI2, or any other real corpus.

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

## What replaces this

The real dataset — 900 pairs (300 per workload: FAQ, code, chat) sampled
from Quora Question Pairs, Stack Overflow duplicate questions, and
ConvAI2-derived same-intent pairs, each carrying the original corpus label
plus the author's blind re-annotation — is produced by:

1. `scripts/sample_dataset.py` pointed at the real downloaded corpus files
   (see `levy/dataset/sampling.py` docstrings for the exact expected raw
   format per corpus).
2. `scripts/annotate_dataset.py` run by the author in blind mode (original
   labels are never shown during annotation).
3. `scripts/compute_kappa.py` to confirm Cohen's kappa > 0.7 over the full
   900 pairs.

See `data/DATASHEET.md` for the full protocol, corpus licences, and
limitations, and `openspec/changes/add-ground-truth-dataset/proposal.md`
for the platform-tooling vs. data-production split (LEV-3).

Once the real 900-pair dataset is produced, it replaces
`ground_truth.csv` / `ground_truth.json` in place (same filenames, same
schema) — no downstream code should need to change.
