# data/

Ground-truth dataset for the Levy study (deliverable D2): 900 annotated query
pairs, 300 per workload (FAQ, code, chat).

The real query text is **not in this repository**. Quora Question Pairs grants
no redistribution right and SODD is CC BY-NC-SA 4.0, so what is published is
which pairs were sampled and how they are labelled; you supply the corpora and
rebuild the text locally. `scripts/audit_release.sh` enforces this rather than
trusting it.

## What is committed, and what is generated

| File | What it is | Tracked |
|---|---|---|
| `ground_truth.ids.csv` | **The released dataset** — identifiers and labels for the 900 pairs (`original_label` and the blind re-annotation `author_label`), no query text | yes |
| `ground_truth.ids.meta.json` | Sampling manifest: per-workload seed, positive ratio, adapter options, corpus snapshots, input checksums, tool versions | yes |
| `corpora.json` | Corpus provenance registry: URL, snapshot, licence, filenames, SHA-256, citation. Read by code | yes |
| `ground_truth.csv` / `.json` | 15 **synthetic fixture pairs** (5 per workload, obviously fake text, `source_corpus = "synthetic-fixture"`). The offline test suite and the defaults of `scripts/reproduce.sh` depend on them. Not research data | yes |
| `annotation_progress.json` | The blind re-annotation's per-answer progress: each label fingerprinted with its pair's `source_pair_id`, plus the session's presentation order and seed. No query text | yes |
| `raw/` | Acquired third-party corpora — directories tracked, contents never committed (see `raw/README.md`) | dirs only |
| `ground_truth.full.csv` / `.json` | The rehydrated working dataset, query text included | no — gitignored |
| `backups/` | Timestamped copies made before any overwrite of the files above (`levy/dataset/backup.py`). Never deleted by the tooling; snapshots of `ground_truth.full.*` carry corpus text | no — gitignored |

There is exactly one ground truth, at the canonical paths above. A re-sample or
re-annotation rewrites them, after backing up what it overwrites; it never
creates a second, versioned copy.

No metric computed against the 15 fixture pairs is meaningful. They only
exercise code paths.

## Reproducing the real dataset

The sequence is acquire → sample → rehydrate → annotate → kappa. The procedure,
with every command and the manual download steps, is in
[`docs/DATA_PRODUCTION.md`](../docs/DATA_PRODUCTION.md), the single place it is
written down.

**A reader reproducing the study only needs to fetch the corpora and
rehydrate:**

```bash
python scripts/fetch_corpora.py       # populates data/raw/, checksum-verified
python scripts/rehydrate_dataset.py   # ids + data/raw/ -> data/ground_truth.full.*
```

Rehydration reconstructs a byte-identical copy of the dataset that was sampled.
See [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) for running the experiments
against it.

## Current state

| Workload | Corpus | Seed |
|---|---|---:|
| faq | Quora Question Pairs | 42 |
| code | SODD | 8484 |
| chat | Twitter PIT-2015 | 4242 |

All 900 pairs are blind re-annotated. Cohen's κ between the corpus label and
the author's label is **0.5000** overall (faq 0.5267, code 0.4200, chat 0.5533),
below the κ > 0.7 criterion. This is reported as a finding, not worked around;
the breakdown is in `DATASHEET.md` §4.

## One workload at a time

`scripts/sample_dataset.py --workload chat` re-draws a single workload in place.
It reads only that workload's corpus, replaces that workload's rows, clears
their `author_label` so annotation presents exactly them, and leaves the other
workloads' rows and labels untouched. New pairs exclude every `source_pair_id`
already in the dataset, and the replaced pairs' entries are dropped from
`annotation_progress.json`. `scripts/annotate_dataset.py --workload chat` then
annotates just those. Details: [`docs/DATA_PRODUCTION.md`](../docs/DATA_PRODUCTION.md).

See [`DATASHEET.md`](DATASHEET.md) for corpus licences, the sampling protocol,
corpus choices and known limitations.
