# data/raw/ — third-party corpora, never committed

This tree is where the raw corpora the study samples from are acquired to. The
**directories are tracked** (via `.gitkeep`) so a clean clone has somewhere to
acquire into; **their contents are gitignored** and never reach the remote.

That is a licence requirement, not a size preference: Quora Question Pairs is
released under Quora's Terms of Service with no redistribution grant, and SODD
is CC BY-NC-SA 4.0 — neither can be redistributed from an Apache-2.0 public
repository. What this repository publishes instead is
`data/ground_truth.ids.csv`: identifiers and labels, no query text. See
`data/README.md` for the full acquire → sample → rehydrate sequence.

The ignore rules live in the repository `.gitignore` (`data/raw/**` plus
negations for the keepers and this file). `scripts/audit_release.sh` enforces
the outcome: it fails if any tracked file carries corpus text.

## What belongs in each directory

Filenames, source URLs, licences and pinned SHA-256 checksums are recorded in
`data/corpora.json`, which is the single source those facts are read from —
this table is a map, not a second copy of the registry.

| Directory | Corpus | Workload | Expected files | Acquisition |
|---|---|---|---|---|
| `quora-qqp/` | Quora Question Pairs | `faq` | `train.tsv` | manual — the hosting platform requires accepting terms |
| `sodd/` | Stack Overflow Duplicity Dataset (MQDD) | `code` | `SODD_train.parquet.gzip`, `SODD_dev.parquet.gzip` | manual — hosted on Google Drive |
| `twitter-pit2015/` | Twitter PIT-2015 (SemEval-2015 Task 1) | `chat` | `train.data`, `dev.data` | automatic |

## Which script populates them

```bash
python scripts/fetch_corpora.py
```

It downloads what it can, verifies every file against the checksum recorded in
`data/corpora.json`, skips files already present that already match, and exits
non-zero printing the exact URL, filename and expected checksum for each corpus
that needs a human step. Run it again after satisfying those steps; it is
idempotent.

On a first acquisition the registry checksums are `null` (nothing is pinned
yet). Pin them once, from the files you actually hold:

```bash
python scripts/fetch_corpora.py --pin
```

After pinning, a checksum mismatch is a hard failure — that is what turns an
upstream re-release into a loud error rather than silent divergence in results.
