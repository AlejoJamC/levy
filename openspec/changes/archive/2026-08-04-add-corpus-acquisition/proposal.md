## Why

The dataset platform shipped (LEV-3) but it cannot acquire data: `levy/dataset/sampling.py` states outright that it downloads nothing and requires raw corpus files to already exist on disk, and two of its three adapters point at corpora that are no longer being used. `data/ground_truth.{csv,json}` therefore still holds 15 synthetic fixture pairs instead of the real 900. Separately, everything the platform can write carries the query text — `levy/dataset/io.py` `FIELDNAMES` includes `query_1` and `query_2` — but the FAQ corpus (Quora Question Pairs) is released under Quora's Terms of Service with no redistribution grant, so committing that text is not an option for a public Apache-2.0 repository.

This change closes both gaps so that D2 data production (LEV-11: blind re-annotation, Cohen's kappa, final datasheet numbers) can start from a real, sampled, licence-clean 900-pair corpus produced by one command.

## What Changes

- **New corpus acquisition layer**: `scripts/fetch_corpora.py` downloads all three corpora into `data/raw/<corpus>/`, idempotently and checksum-verified. Where a corpus requires a human acquisition step, the script exits non-zero naming the exact URL, filename and expected SHA-256 rather than proceeding partially.
- **`data/raw/` exists in the tree but never carries content to the remote**: `.gitkeep` files per corpus directory, with contents gitignored via `data/raw/**` plus a `!data/raw/**/.gitkeep` negation.
- **Corpus adapters replaced to match the final corpora**: add `SODDSource` (gzipped parquet) for the `code` workload and `TwitterPIT2015Source` (tab-separated) for the `chat` workload. `QuoraQQPSource` is unchanged.
- **BREAKING**: `StackOverflowDuplicatesSource` and `ConvAI2Source` are removed, along with the `--stackoverflow-csv` and `--convai2-json` flags on `scripts/sample_dataset.py`. Both adapters target corpora that are no longer part of the study.
- **Pre-flight validation before sampling**: one pass reporting every problem at once — file presence, checksum match, required columns per adapter, per-class pool sizes for all three workloads simultaneously, label value domains, and absence of cross-workload corpus overlap. Nothing is written on failure.
- **`--require-real` on `scripts/sample_dataset.py`**: makes a `MockCorpusSource` fallback a hard error, so a production run cannot silently emit `source_corpus="mock"`.
- **Licence-safe distribution of D2**: a new ids-only record type in its own file (`pair_id`, `workload`, `source_corpus`, `source_pair_id`, `original_label`, `author_label`, `metadata` — no query text) plus `scripts/rehydrate_dataset.py` which reconstructs the full working dataset from that file and a populated `data/raw/`.
- **Provenance made machine-readable**: `data/corpora.json` records per corpus the canonical URL, snapshot version, licence, expected filenames, SHA-256 checksums and citation; a run manifest records checksums, seed, `positive_ratio`, adapter options and tool versions. Both are read by code, not merely documented.
- **Enforced gate on what reaches the remote**: `scripts/audit_release.sh` gains a check that fails if any tracked file contains raw corpus text, with the failure path verified by planting a violation.

### Final corpora

| Workload | Corpus | Positive label | Licence |
|---|---|---|---|
| `faq` | Quora Question Pairs | `is_duplicate == 1`, Quora's human labelling | Quora Terms of Service, non-commercial, no redistribution grant |
| `code` | SODD — Stack Overflow Duplicity Dataset (MQDD, Pasek et al., RANLP 2023) | `label == 0` (`duplicates`) = Stack Overflow duplicate closures from the archive.org SO dump of June 2020 | CC BY-NC-SA 4.0 |
| `chat` | Twitter PIT-2015 (SemEval-2015 Task 1, Xu et al.) | binary paraphrase label, crowdsourced via Amazon Mechanical Turk | SemEval-2015 shared-task release, recorded in `data/corpora.json` |

All three are distributed from this repo as ids plus labels only. That treatment is uniform regardless of each licence's individual redistribution terms.

### Deviations from the frozen documents

Surfaced here rather than resolved silently, per the project rule. These are decisions; supervisor sign-off is tracked in LEV-11.

1. **Code workload corpus**: "Stack Overflow duplicate questions" → SODD. Same underlying source (Stack Overflow duplicate closures) in a published pre-processed release, avoiding the current Stack Exchange dump access situation.
2. **Chat workload corpus**: ConvAI2 → Twitter PIT-2015. ConvAI2 ships dialogues, not pair-level human same-intent labels, so it cannot supply the original human label that the Cohen's kappa criterion compares against.
3. **D2 release format**: "CSV and JSON formats" carrying query text → ids plus labels plus a rehydration script, because Quora Question Pairs grants no redistribution right. This is the approach Google uses for PAWS-QQP. The ±5% replication criterion is preserved through checksums of the raw inputs.

## Capabilities

### New Capabilities
- `corpus-acquisition`: fetching raw corpora into a fixed, content-ignored location; per-corpus provenance and checksum registry; pre-flight technical validation of the raw inputs; licence-safe ids-only distribution of the sampled dataset and its rehydration.

### Modified Capabilities
- `ground-truth-dataset`: three requirements change. "Seeded, stratified sampling from a corpus source" — the adapter set is replaced (SODD and PIT-2015 in, Stack Overflow duplicates CSV and ConvAI2 out), adapters must declare their label mapping, content-affecting adapter options become explicit, and sampling gains a pre-flight validation gate plus a mock-refusing mode. "CSV/JSON persistence" — an ids-only distribution format is added alongside the full-dataset format, as a distinct type rather than a weakened `QueryPair`, and remains excluded as a harness input. "Offline CLIs over the dataset platform" — rehydration joins the covered entry points, and corpus acquisition is named as the sole network-touching entry point, excluded from the test suite.
- `release-packaging`: the "Recorded, re-runnable release audit" requirement changes — the audit additionally fails when any tracked file contains third-party corpus text.

## Impact

- **Code**: `levy/dataset/sampling.py` (adapters added and removed, validation), `levy/dataset/io.py` (ids-only writer/reader alongside the existing full-dataset one), `levy/dataset/schema.py` (new ids-only record type; `QueryPair`'s non-empty-text invariant is deliberately left untouched).
- **Scripts**: new `scripts/fetch_corpora.py` and `scripts/rehydrate_dataset.py`; `scripts/sample_dataset.py` flags change; `scripts/audit_release.sh` gains a check.
- **Repository layout**: `data/raw/` tree with keepers, `data/corpora.json`, `.gitignore` entries. `data/ground_truth.{csv,json}` keep their 15 synthetic fixture pairs — synthetic text carries no third-party licence, and the offline test suite plus the `scripts/reproduce.sh` defaults depend on them.
- **Dependencies**: parquet reading (pandas with pyarrow) for SODD, added to `environment.yml` and `pyproject.toml`.
- **Tests**: new offline tests over committed fixtures; network-touching code stays outside the pytest suite, consistent with the repo's offline-test convention. The ≥90% branch-coverage gate on `levy/` continues to apply.
- **Docs**: `data/DATASHEET.md`, `data/README.md`, `docs/REPRODUCTION.md`, `CLAUDE.md`.
- **Out of scope** (LEV-11): blind re-annotation of the 900 pairs, the Cohen's kappa result, final datasheet numbers, supervisor sign-off.
