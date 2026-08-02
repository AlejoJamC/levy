## 1. Repository layout and provenance registry

- [x] 1.1 Create `data/raw/` with `.gitkeep`, plus `data/raw/quora-qqp/`, `data/raw/sodd/`, `data/raw/twitter-pit2015/` each with `.gitkeep`
- [x] 1.2 Add `.gitignore` rules ignoring `data/raw/**` with a `!data/raw/**/.gitkeep` negation
- [x] 1.3 Verify the ignore rules with `git check-ignore -v` on a planted file in each corpus directory, and confirm `git status` stays clean
- [x] 1.4 Write `data/raw/README.md`: what belongs in each directory, that contents are never committed, and which script populates them
- [x] 1.5 Create `data/corpora.json` with the three corpora — canonical URL, snapshot/version, licence, expected filenames, `sha256: null`, citation
- [x] 1.6 Add `levy/dataset/corpora.py` reading and validating `data/corpora.json`, raising a typed error naming the missing corpus or field
- [x] 1.7 Add `pyarrow` to `environment.yml` (conda-forge) and to `pyproject.toml`; confirm `pandas` is already present

## 2. Corpus adapters

- [x] 2.1 Add committed fixtures: a small QQP-format TSV, a small SODD-format parquet shard, a small PIT-2015-format TSV, each with real column structure and synthetic content
- [x] 2.2 Implement the stdlib `HTMLParser` normaliser: drop tags, keep code-block text content, collapse whitespace; unit-test it on malformed markup and on markup containing code
- [x] 2.3 Implement `SODDSource(CorpusSource)` reading gzipped parquet per shard with a column subset, positive `label == 0`, negative `label == 3` by default, classes `1`/`2` exposed behind a hard-negative option
- [x] 2.4 Implement `TwitterPIT2015Source(CorpusSource)` reading the tab-separated release, using train and dev splits only, rejecting the graded test split rather than coercing its 0–5 scale
- [x] 2.5 Have both adapters declare their positive/negative label mapping and their label domain, so validation can check for unexpected values
- [x] 2.6 Delete `StackOverflowDuplicatesSource` and `ConvAI2Source`, and the `--stackoverflow-csv` / `--convai2-json` flags on `scripts/sample_dataset.py`, in one commit
- [x] 2.7 Update `levy/dataset/sampling.py` module docstring: adapters no longer name dropped corpora, and the fallback-corpora note reflects the current set
- [x] 2.8 Unit-test both adapters against their fixtures: field mapping, label mapping, option effects, unexpected-label rejection

## 3. Identifiers-only distribution format

- [x] 3.1 Add `DistributionRecord` to `levy/dataset/schema.py` with `pair_id`, `workload`, `source_corpus`, `source_pair_id`, `original_label`, `author_label`, `metadata`, and its own validation
- [x] 3.2 Add a test asserting `QueryPair` still rejects empty and whitespace-only query text, so the invariant is pinned against future relaxation
- [x] 3.3 Add reader and writer for the distribution format in `levy/dataset/io.py`, with their own required-field list and error messages naming file and row
- [x] 3.4 Add `to_distribution_records(pairs)` producing distribution records in the sampled order of the input pairs
- [x] 3.5 Make `load_dataset` reject a distribution file with an error directing the caller to rehydrate first, rather than failing on absent text
- [x] 3.6 Unit-test the distribution format: round-trip, no text in any field, missing-field error, `author_label` absent decoding to `None`

## 4. Pre-flight validation and mock refusal

- [x] 4.1 Add `levy/dataset/validation.py` returning a structured report of all findings rather than raising on the first
- [x] 4.2 Implement the checks: file presence and readability, checksum agreement with `data/corpora.json`, required fields per adapter, positive and negative pool sufficiency for all three workloads at once, label values within the declared domain, no two workloads resolving to the same `source_corpus`
- [x] 4.3 Wire validation into `scripts/sample_dataset.py` so it runs before sampling and writes nothing on failure
- [x] 4.4 Add `--require-real` to `scripts/sample_dataset.py` making a `MockCorpusSource` fallback a hard error naming the missing corpus
- [x] 4.5 Unit-test validation offline: two simultaneous problems both reported, nothing written on failure, pool shortfall detected pre-sampling, cross-workload overlap rejected, unexpected label value reported

## 5. Acquisition and rehydration executables

- [x] 5.1 Write `scripts/fetch_corpora.py` downloading SODD parquet shards and the PIT-2015 release into their `data/raw/` directories, verifying each against `data/corpora.json`
- [x] 5.2 Make acquisition idempotent: skip files already present whose checksum matches; exit non-zero on mismatch naming the file and both checksums
- [x] 5.3 Implement the QQP manual-step path: exit non-zero printing canonical URL, expected filename and expected SHA-256, leaving no partially acquired corpus
- [x] 5.4 Add `--pin` writing computed checksums back into `data/corpora.json` for entries whose `sha256` is `null`
- [x] 5.5 Write `scripts/rehydrate_dataset.py` reconstructing the full dataset from the distribution file plus `data/raw/`, using one filtered pass per corpus and preserving the distribution file's `pair_id` order
- [x] 5.6 Make rehydration exit non-zero on an unpopulated or incomplete `data/raw/`, naming the missing corpus and the acquisition script, writing no partial dataset
- [x] 5.7 Make rehydration exit non-zero on a `source_pair_id` absent from its corpus, naming the record and the corpus, rather than dropping the pair
- [x] 5.8 Write the sampling sidecar `data/ground_truth.ids.meta.json` — seed, positive ratio, adapter options, corpus versions and pinned checksums, tool versions — and verify it contains no query text
- [x] 5.9 Add the byte-identical round-trip test on fixtures: sample → distribution records → rehydrate → compare bytes against the originally sampled file
- [x] 5.10 Confirm no acquisition code path is reachable from the pytest suite

## 6. Enforced gate on what reaches the remote

- [x] 6.1 Extend `scripts/audit_release.sh` with a check asserting no tracked file under `data/` carries query-text columns
- [x] 6.2 Add a spot-check comparing tracked file contents against strings sampled from a populated `data/raw/`, skipping cleanly when `data/raw/` is empty
- [x] 6.3 Verify the failure path by planting corpus text in a tracked file and confirming a non-zero exit naming the file
- [x] 6.4 Confirm the check passes with the 15 synthetic fixture pairs present, since synthetic text carries no third-party licence
- [x] 6.5 Add `data/ground_truth.full.csv` and `data/ground_truth.full.json` to `.gitignore`

## 7. Production run

- [ ] 7.1 Acquire all three corpora and pin their checksums in `data/corpora.json`
- [ ] 7.2 Run pre-flight validation against the real corpora and resolve every finding before sampling
- [ ] 7.3 Sample 900 pairs with `--require-real`, 300 per workload at a 0.5 positive ratio, seed 42
- [ ] 7.4 Rehydrate and confirm the round-trip is byte-identical on the real 900-pair dataset, not only on fixtures
- [ ] 7.5 Commit `data/ground_truth.ids.csv` and `data/ground_truth.ids.meta.json` only; confirm `git status` shows nothing under `data/raw/` and no `.full.` file
- [ ] 7.6 Run `scripts/audit_release.sh` against the populated tree and confirm it exits zero

## 8. Tests and documentation

- [x] 8.1 Add `tests/test_corpus_acquisition.py` covering the registry reader, validation report, distribution format and rehydration round-trip; all offline
- [x] 8.2 Extend `tests/test_dataset.py` for the new adapters and the removed ones, and update any test referencing the deleted CLI flags
- [x] 8.3 Run `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` and confirm it passes
- [x] 8.4 Update `data/DATASHEET.md` §2 corpus table, §3 sampling protocol and §6 distribution for QQP / SODD / PIT-2015, their licences and the identifiers-only model, leaving the LEV-11 markers for kappa, final counts and per-workload prevalence
- [x] 8.5 Record the three frozen-document deviations in `data/DATASHEET.md` as decisions with their rationale
- [x] 8.6 Update `data/README.md`: what is committed versus generated, and the acquisition-then-sample-then-rehydrate sequence
- [x] 8.7 Add the corpus-acquisition step to `docs/REPRODUCTION.md` before the pipeline, referencing the scripts rather than restating commands
- [x] 8.8 Update `CLAUDE.md`: `levy/dataset/` and `data/` descriptions, the commands section, and the known-gap #6 wording
- [x] 8.9 Run `openspec validate --all` and confirm it passes
