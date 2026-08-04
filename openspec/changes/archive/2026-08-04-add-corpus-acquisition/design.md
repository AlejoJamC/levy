## Context

The dataset platform from LEV-3 is complete but sealed at both ends. At the input end, `levy/dataset/sampling.py:5` declares that it downloads nothing and requires raw corpus files to already exist; two of its three adapters (`StackOverflowDuplicatesSource`, `ConvAI2Source`) target corpora the study no longer uses. At the output end, everything `levy/dataset/io.py` writes carries `query_1` and `query_2` (`FIELDNAMES`, line 25), and the FAQ corpus is released under Quora's Terms of Service with no redistribution grant — so the one artifact the platform can produce is the one artifact this repository cannot publish.

The consequence is visible in the tree: `data/ground_truth.{csv,json}` still hold 15 synthetic fixture pairs, and LEV-11 (blind re-annotation, Cohen's kappa, final datasheet numbers) has nothing real to annotate.

Constraints that shape the design:

- **`QueryPair` rejects empty text.** `validate_query_pair` raises when `query_1` or `query_2` is empty or whitespace (`levy/dataset/schema.py:136-139`), and `__post_init__` calls it on every construction. The LEV-4 harness replays against that invariant. An identifiers-only artifact therefore cannot be a `QueryPair` with blanked columns.
- **Offline-first is a repo-wide convention.** Every external dependency has an ABC plus a mock, and the test suite runs with no network. Acquisition breaks that pattern by necessity, so it must be quarantined rather than woven in.
- **Byte-stability is load-bearing.** The ±5% replication criterion, and the existing byte-identical re-run guarantees in the harness and analysis layers, mean any normalisation this change introduces must be deterministic and recorded.
- **The repository is public under Apache 2.0.** Two of the three corpora carry terms incompatible with that (Quora ToS: no redistribution; SODD: CC BY-NC-SA 4.0, non-commercial and share-alike).

## Goals / Non-Goals

**Goals:**

- One command populates `data/raw/` with all three corpora, checksum-verified and idempotently.
- A second command produces the real 900-pair dataset, refusing to substitute synthetic data.
- The artifact committed to the remote contains no third-party query text, and that property is enforced by the release audit rather than trusted.
- A third party reconstructs a byte-identical working dataset from the committed artifact plus their own corpus downloads.
- Adapter choices that change sampled content are explicit and recorded, not implicit.

**Non-Goals:**

- Blind re-annotation, kappa, and final datasheet numbers — LEV-11.
- Supervisor sign-off on the three frozen-document deviations — LEV-11.
- Automating any acquisition step that requires accepting terms on a hosting platform. This repository does not handle credentials.
- Changing the harness input contract. The harness continues to load the full-dataset format via `load_dataset`.

## Decisions

### The identifiers-only artifact is a separate dataclass, not a relaxed `QueryPair`

A new `DistributionRecord` dataclass carries `pair_id`, `workload`, `source_corpus`, `source_pair_id`, `original_label`, `author_label`, `metadata`, with its own reader/writer in `levy/dataset/io.py`.

*Alternative rejected:* make `query_1` / `query_2` optional on `QueryPair`. That weakens an invariant the LEV-4 replay path depends on, and it makes "is this pair usable?" a runtime question at every call site instead of a construction-time guarantee. A dataset that cannot be replayed should not be representable as a replayable pair.

### Rehydration filters a single pass over the corpus; the `CorpusSource` ABC does not grow

`CorpusSource` keeps its single abstract method `iter_candidates()`. Rehydration builds the set of `source_pair_id`s it needs per corpus, makes one linear pass, and collects matches. 900 identifiers against SODD's ~1.4M rows is one scan.

*Alternative rejected:* add a `lookup(ids)` method to the ABC. It would force every adapter — including `MockCorpusSource` — to implement random access that only one caller needs, and indexed lookup buys nothing at this scale.

### Byte-identity of the round-trip comes from ordering, not from re-derivation

The distribution file preserves the sampled order of `pair_id`s. Rehydration emits records in the order the distribution file lists them, then writes through the existing `save_csv` / `save_json`, which are already timestamp-free and deterministic. Byte-identity therefore follows from ordering plus the existing writers, with nothing new to guarantee.

### HTML normalisation for SODD uses the standard library

`first_post` / `second_post` are HTML containing code snippets. Normalisation is a `html.parser.HTMLParser` subclass: drop tags, keep the text content of code blocks, collapse runs of whitespace. The rule is named in the sidecar metadata.

*Alternative rejected:* BeautifulSoup or lxml. Both add a dependency, and both can change their output across versions — which would silently break byte-identity for anyone rehydrating with a different version installed. Stdlib parsing pins the behaviour to the Python version already recorded.

### Checksums are pinned in the tracked registry, recorded on first acquisition

`data/corpora.json` carries a `sha256` field per expected file. A `null` value means "not yet pinned"; running acquisition with `--pin` computes and writes it. After pinning, a mismatch is a hard failure.

*Alternative rejected:* compute checksums dynamically every run and never pin. That detects corruption but not upstream re-release, which is exactly the divergence the replication criterion needs to catch.

### QQP acquisition prints a manual step rather than integrating with Kaggle

Acquisition handles PIT-2015 directly. QQP requires accepting terms on the hosting platform, so acquisition exits non-zero printing the canonical URL, the expected filename and the expected SHA-256.

*Alternative rejected:* integrate the Kaggle API. It requires storing a credential token, which this repository does not do, and it would put a credential path into the one script a replicator is most likely to run.

**Correction recorded during implementation:** this decision originally asserted that SODD was also a plain repository download. It is not — the MQDD authors distribute it as a **Google Drive folder** (`kiv-air/StackOverflowDataset`), which has no stable per-file direct-download URL. SODD therefore takes the same manual path as QQP. No mechanism changed: `manual` is a per-corpus flag in `data/corpora.json`, so this is a registry value, not a second code path. Two of three corpora needing a human step makes the "not literally one command end to end" trade-off below more prominent than anticipated, but every other stage remains automated and idempotent.

Also corrected: PIT-2015's `train.data` and `dev.data` are not served as loose files — they ship inside `data/SemEval-PIT2015-github.zip` in the shared-task repository. A file entry may therefore declare `archive_member`, and acquisition extracts the unique member with that basename. Matching on basename rather than full path keeps the archive's internal layout out of the registry, and zero-or-several matches is an error rather than a pick.

### File naming keeps the three roles unambiguous

| Path | Role | Tracked |
|---|---|---|
| `data/ground_truth.csv` / `.json` | 15 synthetic fixture pairs, unchanged — the offline test suite and `scripts/reproduce.sh` defaults depend on them | yes |
| `data/ground_truth.ids.csv` | the D2 distribution artifact: identifiers and labels, no text | yes |
| `data/ground_truth.ids.meta.json` | sampling sidecar: seed, positive ratio, adapter options, corpus versions and checksums | yes |
| `data/ground_truth.full.csv` / `.json` | rehydrated working dataset, the annotation and replay input | no |

The real dataset reaches the harness through the existing `--dataset` flag on `scripts/run_experiments.py`. The fixture defaults are untouched, so the offline reproduce path keeps working unchanged.

### One tracked sidecar, not a tracked registry plus a local manifest

Everything a replicator needs to prove identical inputs — seed, positive ratio, adapter options, corpus versions, raw-file checksums — goes into the single tracked `data/ground_truth.ids.meta.json`. It contains no query text, so tracking it is safe.

*Alternative rejected:* a separate gitignored run manifest. It would duplicate the same fields with no reader, and the publishable copy is the one that matters.

### Cross-workload overlap is checked by corpus identity, not by content

Pre-flight validation fails if two workloads resolve to the same `source_corpus`. The guard exists to prevent a configuration mistake that would collapse the workload contrast the workload hypothesis tests; content-level near-duplicate detection across corpora is a different and much larger problem, and not the failure mode at hand.

### Pre-flight validation returns a report; the CLI renders it

Validation is a function returning a structured report of all findings. The CLI prints it and derives its exit code from it. This keeps the multi-problem-in-one-pass behaviour unit-testable offline against fixtures, rather than only observable through process exit codes.

### Acquisition is excluded from the test suite; adapters are not

`scripts/fetch_corpora.py` performs network access and is not exercised by pytest. The adapters, the validation report, the distribution writer/reader and the rehydration round-trip are all tested against small committed fixtures — real-format samples small enough to carry no meaningful corpus content. This is the same boundary the repo already draws for provider internals.

## Risks / Trade-offs

- **Upstream re-releases a corpus and checksums stop matching** → the pinned checksum turns this into a loud failure naming both values, instead of silent divergence in results. The datasheet records the pinned snapshot, so the mismatch is diagnosable.
- **SODD's ~1.4M rows across 9 parquet shards are large in memory** → read only the columns each adapter needs, per shard, filtering as it goes and discarding the rest. Never materialise all shards at once.
- **A workload's positive or negative pool turns out too small for 300 pairs at a 0.5 ratio** → pre-flight validation reports the shortfall for all three workloads before anything is written, so the substitution decision is made with full information. The frozen Proposal's Risk 1 contingency path covers corpus substitution.
- **QQP's manual acquisition step means the pipeline is not literally one command end to end** → the step is announced with an exact URL, filename and checksum, is idempotent once satisfied, and is documented in the reproduction guide. Every other stage remains automated.
- **SODD's non-commercial, share-alike terms conflict with the repository's Apache 2.0 licence** → identifiers-only distribution means no SODD text is ever redistributed by this repository, so its terms do not propagate to anyone consuming this repo.
- **Stdlib HTML stripping is less robust than a real parser on malformed markup** → the normalisation rule is recorded in the sidecar and applied identically at sampling and rehydration time, so any imperfection is reproducible rather than divergent. Correctness of the semantic content matters less here than both parties producing the same bytes.
- **The removed CLI flags are a breaking change** → they configured adapters for corpora the study no longer uses, so no working invocation can depend on them. The spec delta records the migration.

## Migration Plan

1. Add `data/raw/` with keepers and the `.gitignore` rules; verify with `git check-ignore -v` and a clean-clone check.
2. Add `data/corpora.json` with the three corpora and `null` checksums.
3. Add the SODD and PIT-2015 adapters and their fixtures; remove `StackOverflowDuplicatesSource`, `ConvAI2Source` and their CLI flags in the same commit so no dead adapter is ever on `main`.
4. Add the `DistributionRecord` type, its reader/writer, and the round-trip test — before any real data exists, so the licence-safe path is proven on fixtures first.
5. Add pre-flight validation and the `--require-real` mode.
6. Add `scripts/fetch_corpora.py` and `scripts/rehydrate_dataset.py`.
7. Extend `scripts/audit_release.sh` with the corpus-text gate and verify its failure path by planting a violation.
8. Run acquisition, pin the checksums, sample the 900 pairs, and commit only `data/ground_truth.ids.csv` and its sidecar.
9. Update `data/DATASHEET.md`, `data/README.md`, `docs/REPRODUCTION.md` and `CLAUDE.md`.

Rollback: steps 1 through 7 are additive except the adapter removal in step 3, which is revertible on its own. Nothing in the existing offline reproduce path changes, because the fixture defaults and the harness input contract are untouched.
