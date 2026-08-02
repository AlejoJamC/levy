"""
Seeded, stratified sampling pipeline for the ground-truth dataset (LEV-3 / D2,
extended by LEV-12).

This module builds `QueryPair` lists from labeled candidate pairs exposed by a
`CorpusSource`. It does **not** download anything: adapters read local raw
files already present under `data/raw/`, in the format documented on each
adapter class. Acquiring those files is `scripts/fetch_corpora.py`, the only
entry point in the repository permitted to touch the network.

Adapters exist for exactly the corpora the study uses — Quora Question Pairs
(`faq`), the Stack Overflow Duplicity Dataset (`code`) and Twitter PIT-2015
(`chat`) — plus `MockCorpusSource` for offline smoke tests. Each adapter
declares, alongside its parsing:

- `label_mapping()`: which native label values mean duplicate, which mean
  non-duplicate, and the full domain of values the corpus may legally carry.
  Anything outside the domain is rejected rather than coerced; anything inside
  it but in neither class (a debatable pair, say) is skipped.
- `options()`: every adapter choice that changes which pairs are produced or
  how their text is normalised. These land in the run manifest, so a
  replicator can reproduce the same sample.
- `check_fields()`: a cheap, read-the-header-only check that
  `levy.dataset.validation` runs before any sampling happens.

Fallback corpora named in the frozen risk plan (Project_Proposal.md Risk 1) if
a primary corpus is unavailable or too small: MS MARCO (FAQ/QA fallback),
CodeSearchNet (code fallback), DailyDialog (chat fallback). Adapters for those
follow the same `CorpusSource` interface; they are not implemented here
because no primary-corpus shortfall has been observed yet — add an adapter
only if/when the author needs one.
"""

import csv
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from levy.dataset.normalize import NORMALISATION_RULE, normalize_html
from levy.dataset.schema import QueryPair, WORKLOADS

AnyPath = Union[str, "os.PathLike"]


@dataclass(frozen=True)
class RawCandidatePair:
    """A labeled pair as it comes out of a corpus, before dataset assembly."""

    source_pair_id: str
    query_1: str
    query_2: str
    label: int  # 0 or 1, mapped from the corpus's original human label


@dataclass(frozen=True)
class LabelMapping:
    """
    How an adapter maps a corpus's native labels onto the binary study label.

    `domain` is every native value the corpus may legally carry. Values in
    `domain` but in neither `positive` nor `negative` are deliberately excluded
    from sampling (PIT-2015's debatable 2-vs-3 vote, SODD's accepted-answer
    class). Values outside `domain` are a corruption or a format change and are
    reported, never coerced.
    """

    positive: Tuple[Any, ...]
    negative: Tuple[Any, ...]
    domain: Tuple[Any, ...]

    def classify(self, native: Any) -> Optional[int]:
        """1, 0, or None for an in-domain value that is deliberately excluded."""
        if native in self.positive:
            return 1
        if native in self.negative:
            return 0
        return None

    def in_domain(self, native: Any) -> bool:
        return native in self.domain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive": list(self.positive),
            "negative": list(self.negative),
            "domain": list(self.domain),
        }


class CorpusSourceError(ValueError):
    """Raised for malformed raw corpus files or corpus configuration errors."""


class CorpusSource(ABC):
    """
    Yields candidate labeled pairs for one workload from one corpus.

    Concrete adapters wrap a specific published raw format (documented on
    each subclass) and never perform network access — the raw files must
    already exist on disk (`scripts/fetch_corpora.py` puts them there).
    """

    #: workload this source's candidates belong to (one of `WORKLOADS`)
    workload: str
    #: short, stable name recorded as `QueryPair.source_corpus`
    name: str
    #: key of this corpus in `data/corpora.json`; None for synthetic sources
    corpus_key: Optional[str] = None

    @abstractmethod
    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        """Yield every candidate pair this source has available."""
        raise NotImplementedError

    def label_mapping(self) -> LabelMapping:
        """Which native label values this adapter treats as positive/negative."""
        return LabelMapping(positive=(1,), negative=(0,), domain=(0, 1))

    def options(self) -> Dict[str, Any]:
        """Adapter choices affecting sampled content, for the run manifest."""
        return {}

    def source_files(self) -> List[Path]:
        """Raw files this adapter reads, in the order it reads them."""
        return []

    def check_fields(self) -> List[str]:
        """
        Cheap pre-flight check of the raw files' field/column structure.

        Returns a list of problem descriptions (empty when fine) rather than
        raising, so `levy.dataset.validation` can report every problem across
        every workload in one pass.
        """
        return []


class QuoraQQPSource(CorpusSource):
    """
    Adapter for the Quora Question Pairs (QQP) corpus, workload="faq".

    Expected raw format: the publicly released QQP TSV file, tab-separated,
    with a header row containing at least the columns:
        id, qid1, qid2, question1, question2, is_duplicate
    (`is_duplicate` is 0/1.) This is the format of the GLUE QQP TSV export;
    the Kaggle release ships the same columns comma-separated and must be
    converted to TSV first (see `data/corpora.json`).
    """

    workload = "faq"
    name = "quora-qqp"
    corpus_key = "quora-qqp"

    REQUIRED_COLUMNS = ("id", "question1", "question2", "is_duplicate")

    def __init__(self, path: AnyPath) -> None:
        self.path = Path(path)

    def source_files(self) -> List[Path]:
        return [self.path]

    def label_mapping(self) -> LabelMapping:
        return LabelMapping(positive=(1,), negative=(0,), domain=(0, 1))

    def check_fields(self) -> List[str]:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                fieldnames = csv.DictReader(fh, delimiter="\t").fieldnames or []
        except OSError as exc:
            return [f"{self.path}: cannot read QQP TSV: {exc}"]
        missing = [c for c in self.REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            return [f"{self.path}: QQP TSV missing columns {missing}"]
        return []

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        mapping = self.label_mapping()
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            missing = set(self.REQUIRED_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise CorpusSourceError(
                    f"{self.path}: QQP TSV missing columns {sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):  # header is line 1
                if not row.get("question1") or not row.get("question2"):
                    continue
                try:
                    native = int(row["is_duplicate"])
                except (TypeError, ValueError):
                    raise CorpusSourceError(
                        f"{self.path}:{row_number}: is_duplicate is not an integer: "
                        f"{row['is_duplicate']!r}"
                    ) from None
                if not mapping.in_domain(native):
                    raise CorpusSourceError(
                        f"{self.path}:{row_number}: is_duplicate {native!r} outside the "
                        f"declared domain {list(mapping.domain)}"
                    )
                label = mapping.classify(native)
                if label is None:  # pragma: no cover - domain == positive|negative here
                    continue
                yield RawCandidatePair(
                    source_pair_id=str(row["id"]),
                    query_1=row["question1"],
                    query_2=row["question2"],
                    label=label,
                )


class SODDSource(CorpusSource):
    """
    Adapter for SODD — the Stack Overflow Duplicity Dataset released alongside
    MQDD (Pasek et al., RANLP 2023) — workload="code".

    Expected raw format: one gzip-compressed parquet file per split
    (`SODD_train.parquet.gzip`, `SODD_dev.parquet.gzip`), with the columns
    `first_post`, `second_post`, `first_author`, `second_author`, `label`,
    `page`. Only the three the study needs are read; the author columns are
    usernames and are deliberately never loaded.

    Native label classes:
        0 = duplicates            -> positive
        1 = similar (fulltext search)
        2 = similar (tags)
        3 = different             -> negative (default)
        4 = accepted answer       -> always excluded (not a question pair)

    Classes 1 and 2 are near-misses. Using them as negatives makes the negative
    pool adversarially hard, which changes what a false positive means — so it
    is an explicit option (`hard_negatives`, default off) recorded in the run
    manifest, not an implicit default.

    `first_post` / `second_post` are HTML; they are reduced to plain text by
    `levy.dataset.normalize.normalize_html`, identically here and at
    rehydration time.
    """

    workload = "code"
    name = "sodd"
    corpus_key = "sodd"

    REQUIRED_COLUMNS = ("first_post", "second_post", "label")
    LABEL_DOMAIN = (0, 1, 2, 3, 4)
    #: rows read per batch; keeps a ~1.4M-row shard off the heap
    BATCH_SIZE = 4096

    def __init__(self, paths: Sequence[AnyPath], hard_negatives: bool = False) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise CorpusSourceError("SODDSource needs at least one parquet shard")
        self.hard_negatives = bool(hard_negatives)

    def source_files(self) -> List[Path]:
        return list(self.paths)

    def label_mapping(self) -> LabelMapping:
        negative = (1, 2, 3) if self.hard_negatives else (3,)
        return LabelMapping(positive=(0,), negative=negative, domain=self.LABEL_DOMAIN)

    def options(self) -> Dict[str, Any]:
        return {
            "hard_negatives": self.hard_negatives,
            "shards": [p.name for p in self.paths],
            "normalisation": NORMALISATION_RULE,
        }

    @staticmethod
    def _shard_key(path: Path) -> str:
        """`SODD_train.parquet.gzip` -> `SODD_train`; the id prefix per shard."""
        return path.name.split(".")[0]

    def check_fields(self) -> List[str]:
        import pyarrow.parquet as pq  # local: keeps parquet optional for other adapters

        problems: List[str] = []
        for path in self.paths:
            try:
                schema = pq.ParquetFile(path).schema_arrow
            except Exception as exc:  # noqa: BLE001 - any parquet failure is a finding
                problems.append(f"{path}: cannot read SODD parquet: {exc}")
                continue
            missing = [c for c in self.REQUIRED_COLUMNS if c not in schema.names]
            if missing:
                problems.append(f"{path}: SODD parquet missing columns {missing}")
        return problems

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        import pyarrow.parquet as pq

        mapping = self.label_mapping()
        for path in self.paths:
            shard = self._shard_key(path)
            try:
                parquet_file = pq.ParquetFile(path)
            except Exception as exc:  # noqa: BLE001
                raise CorpusSourceError(f"{path}: cannot read SODD parquet: {exc}") from exc
            missing = [c for c in self.REQUIRED_COLUMNS if c not in parquet_file.schema_arrow.names]
            if missing:
                raise CorpusSourceError(f"{path}: SODD parquet missing columns {missing}")

            row_index = 0
            for batch in parquet_file.iter_batches(
                batch_size=self.BATCH_SIZE, columns=list(self.REQUIRED_COLUMNS)
            ):
                first_posts = batch.column("first_post").to_pylist()
                second_posts = batch.column("second_post").to_pylist()
                labels = batch.column("label").to_pylist()
                for first, second, native in zip(first_posts, second_posts, labels):
                    index, row_index = row_index, row_index + 1
                    if native is None or not mapping.in_domain(int(native)):
                        raise CorpusSourceError(
                            f"{path} row {index}: label {native!r} outside the declared "
                            f"domain {list(mapping.domain)}"
                        )
                    label = mapping.classify(int(native))
                    if label is None:
                        continue  # in-domain but deliberately excluded
                    query_1 = normalize_html(first or "")
                    query_2 = normalize_html(second or "")
                    if not query_1 or not query_2:
                        continue
                    yield RawCandidatePair(
                        source_pair_id=f"{shard}:{index}",
                        query_1=query_1,
                        query_2=query_2,
                        label=label,
                    )


class TwitterPIT2015Source(CorpusSource):
    """
    Adapter for the Twitter Paraphrase Corpus (SemEval-2015 Task 1, PIT),
    workload="chat".

    Expected raw format: the shared-task release's tab-separated data files
    (`train.data`, `dev.data`), no header row, seven columns:
        Topic_Id, Topic_Name, Sent_1, Sent_2, Label, Sent_1_tag, Sent_2_tag

    Train and dev carry a crowdsourced vote count in the `Label` column, e.g.
    `(4, 1)` = four of five annotators judged the pair a paraphrase:
        5, 4, 3 yes-votes -> positive
        1, 0 yes-votes    -> negative
        2 yes-votes       -> debatable, excluded by the task's own guidance

    The **test split is rejected**, not read: it carries a single expert grade
    on a 0-5 scale instead of a vote count. Coercing one scale onto the other
    would mix two different label scales into one sample, so the adapter fails
    on a graded file rather than guessing.

    The release has no per-pair identifier, so `source_pair_id` is
    `"<split>:<line number>"` — stable because the file itself is
    checksum-pinned in `data/corpora.json`.
    """

    workload = "chat"
    name = "twitter-pit2015"
    corpus_key = "twitter-pit2015"

    N_COLUMNS = 7
    #: yes-vote counts, out of five annotators
    LABEL_DOMAIN = (0, 1, 2, 3, 4, 5)
    _VOTE_RE = re.compile(r"^\(\s*(\d)\s*,\s*(\d)\s*\)$")

    def __init__(self, paths: Sequence[AnyPath]) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise CorpusSourceError("TwitterPIT2015Source needs at least one data file")

    def source_files(self) -> List[Path]:
        return list(self.paths)

    def label_mapping(self) -> LabelMapping:
        return LabelMapping(positive=(3, 4, 5), negative=(0, 1), domain=self.LABEL_DOMAIN)

    def options(self) -> Dict[str, Any]:
        return {"splits": [p.stem for p in self.paths]}

    def _parse_votes(self, raw: str, where: str) -> int:
        """Yes-vote count from a `(4, 1)` label, or an error naming the scale clash."""
        match = self._VOTE_RE.match(raw.strip())
        if match is None:
            raise CorpusSourceError(
                f"{where}: label {raw!r} is not a PIT-2015 vote count such as '(4, 1)'. "
                "The graded 0-5 test split is deliberately rejected rather than coerced "
                "onto the vote scale — use train.data and dev.data only."
            )
        return int(match.group(1))

    def check_fields(self) -> List[str]:
        problems: List[str] = []
        for path in self.paths:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    first_line = fh.readline()
            except OSError as exc:
                problems.append(f"{path}: cannot read PIT-2015 data file: {exc}")
                continue
            if not first_line.strip():
                problems.append(f"{path}: PIT-2015 data file is empty")
                continue
            columns = first_line.rstrip("\n").split("\t")
            if len(columns) != self.N_COLUMNS:
                problems.append(
                    f"{path}: PIT-2015 data file has {len(columns)} tab-separated columns, "
                    f"expected {self.N_COLUMNS}"
                )
                continue
            if self._VOTE_RE.match(columns[4].strip()) is None:
                problems.append(
                    f"{path}: label column is {columns[4]!r}, not a vote count such as "
                    "'(4, 1)' — this looks like the graded test split, which is rejected"
                )
        return problems

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        mapping = self.label_mapping()
        for path in self.paths:
            split = path.stem
            with path.open("r", encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, start=1):
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    columns = line.split("\t")
                    if len(columns) != self.N_COLUMNS:
                        raise CorpusSourceError(
                            f"{path}:{line_number}: expected {self.N_COLUMNS} tab-separated "
                            f"columns, found {len(columns)}"
                        )
                    sent_1, sent_2, raw_label = columns[2], columns[3], columns[4]
                    votes = self._parse_votes(raw_label, f"{path}:{line_number}")
                    if not mapping.in_domain(votes):
                        raise CorpusSourceError(
                            f"{path}:{line_number}: vote count {votes!r} outside the declared "
                            f"domain {list(mapping.domain)}"
                        )
                    label = mapping.classify(votes)
                    if label is None:
                        continue  # debatable (2 of 5), excluded by the task's guidance
                    if not sent_1.strip() or not sent_2.strip():
                        continue
                    yield RawCandidatePair(
                        source_pair_id=f"{split}:{line_number}",
                        query_1=sent_1,
                        query_2=sent_2,
                        label=label,
                    )


class MockCorpusSource(CorpusSource):
    """
    Deterministic, fully synthetic `CorpusSource` for tests and offline
    smoke tests. Generates `n_candidates` candidate pairs (balanced 50/50
    positive/negative) with a fixed seed — no file access.

    A production run must never reach this: `scripts/sample_dataset.py
    --require-real` turns a fallback to this source into a hard error.
    """

    name = "mock"

    def __init__(self, workload: str, n_candidates: int = 40, seed: int = 0) -> None:
        if workload not in WORKLOADS:
            raise CorpusSourceError(f"Unknown workload {workload!r}; expected one of {WORKLOADS}")
        self.workload = workload
        self.n_candidates = n_candidates
        self.seed = seed

    def options(self) -> Dict[str, Any]:
        return {"n_candidates": self.n_candidates, "seed": self.seed}

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        rng = random.Random(self.seed)
        for i in range(self.n_candidates):
            label = i % 2  # exactly balanced, deterministic
            suffix = rng.randint(1000, 9999)
            yield RawCandidatePair(
                source_pair_id=f"mock-{self.workload}-{i:04d}",
                query_1=f"[mock:{self.workload}] question variant A #{i} ({suffix})",
                query_2=f"[mock:{self.workload}] question variant B #{i} ({suffix})",
                label=label,
            )


#: Adapter name (as recorded in `data/corpora.json`) -> class. Sampling and
#: rehydration both build sources through `make_source`, so a corpus's adapter
#: is chosen by the registry rather than hard-coded twice.
ADAPTERS: Dict[str, type] = {
    "QuoraQQPSource": QuoraQQPSource,
    "SODDSource": SODDSource,
    "TwitterPIT2015Source": TwitterPIT2015Source,
}


def make_source(adapter: str, paths: Sequence[AnyPath], **options: Any) -> CorpusSource:
    """
    Build the adapter named `adapter` over `paths`.

    Single-file adapters take one path; multi-file ones take the sequence.
    Unknown adapter names are an error naming what is available, never a guess.
    """
    try:
        cls = ADAPTERS[adapter]
    except KeyError:
        raise CorpusSourceError(
            f"Unknown corpus adapter {adapter!r}; known adapters: {sorted(ADAPTERS)}"
        ) from None
    paths = list(paths)
    if cls is QuoraQQPSource:
        if len(paths) != 1:
            raise CorpusSourceError(
                f"{adapter} reads exactly one file, got {len(paths)}"
            )
        return cls(paths[0], **options)
    return cls(paths, **options)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_workload(
    source: CorpusSource,
    n: int,
    seed: int,
    positive_ratio: float = 0.5,
) -> List[QueryPair]:
    """
    Deterministically sample `n` `QueryPair`s for `source.workload` from
    `source`, stratified so that `round(n * positive_ratio)` pairs have
    `original_label == 1` and the rest have `original_label == 0`.

    Determinism: candidates are sorted by `source_pair_id` before shuffling
    (so iteration order from the underlying file/generator never matters),
    then shuffled and sliced using `random.Random(seed)`. The same
    `(source, n, seed, positive_ratio)` always yields the same pairs in the
    same order; a different `seed` yields a different sample (with high
    probability).

    Raises `CorpusSourceError` if `source` does not have enough candidates
    in either class to satisfy the requested stratification.
    """
    positives: List[RawCandidatePair] = []
    negatives: List[RawCandidatePair] = []
    for candidate in source.iter_candidates():
        (positives if candidate.label == 1 else negatives).append(candidate)

    positives.sort(key=lambda c: c.source_pair_id)
    negatives.sort(key=lambda c: c.source_pair_id)

    n_pos = round(n * positive_ratio)
    n_neg = n - n_pos

    if len(positives) < n_pos:
        raise CorpusSourceError(
            f"{source.name}/{source.workload}: requested {n_pos} positive pairs, "
            f"only {len(positives)} available"
        )
    if len(negatives) < n_neg:
        raise CorpusSourceError(
            f"{source.name}/{source.workload}: requested {n_neg} negative pairs, "
            f"only {len(negatives)} available"
        )

    rng = random.Random(seed)
    chosen_pos = rng.sample(positives, n_pos)
    chosen_neg = rng.sample(negatives, n_neg)
    chosen = chosen_pos + chosen_neg
    rng.shuffle(chosen)

    pairs: List[QueryPair] = []
    for i, candidate in enumerate(chosen):
        pairs.append(
            QueryPair(
                pair_id=f"{source.workload}-{i:04d}",
                workload=source.workload,
                source_corpus=source.name,
                source_pair_id=candidate.source_pair_id,
                query_1=candidate.query_1,
                query_2=candidate.query_2,
                original_label=candidate.label,
            )
        )
    return pairs


def sample_dataset(
    sources: Dict[str, CorpusSource],
    n_per_workload: int = 300,
    seed: int = 42,
    positive_ratio: float = 0.5,
) -> List[QueryPair]:
    """
    Build the full ground-truth dataset across all workloads.

    `sources` maps workload name -> `CorpusSource` for that workload; must
    cover exactly `WORKLOADS`. Each workload is sampled independently with
    the same `seed` (the seed's effect is scoped per-workload by construction
    of `random.Random(seed)` inside `sample_workload`, combined with each
    workload's own candidate pool, so workloads do not draw from a shared
    stream).
    """
    missing = set(WORKLOADS) - set(sources.keys())
    if missing:
        raise CorpusSourceError(f"Missing sources for workload(s): {sorted(missing)}")

    all_pairs: List[QueryPair] = []
    for workload in WORKLOADS:
        source = sources[workload]
        if source.workload != workload:
            raise CorpusSourceError(
                f"Source for {workload!r} reports workload {source.workload!r}"
            )
        all_pairs.extend(
            sample_workload(source, n=n_per_workload, seed=seed, positive_ratio=positive_ratio)
        )
    return all_pairs
