"""
Seeded, stratified sampling pipeline for the ground-truth dataset (LEV-3 / D2).

This module builds `QueryPair` lists from labeled candidate pairs exposed by a
`CorpusSource`. It does **not** download anything: adapters read local raw
files already present on disk, in the format documented on each adapter
class. Producing the real dataset (running this against the actual Quora
QQP / Stack Overflow / ConvAI2 downloads and committing 900 real-world pairs)
is an author data-production task, tracked separately from this platform
tooling — see `openspec/changes/add-ground-truth-dataset/proposal.md`.

Fallback corpora named in the frozen risk plan (Project_Proposal.md Risk 1)
if a primary corpus is unavailable or too small: MS MARCO (FAQ/QA fallback),
CodeSearchNet (code fallback), DailyDialog (chat fallback). Adapters for
those follow the same `CorpusSource` interface; they are not implemented here
because no primary-corpus shortfall has been observed yet — add an adapter
only if/when the author needs one.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Union

from levy.dataset.schema import QueryPair, WORKLOADS

AnyPath = Union[str, "os.PathLike"]


@dataclass(frozen=True)
class RawCandidatePair:
    """A labeled pair as it comes out of a corpus, before dataset assembly."""

    source_pair_id: str
    query_1: str
    query_2: str
    label: int  # 0 or 1, the corpus's original human label


class CorpusSourceError(ValueError):
    """Raised for malformed raw corpus files or corpus configuration errors."""


class CorpusSource(ABC):
    """
    Yields candidate labeled pairs for one workload from one corpus.

    Concrete adapters wrap a specific published raw format (documented on
    each subclass) and never perform network access — the raw file must
    already exist on disk.
    """

    #: workload this source's candidates belong to (one of `WORKLOADS`)
    workload: str
    #: short, stable name recorded as `QueryPair.source_corpus`
    name: str

    @abstractmethod
    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        """Yield every candidate pair this source has available."""
        raise NotImplementedError


class QuoraQQPSource(CorpusSource):
    """
    Adapter for the Quora Question Pairs (QQP) corpus, workload="faq".

    Expected raw format: the publicly released QQP TSV file, tab-separated,
    with a header row containing at least the columns:
        id, qid1, qid2, question1, question2, is_duplicate
    (`is_duplicate` is 0/1.) This is the format of the Kaggle "Quora Question
    Pairs" release and the GLUE QQP TSV export.
    """

    workload = "faq"
    name = "quora-qqp"

    def __init__(self, path: AnyPath) -> None:
        self.path = Path(path)

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            required = {"id", "question1", "question2", "is_duplicate"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise CorpusSourceError(
                    f"{self.path}: QQP TSV missing columns {sorted(missing)}"
                )
            for row in reader:
                if not row.get("question1") or not row.get("question2"):
                    continue
                yield RawCandidatePair(
                    source_pair_id=str(row["id"]),
                    query_1=row["question1"],
                    query_2=row["question2"],
                    label=int(row["is_duplicate"]),
                )


class StackOverflowDuplicatesSource(CorpusSource):
    """
    Adapter for a Stack Overflow duplicate-questions export, workload="code".

    Expected raw format: a CSV file with a header row containing at least:
        pair_id, question1_id, question1, question2_id, question2, is_duplicate
    (`is_duplicate` is 0/1.) This matches the commonly published "Stack
    Overflow duplicate questions" CSV exports (Stack Exchange Data Dump
    derivatives). If the concrete file the author downloads uses different
    column names, either rename the columns before use or adjust this
    adapter's `required` set — do not silently guess column mappings.
    """

    workload = "code"
    name = "stackoverflow-duplicates"

    def __init__(self, path: AnyPath) -> None:
        self.path = Path(path)

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"pair_id", "question1", "question2", "is_duplicate"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise CorpusSourceError(
                    f"{self.path}: Stack Overflow CSV missing columns {sorted(missing)}"
                )
            for row in reader:
                if not row.get("question1") or not row.get("question2"):
                    continue
                yield RawCandidatePair(
                    source_pair_id=str(row["pair_id"]),
                    query_1=row["question1"],
                    query_2=row["question2"],
                    label=int(row["is_duplicate"]),
                )


class ConvAI2Source(CorpusSource):
    """
    Adapter for a ConvAI2-derived same-intent pair file, workload="chat".

    Raw ConvAI2 (PersonaChat) releases are multi-turn dialogues, not labeled
    duplicate-intent pairs, so a derivation step (pairing utterances and
    labeling same-intent vs. different-intent) is required before this
    adapter can run. That derivation is an author data-production task; this
    adapter only consumes its output.

    Expected raw format: a JSON file containing a list of objects, each with
    at least:
        {"pair_id": "...", "utterance_1": "...", "utterance_2": "...",
         "same_intent": 0 or 1}
    """

    workload = "chat"
    name = "convai2"

    def __init__(self, path: AnyPath) -> None:
        self.path = Path(path)

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        with self.path.open("r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
            except json.JSONDecodeError as exc:
                raise CorpusSourceError(f"{self.path}: invalid JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise CorpusSourceError(f"{self.path}: expected a JSON list of pair objects")
        required = {"pair_id", "utterance_1", "utterance_2", "same_intent"}
        for index, item in enumerate(raw):
            missing = required - set(item.keys())
            if missing:
                raise CorpusSourceError(
                    f"{self.path}: item[{index}] missing fields {sorted(missing)}"
                )
            yield RawCandidatePair(
                source_pair_id=str(item["pair_id"]),
                query_1=item["utterance_1"],
                query_2=item["utterance_2"],
                label=int(item["same_intent"]),
            )


class MockCorpusSource(CorpusSource):
    """
    Deterministic, fully synthetic `CorpusSource` for tests and offline
    smoke tests. Generates `n_candidates` candidate pairs (balanced 50/50
    positive/negative) with a fixed seed — no file access.
    """

    name = "mock"

    def __init__(self, workload: str, n_candidates: int = 40, seed: int = 0) -> None:
        if workload not in WORKLOADS:
            raise CorpusSourceError(f"Unknown workload {workload!r}; expected one of {WORKLOADS}")
        self.workload = workload
        self.n_candidates = n_candidates
        self.seed = seed

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
