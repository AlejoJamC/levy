"""
Per-workload re-sampling of the single ground-truth dataset.

The dataset is one artifact of 900 pairs, and by the time anything here runs it
is fully annotated: every pair carries an `author_label` from a finished blind
re-annotation. So "re-sample the chat workload" is not a fresh build — it is
surgery on a live dataset. This module holds the three operations that surgery
needs, kept out of `scripts/sample_dataset.py` so they can be tested directly:

1. **Exclusion.** `ExcludingCorpusSource` hides candidates whose
   `source_pair_id` is already in the dataset, so a re-sample draws pairs the
   previous sample did not use. It wraps rather than subclasses the adapter, so
   pool-sufficiency validation counts the *post-exclusion* pool and a shortfall
   surfaces before anything is written.
2. **Splicing.** `splice_workload` replaces one workload's rows in place,
   passing every other row through as the *same object* — the other workloads'
   `author_label` values are not copied, re-derived, or re-validated, they are
   simply never touched.
3. **Alignment.** `check_ids_alignment` refuses to proceed when the published
   identifiers file and the working dataset disagree about the workloads that
   are *not* being re-sampled, which would mean one of the two is stale and the
   merge would silently publish whichever the code happened to read.

On labels: a freshly sampled `QueryPair` has `author_label is None`, which is
exactly the "cleared" state the annotation step looks for. `clear_author_labels`
states that intent explicitly rather than relying on the constructor default,
because the whole point of a re-sample is that the new pairs are unannotated
while the other 600 stay annotated.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from levy.dataset.sampling import (
    CorpusSource,
    CorpusSourceError,
    LabelMapping,
    RawCandidatePair,
)
from levy.dataset.schema import DistributionRecord, QueryPair, WORKLOADS


class WorkloadUpdateError(RuntimeError):
    """Raised when a per-workload update cannot be performed safely."""


# ---------------------------------------------------------------------------
# Exclusion of already-sampled pairs
# ---------------------------------------------------------------------------

class ExcludingCorpusSource(CorpusSource):
    """
    A `CorpusSource` with some candidates hidden by `source_pair_id`.

    Delegates every declaration — workload, corpus name, registry key, label
    mapping, adapter options, source files, field checks — to the wrapped
    adapter, so validation, the run manifest and rehydration all see the real
    corpus. Only `iter_candidates()` differs.

    Wrapping (rather than filtering after sampling) is what makes a shortfall
    honest: `sample_workload` and `validate_sources` both count what this
    yields, so "300 requested, 212 available after excluding 300 already
    sampled" is reported before any file is written, not discovered as a short
    dataset afterwards.
    """

    def __init__(self, source: CorpusSource, excluded_source_pair_ids: Iterable[str]) -> None:
        self.source = source
        self.excluded: Set[str] = {str(value) for value in excluded_source_pair_ids}
        self.workload = source.workload
        self.name = source.name
        self.corpus_key = source.corpus_key

    def iter_candidates(self) -> Iterator[RawCandidatePair]:
        for candidate in self.source.iter_candidates():
            if candidate.source_pair_id in self.excluded:
                continue
            yield candidate

    def label_mapping(self) -> LabelMapping:
        return self.source.label_mapping()

    def options(self) -> Dict[str, object]:
        return self.source.options()

    def source_files(self) -> List:
        return self.source.source_files()

    def check_fields(self) -> List[str]:
        return self.source.check_fields()


def source_pair_ids(pairs: Sequence[QueryPair], workload: Optional[str] = None) -> Set[str]:
    """
    Every `source_pair_id` in `pairs`, optionally restricted to one workload.

    Restricted to a workload is the right scope for exclusion: two corpora can
    legitimately use the same identifier string (`"train:12"` means one thing in
    PIT-2015 and another in SODD), so excluding across workloads would drop
    candidates for no reason.
    """
    return {
        pair.source_pair_id
        for pair in pairs
        if workload is None or pair.workload == workload
    }


# ---------------------------------------------------------------------------
# In-place replacement of one workload's rows
# ---------------------------------------------------------------------------

@dataclass
class SpliceResult:
    """Outcome of splicing a new workload sample into the existing dataset."""

    pairs: List[QueryPair]  # the complete dataset after replacement
    replaced: List[QueryPair] = field(default_factory=list)  # the new rows
    removed: List[QueryPair] = field(default_factory=list)  # the rows they displaced
    appended: bool = False  # True when the workload was absent and was added


def splice_workload(
    current: Sequence[QueryPair], workload: str, new_pairs: Sequence[QueryPair]
) -> SpliceResult:
    """
    Return the dataset with `workload`'s rows replaced by `new_pairs`.

    Rows of other workloads are passed through by identity, at their original
    positions: for the ordinary 300-for-300 replacement the output differs from
    the input in exactly the replaced block's rows, so the other 600 rows —
    including their `author_label`s — are byte-identical by construction rather
    than by careful copying.

    A larger new sample puts its surplus immediately after the workload's last
    existing row; a smaller one drops the trailing rows. A workload absent from
    `current` is appended, with `appended=True` so the caller can say so.
    """
    if workload not in WORKLOADS:
        raise WorkloadUpdateError(f"Unknown workload {workload!r}; expected one of {WORKLOADS}")
    foreign = {pair.workload for pair in new_pairs} - {workload}
    if foreign:
        raise WorkloadUpdateError(
            f"new pairs for workload {workload!r} also carry workload(s) {sorted(foreign)}"
        )

    positions = [index for index, pair in enumerate(current) if pair.workload == workload]
    if not positions:
        return SpliceResult(
            pairs=list(current) + list(new_pairs),
            replaced=list(new_pairs),
            removed=[],
            appended=True,
        )

    last_position = positions[-1]
    queue = list(new_pairs)
    out: List[QueryPair] = []
    removed: List[QueryPair] = []

    for index, pair in enumerate(current):
        if pair.workload != workload:
            out.append(pair)
            continue
        removed.append(pair)
        if queue:
            out.append(queue.pop(0))
        if index == last_position and queue:
            out.extend(queue)
            queue = []

    return SpliceResult(
        pairs=out, replaced=list(new_pairs), removed=removed, appended=False
    )


def clear_author_labels(pairs: Iterable[QueryPair]) -> int:
    """
    Drop `author_label` from `pairs`, returning how many carried one.

    Called on the newly sampled rows only. A fresh sample has none, so this is
    normally a no-op — it exists so the "the re-sampled workload comes back
    unannotated, everything else keeps its labels" contract is written down in
    code rather than inferred from a dataclass default.
    """
    cleared = 0
    for pair in pairs:
        if pair.author_label is not None:
            pair.author_label = None
            cleared += 1
    return cleared


def verify_disjoint(new_pairs: Sequence[QueryPair], previous_ids: Set[str]) -> None:
    """
    Assert that a re-sample drew nothing the previous sample held.

    `ExcludingCorpusSource` already guarantees this; checking it again here is
    cheap and turns a future bug in the exclusion path into an error before the
    write instead of a duplicated pair in the published dataset.
    """
    overlap = sorted(pair.source_pair_id for pair in new_pairs if pair.source_pair_id in previous_ids)
    if overlap:
        raise WorkloadUpdateError(
            f"re-sampled pairs overlap the pairs they replace: {overlap[:5]}"
            f"{' ...' if len(overlap) > 5 else ''} "
            "(exclusion failed; nothing written)"
        )


# ---------------------------------------------------------------------------
# Alignment between the working dataset and the published identifiers
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS = ("pair_id", "workload", "source_corpus", "source_pair_id", "original_label")


def _identity(record) -> Tuple:
    return tuple(getattr(record, name) for name in _IDENTITY_FIELDS)


def check_ids_alignment(
    pairs: Sequence[QueryPair],
    records: Sequence[DistributionRecord],
    untouched_workloads: Iterable[str],
) -> None:
    """
    Verify that the working dataset and the published identifiers file describe
    the same pairs for the workloads this run is *not* re-sampling.

    Only identity is compared, not `author_label`: annotation writes the working
    dataset, and refreshing the identifiers file from it is a later step, so a
    label present in one and absent in the other is a normal intermediate state
    (the working dataset wins). A disagreement about *which* pair a row is means
    one of the two files is stale, and merging them would publish whichever the
    code happened to read first — so that is an error, and nothing is written.
    """
    untouched = set(untouched_workloads)
    dataset_rows = [_identity(p) for p in pairs if p.workload in untouched]
    ids_rows = [_identity(r) for r in records if r.workload in untouched]

    if len(dataset_rows) != len(ids_rows):
        raise WorkloadUpdateError(
            f"the working dataset and the identifiers file disagree on the untouched "
            f"workloads {sorted(untouched)}: {len(dataset_rows)} row(s) vs {len(ids_rows)}. "
            "One of the two is stale — rebuild the working dataset with "
            "`python scripts/rehydrate_dataset.py` before re-sampling. Nothing written."
        )
    for position, (from_dataset, from_ids) in enumerate(zip(dataset_rows, ids_rows)):
        if from_dataset != from_ids:
            raise WorkloadUpdateError(
                f"the working dataset and the identifiers file disagree at untouched-row "
                f"{position}: dataset has {from_dataset}, identifiers file has {from_ids}. "
                "One of the two is stale — rebuild the working dataset with "
                "`python scripts/rehydrate_dataset.py` before re-sampling. Nothing written."
            )


def resample_shortfall_message(workload: str, excluded: int, exc: CorpusSourceError) -> str:
    """The shortfall error a re-sample raises, naming the workload and the exclusion."""
    return (
        f"workload {workload!r}: cannot draw a fresh sample — {exc}. "
        f"{excluded} pair(s) already in the ground truth were excluded from the "
        "candidate pool, so the pool is smaller than a first-time sample's. "
        "Nothing written."
    )
