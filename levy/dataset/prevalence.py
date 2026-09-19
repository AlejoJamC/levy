"""
Duplicate prevalence of the source pools (LEV-21).

Before stratified sampling, each corpus adapter filters its raw rows into a
pool: rows whose native label maps to the positive or negative class, with
non-empty text. `sample_workload` then draws a fixed number from each class, so
the class balance of the *sample* says nothing about the corpus. This module
measures the balance of the pool the sampler drew from.

Two prevalences are computed per workload, over different denominators:

- A: positives / (positives + negatives + excluded-band rows) — the whole
  corpus as the adapter's label domain sees it;
- B: positives / (positives + negatives) — the binary-eligible pool the
  sample was drawn from.

Positive and negative counts are the candidates `iter_candidates()` yields, i.e.
exactly the sampler's pool. The excluded-band count and the per-native-label
breakdown come from `CorpusSource.native_label_counts()`, which counts rows
before the empty-text filter; the gap between the two is reported as
`dropped_empty_text_*`. Nothing here samples, and nothing touches the network
or the ground-truth dataset.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from levy.dataset.sampling import CorpusSource


@dataclass(frozen=True)
class PoolCounts:
    workload: str
    corpus: str
    positive: int
    negative: int
    excluded: int
    dropped_empty_text_positive: int
    dropped_empty_text_negative: int
    native_label_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def eligible(self) -> int:
        """Binary-eligible pool: positive + negative."""
        return self.positive + self.negative

    @property
    def pool_size(self) -> int:
        """Full filtered pool: eligible rows plus the excluded band."""
        return self.eligible + self.excluded

    @property
    def prevalence_a(self) -> Optional[float]:
        """positives / full filtered pool; None when the pool is empty."""
        return self.positive / self.pool_size if self.pool_size else None

    @property
    def prevalence_b(self) -> Optional[float]:
        """positives / binary-eligible pool; None when it is empty."""
        return self.positive / self.eligible if self.eligible else None


def measure_pool(source: CorpusSource) -> PoolCounts:
    """
    Count `source`'s pool.

    Raises `ValueError` if the per-label census and the candidate stream
    disagree in a way empty-text filtering cannot explain (fewer census rows
    than candidates in a class).
    """
    mapping = source.label_mapping()

    positive = negative = 0
    for candidate in source.iter_candidates():
        if candidate.label == 1:
            positive += 1
        else:
            negative += 1

    census = source.native_label_counts()
    census_positive = sum(n for label, n in census.items() if label in mapping.positive)
    census_negative = sum(n for label, n in census.items() if label in mapping.negative)
    excluded = sum(
        n
        for label, n in census.items()
        if label not in mapping.positive and label not in mapping.negative
    )

    if census_positive < positive or census_negative < negative:
        raise ValueError(
            f"{source.name}/{source.workload}: label census ({census_positive} positive, "
            f"{census_negative} negative) is smaller than the candidate stream "
            f"({positive} positive, {negative} negative)"
        )

    return PoolCounts(
        workload=source.workload,
        corpus=source.name,
        positive=positive,
        negative=negative,
        excluded=excluded,
        dropped_empty_text_positive=census_positive - positive,
        dropped_empty_text_negative=census_negative - negative,
        native_label_counts=dict(census),
    )
