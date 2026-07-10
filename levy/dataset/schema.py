"""
Ground-truth dataset schema (LEV-3 / D2).

Defines the workload vocabulary and the `QueryPair` record that is the unit
of the 900-pair ground-truth dataset described in the frozen S&D Report
("900 query pairs (300 per workload) ... Each pair retains the original
human label and the author's blind re-annotation").

This is the contract LEV-4 (experiment harness) replays against:
for each pair, `query_1` is submitted first (always a miss, populates the
cache) and `query_2` is submitted second (hit/miss decision); the decision
is compared against the ground-truth label (`author_label` if present,
else `original_label`) to accumulate TP/FP/TN/FN.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Workload vocabulary
# ---------------------------------------------------------------------------

WORKLOAD_FAQ = "faq"
WORKLOAD_CODE = "code"
WORKLOAD_CHAT = "chat"

WORKLOADS: Tuple[str, ...] = (WORKLOAD_FAQ, WORKLOAD_CODE, WORKLOAD_CHAT)


class QueryPairValidationError(ValueError):
    """Raised when a `QueryPair` (or a raw record destined to become one) is invalid."""


@dataclass
class QueryPair:
    """
    A single labeled query pair in the ground-truth dataset.

    Fields:
        pair_id: stable identifier within the released dataset, e.g. "faq-0001".
        workload: one of `WORKLOADS` ("faq", "code", "chat").
        source_corpus: name of the originating public corpus (e.g. "quora-qqp",
            "stackoverflow-duplicates", "convai2", or "synthetic-fixture" for
            placeholder data), for traceability back to the source dataset.
        source_pair_id: identifier of this pair within `source_corpus`, so the
            sampling can be audited/reproduced against the raw corpus file.
        query_1: first query in the pair. Per Algorithm 1 (S&D Report), this is
            the query submitted first during replay (always a cache miss).
        query_2: second query in the pair; the cache's hit/miss decision on
            this query is compared against the ground-truth label.
        original_label: binary duplicate/similarity label (1 = duplicate /
            same-intent, 0 = not) from the source corpus's original human
            annotators.
        author_label: the author's blind re-annotation (Optional[int], None
            until the author has annotated the pair). Computed independently
            of `original_label` — see `levy.dataset.annotation`.
        metadata: free-form extra fields (e.g. `{"provenance": "synthetic-fixture"}`).
    """

    pair_id: str
    workload: str
    source_corpus: str
    source_pair_id: str
    query_1: str
    query_2: str
    original_label: int
    author_label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_query_pair(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "workload": self.workload,
            "source_corpus": self.source_corpus,
            "source_pair_id": self.source_pair_id,
            "query_1": self.query_1,
            "query_2": self.query_2,
            "original_label": self.original_label,
            "author_label": self.author_label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPair":
        try:
            return cls(
                pair_id=str(data["pair_id"]),
                workload=str(data["workload"]),
                source_corpus=str(data["source_corpus"]),
                source_pair_id=str(data["source_pair_id"]),
                query_1=str(data["query_1"]),
                query_2=str(data["query_2"]),
                original_label=int(data["original_label"]),
                author_label=(
                    None
                    if data.get("author_label") in (None, "")
                    else int(data["author_label"])
                ),
                metadata=dict(data.get("metadata") or {}),
            )
        except KeyError as exc:
            raise QueryPairValidationError(f"Missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise QueryPairValidationError(f"Invalid field value: {exc}") from exc

    def ground_truth_label(self) -> int:
        """
        The label to evaluate replay decisions against (LEV-4 contract):
        the author's blind re-annotation if available, else the original
        corpus label.
        """
        return self.author_label if self.author_label is not None else self.original_label


def validate_query_pair(pair: QueryPair, context: Optional[str] = None) -> None:
    """
    Validate a `QueryPair`'s invariants. Raises `QueryPairValidationError` with
    a message that includes `context` (e.g. "row 12") when given, so loaders
    can point to the offending record.
    """
    prefix = f"{context}: " if context else ""

    if not pair.pair_id:
        raise QueryPairValidationError(f"{prefix}pair_id must be non-empty")
    if pair.workload not in WORKLOADS:
        raise QueryPairValidationError(
            f"{prefix}workload {pair.workload!r} must be one of {WORKLOADS}"
        )
    if not pair.source_corpus:
        raise QueryPairValidationError(f"{prefix}source_corpus must be non-empty")
    if not pair.source_pair_id:
        raise QueryPairValidationError(f"{prefix}source_pair_id must be non-empty")
    if not pair.query_1 or not pair.query_1.strip():
        raise QueryPairValidationError(f"{prefix}query_1 must be a non-empty string")
    if not pair.query_2 or not pair.query_2.strip():
        raise QueryPairValidationError(f"{prefix}query_2 must be a non-empty string")
    if pair.original_label not in (0, 1):
        raise QueryPairValidationError(
            f"{prefix}original_label must be 0 or 1, got {pair.original_label!r}"
        )
    if pair.author_label is not None and pair.author_label not in (0, 1):
        raise QueryPairValidationError(
            f"{prefix}author_label must be 0, 1, or None, got {pair.author_label!r}"
        )
