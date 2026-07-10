"""
Ground-truth dataset tooling (LEV-3 / D2): schema, CSV/JSON I/O, seeded
stratified sampling, blind re-annotation, and Cohen's kappa.

Producing the real 900-pair dataset (running `sampling` against the actual
Quora QQP / Stack Overflow / ConvAI2 corpora, the author's blind
re-annotation session, and the final kappa result) is a data-production task
performed by the author, not part of this package. See
`openspec/changes/add-ground-truth-dataset/proposal.md` for the platform vs.
data-production split.
"""

from levy.dataset.schema import (
    QueryPair,
    QueryPairValidationError,
    WORKLOADS,
    WORKLOAD_CHAT,
    WORKLOAD_CODE,
    WORKLOAD_FAQ,
    validate_query_pair,
)
from levy.dataset.io import (
    DatasetValidationError,
    load_csv,
    load_dataset,
    load_json,
    save_csv,
    save_dataset,
    save_json,
)
from levy.dataset.sampling import (
    CorpusSource,
    CorpusSourceError,
    ConvAI2Source,
    MockCorpusSource,
    QuoraQQPSource,
    RawCandidatePair,
    StackOverflowDuplicatesSource,
    sample_dataset,
    sample_workload,
)
from levy.dataset.annotation import AnnotationSummary, BlindAnnotationSession
from levy.dataset.kappa import KappaReport, KappaResult, cohen_kappa, kappa_report

__all__ = [
    "QueryPair",
    "QueryPairValidationError",
    "WORKLOADS",
    "WORKLOAD_CHAT",
    "WORKLOAD_CODE",
    "WORKLOAD_FAQ",
    "validate_query_pair",
    "DatasetValidationError",
    "load_csv",
    "load_dataset",
    "load_json",
    "save_csv",
    "save_dataset",
    "save_json",
    "CorpusSource",
    "CorpusSourceError",
    "ConvAI2Source",
    "MockCorpusSource",
    "QuoraQQPSource",
    "RawCandidatePair",
    "StackOverflowDuplicatesSource",
    "sample_dataset",
    "sample_workload",
    "AnnotationSummary",
    "BlindAnnotationSession",
    "KappaReport",
    "KappaResult",
    "cohen_kappa",
    "kappa_report",
]
