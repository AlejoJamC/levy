"""
Ground-truth dataset tooling (LEV-3 / D2, extended by LEV-12): schema, CSV/JSON
I/O, the identifiers-only distribution format, the corpus provenance registry,
seeded stratified sampling, pre-flight validation, blind re-annotation, and
Cohen's kappa.

Nothing in this package touches the network: adapters read raw corpora already
present under `data/raw/`, which `scripts/fetch_corpora.py` acquires.

Producing the real 900-pair dataset (acquiring the corpora, sampling, the
author's blind re-annotation session, and the final kappa result) is a
data-production task performed by the author, not part of this package. See
`openspec/changes/add-ground-truth-dataset/proposal.md` for the platform vs.
data-production split.
"""

from levy.dataset.schema import (
    DistributionRecord,
    QueryPair,
    QueryPairValidationError,
    WORKLOADS,
    WORKLOAD_CHAT,
    WORKLOAD_CODE,
    WORKLOAD_FAQ,
    validate_distribution_record,
    validate_query_pair,
)
from levy.dataset.io import (
    DatasetValidationError,
    load_csv,
    load_dataset,
    load_distribution_csv,
    load_json,
    save_csv,
    save_dataset,
    save_distribution_csv,
    save_json,
    to_distribution_records,
)
from levy.dataset.corpora import (
    CorpusEntry,
    CorpusFile,
    CorpusRegistry,
    CorpusRegistryError,
    load_registry,
    sha256_file,
)
from levy.dataset.sampling import (
    CorpusSource,
    CorpusSourceError,
    LabelMapping,
    MockCorpusSource,
    QuoraQQPSource,
    RawCandidatePair,
    SODDSource,
    TwitterPIT2015Source,
    sample_dataset,
    sample_workload,
)
from levy.dataset.annotation import AnnotationSummary, BlindAnnotationSession
from levy.dataset.kappa import KappaReport, KappaResult, cohen_kappa, kappa_report

__all__ = [
    "DistributionRecord",
    "QueryPair",
    "QueryPairValidationError",
    "WORKLOADS",
    "WORKLOAD_CHAT",
    "WORKLOAD_CODE",
    "WORKLOAD_FAQ",
    "validate_distribution_record",
    "validate_query_pair",
    "DatasetValidationError",
    "load_csv",
    "load_dataset",
    "load_distribution_csv",
    "load_json",
    "save_csv",
    "save_dataset",
    "save_distribution_csv",
    "save_json",
    "to_distribution_records",
    "CorpusEntry",
    "CorpusFile",
    "CorpusRegistry",
    "CorpusRegistryError",
    "load_registry",
    "sha256_file",
    "CorpusSource",
    "CorpusSourceError",
    "LabelMapping",
    "MockCorpusSource",
    "QuoraQQPSource",
    "RawCandidatePair",
    "SODDSource",
    "TwitterPIT2015Source",
    "sample_dataset",
    "sample_workload",
    "AnnotationSummary",
    "BlindAnnotationSession",
    "KappaReport",
    "KappaResult",
    "cohen_kappa",
    "kappa_report",
]
