"""
Pre-flight validation of raw corpus inputs (LEV-12).

Sampling the 900-pair dataset is preceded by one validation pass that reports
*every* problem it finds rather than aborting on the first. That matters
because the expensive part of a shortfall is discovering it: learning that the
`code` workload is 40 positives short only after fixing an unrelated `chat`
problem costs another full pass over a ~1.4M-row corpus, and — worse — invites
resolving the substitution question one workload at a time instead of with the
whole picture in view.

Validation returns a structured `ValidationReport`; the CLI renders it and
derives its exit code from it. Keeping the decision in a returned value rather
than in a raised exception is what makes the multi-problem behaviour testable
offline against fixtures instead of only observable through a process exit code.

Checks, in the order they run:

1. **Cross-workload corpus overlap** — two workloads resolving to the same
   corpus collapses the workload contrast the study's workload hypothesis
   tests, so it is a configuration error, not a warning.
2. **File presence and readability** — per adapter, every file it will read.
3. **Checksum agreement** with `data/corpora.json`, for files whose checksum is
   pinned. An unpinned file is reported as a note, not a finding: there is
   nothing yet to disagree with.
4. **Required fields** — each adapter's own cheap header/schema check.
5. **Pool sufficiency and label domain** — one pass per corpus counting
   positives and negatives, for all workloads, comparing against what the
   requested stratified sample needs. A label outside the adapter's declared
   domain surfaces here as a finding rather than a coerced value.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from levy.dataset.corpora import (
    CorpusRegistry,
    CorpusRegistryError,
    load_registry,
    sha256_file,
)
from levy.dataset.sampling import CorpusSource, CorpusSourceError

PathLike = Union[str, Path]

CHECK_OVERLAP = "corpus-overlap"
CHECK_PRESENCE = "file-presence"
CHECK_CHECKSUM = "checksum"
CHECK_FIELDS = "required-fields"
CHECK_POOL = "class-pool"
CHECK_LABELS = "label-domain"


@dataclass(frozen=True)
class Finding:
    """One validation problem, attributable to a check and (usually) a workload."""

    check: str
    message: str
    workload: Optional[str] = None
    corpus: Optional[str] = None

    def render(self) -> str:
        where = "/".join(part for part in (self.workload, self.corpus) if part)
        return f"[{self.check}] {where + ': ' if where else ''}{self.message}"


@dataclass(frozen=True)
class PoolSizes:
    """Candidate counts for one workload, against what the sample requires."""

    workload: str
    corpus: str
    positive_available: int
    negative_available: int
    positive_required: int
    negative_required: int

    @property
    def sufficient(self) -> bool:
        return (
            self.positive_available >= self.positive_required
            and self.negative_available >= self.negative_required
        )


@dataclass
class ValidationReport:
    """Every finding from one validation pass, plus what was measured."""

    findings: List[Finding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    pools: Dict[str, PoolSizes] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, check: str, message: str, workload=None, corpus=None) -> None:
        self.findings.append(
            Finding(check=check, message=message, workload=workload, corpus=corpus)
        )

    def render(self) -> str:
        lines: List[str] = []
        for pool in self.pools.values():
            status = "ok" if pool.sufficient else "SHORT"
            lines.append(
                f"  pool  {pool.workload:<5} {pool.corpus:<16} "
                f"positives {pool.positive_available}/{pool.positive_required}, "
                f"negatives {pool.negative_available}/{pool.negative_required}  [{status}]"
            )
        for note in self.notes:
            lines.append(f"  note  {note}")
        for finding in self.findings:
            lines.append(f"  FAIL  {finding.render()}")
        if self.ok:
            lines.append(f"  OK    {len(self.pools)} workload(s) validated, no findings")
        else:
            lines.append(f"  FAILED with {len(self.findings)} finding(s)")
        return "\n".join(lines)


def validate_sources(
    sources: Dict[str, CorpusSource],
    n_per_workload: int,
    positive_ratio: float = 0.5,
    registry: Optional[CorpusRegistry] = None,
    raw_root: Optional[PathLike] = None,
    count_pools: bool = True,
) -> ValidationReport:
    """
    Validate every workload's corpus source in one pass.

    `registry` defaults to `data/corpora.json`; a registry that cannot be read
    becomes a finding rather than an exception, so the rest of the checks still
    run. `count_pools=False` skips the full corpus scan when only the cheap
    structural checks are wanted.
    """
    report = ValidationReport()

    if registry is None:
        try:
            registry = load_registry()
        except CorpusRegistryError as exc:
            report.add(CHECK_CHECKSUM, str(exc))

    _check_overlap(sources, report)

    n_pos_required = round(n_per_workload * positive_ratio)
    n_neg_required = n_per_workload - n_pos_required

    for workload in sorted(sources):
        source = sources[workload]
        structurally_ok = True

        files = source.source_files()
        for path in files:
            if not Path(path).is_file():
                report.add(
                    CHECK_PRESENCE,
                    f"{path} is missing; acquire it with `python scripts/fetch_corpora.py`",
                    workload,
                    source.name,
                )
                structurally_ok = False
            elif not _is_readable(path):
                report.add(CHECK_PRESENCE, f"{path} is not readable", workload, source.name)
                structurally_ok = False

        if structurally_ok and registry is not None and source.corpus_key:
            structurally_ok &= _check_checksums(source, registry, raw_root, report, workload)

        if structurally_ok:
            for problem in source.check_fields():
                report.add(CHECK_FIELDS, problem, workload, source.name)
                structurally_ok = False

        if not (structurally_ok and count_pools):
            continue

        try:
            positives = negatives = 0
            for candidate in source.iter_candidates():
                if candidate.label == 1:
                    positives += 1
                else:
                    negatives += 1
        except CorpusSourceError as exc:
            # Covers a label outside the adapter's declared domain, reported
            # rather than coerced — and scoped to this workload, so the other
            # workloads are still checked in the same pass.
            report.add(CHECK_LABELS, str(exc), workload, source.name)
            continue

        pool = PoolSizes(
            workload=workload,
            corpus=source.name,
            positive_available=positives,
            negative_available=negatives,
            positive_required=n_pos_required,
            negative_required=n_neg_required,
        )
        report.pools[workload] = pool
        if positives < n_pos_required:
            report.add(
                CHECK_POOL,
                f"needs {n_pos_required} positive pairs, only {positives} available",
                workload,
                source.name,
            )
        if negatives < n_neg_required:
            report.add(
                CHECK_POOL,
                f"needs {n_neg_required} negative pairs, only {negatives} available",
                workload,
                source.name,
            )

    return report


def _check_overlap(sources: Dict[str, CorpusSource], report: ValidationReport) -> None:
    by_corpus: Dict[str, List[str]] = {}
    for workload, source in sources.items():
        # Keyed on the *registry* corpus, not the display name: three
        # `MockCorpusSource`s all report `source_corpus="mock"` but are three
        # independent synthetic pools, not one shared corpus. The failure mode
        # this guard exists for — two workloads pointed at the same real
        # corpus — is a real-corpus mistake, and a synthetic source is already
        # refused outright by `--require-real`.
        if not source.corpus_key:
            continue
        by_corpus.setdefault(source.corpus_key, []).append(workload)
    for corpus, workloads in sorted(by_corpus.items()):
        if len(workloads) > 1:
            report.add(
                CHECK_OVERLAP,
                f"workloads {sorted(workloads)} all draw from corpus {corpus!r}; a shared "
                "corpus removes the workload contrast the workload hypothesis tests",
                corpus=corpus,
            )


def _check_checksums(
    source: CorpusSource,
    registry: CorpusRegistry,
    raw_root: Optional[PathLike],
    report: ValidationReport,
    workload: str,
) -> bool:
    try:
        entry = registry.get(source.corpus_key)
    except CorpusRegistryError as exc:
        report.add(CHECK_CHECKSUM, str(exc), workload, source.name)
        return False

    expected = {f.filename: f.sha256 for f in entry.files}
    ok = True
    for path in source.source_files():
        path = Path(path)
        pinned = expected.get(path.name)
        if pinned is None:
            report.notes.append(
                f"{workload}/{source.name}: {path.name} has no pinned checksum in "
                f"{registry.path.name if registry.path else 'the registry'} — "
                "run `python scripts/fetch_corpora.py --pin` once to pin it"
            )
            continue
        actual = sha256_file(path)
        if actual != pinned:
            report.add(
                CHECK_CHECKSUM,
                f"{path}: SHA-256 {actual} does not match the pinned {pinned}",
                workload,
                source.name,
            )
            ok = False
    return ok


def _is_readable(path: PathLike) -> bool:
    try:
        with Path(path).open("rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False
