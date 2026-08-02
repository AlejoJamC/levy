"""
Corpus provenance registry (LEV-12).

`data/corpora.json` records, per corpus, everything needed to acquire it and
to prove which snapshot was used: canonical URL, snapshot identifier, licence,
expected filenames, SHA-256 checksums and citation. This module is the only
reader of that file — `scripts/fetch_corpora.py`, `levy/dataset/validation.py`
and `scripts/rehydrate_dataset.py` all go through it rather than embedding
their own copies of the same facts, so changing a filename or a checksum in
the registry is observed everywhere without another file being edited.

A `sha256` of `None` means "not yet pinned". `scripts/fetch_corpora.py --pin`
records the checksums of the files actually held; after that, a mismatch is a
hard failure rather than a silent upstream re-release.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]

#: Repository-root-relative location of the registry, resolved absolutely so
#: callers work regardless of their working directory.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "corpora.json"

#: Where acquisition puts raw corpora. Contents are gitignored (see data/raw/README.md).
DEFAULT_RAW_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

_REQUIRED_ENTRY_FIELDS = (
    "title",
    "workload",
    "adapter",
    "snapshot",
    "canonical_url",
    "licence",
    "citation",
    "files",
)
_REQUIRED_FILE_FIELDS = ("filename", "url", "sha256")

_CHUNK = 1024 * 1024


class CorpusRegistryError(ValueError):
    """Raised when the corpus registry is missing, malformed, or lacks an entry."""


@dataclass(frozen=True)
class CorpusFile:
    """
    One expected file of a corpus, with its download URL and pinned checksum.

    When `archive_member` is set, `url` points at a zip archive and this file
    is the unique member of it whose basename is `archive_member` — the
    PIT-2015 release ships `train.data` and `dev.data` inside one zip, and
    matching on basename means the registry does not have to encode the
    archive's internal directory layout.
    """

    filename: str
    url: Optional[str]
    sha256: Optional[str]
    archive_member: Optional[str] = None

    @property
    def is_pinned(self) -> bool:
        return self.sha256 is not None

    @property
    def in_archive(self) -> bool:
        return self.archive_member is not None


@dataclass(frozen=True)
class CorpusEntry:
    """Provenance for one corpus, as recorded in `data/corpora.json`."""

    key: str
    title: str
    workload: str
    adapter: str
    snapshot: str
    canonical_url: str
    licence: str
    citation: str
    files: Tuple[CorpusFile, ...]
    download_url: Optional[str] = None
    licence_url: Optional[str] = None
    redistributable: bool = False
    manual: bool = False
    manual_reason: Optional[str] = None
    notes: Optional[str] = None

    @property
    def acquisition_url(self) -> str:
        """The URL a human should open to obtain this corpus."""
        return self.download_url or self.canonical_url

    def directory(self, raw_root: PathLike = DEFAULT_RAW_ROOT) -> Path:
        """This corpus's directory under the raw-corpus root."""
        return Path(raw_root) / self.key

    def paths(self, raw_root: PathLike = DEFAULT_RAW_ROOT) -> List[Path]:
        """Expected on-disk location of every file of this corpus, in registry order."""
        directory = self.directory(raw_root)
        return [directory / f.filename for f in self.files]

    def missing_paths(self, raw_root: PathLike = DEFAULT_RAW_ROOT) -> List[Path]:
        """Expected files that are not present on disk."""
        return [path for path in self.paths(raw_root) if not path.is_file()]

    def provenance(self) -> Dict[str, Any]:
        """
        The subset of this entry that belongs in a run manifest: enough for a
        third party to prove they hold the same inputs, with no query text.
        """
        return {
            "title": self.title,
            "snapshot": self.snapshot,
            "canonical_url": self.canonical_url,
            "licence": self.licence,
            "files": [
                {"filename": f.filename, "sha256": f.sha256} for f in self.files
            ],
        }


@dataclass(frozen=True)
class CorpusRegistry:
    """All corpora the study depends on, keyed by short corpus name."""

    schema_version: int
    entries: Dict[str, CorpusEntry] = field(default_factory=dict)
    path: Optional[Path] = None

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def keys(self) -> List[str]:
        return list(self.entries.keys())

    def get(self, key: str) -> CorpusEntry:
        """
        The entry for `key`, or `CorpusRegistryError` naming the missing entry —
        never a guessed URL or filename.
        """
        try:
            return self.entries[key]
        except KeyError:
            raise CorpusRegistryError(
                f"{self.path or 'corpus registry'}: no entry for corpus {key!r}; "
                f"known corpora: {sorted(self.entries)}"
            ) from None

    def for_workload(self, workload: str) -> CorpusEntry:
        """The single entry declaring `workload`, or an error naming the ambiguity."""
        matches = [entry for entry in self.entries.values() if entry.workload == workload]
        if not matches:
            raise CorpusRegistryError(
                f"{self.path or 'corpus registry'}: no corpus declares workload {workload!r}"
            )
        if len(matches) > 1:
            raise CorpusRegistryError(
                f"{self.path or 'corpus registry'}: workload {workload!r} is claimed by "
                f"more than one corpus: {sorted(m.key for m in matches)}"
            )
        return matches[0]


def _parse_file(corpus_key: str, index: int, raw: Any, source: str) -> CorpusFile:
    if not isinstance(raw, dict):
        raise CorpusRegistryError(
            f"{source}: corpus {corpus_key!r} files[{index}] must be an object"
        )
    missing = [name for name in _REQUIRED_FILE_FIELDS if name not in raw]
    if missing:
        raise CorpusRegistryError(
            f"{source}: corpus {corpus_key!r} files[{index}] is missing field(s) {missing}"
        )
    if not raw["filename"]:
        raise CorpusRegistryError(
            f"{source}: corpus {corpus_key!r} files[{index}] has an empty filename"
        )
    archive_member = raw.get("archive_member")
    return CorpusFile(
        filename=str(raw["filename"]),
        url=raw["url"] if raw["url"] is None else str(raw["url"]),
        sha256=raw["sha256"] if raw["sha256"] is None else str(raw["sha256"]).lower(),
        archive_member=None if archive_member is None else str(archive_member),
    )


def _parse_entry(key: str, raw: Any, source: str) -> CorpusEntry:
    if not isinstance(raw, dict):
        raise CorpusRegistryError(f"{source}: corpus {key!r} must be an object")
    missing = [name for name in _REQUIRED_ENTRY_FIELDS if name not in raw]
    if missing:
        raise CorpusRegistryError(
            f"{source}: corpus {key!r} is missing field(s) {missing}"
        )
    files_raw = raw["files"]
    if not isinstance(files_raw, list) or not files_raw:
        raise CorpusRegistryError(
            f"{source}: corpus {key!r} must declare a non-empty 'files' list"
        )
    files = tuple(
        _parse_file(key, index, item, source) for index, item in enumerate(files_raw)
    )
    manual = bool(raw.get("manual", False))
    if not manual:
        unfetchable = [f.filename for f in files if not f.url]
        if unfetchable:
            raise CorpusRegistryError(
                f"{source}: corpus {key!r} is not marked manual but file(s) "
                f"{unfetchable} have no url"
            )
    return CorpusEntry(
        key=key,
        title=str(raw["title"]),
        workload=str(raw["workload"]),
        adapter=str(raw["adapter"]),
        snapshot=str(raw["snapshot"]),
        canonical_url=str(raw["canonical_url"]),
        licence=str(raw["licence"]),
        citation=str(raw["citation"]),
        files=files,
        download_url=raw.get("download_url"),
        licence_url=raw.get("licence_url"),
        redistributable=bool(raw.get("redistributable", False)),
        manual=manual,
        manual_reason=raw.get("manual_reason"),
        notes=raw.get("notes"),
    )


def load_registry(path: Optional[PathLike] = None) -> CorpusRegistry:
    """
    Read and validate `data/corpora.json` (or `path`).

    Raises `CorpusRegistryError` naming the file and the offending corpus or
    field; never returns a partially parsed registry.
    """
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    try:
        text = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusRegistryError(f"{registry_path}: cannot read corpus registry: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CorpusRegistryError(f"{registry_path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CorpusRegistryError(f"{registry_path}: expected a JSON object at the top level")
    if "corpora" not in raw or not isinstance(raw["corpora"], dict):
        raise CorpusRegistryError(f"{registry_path}: missing a 'corpora' object")

    source = str(registry_path)
    entries = {
        key: _parse_entry(key, value, source) for key, value in raw["corpora"].items()
    }
    if not entries:
        raise CorpusRegistryError(f"{registry_path}: registry declares no corpora")
    return CorpusRegistry(
        schema_version=int(raw.get("schema_version", 1)),
        entries=entries,
        path=registry_path,
    )


def sha256_file(path: PathLike) -> str:
    """SHA-256 of a file's contents, streamed so large corpora are not loaded whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_checksums(
    registry_path: Optional[PathLike] = None,
    raw_root: PathLike = DEFAULT_RAW_ROOT,
) -> Dict[str, str]:
    """
    Compute and record the SHA-256 of every present file whose registry entry
    is not yet pinned, writing the registry back in place.

    Returns a `{"<corpus>/<filename>": "<sha256>"}` map of what was newly
    pinned. Already-pinned entries are left alone — re-pinning would defeat the
    point of pinning, which is to make an upstream change loud.
    """
    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    registry = load_registry(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    pinned: Dict[str, str] = {}
    for key, entry in registry.entries.items():
        for index, corpus_file in enumerate(entry.files):
            if corpus_file.is_pinned:
                continue
            on_disk = entry.directory(raw_root) / corpus_file.filename
            if not on_disk.is_file():
                continue
            checksum = sha256_file(on_disk)
            raw["corpora"][key]["files"][index]["sha256"] = checksum
            pinned[f"{key}/{corpus_file.filename}"] = checksum

    if pinned:
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return pinned
