#!/usr/bin/env python
"""
Acquire the raw corpora the study samples from (LEV-12).

This is the **only** entry point in the repository that touches the network,
and it is deliberately excluded from the test suite. Everything downstream —
sampling, validation, rehydration, the harness — reads local files.

What it does, per corpus in `data/corpora.json`:

- Downloads each expected file into `data/raw/<corpus>/`, extracting it from a
  release archive when the registry says so.
- Verifies every file against its pinned SHA-256. A download is written to a
  `.part` file and only moved into place once it verifies, so a mismatch never
  leaves a half-acquired corpus behind.
- Skips files already present whose checksum matches, so re-running is cheap
  and idempotent.
- For corpora that cannot be acquired without a human step (accepting terms on
  the hosting platform, a Google Drive folder), prints the exact URL, filename
  and expected checksum, and exits non-zero.

Usage:
    python scripts/fetch_corpora.py                  # acquire what it can
    python scripts/fetch_corpora.py --corpus sodd    # one corpus only
    python scripts/fetch_corpora.py --pin            # record checksums of what you hold
    python scripts/fetch_corpora.py --verify-only    # check, download nothing

On a first acquisition every checksum in the registry is `null`: nothing is
pinned, so nothing can mismatch. Run `--pin` once you hold the files you intend
to use, and from then on an upstream re-release is a loud failure rather than a
silent divergence in results.
"""

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.corpora import (
    DEFAULT_RAW_ROOT,
    DEFAULT_REGISTRY_PATH,
    CorpusEntry,
    CorpusFile,
    CorpusRegistryError,
    load_registry,
    pin_checksums,
    sha256_file,
)

USER_AGENT = "levy-corpus-fetch/1.0 (+https://github.com/AlejoJamC/levy)"
_TIMEOUT_SECONDS = 300


class AcquisitionError(RuntimeError):
    """Raised when a corpus file cannot be acquired or fails verification."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Corpus provenance registry (default: data/corpora.json)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT, help="Where to acquire into (default: data/raw)")
    parser.add_argument("--corpus", action="append", default=None, help="Limit to this corpus key; repeatable (default: all)")
    parser.add_argument("--pin", action="store_true", help="Record the SHA-256 of every present, not-yet-pinned file back into the registry")
    parser.add_argument("--verify-only", action="store_true", help="Verify what is already present; download nothing")
    return parser


def _download(url: str, destination: Path) -> None:
    """Fetch `url` to `destination`, streamed so a large archive is not held in memory."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            with destination.open("wb") as fh:
                shutil.copyfileobj(response, fh)
    except (urllib.error.URLError, OSError) as exc:
        raise AcquisitionError(f"{url}: download failed: {exc}") from exc


def _extract_member(archive: Path, member_basename: str, destination: Path) -> None:
    """
    Extract the unique member of `archive` whose basename is `member_basename`.

    Matching on basename keeps the archive's internal directory layout out of
    the registry. Zero or several matches is an error, never a pick.
    """
    try:
        with zipfile.ZipFile(archive) as zf:
            matches = [
                name
                for name in zf.namelist()
                if not name.endswith("/") and Path(name).name == member_basename
            ]
            if not matches:
                raise AcquisitionError(
                    f"{archive.name}: no member named {member_basename!r} in the archive"
                )
            if len(matches) > 1:
                raise AcquisitionError(
                    f"{archive.name}: {len(matches)} members named {member_basename!r} "
                    f"({matches}); the registry cannot tell which is meant"
                )
            with zf.open(matches[0]) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"{archive}: not a readable zip archive: {exc}") from exc


def _verify(path: Path, corpus_file: CorpusFile) -> None:
    """Raise unless `path` matches its pinned checksum (unpinned always passes)."""
    if not corpus_file.is_pinned:
        return
    actual = sha256_file(path)
    if actual != corpus_file.sha256:
        raise AcquisitionError(
            f"{path}: SHA-256 mismatch\n"
            f"        expected (pinned): {corpus_file.sha256}\n"
            f"        actual   (on disk): {actual}\n"
            "        The upstream release may have changed. Do not use this file; "
            "either re-acquire it or update the pin deliberately."
        )


def acquire_file(
    entry: CorpusEntry,
    corpus_file: CorpusFile,
    raw_dir: Path,
    archive_cache: Dict[str, Path],
    verify_only: bool,
) -> str:
    """
    Acquire one file. Returns a short status word for the run summary.

    The download lands in a sibling `.part` file and is only moved into place
    once it verifies, so a failure never leaves something that looks acquired.
    """
    target = entry.directory(raw_dir) / corpus_file.filename

    if target.is_file():
        _verify(target, corpus_file)
        return "present" if corpus_file.is_pinned else "present (unpinned)"

    if verify_only:
        return "absent"
    if entry.manual or not corpus_file.url:
        return "manual"

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        if corpus_file.in_archive:
            archive = archive_cache.get(corpus_file.url)
            if archive is None:
                archive = Path(tempfile.mkdtemp(prefix="levy-corpus-")) / "download.zip"
                print(f"    downloading archive {corpus_file.url}", file=sys.stderr)
                _download(corpus_file.url, archive)
                archive_cache[corpus_file.url] = archive
            _extract_member(archive, corpus_file.archive_member, part)
        else:
            print(f"    downloading {corpus_file.url}", file=sys.stderr)
            _download(corpus_file.url, part)
        _verify(part, corpus_file)
    except AcquisitionError:
        if part.exists():
            # Kept, not deleted: a mismatching file is evidence worth inspecting.
            # It is not moved into place, so it is not treated as acquired.
            print(f"    rejected download left at {part}", file=sys.stderr)
        raise
    part.replace(target)
    return "acquired"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        registry = load_registry(args.registry)
    except CorpusRegistryError as exc:
        print(f"[fetch_corpora] {exc}", file=sys.stderr)
        return 2

    keys = args.corpus or registry.keys()
    try:
        entries = [registry.get(key) for key in keys]
    except CorpusRegistryError as exc:
        print(f"[fetch_corpora] {exc}", file=sys.stderr)
        return 2

    archive_cache: Dict[str, Path] = {}
    manual: List[CorpusEntry] = []
    failures: List[str] = []
    absent: List[str] = []

    print(f"[fetch_corpora] registry: {registry.path}")
    print(f"[fetch_corpora] target:   {args.raw_dir}")
    for entry in entries:
        print(f"\n  {entry.key} ({entry.title}) — {entry.licence}")
        if not args.verify_only:
            entry.directory(args.raw_dir).mkdir(parents=True, exist_ok=True)
        needs_manual = False
        for corpus_file in entry.files:
            try:
                status = acquire_file(
                    entry, corpus_file, args.raw_dir, archive_cache, args.verify_only
                )
            except AcquisitionError as exc:
                print(f"    {corpus_file.filename}: FAILED", file=sys.stderr)
                print(f"      {exc}", file=sys.stderr)
                failures.append(f"{entry.key}/{corpus_file.filename}")
                continue
            print(f"    {corpus_file.filename}: {status}")
            needs_manual |= status == "manual"
            if status == "absent":
                absent.append(f"{entry.key}/{corpus_file.filename}")
        if needs_manual:
            manual.append(entry)

    for archive in archive_cache.values():
        shutil.rmtree(archive.parent, ignore_errors=True)

    if args.pin:
        pinned = pin_checksums(args.registry, args.raw_dir)
        if pinned:
            print(f"\n[fetch_corpora] pinned {len(pinned)} checksum(s) into {args.registry}:")
            for name, checksum in sorted(pinned.items()):
                print(f"    {name}  {checksum}")
        else:
            print("\n[fetch_corpora] --pin: nothing to pin (all present files already pinned)")

    if manual:
        print("\n" + "=" * 70, file=sys.stderr)
        print("Manual acquisition required — these cannot be downloaded here:", file=sys.stderr)
        for entry in manual:
            print(f"\n  {entry.key} ({entry.title})", file=sys.stderr)
            print(f"    reason:   {entry.manual_reason or 'requires a human step'}", file=sys.stderr)
            print(f"    open:     {entry.acquisition_url}", file=sys.stderr)
            print(f"    licence:  {entry.licence}", file=sys.stderr)
            print(f"    save to:  {entry.directory(args.raw_dir)}", file=sys.stderr)
            for corpus_file in entry.files:
                expected = corpus_file.sha256 or "not yet pinned — pin it with --pin"
                print(f"      - {corpus_file.filename}", file=sys.stderr)
                print(f"        expected SHA-256: {expected}", file=sys.stderr)
            if entry.notes:
                print(f"    note:     {entry.notes}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("Re-run this script once those files are in place; it is idempotent.", file=sys.stderr)

    if failures:
        print(f"\n[fetch_corpora] {len(failures)} file(s) failed: {failures}", file=sys.stderr)
        return 1
    if manual:
        return 1
    if absent:
        print(
            f"\n[fetch_corpora] --verify-only: {len(absent)} file(s) not acquired: {absent}",
            file=sys.stderr,
        )
        return 1
    print("\n[fetch_corpora] all corpora present and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
