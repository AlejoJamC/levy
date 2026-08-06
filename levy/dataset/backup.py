"""
Timestamped backups for ground-truth and results artifacts.

There is exactly **one** ground-truth dataset, at its canonical paths
(`data/ground_truth.ids.csv`, its sidecar, and the gitignored
`data/ground_truth.full.{csv,json}`). Every re-sample and every re-annotation
writes back to those same paths — no `_v2`, no `_new`, no dated working copy
living next to the real one. The one-dataset rule is what makes "the ground
truth" a thing that can be named at all.

That rule is only safe if an overwrite is recoverable, so this module is the
other half of it: **before** any write that would modify an existing artifact,
the current bytes are copied to `<dir>/backups/<stem>.<UTC timestamp><suffix>`,
and a backup that cannot be created aborts the write rather than proceeding.
Backups are never deleted and never overwritten — a second backup of the same
file in the same second gets a `-001` discriminator rather than clobbering the
first.

Timestamps use ISO 8601 *basic* format (`20260806T215233Z`) rather than the
extended format: the extended form's colons are illegal in filenames on
Windows and need quoting in every shell, and a backup you cannot `cp` is a
backup you will not use.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

PathLike = Union[str, Path]

#: Subdirectory, relative to the file being protected, that backups land in.
BACKUP_DIRNAME = "backups"

#: Largest `-NNN` discriminator tried before giving up on a same-second clash.
_MAX_DISCRIMINATORS = 1000


class BackupError(RuntimeError):
    """
    Raised when an existing artifact could not be backed up.

    Callers must treat this as fatal for the whole write: the invariant is
    "back up, then write", so a failed backup means nothing is written.
    """


def backup_timestamp(now: Optional[datetime] = None) -> str:
    """
    A filename-safe UTC ISO 8601 basic-format stamp, e.g. `20260806T215233Z`.

    Pass `now` to make a caller's naming deterministic under test; the default
    reads the clock in UTC (never local time — these names are compared across
    machines).
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def resolve_backup_dir(path: PathLike, backup_dir: Optional[PathLike] = None) -> Path:
    """Where backups of `path` go: `backup_dir` if given, else `<parent>/backups`."""
    return Path(backup_dir) if backup_dir is not None else Path(path).parent / BACKUP_DIRNAME


def _candidate_names(path: Path, timestamp: str) -> Iterable[str]:
    """
    `ground_truth.ids.csv` -> `ground_truth.ids.20260806T215233Z.csv`, then
    `...Z-001.csv`, `...Z-002.csv`, ... so two backups in the same second
    coexist instead of one destroying the other.

    Only the final suffix is treated as the extension: `ground_truth.ids.csv`
    keeps `ground_truth.ids` as its stem, so the backup is still recognisably
    the ids file rather than something called `ground_truth`.
    """
    stem, suffix = path.stem, path.suffix
    yield f"{stem}.{timestamp}{suffix}"
    for index in range(1, _MAX_DISCRIMINATORS):
        yield f"{stem}.{timestamp}-{index:03d}{suffix}"


def backup_file(
    path: PathLike,
    timestamp: Optional[str] = None,
    backup_dir: Optional[PathLike] = None,
) -> Optional[Path]:
    """
    Copy `path` into its backup directory, returning the backup's path.

    Returns `None` when `path` does not exist: there is nothing to protect, so
    the caller's write creates a new file rather than modifying one. Any other
    failure — an unwritable directory, a full disk, a `backup_dir` that is
    actually a file — raises `BackupError`, and the caller must not write.
    """
    path = Path(path)
    if not path.exists():
        return None
    if not path.is_file():
        raise BackupError(f"{path}: not a regular file, refusing to back it up")

    directory = resolve_backup_dir(path, backup_dir)
    stamp = timestamp or backup_timestamp()

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupError(
            f"{path}: cannot create backup directory {directory}: {exc}. "
            "Nothing was written — the current dataset is intact."
        ) from exc

    for name in _candidate_names(path, stamp):
        target = directory / name
        if target.exists():
            continue
        try:
            # copy2, not move: the original must stay in place until the
            # caller has written its replacement.
            shutil.copy2(path, target)
        except OSError as exc:
            raise BackupError(
                f"{path}: cannot write backup {target}: {exc}. "
                "Nothing was written — the current dataset is intact."
            ) from exc
        return target

    raise BackupError(
        f"{path}: exhausted {_MAX_DISCRIMINATORS} backup names for timestamp {stamp} "
        f"in {directory}; refusing to overwrite an existing backup"
    )


def backup_files(
    paths: Iterable[PathLike],
    timestamp: Optional[str] = None,
    backup_dir: Optional[PathLike] = None,
) -> Dict[Path, Path]:
    """
    Back up several artifacts under **one** shared timestamp, so a multi-file
    write (dataset CSV + JSON + ids + sidecar) has a single restore point
    instead of four names that have to be correlated by hand.

    Every existing file is backed up before this returns; the first failure
    raises `BackupError` so the caller writes nothing. Returns
    `{original: backup}`, omitting paths that did not exist.
    """
    stamp = timestamp or backup_timestamp()
    made: Dict[Path, Path] = {}
    unique: List[Path] = []
    for candidate in paths:
        candidate = Path(candidate)
        if candidate not in unique:
            unique.append(candidate)
    for path in unique:
        created = backup_file(path, timestamp=stamp, backup_dir=backup_dir)
        if created is not None:
            made[path] = created
    return made


def describe_backups(made: Dict[Path, Path]) -> str:
    """One line per backup, for a CLI to print. Empty string when none were needed."""
    return "\n".join(
        f"[backup] {original} -> {backup}" for original, backup in sorted(made.items())
    )
