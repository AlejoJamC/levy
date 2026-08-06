"""
Blind re-annotation tooling for the ground-truth dataset (LEV-3 / D2).

The frozen S&D Report requires "the author's blind re-annotations" of the
900 sampled pairs, to be compared against the original corpus labels via
Cohen's kappa (`levy.dataset.kappa`). "Blind" means the annotator must not
see `original_label` (or `source_corpus` / `source_pair_id`, which could hint
at the label) while annotating — only `query_1` and `query_2` are shown.

`BlindAnnotationSession` is a synchronous, resumable CLI-driven loop:
  - progress (recorded `author_label`s) is persisted to a JSON file after
    every single answer, so a 900-pair session can be interrupted (Ctrl-C,
    crash, closed terminal) and resumed later without losing work;
  - an existing `author_label` (whether already present in the dataset file
    or recorded in a prior progress file) is never overwritten unless the
    caller explicitly passes `overwrite=True`;
  - I/O is injected (`input_fn`, `output_fn`) so the whole flow is testable
    without a real terminal.

**Presentation order** is part of the instrument, not a detail. Annotating
in file order means one solid block per workload in whatever sequence the
sampler happened to emit, which lets position stand in for content: after
forty consecutive near-duplicates the next pair is judged against the run of
answers rather than on its own. So:
  - workload blocks are presented in `DEFAULT_WORKLOAD_ORDER` (faq, chat,
    code) — code last, because its pairs are the longest and most tiring to
    read, and fatigue there costs the most;
  - within a block the order is shuffled under an explicit `order_seed`,
    recorded in the progress file rather than left to vary per run;
  - the fully resolved order is persisted, so an interrupted session resumes
    in the order it started in instead of re-deriving one.

**Stale progress.** A single workload can be re-sampled (see
`levy.dataset.workload_update`), which reuses that workload's `pair_id`s for
different pairs. A progress file keyed on `pair_id` alone would then re-apply
the old labels to the new pairs — silently, and to exactly the pairs that were
supposed to come back unannotated. Every recorded label therefore carries its
pair's `source_pair_id` as a fingerprint; an entry whose fingerprint no longer
matches is dropped and counted, never applied.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from levy.dataset.backup import backup_file
from levy.dataset.schema import (
    WORKLOAD_CHAT,
    WORKLOAD_CODE,
    WORKLOAD_FAQ,
    WORKLOADS,
    QueryPair,
)

PathLike = Union[str, Path]

_VALID_ANSWERS = {"0": 0, "1": 1}
_SKIP = "s"
_QUIT = "q"

#: Order the workload blocks are presented in unless the caller overrides it.
#: Code last: its pairs are Stack Overflow posts, far longer than a Quora
#: question or a tweet, so leaving them until the end keeps reading fatigue
#: from landing on the two workloads that are quick to judge.
DEFAULT_WORKLOAD_ORDER: Tuple[str, ...] = (WORKLOAD_FAQ, WORKLOAD_CHAT, WORKLOAD_CODE)

#: Progress-file schema version. v1 was a flat `{pair_id: label}` map; v2 adds
#: the resolved presentation order, its seed, and per-label fingerprints.
PROGRESS_VERSION = 2


class AnnotationOrderError(ValueError):
    """Raised for a malformed `--workload-order` / workload selection."""


@dataclass
class AnnotationSummary:
    """Outcome of one `BlindAnnotationSession.run()` call."""

    total_pairs: int
    already_labeled: int  # had an author_label before this run started
    newly_labeled: int  # labeled during this run
    skipped: int  # explicitly skipped this run
    quit_early: bool
    session_limit_reached: bool = False  # stopped because --session-limit was hit
    stale_progress_dropped: int = 0  # progress entries whose pair was re-sampled
    selected_pairs: int = 0  # pairs in scope after the workload restriction


def parse_workload_order(text: str) -> Tuple[str, ...]:
    """
    Parse a `faq,chat,code` presentation order.

    Rejects an unknown workload and a repeated one — both are almost certainly
    a typo, and guessing at the intent would silently change the order the
    session is conducted in.
    """
    names = [part.strip() for part in str(text).split(",")]
    names = [name for name in names if name]
    if not names:
        raise AnnotationOrderError(
            f"workload order is empty; expected a comma-separated sequence of {list(WORKLOADS)}"
        )
    seen: List[str] = []
    for name in names:
        if name not in WORKLOADS:
            raise AnnotationOrderError(
                f"unknown workload {name!r} in workload order; expected one of {list(WORKLOADS)}"
            )
        if name in seen:
            raise AnnotationOrderError(
                f"workload {name!r} appears more than once in workload order {names}"
            )
        seen.append(name)
    return tuple(seen)


def validate_workloads(workloads: Optional[Iterable[str]]) -> Optional[Tuple[str, ...]]:
    """Normalise a workload restriction: deduplicated, order-preserving, validated."""
    if workloads is None:
        return None
    normalised: List[str] = []
    for name in workloads:
        if name not in WORKLOADS:
            raise AnnotationOrderError(
                f"unknown workload {name!r}; expected one of {list(WORKLOADS)}"
            )
        if name not in normalised:
            normalised.append(name)
    if not normalised:
        return None
    return tuple(normalised)


def resolve_presentation_order(
    pairs: Sequence[QueryPair],
    workloads: Optional[Sequence[str]] = None,
    workload_order: Sequence[str] = DEFAULT_WORKLOAD_ORDER,
    order_seed: int = 0,
) -> List[str]:
    """
    The `pair_id` sequence a session presents: workload blocks in
    `workload_order`, shuffled within each block under `order_seed`.

    Each block is shuffled with its own `Random(f"{order_seed}:{workload}")`,
    not with one stream shared across blocks, so restricting a session to one
    workload does not change the order the other workloads would have been
    presented in. `random.Random` seeded from a string is stable across Python
    versions, which is what makes a recorded seed reproducible.

    A selected workload absent from `workload_order` is appended after the named
    blocks in `WORKLOADS` order rather than dropped — the caller is told, so an
    incomplete `--workload-order` cannot silently omit pairs.
    """
    selected = set(workloads) if workloads else set(WORKLOADS)
    blocks = [name for name in workload_order if name in selected]
    blocks += [name for name in WORKLOADS if name in selected and name not in blocks]

    order: List[str] = []
    for workload in blocks:
        block = [pair.pair_id for pair in pairs if pair.workload == workload]
        random.Random(f"{order_seed}:{workload}").shuffle(block)
        order.extend(block)
    return order


def unordered_workloads(
    workloads: Optional[Sequence[str]], workload_order: Sequence[str]
) -> List[str]:
    """Selected workloads that `workload_order` does not name (for a note to the user)."""
    selected = list(workloads) if workloads else list(WORKLOADS)
    return [name for name in selected if name not in set(workload_order)]


def read_progress_labels(raw: Dict) -> Dict[str, Dict]:
    """
    Normalise both progress formats to `{pair_id: {"label", "source_pair_id"}}`.

    v1 (a flat `{pair_id: label}` map) carries no fingerprint, so its entries can
    only be applied on `pair_id` alone — as they always were, and which is why
    `prune_progress` exists. Only v2 entries can be recognised as stale.
    """
    if raw.get("version"):
        labels: Dict[str, Dict] = {}
        for pair_id, entry in (raw.get("labels") or {}).items():
            if isinstance(entry, dict):
                labels[pair_id] = {
                    "label": int(entry["label"]),
                    "source_pair_id": entry.get("source_pair_id"),
                }
            else:  # tolerate a hand-edited bare label inside a v2 file
                labels[pair_id] = {"label": int(entry), "source_pair_id": None}
        return labels
    return {
        str(pair_id): {"label": int(label), "source_pair_id": None}
        for pair_id, label in raw.items()
    }


def prune_progress(
    progress_path: PathLike,
    dropped_pair_ids: Iterable[str],
    pairs: Sequence[QueryPair],
) -> int:
    """
    Remove `dropped_pair_ids` from a progress file and rewrite it as v2, with a
    fingerprint attached to every label that survives. Returns how many were
    dropped; a missing file is 0.

    **The caller must have backed the file up already** — it holds hours of
    annotation work. `scripts/sample_dataset.py` backs it up in the same group as
    the dataset files, under one timestamp, before any write happens.

    This is the primary defence against a re-sampled workload inheriting the
    previous sample's answers, and it belongs here rather than in the session
    because only the re-sample knows a replacement happened. The session's
    per-label fingerprint is the second line: it catches the case where the
    dataset was replaced by some other means, but it cannot help a v1 progress
    file, which carries no fingerprints to compare — and v1 is exactly what a
    file written before this existed is.
    """
    progress_path = Path(progress_path)
    if not progress_path.exists():
        return 0

    with progress_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise AnnotationOrderError(
            f"{progress_path}: expected a JSON object, got {type(raw).__name__}"
        )

    labels = read_progress_labels(raw)
    dropped = {str(pair_id) for pair_id in dropped_pair_ids}
    kept = {
        pair_id: entry for pair_id, entry in labels.items() if pair_id not in dropped
    }

    by_id = {pair.pair_id: pair for pair in pairs}
    for pair_id, entry in kept.items():
        pair = by_id.get(pair_id)
        if pair is not None:
            entry["source_pair_id"] = pair.source_pair_id

    payload = {
        "version": PROGRESS_VERSION,
        "order_seed": raw.get("order_seed", 0),
        "workload_order": raw.get("workload_order") or list(DEFAULT_WORKLOAD_ORDER),
        "workloads": raw.get("workloads"),
        # The recorded order is dropped with the pairs it ordered: a re-sample
        # changes what there is to present, so the next session derives a fresh
        # order over the new pairs rather than resuming a stale one.
        "order": [],
        "labels": {pair_id: dict(entry) for pair_id, entry in sorted(kept.items())},
    }
    with progress_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return len(labels) - len(kept)


class BlindAnnotationSession:
    """
    Resumable blind re-annotation loop over a list of `QueryPair`s.

    Usage:
        pairs = load_dataset("data/ground_truth.full.json")
        session = BlindAnnotationSession(pairs, progress_path="progress.json")
        summary = session.run()  # prompts via input()/print() by default
        save_dataset(pairs, "data/ground_truth.full.csv", "data/ground_truth.full.json")
    """

    def __init__(
        self,
        pairs: List[QueryPair],
        progress_path: PathLike,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        overwrite: bool = False,
        workloads: Optional[Sequence[str]] = None,
        workload_order: Sequence[str] = DEFAULT_WORKLOAD_ORDER,
        order_seed: int = 0,
        session_limit: Optional[int] = None,
        backup_dir: Optional[PathLike] = None,
    ) -> None:
        self.pairs = pairs
        self.progress_path = Path(progress_path)
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.overwrite = overwrite
        self.workloads = validate_workloads(workloads)
        self.workload_order = tuple(workload_order)
        self.order_seed = int(order_seed)
        if session_limit is not None and session_limit < 1:
            raise AnnotationOrderError(
                f"session limit must be at least 1, got {session_limit!r}"
            )
        self.session_limit = session_limit
        self.backup_dir = backup_dir
        self.stale_progress_dropped = 0

        raw = self._load_progress_file()
        self._labels: Dict[str, Dict] = read_progress_labels(raw)
        self._apply_progress()

        self.order: List[str] = self._resolve_order(raw)
        # One backup per session, taken before the progress file is first
        # rewritten — the file holds hours of annotation work, and this run may
        # prune stale entries out of it.
        backup_file(self.progress_path, backup_dir=self.backup_dir)
        self._save_progress()

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def _load_progress_file(self) -> Dict:
        if not self.progress_path.exists():
            return {}
        with self.progress_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise AnnotationOrderError(
                f"{self.progress_path}: expected a JSON object, got {type(raw).__name__}"
            )
        return raw

    def _save_progress(self) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PROGRESS_VERSION,
            "order_seed": self.order_seed,
            "workload_order": list(self.workload_order),
            "workloads": list(self.workloads) if self.workloads else None,
            "order": list(self.order),
            "labels": {
                pair_id: dict(entry) for pair_id, entry in sorted(self._labels.items())
            },
        }
        with self.progress_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def _apply_progress(self) -> None:
        """
        Merge previously recorded progress-file labels into `self.pairs`.

        An entry whose fingerprint disagrees with the pair now holding that
        `pair_id` belongs to a pair that has since been re-sampled away: it is
        dropped from the progress file and counted, not applied.
        """
        by_id = {pair.pair_id: pair for pair in self.pairs}
        stale: List[str] = []
        for pair_id, entry in self._labels.items():
            pair = by_id.get(pair_id)
            if pair is None:
                continue  # belongs to another dataset slice; leave it alone
            fingerprint = entry.get("source_pair_id")
            if fingerprint is not None and fingerprint != pair.source_pair_id:
                stale.append(pair_id)
                continue
            if pair.author_label is None or self.overwrite:
                pair.author_label = entry["label"]
        for pair_id in stale:
            del self._labels[pair_id]
        self.stale_progress_dropped = len(stale)
        if stale:
            self.output_fn(
                f"[annotate] dropped {len(stale)} progress entry/entries whose pair was "
                f"re-sampled (e.g. {stale[0]}); those pairs are unannotated."
            )

    # ------------------------------------------------------------------
    # Presentation order
    # ------------------------------------------------------------------

    def _resolve_order(self, raw: Dict) -> List[str]:
        """
        Reuse the recorded order when resuming; derive a fresh one otherwise.

        The persisted order wins over a freshly derived one even if `order_seed`
        or `--workload-order` changed since — a session's order is fixed once it
        starts, and silently re-deriving it mid-way would present some pairs
        twice and others never. Pairs in scope but absent from the recorded
        order (a re-sampled workload, say) are appended in derived order.
        """
        in_scope = [
            pair.pair_id
            for pair in self.pairs
            if self.workloads is None or pair.workload in self.workloads
        ]
        scope = set(in_scope)
        recorded = [pair_id for pair_id in (raw.get("order") or []) if pair_id in scope]

        if not recorded:
            return resolve_presentation_order(
                self.pairs,
                workloads=self.workloads,
                workload_order=self.workload_order,
                order_seed=self.order_seed,
            )

        recorded_seed = raw.get("order_seed")
        if recorded_seed is not None and int(recorded_seed) != self.order_seed:
            self.output_fn(
                f"[annotate] resuming in the order recorded at seed {recorded_seed}, "
                f"not the requested seed {self.order_seed} — an in-progress session's "
                "order is fixed once it starts."
            )

        known = set(recorded)
        derived = resolve_presentation_order(
            self.pairs,
            workloads=self.workloads,
            workload_order=self.workload_order,
            order_seed=self.order_seed,
        )
        appended = [pair_id for pair_id in derived if pair_id not in known]
        if appended:
            self.output_fn(
                f"[annotate] {len(appended)} pair(s) are not in the recorded order "
                "(newly sampled since it was written); appending them after it."
            )
        return recorded + appended

    # ------------------------------------------------------------------
    # Session loop
    # ------------------------------------------------------------------

    def selected_pairs(self) -> List[QueryPair]:
        """Pairs in scope this session, in presentation order."""
        by_id = {pair.pair_id: pair for pair in self.pairs}
        return [by_id[pair_id] for pair_id in self.order if pair_id in by_id]

    def run(self) -> AnnotationSummary:
        """
        Run the blind annotation loop over the in-scope pairs lacking an
        `author_label` (or all of them, if `overwrite=True`), in the resolved
        presentation order.

        Returns as soon as every pair is labeled, the session limit is reached,
        the annotator quits (`q`), or input is exhausted (e.g. piped stdin ends)
        — in all cases already recorded progress is preserved on disk.
        """
        selected = self.selected_pairs()
        already_labeled = sum(1 for p in selected if p.author_label is not None)
        newly_labeled = 0
        skipped = 0
        quit_early = False
        limit_reached = False

        to_annotate = [p for p in selected if self.overwrite or p.author_label is None]
        if self.session_limit is not None:
            self.output_fn(
                f"[annotate] session limit: stopping after {self.session_limit} "
                f"labeled pair(s) of {len(to_annotate)} outstanding."
            )

        for index, pair in enumerate(to_annotate, start=1):
            self.output_fn(f"\n--- Pair {index}/{len(to_annotate)} ({pair.pair_id}) ---")
            self.output_fn(f"Query 1: {pair.query_1}")
            self.output_fn(f"Query 2: {pair.query_2}")
            self.output_fn("Are these the same question/intent? [1=yes, 0=no, s=skip, q=quit]")

            try:
                raw_answer = self.input_fn("> ").strip().lower()
            except (EOFError, StopIteration):
                quit_early = True
                break

            if raw_answer == _QUIT:
                quit_early = True
                break
            if raw_answer == _SKIP:
                skipped += 1
                continue
            if raw_answer not in _VALID_ANSWERS:
                self.output_fn(f"Unrecognized answer {raw_answer!r}; skipping this pair.")
                skipped += 1
                continue

            label = _VALID_ANSWERS[raw_answer]
            pair.author_label = label
            self._labels[pair.pair_id] = {
                "label": label,
                "source_pair_id": pair.source_pair_id,
            }
            self._save_progress()
            newly_labeled += 1

            if self.session_limit is not None and newly_labeled >= self.session_limit:
                limit_reached = True
                self.output_fn(
                    f"\n[annotate] session limit of {self.session_limit} reached — "
                    "stopping here. Re-run the same command to continue in this order."
                )
                break

        return AnnotationSummary(
            total_pairs=len(self.pairs),
            already_labeled=already_labeled,
            newly_labeled=newly_labeled,
            skipped=skipped,
            quit_early=quit_early,
            session_limit_reached=limit_reached,
            stale_progress_dropped=self.stale_progress_dropped,
            selected_pairs=len(selected),
        )

    def remaining_count(self) -> int:
        """Number of in-scope pairs still lacking an `author_label`."""
        return sum(1 for p in self.selected_pairs() if p.author_label is None)
