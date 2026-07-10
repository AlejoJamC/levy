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
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from levy.dataset.schema import QueryPair

PathLike = Union[str, Path]

_VALID_ANSWERS = {"0": 0, "1": 1}
_SKIP = "s"
_QUIT = "q"


@dataclass
class AnnotationSummary:
    """Outcome of one `BlindAnnotationSession.run()` call."""

    total_pairs: int
    already_labeled: int  # had an author_label before this run started
    newly_labeled: int  # labeled during this run
    skipped: int  # explicitly skipped this run
    quit_early: bool


class BlindAnnotationSession:
    """
    Resumable blind re-annotation loop over a list of `QueryPair`s.

    Usage:
        pairs = load_dataset("data/ground_truth.json")
        session = BlindAnnotationSession(pairs, progress_path="progress.json")
        summary = session.run()  # prompts via input()/print() by default
        save_dataset(pairs, "data/ground_truth.csv", "data/ground_truth.json")
    """

    def __init__(
        self,
        pairs: List[QueryPair],
        progress_path: PathLike,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        overwrite: bool = False,
    ) -> None:
        self.pairs = pairs
        self.progress_path = Path(progress_path)
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.overwrite = overwrite

        self._progress: Dict[str, int] = self._load_progress()
        self._apply_progress()

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def _load_progress(self) -> Dict[str, int]:
        if not self.progress_path.exists():
            return {}
        with self.progress_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_progress(self) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("w", encoding="utf-8") as fh:
            json.dump(self._progress, fh, indent=2, sort_keys=True)

    def _apply_progress(self) -> None:
        """Merge previously recorded progress-file labels into `self.pairs`."""
        by_id = {pair.pair_id: pair for pair in self.pairs}
        for pair_id, label in self._progress.items():
            pair = by_id.get(pair_id)
            if pair is None:
                continue
            if pair.author_label is None or self.overwrite:
                pair.author_label = label

    # ------------------------------------------------------------------
    # Session loop
    # ------------------------------------------------------------------

    def run(self) -> AnnotationSummary:
        """
        Run the blind annotation loop over all pairs lacking an
        `author_label` (or all pairs, if `overwrite=True`).

        Returns as soon as every pair is labeled, the annotator quits (`q`),
        or input is exhausted (e.g. piped stdin ends) — in all cases already
        recorded progress is preserved on disk.
        """
        already_labeled = sum(1 for p in self.pairs if p.author_label is not None)
        newly_labeled = 0
        skipped = 0
        quit_early = False

        to_annotate = [
            p for p in self.pairs if self.overwrite or p.author_label is None
        ]

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
            self._progress[pair.pair_id] = label
            self._save_progress()
            newly_labeled += 1

        return AnnotationSummary(
            total_pairs=len(self.pairs),
            already_labeled=already_labeled,
            newly_labeled=newly_labeled,
            skipped=skipped,
            quit_early=quit_early,
        )

    def remaining_count(self) -> int:
        """Number of pairs still lacking an `author_label`."""
        return sum(1 for p in self.pairs if p.author_label is None)
