"""
Merge partial harness runs into one complete result set (LEV-4 / D3).

The grid is 30 configurations, and the analysis pipeline (LEV-8) requires all
30 in a single `results.csv`. But a production run does not always happen in one
sitting: `scripts/run_experiments.py --workloads chat` produces 10 rows, a
re-sampled workload has to be re-run on its own, and a long sweep gets
interrupted. Without a merge step the only way to get 30 rows is to re-run
everything, which is what makes people re-run the two workloads that were
already fine.

This module merges at the level of **raw CSV rows**, not DataFrames: the harness
writes pre-formatted strings (`f"{threshold:.2f}"`, `f"{precision:.6f}"`), so
passing them through untouched keeps a merged file byte-identical to what a
single full run would have written, and keeps the determinism guarantee the
harness makes. Parsing to floats and reformatting would not.

Two failures are loud by design, because both produce a file that *looks*
complete:

- **Duplicate `config_id`.** Two runs covering the same cell means one of them
  is stale, and silently keeping either would publish statistics nobody chose.
  The error names the cell and both files.
- **Incomplete grid.** A merged set missing cells, or carrying cells outside the
  frozen grid, is rejected here rather than in the middle of a two-way ANOVA
  that assumes a balanced design.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from levy.experiment.config import ExperimentConfig, full_grid
from levy.experiment.runner import DECISIONS_FIELDNAMES, RESULTS_FIELDNAMES

PathLike = Union[str, Path]

RESULTS_FILENAME = "results.csv"
DECISIONS_FILENAME = "decisions.csv"
RUN_META_FILENAME = "run_meta.json"

#: `run_meta.json` keys whose values must agree across every merged run: a
#: result set whose halves came from different datasets or different embedding
#: providers is not one experiment, and the ±5% replication check would compare
#: it against a reference it never matched.
_MUST_AGREE = ("dataset_path", "embedding_provider", "llm_provider")


class ResultsMergeError(RuntimeError):
    """Raised when harness runs cannot be merged into one valid result set."""


@dataclass
class MergeInput:
    """One harness output directory being merged."""

    directory: Path
    results: List[Dict[str, str]]
    decisions: Optional[List[Dict[str, str]]] = None
    run_meta: Dict = field(default_factory=dict)

    @property
    def config_ids(self) -> List[str]:
        return [row["config_id"] for row in self.results]


def read_rows(path: PathLike, fieldnames: Sequence[str]) -> List[Dict[str, str]]:
    """
    Read a harness CSV as raw strings, validating the column contract.

    Values are never coerced — see the module docstring: their on-disk
    formatting *is* the contract.
    """
    path = Path(path)
    if not path.is_file():
        raise ResultsMergeError(f"{path}: not found")
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [name for name in fieldnames if name not in (reader.fieldnames or [])]
        if missing:
            raise ResultsMergeError(
                f"{path}: missing required column(s): {', '.join(missing)}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ResultsMergeError(f"{path}: contains no rows")
    return rows


def load_input(directory: PathLike) -> MergeInput:
    """
    Load one harness output directory. `results.csv` is required;
    `decisions.csv` and `run_meta.json` are loaded when present.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ResultsMergeError(f"{directory}: harness output directory not found")

    results = read_rows(directory / RESULTS_FILENAME, RESULTS_FIELDNAMES)

    decisions_path = directory / DECISIONS_FILENAME
    decisions = (
        read_rows(decisions_path, DECISIONS_FIELDNAMES) if decisions_path.is_file() else None
    )

    meta_path = directory / RUN_META_FILENAME
    run_meta: Dict = {}
    if meta_path.is_file():
        try:
            run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ResultsMergeError(f"{meta_path}: not valid JSON: {exc}") from exc
        if not isinstance(run_meta, dict):
            raise ResultsMergeError(f"{meta_path}: expected a JSON object")

    return MergeInput(
        directory=directory, results=results, decisions=decisions, run_meta=run_meta
    )


def merge_results(inputs: Sequence[MergeInput]) -> List[Dict[str, str]]:
    """
    Concatenate every input's result rows, rejecting a `config_id` that appears
    twice — within one file or across two.
    """
    if not inputs:
        raise ResultsMergeError("no harness runs to merge")

    seen: Dict[str, Path] = {}
    merged: List[Dict[str, str]] = []
    for item in inputs:
        for row in item.results:
            config_id = row["config_id"]
            previous = seen.get(config_id)
            if previous is not None:
                where = (
                    f"twice in {item.directory / RESULTS_FILENAME}"
                    if previous == item.directory
                    else f"in both {previous / RESULTS_FILENAME} and "
                    f"{item.directory / RESULTS_FILENAME}"
                )
                raise ResultsMergeError(
                    f"duplicate configuration {config_id!r} — it appears {where}. "
                    "One of the two runs is stale; drop it (or re-run the cell once) "
                    "rather than letting the merge pick a winner. Nothing written."
                )
            seen[config_id] = item.directory
            merged.append(row)
    return merged


def merge_decisions(inputs: Sequence[MergeInput]) -> Optional[List[Dict[str, str]]]:
    """
    Concatenate the per-pair audit logs, or return `None` when no input has one.

    A mix — some runs with `decisions.csv`, some without — is an error: the
    merged log would silently cover only part of the grid, which is worse than
    not having one at all.
    """
    with_log = [item for item in inputs if item.decisions is not None]
    if not with_log:
        return None
    if len(with_log) != len(inputs):
        missing = [
            str(item.directory) for item in inputs if item.decisions is None
        ]
        raise ResultsMergeError(
            f"some runs have no {DECISIONS_FILENAME}: {missing}. Merging would produce an "
            "audit log covering only part of the grid. Nothing written."
        )

    seen: Dict[Tuple[str, str], Path] = {}
    merged: List[Dict[str, str]] = []
    for item in inputs:
        for row in item.decisions or []:
            key = (row["config_id"], row["pair_id"])
            previous = seen.get(key)
            if previous is not None:
                raise ResultsMergeError(
                    f"duplicate decision row for configuration {key[0]!r} / pair {key[1]!r} "
                    f"in {previous / DECISIONS_FILENAME} and "
                    f"{item.directory / DECISIONS_FILENAME}. Nothing written."
                )
            seen[key] = item.directory
            merged.append(row)
    return merged


def check_grid_coverage(rows: Sequence[Dict[str, str]]) -> None:
    """
    Require the merged rows to be exactly the frozen 2 x 3 x 5 grid.

    Both directions matter: a missing cell breaks the balanced design the ANOVA
    assumes, and an unexpected `config_id` means a run outside the frozen grid
    was merged in, which would change what the analysis is a statement about.
    """
    expected = [config.config_id for config in full_grid()]
    present = [row["config_id"] for row in rows]

    missing = [config_id for config_id in expected if config_id not in set(present)]
    unexpected = sorted(set(present) - set(expected))

    problems: List[str] = []
    if missing:
        problems.append(
            f"{len(missing)} of {len(expected)} configuration(s) missing: {missing}"
        )
    if unexpected:
        problems.append(
            f"{len(unexpected)} configuration(s) outside the frozen grid: {unexpected}"
        )
    if problems:
        raise ResultsMergeError(
            "the merged result set does not cover the frozen grid "
            f"({len(expected)} configurations = 2 models x 3 workloads x 5 thresholds): "
            + "; ".join(problems)
            + ". Run the missing cells with `python scripts/run_experiments.py` and merge "
            "again. Nothing written."
        )


def sort_by_grid(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Order rows as `full_grid()` enumerates them, so a merged file is
    indistinguishable from a single full run's output.

    Rows outside the grid keep their relative order at the end; in practice
    `check_grid_coverage` has already rejected them, but sorting is not the
    place to lose data.
    """
    rank = {config.config_id: index for index, config in enumerate(full_grid())}
    fallback = len(rank)
    return sorted(
        rows, key=lambda row: (rank.get(row["config_id"], fallback), row["config_id"])
    )


def sort_decisions_by_grid(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Same grid order for the audit log, preserving each configuration's pair order."""
    rank = {config.config_id: index for index, config in enumerate(full_grid())}
    fallback = len(rank)
    indexed = list(enumerate(rows))
    return [
        row
        for _, row in sorted(
            indexed, key=lambda item: (rank.get(item[1]["config_id"], fallback), item[0])
        )
    ]


def write_rows(path: PathLike, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    """Write raw rows back out in the canonical column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def merge_run_meta(inputs: Sequence[MergeInput], timestamp: str) -> Dict:
    """
    Build the merged sidecar: the keys downstream tools read
    (`dataset_path`, `embedding_provider`), the union of the grids, and each
    source run recorded so the provenance of every cell stays traceable.
    """
    agreed: Dict[str, object] = {}
    for key in _MUST_AGREE:
        values = {
            json.dumps(item.run_meta[key], sort_keys=True): item.directory
            for item in inputs
            if key in item.run_meta
        }
        if len(values) > 1:
            detail = ", ".join(
                f"{directory}: {value}" for value, directory in sorted(values.items())
            )
            raise ResultsMergeError(
                f"runs disagree on {key!r} ({detail}). Merging them would describe one "
                "experiment as if it had a single input. Nothing written."
            )
        if values:
            agreed[key] = json.loads(next(iter(values)))

    model_identities: Dict[str, object] = {}
    for item in inputs:
        model_identities.update(item.run_meta.get("model_identities") or {})

    grid: List[Dict[str, object]] = []
    for config in full_grid():
        grid.append(
            {
                "model": config.model,
                "workload": config.workload,
                "threshold": config.threshold,
            }
        )

    return {
        "generated_by": "scripts/merge_results.py",
        "generated_at_utc": timestamp,
        **agreed,
        "n_configurations": len(grid),
        "grid": grid,
        "model_identities": model_identities,
        "merged_from": [
            {
                "directory": str(item.directory),
                "n_configurations": len(item.results),
                "config_ids": item.config_ids,
                "run_meta": item.run_meta,
            }
            for item in inputs
        ],
        "latency": {
            "note": (
                "Merged from separate harness runs, so no single elapsed time describes "
                "this result set; per-run timings are under 'merged_from'. LLM latency is "
                "synthetic in all of them (MockLLMClient sleeps a fixed 0.5s per call)."
            )
        },
    }


def configs_for_ids(config_ids: Sequence[str]) -> List[ExperimentConfig]:
    """The grid cells named by `config_ids`, in grid order (used for reporting)."""
    wanted = set(config_ids)
    return [config for config in full_grid() if config.config_id in wanted]
