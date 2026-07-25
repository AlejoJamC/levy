"""
D3 bundle assembly (LEV-8).

One call turns a harness output directory into the full evidence bundle:

    anova.csv             hypothesis table for H0_1 / H0_2 / H0_3
    tukey.csv             pairwise post-hoc comparisons (headers only if skipped)
    tukey_status.csv      per-effect ran/skipped statement and reason
    curves_hit_rate.csv   threshold vs hit rate, per (model, workload)
    curves_precision.csv  threshold vs precision, per (model, workload)
    kappa.json            Cohen's kappa, sourced from levy.dataset (not recomputed)
    figures/              curve_hit_rate.{png,pdf}, curve_precision.{png,pdf}
    analysis_meta.json    input paths, library versions, diagnostics, timestamp

Every CSV is timestamp-free and byte-stable for identical input, mirroring
LEV-4's determinism convention. Versions and the generation timestamp live in
`analysis_meta.json` and nowhere else.
"""

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import pandas as pd
import scipy
import statsmodels

from levy.analysis.curves import build_curve_tables, write_curve_figures
from levy.analysis.hypothesis import (
    DEFAULT_ALPHA,
    AnovaResult,
    TukeyResult,
    run_tukey_hsd,
    run_two_way_anova,
)
from levy.analysis.io import HarnessOutputs, load_harness_outputs
from levy.dataset.io import load_dataset
from levy.dataset.kappa import KappaResult, kappa_report

PathLike = Union[str, Path]

#: Frozen annotation-validity bar (S&D Report): Cohen's kappa > 0.7.
KAPPA_THRESHOLD = 0.7

FIGURES_DIRNAME = "figures"

ANOVA_FILENAME = "anova.csv"
TUKEY_FILENAME = "tukey.csv"
TUKEY_STATUS_FILENAME = "tukey_status.csv"
CURVES_HIT_RATE_FILENAME = "curves_hit_rate.csv"
CURVES_PRECISION_FILENAME = "curves_precision.csv"
KAPPA_FILENAME = "kappa.json"
META_FILENAME = "analysis_meta.json"

#: The CSVs whose byte-stability across re-runs is part of the contract.
DETERMINISTIC_TABLES = (
    ANOVA_FILENAME,
    TUKEY_FILENAME,
    TUKEY_STATUS_FILENAME,
    CURVES_HIT_RATE_FILENAME,
    CURVES_PRECISION_FILENAME,
)


@dataclass
class AnalysisBundle:
    """What `build_analysis_bundle` produced, for callers and tests."""

    out_dir: Path
    anova: AnovaResult
    tukey: TukeyResult
    hit_rate_table: pd.DataFrame
    precision_table: pd.DataFrame
    kappa: Dict[str, Any]
    figures: List[Path]
    meta: Dict[str, Any]


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    """Deterministic CSV: fixed column order, no index, no timestamps."""
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def _kappa_result_to_dict(result: KappaResult) -> Dict[str, Any]:
    """Serialize LEV-3's `KappaResult`; the kappa itself is never recomputed here."""
    return {
        "n_annotated": result.n_annotated,
        "n_excluded_unannotated": result.n_excluded_unannotated,
        "observed_agreement": result.observed_agreement,
        "expected_agreement": result.expected_agreement,
        "kappa": result.kappa,
        "confusion": dict(result.confusion),
    }


def build_kappa_section(
    dataset_path: Optional[PathLike],
    threshold: float = KAPPA_THRESHOLD,
) -> Dict[str, Any]:
    """
    Build the bundle's kappa section by calling LEV-3's `kappa_report` over
    the dataset -- the computation is not reimplemented here.

    The section carries its own provenance: the source corpora behind the
    pairs, and a `fixture_only` flag that is true whenever any pair comes
    from the synthetic placeholder fixture, so a fixture-derived kappa can
    never be mistaken for the released dataset's value.
    """
    if dataset_path is None:
        return {
            "status": "unavailable",
            "reason": (
                "no dataset path supplied and the harness run_meta.json did not "
                "record one; pass --dataset to include the kappa section"
            ),
            "threshold": threshold,
            "source": "levy.dataset.kappa.kappa_report",
        }

    path = Path(dataset_path)
    if not path.is_file():
        return {
            "status": "unavailable",
            "reason": f"dataset file not found: {path}",
            "dataset_path": str(path),
            "threshold": threshold,
            "source": "levy.dataset.kappa.kappa_report",
        }

    pairs = load_dataset(path)
    report = kappa_report(pairs)

    corpora = sorted({pair.source_corpus for pair in pairs})
    fixture_only = any("synthetic" in corpus for corpus in corpora)
    fully_annotated = report.overall.n_excluded_unannotated == 0

    passes: Optional[bool]
    if report.overall.kappa is None or not fully_annotated:
        passes = None
    else:
        passes = bool(report.overall.kappa > threshold)

    return {
        "status": "computed",
        "source": "levy.dataset.kappa.kappa_report",
        "dataset_path": str(path),
        "n_pairs": len(pairs),
        "threshold": threshold,
        "fully_annotated": fully_annotated,
        "passes_threshold": passes,
        "provenance": {
            "source_corpora": corpora,
            "fixture_only": fixture_only,
            "note": (
                "FIXTURE ONLY -- this kappa is derived from placeholder data and is "
                "not the dissertation's reported annotation-validity result."
                if fixture_only
                else "Derived from the released dataset."
            ),
        },
        "overall": _kappa_result_to_dict(report.overall),
        "per_workload": {
            workload: _kappa_result_to_dict(result)
            for workload, result in report.per_workload.items()
        },
    }


def _tukey_status_table(tukey: TukeyResult) -> pd.DataFrame:
    rows = [
        {
            "effect": effect,
            "ran": reason.startswith("ran"),
            "reason": reason,
        }
        for effect, reason in sorted(tukey.per_effect.items())
    ]
    rows.append({"effect": "__overall__", "ran": tukey.ran, "reason": tukey.statement})
    return pd.DataFrame(rows, columns=["effect", "ran", "reason"])


def _library_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy.__version__,
    }


def build_analysis_bundle(
    harness_dir: PathLike,
    out_dir: PathLike,
    dataset_path: Optional[PathLike] = None,
    alpha: float = DEFAULT_ALPHA,
    kappa_threshold: float = KAPPA_THRESHOLD,
) -> AnalysisBundle:
    """
    Run the whole D3 pipeline over a harness output directory.

    `dataset_path` is used only for the kappa section; when omitted it falls
    back to the `dataset_path` recorded in the harness's `run_meta.json`, so
    the bundle stays tied to the data the results came from.
    """
    outputs: HarnessOutputs = load_harness_outputs(harness_dir)
    results = outputs.results

    anova = run_two_way_anova(results, alpha=alpha)
    tukey = run_tukey_hsd(results, anova, alpha=alpha)
    hit_rate_table, precision_table = build_curve_tables(results)

    resolved_dataset = dataset_path or outputs.run_meta.get("dataset_path")
    kappa = build_kappa_section(resolved_dataset, threshold=kappa_threshold)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_table(anova.table, out_dir / ANOVA_FILENAME)
    _write_table(tukey.table, out_dir / TUKEY_FILENAME)
    _write_table(_tukey_status_table(tukey), out_dir / TUKEY_STATUS_FILENAME)
    _write_table(hit_rate_table, out_dir / CURVES_HIT_RATE_FILENAME)
    _write_table(precision_table, out_dir / CURVES_PRECISION_FILENAME)

    with (out_dir / KAPPA_FILENAME).open("w", encoding="utf-8") as fh:
        json.dump(kappa, fh, indent=2)
        fh.write("\n")

    figures = write_curve_figures(hit_rate_table, precision_table, out_dir / FIGURES_DIRNAME)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "harness_dir": str(Path(harness_dir)),
            "n_configurations": outputs.n_configurations,
            "dataset_path_for_kappa": None if resolved_dataset is None else str(resolved_dataset),
            "harness_run_meta": outputs.run_meta,
        },
        "parameters": {"alpha": alpha, "kappa_threshold": kappa_threshold},
        "anova": {
            "response": "fpr",
            "formula": "fpr ~ C(model) * C(workload)",
            "significant_effects": anova.significant_effects,
            "degenerate_response": anova.degenerate,
            "decisions": {
                row["hypothesis"]: row["decision"]
                for _, row in anova.table.iterrows()
                if row["hypothesis"]
            },
            "diagnostics": anova.diagnostics,
        },
        "tukey": {
            "ran": tukey.ran,
            "statement": tukey.statement,
            "per_effect": tukey.per_effect,
        },
        "outputs": {
            "tables": list(DETERMINISTIC_TABLES),
            "kappa": KAPPA_FILENAME,
            "figures": [str(path.relative_to(out_dir)) for path in figures],
        },
        "library_versions": _library_versions(),
        "note": (
            "Tables are timestamp-free and byte-identical across re-runs on identical "
            "input; versions and the generation timestamp live only in this sidecar."
        ),
    }
    with (out_dir / META_FILENAME).open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
        fh.write("\n")

    return AnalysisBundle(
        out_dir=out_dir,
        anova=anova,
        tukey=tukey,
        hit_rate_table=hit_rate_table,
        precision_table=precision_table,
        kappa=kappa,
        figures=figures,
        meta=meta,
    )
