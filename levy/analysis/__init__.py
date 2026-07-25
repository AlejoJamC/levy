"""
Statistical analysis pipeline (LEV-8 / D3).

A pure consumer of the LEV-4 harness output contract: it reads a harness
output directory (`results.csv`, `decisions.csv`, `run_meta.json`) and emits
the dissertation's D3 evidence bundle -- two-way ANOVA on false positive rate
testing H0-1/H0-2/H0-3, conditional Tukey HSD, threshold-selection curve
tables and figures, the Cohen's kappa section (sourced from `levy.dataset`,
never reimplemented here), and a metadata sidecar.

The pipeline is dataset-agnostic: it works against any harness output
directory, whether produced from the committed synthetic fixture or from the
real 900-pair dataset.
"""

from levy.analysis.curves import (
    HIT_RATE_VIABILITY,
    build_curve_tables,
    write_curve_figures,
)
from levy.analysis.hypothesis import (
    AnovaResult,
    TukeyResult,
    run_tukey_hsd,
    run_two_way_anova,
)
from levy.analysis.io import (
    DECISIONS_REQUIRED_COLUMNS,
    HarnessContractError,
    HarnessOutputs,
    RESULTS_REQUIRED_COLUMNS,
    load_harness_outputs,
    load_results,
)
from levy.analysis.replication import (
    ABSOLUTE_FLOOR,
    RELATIVE_TOLERANCE,
    ReplicationReport,
    compare_results,
    format_report,
    tolerance_rule,
)
from levy.analysis.report import build_analysis_bundle

__all__ = [
    "ABSOLUTE_FLOOR",
    "AnovaResult",
    "DECISIONS_REQUIRED_COLUMNS",
    "HIT_RATE_VIABILITY",
    "HarnessContractError",
    "HarnessOutputs",
    "RELATIVE_TOLERANCE",
    "RESULTS_REQUIRED_COLUMNS",
    "ReplicationReport",
    "TukeyResult",
    "build_analysis_bundle",
    "build_curve_tables",
    "compare_results",
    "format_report",
    "load_harness_outputs",
    "load_results",
    "run_tukey_hsd",
    "run_two_way_anova",
    "tolerance_rule",
    "write_curve_figures",
]
