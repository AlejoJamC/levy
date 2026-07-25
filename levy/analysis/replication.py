"""
Replication check against the frozen +/-5% criterion (LEV-8, Success
Criterion 3 of the S&D Report: "replication within +/-5%").

Compares the headline metrics -- precision and recall -- of a candidate
harness run against a reference `results.csv`, configuration by
configuration.

Tolerance rule (documented, auditable, applied uniformly):

    |candidate - reference| <= max(ABSOLUTE_FLOOR, RELATIVE_TOLERANCE * |reference|)

The relative term is the frozen +/-5%. The absolute floor exists because a
relative tolerance collapses to zero near a reference of 0.0, where it would
demand bit-exact equality of a quantity the criterion never meant to pin
that hard; the floor is stated in every report so the criterion stays
auditable rather than implicit.

Under mock providers the harness is byte-deterministic (LEV-4), so a
self-comparison matches exactly; the tolerance exists for real-provider runs.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

#: Frozen criterion: headline metrics must replicate within +/-5%.
RELATIVE_TOLERANCE = 0.05

#: Absolute floor for near-zero reference values (5 percentage points of a
#: rate is meaningless when the reference is ~0; 0.01 is one percentage point).
ABSOLUTE_FLOOR = 0.01

#: The metrics the criterion names.
HEADLINE_METRICS: Tuple[str, ...] = ("precision", "recall")

DIFF_COLUMNS = [
    "config_id",
    "metric",
    "reference",
    "candidate",
    "abs_diff",
    "rel_diff",
    "tolerance",
    "within_tolerance",
]


def tolerance_rule(
    relative: float = RELATIVE_TOLERANCE,
    absolute_floor: float = ABSOLUTE_FLOOR,
) -> str:
    """Human-readable statement of the rule actually applied."""
    return (
        f"|candidate - reference| <= max({absolute_floor}, {relative:.0%} * |reference|) "
        "per configuration, for precision and recall"
    )


@dataclass
class ReplicationReport:
    """
    Outcome of one comparison. `table` has one row per (configuration,
    headline metric) with both values and the computed deviation, so a
    failure is itemized rather than merely asserted.
    """

    table: pd.DataFrame
    passed: bool
    rule: str
    missing_configs: List[str] = field(default_factory=list)
    extra_configs: List[str] = field(default_factory=list)

    @property
    def violations(self) -> pd.DataFrame:
        return self.table.loc[~self.table["within_tolerance"]]


def compare_results(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    relative: float = RELATIVE_TOLERANCE,
    absolute_floor: float = ABSOLUTE_FLOOR,
) -> ReplicationReport:
    """
    Compare `candidate` against `reference` on the headline metrics.

    A configuration present in the reference but absent from the candidate
    (or vice versa) fails the check: replication means reproducing the same
    grid, not a subset of it.
    """
    ref_indexed = reference.set_index("config_id")
    cand_indexed = candidate.set_index("config_id")

    missing = sorted(set(ref_indexed.index) - set(cand_indexed.index))
    extra = sorted(set(cand_indexed.index) - set(ref_indexed.index))

    rows = []
    for config_id in sorted(set(ref_indexed.index) & set(cand_indexed.index)):
        for metric in HEADLINE_METRICS:
            ref_value = float(ref_indexed.loc[config_id, metric])
            cand_value = float(cand_indexed.loc[config_id, metric])
            abs_diff = abs(cand_value - ref_value)
            tolerance = max(absolute_floor, relative * abs(ref_value))
            rows.append(
                {
                    "config_id": config_id,
                    "metric": metric,
                    "reference": ref_value,
                    "candidate": cand_value,
                    "abs_diff": abs_diff,
                    "rel_diff": abs_diff / abs(ref_value) if ref_value else float("inf"),
                    "tolerance": tolerance,
                    "within_tolerance": bool(abs_diff <= tolerance),
                }
            )

    table = pd.DataFrame(rows, columns=DIFF_COLUMNS)
    passed = bool(not missing and not extra and (table.empty or table["within_tolerance"].all()))

    return ReplicationReport(
        table=table,
        passed=passed,
        rule=tolerance_rule(relative, absolute_floor),
        missing_configs=missing,
        extra_configs=extra,
    )


def format_report(report: ReplicationReport, show_all: bool = False) -> str:
    """
    Render the report for a terminal. On failure the per-configuration diff
    table names the configuration, the metric, both values, and the computed
    deviation, per the frozen criterion's auditability requirement.
    """
    lines = [f"Replication rule: {report.rule}"]

    if report.missing_configs:
        lines.append(f"MISSING from candidate run: {', '.join(report.missing_configs)}")
    if report.extra_configs:
        lines.append(f"UNEXPECTED in candidate run: {', '.join(report.extra_configs)}")

    compared = len(report.table)
    lines.append(f"Compared {compared} (configuration, metric) value(s).")

    shown = report.table if show_all else report.violations
    if len(shown):
        heading = "All comparisons:" if show_all else "Out-of-tolerance comparisons:"
        lines.append(heading)
        for _, row in shown.iterrows():
            lines.append(
                f"  {row['config_id']}  {row['metric']}: "
                f"reference={row['reference']:.6f} candidate={row['candidate']:.6f} "
                f"abs_diff={row['abs_diff']:.6f} rel_diff={row['rel_diff']:.4%} "
                f"tolerance={row['tolerance']:.6f} "
                f"{'OK' if row['within_tolerance'] else 'FAIL'}"
            )

    lines.append("RESULT: replication PASSED" if report.passed else "RESULT: replication FAILED")
    return "\n".join(lines)
