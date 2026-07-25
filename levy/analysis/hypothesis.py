"""
Hypothesis testing on false positive rate (LEV-8 / D3).

Implements the frozen S&D Report's statistical analysis plan literally:

  * Two-way ANOVA on false positive rate with factors (embedding model,
    workload) *including their interaction*, over the 30 per-configuration
    FPR values. The 5 thresholds within each (model, workload) cell are the
    replicates the frozen grid provides.
  * The three frozen hypotheses, each explicitly rejected or retained at
    alpha = 0.05:
        H0_1  no embedding-model main effect on FPR
        H0_2  no workload main effect on FPR
        H0_3  no model x workload interaction
  * Tukey HSD post-hoc comparisons, run only for effects the ANOVA found
    significant (over the 6 model x workload cells when the interaction is
    significant), always accompanied by a ran/skipped statement.

Nothing beyond the frozen plan is added: no extra tests, no multiplicity
corrections across the three hypotheses, no auto-"correction" when residual
diagnostics look poor. The diagnostics are reported so the author can
interpret them in the dissertation; interpreting them is not code's job.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DEFAULT_ALPHA = 0.05

#: The dependent variable prescribed by the frozen plan.
RESPONSE = "fpr"

EFFECT_MODEL = "model"
EFFECT_WORKLOAD = "workload"
EFFECT_INTERACTION = "model:workload"

#: Design-matrix term -> (tidy effect name, hypothesis id, hypothesis statement).
_TERMS = {
    "C(model)": (
        EFFECT_MODEL,
        "H0_1",
        "No effect of embedding model on false positive rate",
    ),
    "C(workload)": (
        EFFECT_WORKLOAD,
        "H0_2",
        "No effect of workload type on false positive rate",
    ),
    "C(model):C(workload)": (
        EFFECT_INTERACTION,
        "H0_3",
        "No interaction between embedding model and workload",
    ),
}

ANOVA_COLUMNS = [
    "hypothesis",
    "effect",
    "statement",
    "df",
    "sum_sq",
    "mean_sq",
    "F",
    "p_value",
    "alpha",
    "decision",
]

TUKEY_COLUMNS = [
    "effect",
    "group1",
    "group2",
    "meandiff",
    "p_adj",
    "lower",
    "upper",
    "reject",
]


class AnovaDesignError(ValueError):
    """
    Raised when the supplied results cannot support the frozen two-way design
    with interaction (a factor with fewer than two levels, an empty cell, or
    no within-cell replication).
    """


@dataclass
class AnovaResult:
    """
    The fitted two-way ANOVA.

    `table` is the machine-readable hypothesis table (one row per effect plus
    a `Residual` row for auditability); `significant_effects` lists the tidy
    effect names whose p-value fell below `alpha`, and drives Tukey HSD.

    `degenerate` is True when the response has no variance at all across the
    30 configurations, in which case the F-tests are undefined (NaN) and the
    decisions are reported as `undefined` rather than as retentions.
    """

    table: pd.DataFrame
    alpha: float
    significant_effects: List[str]
    degenerate: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def p_value(self, hypothesis: str) -> float:
        """p-value for a hypothesis id ("H0_1" / "H0_2" / "H0_3")."""
        row = self.table.loc[self.table["hypothesis"] == hypothesis]
        if row.empty:
            raise KeyError(f"Unknown hypothesis: {hypothesis!r}")
        return float(row["p_value"].iloc[0])

    def decision(self, hypothesis: str) -> str:
        """"reject", "retain", or "undefined" for a hypothesis id."""
        row = self.table.loc[self.table["hypothesis"] == hypothesis]
        if row.empty:
            raise KeyError(f"Unknown hypothesis: {hypothesis!r}")
        return str(row["decision"].iloc[0])


@dataclass
class TukeyResult:
    """
    Post-hoc comparisons. `table` is empty (headers only) when no effect was
    significant; `statement` always says whether Tukey ran and why, and
    `per_effect` records that reason per effect.
    """

    table: pd.DataFrame
    ran: bool
    statement: str
    per_effect: Dict[str, str] = field(default_factory=dict)


def _validate_design(frame: pd.DataFrame) -> Dict[str, Any]:
    """Check the crossed design is estimable; return cell-count facts."""
    for column in (RESPONSE, EFFECT_MODEL, EFFECT_WORKLOAD):
        if column not in frame.columns:
            raise AnovaDesignError(f"results are missing the {column!r} column")

    models = sorted(frame[EFFECT_MODEL].unique())
    workloads = sorted(frame[EFFECT_WORKLOAD].unique())
    if len(models) < 2:
        raise AnovaDesignError(
            f"two-way ANOVA needs at least 2 embedding models, got {models}"
        )
    if len(workloads) < 2:
        raise AnovaDesignError(
            f"two-way ANOVA needs at least 2 workloads, got {workloads}"
        )

    counts = frame.groupby([EFFECT_MODEL, EFFECT_WORKLOAD]).size()
    expected_cells = len(models) * len(workloads)
    if len(counts) < expected_cells:
        present = {f"{m}|{w}" for m, w in counts.index}
        missing = sorted(
            f"{m}|{w}" for m in models for w in workloads if f"{m}|{w}" not in present
        )
        raise AnovaDesignError(
            f"the model x workload design has empty cell(s): {', '.join(missing)}"
        )
    if int(counts.min()) < 2:
        thin = sorted(f"{m}|{w}" for (m, w), c in counts.items() if c < 2)
        raise AnovaDesignError(
            "the interaction term needs at least 2 observations (thresholds) per "
            f"cell; under-replicated cell(s): {', '.join(thin)}"
        )

    return {
        "models": list(models),
        "workloads": list(workloads),
        "cell_counts": {f"{m}|{w}": int(c) for (m, w), c in counts.items()},
        "balanced": bool(counts.nunique() == 1),
    }


def _residual_diagnostics(residuals: np.ndarray, frame: pd.DataFrame) -> Dict[str, Any]:
    """
    Standard assumption diagnostics available from the fitted residuals.
    Reported, never acted upon -- interpretation is dissertation-space.
    """
    diagnostics: Dict[str, Any] = {
        "n_observations": int(residuals.size),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if residuals.size > 1 else None,
        "note": (
            "Diagnostics are reported for the author's interpretation; the pipeline "
            "does not modify the frozen analysis plan in response to them."
        ),
    }

    if residuals.size >= 3 and not np.allclose(residuals, residuals[0]):
        shapiro_w, shapiro_p = stats.shapiro(residuals)
        diagnostics["shapiro_wilk"] = {"W": float(shapiro_w), "p_value": float(shapiro_p)}
    else:
        diagnostics["shapiro_wilk"] = {
            "W": None,
            "p_value": None,
            "skipped": "fewer than 3 residuals, or all residuals identical",
        }

    cells = [
        group[RESPONSE].to_numpy()
        for _, group in frame.groupby([EFFECT_MODEL, EFFECT_WORKLOAD], sort=True)
    ]
    spread = [cell for cell in cells if cell.size > 1 and not np.allclose(cell, cell[0])]
    if len(spread) == len(cells) and len(cells) > 1:
        levene_stat, levene_p = stats.levene(*cells)
        diagnostics["levene"] = {"statistic": float(levene_stat), "p_value": float(levene_p)}
    else:
        diagnostics["levene"] = {
            "statistic": None,
            "p_value": None,
            "skipped": "at least one model x workload cell has zero variance",
        }

    return diagnostics


def run_two_way_anova(results: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> AnovaResult:
    """
    Fit `fpr ~ C(model) * C(workload)` and return the hypothesis table.

    Type II sums of squares are used. The frozen grid is balanced (5
    thresholds in every model x workload cell), and for a balanced design
    Types I, II and III coincide -- the choice is documented for the methods
    section rather than being load-bearing.
    """
    design = _validate_design(results)
    frame = results[[RESPONSE, EFFECT_MODEL, EFFECT_WORKLOAD]].copy()

    # A response with no variance at all (e.g. every configuration scoring
    # FPR = 0.0) makes the F-tests undefined. That is reported as such, never
    # laundered into a retention of the null.
    degenerate = bool(frame[RESPONSE].nunique() <= 1)

    with warnings.catch_warnings():
        if degenerate:
            # statsmodels warns about the rank-deficient covariance and the
            # 0/0 F ratio. Both are exactly the degeneracy already detected
            # above and reported in the table and diagnostics, so the raw
            # warnings would only add noise to an answer already given.
            warnings.simplefilter("ignore")
        fitted = ols(f"{RESPONSE} ~ C({EFFECT_MODEL}) * C({EFFECT_WORKLOAD})", data=frame).fit()
        table = anova_lm(fitted, typ=2)

    rows = []
    significant: List[str] = []
    for term, (effect, hypothesis, statement) in _TERMS.items():
        record = table.loc[term]
        df = float(record["df"])
        sum_sq = float(record["sum_sq"])
        f_stat = float(record["F"])
        p_value = float(record["PR(>F)"])
        if math.isnan(p_value):
            decision = "undefined"
        elif p_value < alpha:
            decision = "reject"
            significant.append(effect)
        else:
            decision = "retain"
        rows.append(
            {
                "hypothesis": hypothesis,
                "effect": effect,
                "statement": statement,
                "df": df,
                "sum_sq": sum_sq,
                "mean_sq": sum_sq / df if df else float("nan"),
                "F": f_stat,
                "p_value": p_value,
                "alpha": alpha,
                "decision": decision,
            }
        )

    residual = table.loc["Residual"]
    residual_df = float(residual["df"])
    residual_ss = float(residual["sum_sq"])
    rows.append(
        {
            "hypothesis": "",
            "effect": "Residual",
            "statement": "Within-cell (threshold replicate) variation",
            "df": residual_df,
            "sum_sq": residual_ss,
            "mean_sq": residual_ss / residual_df if residual_df else float("nan"),
            "F": float("nan"),
            "p_value": float("nan"),
            "alpha": alpha,
            "decision": "",
        }
    )

    diagnostics: Dict[str, Any] = {"design": design, "sum_of_squares_type": "II"}
    if degenerate:
        diagnostics["degenerate_response"] = {
            "reason": (
                f"every configuration reports the same {RESPONSE} "
                f"({float(frame[RESPONSE].iloc[0])}); the F-tests are undefined"
            ),
            "consequence": "H0_1/H0_2/H0_3 are reported as 'undefined', not 'retain'",
        }
    diagnostics.update(_residual_diagnostics(np.asarray(fitted.resid, dtype=float), frame))

    return AnovaResult(
        table=pd.DataFrame(rows, columns=ANOVA_COLUMNS),
        alpha=alpha,
        significant_effects=significant,
        degenerate=degenerate,
        diagnostics=diagnostics,
    )


def _tukey_rows(effect: str, labels: pd.Series, values: pd.Series, alpha: float) -> List[dict]:
    tukey = pairwise_tukeyhsd(endog=values.to_numpy(), groups=labels.to_numpy(), alpha=alpha)
    frame = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    return [
        {
            "effect": effect,
            "group1": str(row["group1"]),
            "group2": str(row["group2"]),
            "meandiff": float(row["meandiff"]),
            "p_adj": float(row["p-adj"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "reject": bool(row["reject"]),
        }
        for _, row in frame.iterrows()
    ]


def run_tukey_hsd(
    results: pd.DataFrame,
    anova: AnovaResult,
    alpha: float = DEFAULT_ALPHA,
) -> TukeyResult:
    """
    Run Tukey HSD for each effect the ANOVA found significant.

    Grouping follows the frozen wording: the model main effect compares the
    2 models, the workload main effect the 3 workloads, and a significant
    interaction compares the 6 (model, workload) cells. Effects that were not
    significant are recorded as skipped, with the reason, in `per_effect`.
    """
    frame = results.copy()
    frame["cell"] = frame[EFFECT_MODEL].astype(str) + "|" + frame[EFFECT_WORKLOAD].astype(str)

    groupings = {
        EFFECT_MODEL: frame[EFFECT_MODEL].astype(str),
        EFFECT_WORKLOAD: frame[EFFECT_WORKLOAD].astype(str),
        EFFECT_INTERACTION: frame["cell"],
    }

    rows: List[dict] = []
    per_effect: Dict[str, str] = {}
    for effect, labels in groupings.items():
        if effect not in anova.significant_effects:
            per_effect[effect] = (
                f"skipped: the {effect} F-test was undefined (zero variance in "
                f"{RESPONSE}); no effect to follow up"
                if anova.degenerate
                else f"skipped: ANOVA retained the null for {effect} at alpha={alpha}"
            )
            continue
        # Every grouping here has at least 2 groups: `_validate_design` already
        # rejected any design with a single model or workload level.
        rows.extend(_tukey_rows(effect, labels, frame[RESPONSE], alpha))
        per_effect[effect] = (
            f"ran: ANOVA rejected the null for {effect} at alpha={alpha}; "
            f"{labels.nunique()} groups compared"
        )

    ran = bool(rows)
    if ran:
        which = ", ".join(sorted(anova.significant_effects))
        statement = f"Tukey HSD ran for significant effect(s): {which} (alpha={alpha})."
    elif anova.degenerate:
        statement = (
            f"Tukey HSD was not run: {RESPONSE} has zero variance across all "
            "configurations, so the ANOVA F-tests are undefined and there is no "
            "significant effect to follow up."
        )
    else:
        statement = (
            "Tukey HSD was not run: no effect was significant at "
            f"alpha={alpha} (model, workload, model:workload all retained)."
        )

    return TukeyResult(
        table=pd.DataFrame(rows, columns=TUKEY_COLUMNS),
        ran=ran,
        statement=statement,
        per_effect=per_effect,
    )
