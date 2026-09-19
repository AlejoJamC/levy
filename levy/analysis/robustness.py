"""
Robustness re-analysis of H0_1-H0_3 on the per-decision binary outcome (LEV-22).

Input is the released per-pair decision table (`decisions.csv`). Rows with
`label == 0` are kept and `is_fp = outcome == "FP"` is the Bernoulli
response, one observation per (configuration, negative pair).

Methods, in the order they are fitted:

  1. Exact stratified permutation tests.
     * model factor: model labels permuted within (workload, threshold)
       strata; statistic = difference of the mean per-configuration FPR
       between the two models (also the difference of pooled rates). The
       permutation distribution is the convolution of one hypergeometric per
       stratum, so the p-value is exact, not Monte Carlo. A Monte Carlo run of
       explicit label shuffles is reported beside it as a cross-check.
     * workload factor: workload labels permuted within (model, threshold)
       strata; statistic = sum of squared deviations of the per-workload mean
       per-configuration FPR from their mean. Exact via convolution of one
       multivariate hypergeometric per stratum.
     * model factor, cluster sign-flip (sensitivity): model labels swapped
       jointly across all thresholds of a pair, one swap per pair, enumerated
       exactly. Workloads are nested in pairs, so no cluster-level workload
       permutation exists.
  2. Logistic regression, `is_fp ~ C(model) * C(workload) + C(threshold)`:
     maximum likelihood (statsmodels GLM) and Firth-penalised likelihood
     (Jeffreys-prior penalty). Terms are tested type-II style (main effects in
     the additive model, the interaction against the additive model) by
     likelihood-ratio tests; for Firth, penalised likelihood-ratio tests with
     the penalty taken from the full design (the `logistf` convention), and
     profile penalised-likelihood intervals.
  3. Mixed-effects logistic with a random intercept per `pair_id`, fitted by
     maximum likelihood with adaptive Gauss-Hermite quadrature. Convergence is
     reported, never forced.
  4. Cluster bootstrap over `pair_id`, resampled within workload.

Nothing here reads or writes the frozen ANOVA outputs except to read
`anova.csv` for the comparison row.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import optimize, signal, special, stats

DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 22
DEFAULT_N_BOOT = 10_000
DEFAULT_N_MC = 20_000
GH_NODES = 25

FORMULA_FULL = "C(model) * C(workload) + C(threshold)"
FORMULA_ADDITIVE = "C(model) + C(workload) + C(threshold)"

#: hypothesis id -> (design term, formula of the model the term is tested in)
TERM_TESTS = {
    "H0_1": ("C(model)", FORMULA_ADDITIVE),
    "H0_2": ("C(workload)", FORMULA_ADDITIVE),
    "H0_3": ("C(model):C(workload)", FORMULA_FULL),
    "threshold": ("C(threshold)", FORMULA_ADDITIVE),
}

DECISIONS_REQUIRED = ("model", "workload", "threshold", "pair_id", "label", "outcome")


class RobustnessInputError(ValueError):
    """The decisions table does not have the shape this analysis needs."""


# --------------------------------------------------------------------------- data


def load_negatives(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"model": str, "workload": str, "threshold": str, "pair_id": str, "outcome": str})
    missing = [c for c in DECISIONS_REQUIRED if c not in df.columns]
    if missing:
        raise RobustnessInputError(f"decisions table missing columns: {missing}")
    return negatives_from_frame(df)


def negatives_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    neg = df[df["label"].astype(int) == 0].copy()
    bad = sorted(set(neg["outcome"]) - {"FP", "TN"})
    if bad:
        raise RobustnessInputError(f"negative-class rows with outcomes other than FP/TN: {bad}")
    neg["is_fp"] = (neg["outcome"] == "FP").astype(int)
    return neg[["model", "workload", "threshold", "pair_id", "is_fp"]].reset_index(drop=True)


def event_counts(neg: pd.DataFrame) -> pd.DataFrame:
    levels = [
        ("cell", ["model", "workload", "threshold"]),
        ("model:workload", ["model", "workload"]),
        ("model", ["model"]),
        ("workload", ["workload"]),
        ("threshold", ["threshold"]),
        ("total", []),
    ]
    rows = []
    for level, keys in levels:
        groups = [((), neg)] if not keys else neg.groupby(keys, sort=True)
        for key, g in groups:
            key = key if isinstance(key, tuple) else (key,)
            values = dict(zip(keys, key))
            events = int(g["is_fp"].sum())
            n = int(len(g))
            per_pair = g.groupby("pair_id")["is_fp"].sum()
            rows.append(
                {
                    "level": level,
                    "model": values.get("model", "ALL"),
                    "workload": values.get("workload", "ALL"),
                    "threshold": values.get("threshold", "ALL"),
                    "n": n,
                    "events": events,
                    "rate": events / n,
                    "zero_events": events == 0,
                    "n_pairs": int(per_pair.size),
                    "pairs_with_events": int((per_pair > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------- 1. permutation tests


def _hypergeom_pmf(total: int, events: int, draws: int) -> np.ndarray:
    support = np.arange(0, min(events, draws) + 1)
    return stats.hypergeom.pmf(support, total, events, draws)


def _model_strata(neg: pd.DataFrame) -> Tuple[str, str, List[Tuple[int, int, int]]]:
    models = sorted(neg["model"].unique())
    if len(models) != 2:
        raise RobustnessInputError(f"model permutation needs exactly 2 models, got {models}")
    a, b = models
    strata = []
    for _, g in neg.groupby(["workload", "threshold"], sort=True):
        na = int((g["model"] == a).sum())
        nb = int((g["model"] == b).sum())
        if na != nb or na == 0:
            raise RobustnessInputError("model permutation needs equal, non-zero n per model in every stratum")
        strata.append((na, int(g["is_fp"].sum()), int(g.loc[g["model"] == a, "is_fp"].sum())))
    return a, b, strata


def model_permutation_exact(neg: pd.DataFrame) -> Dict[str, Any]:
    """Exact two-sided p-values for the model factor, labels permuted within
    (workload, threshold) strata.

    Within a stratum with n observations per model and k events, the number of
    events landing on model A is Hypergeom(2n, k, n). Both statistics depend
    only on the per-n-group totals of those counts, so the distribution is the
    product of per-group convolutions.
    """
    a, b, strata = _model_strata(neg)
    n_strata = len(strata)
    total_n = sum(n for n, _, _ in strata)
    groups: Dict[int, List[Tuple[int, int]]] = {}
    for n, k, obs in strata:
        groups.setdefault(n, []).append((k, obs))
    dists = []
    for n, items in sorted(groups.items()):
        pmf = np.array([1.0])
        for k, _ in items:
            pmf = np.convolve(pmf, _hypergeom_pmf(2 * n, k, n))
        dists.append((n, sum(k for k, _ in items), pmf, sum(o for _, o in items)))

    def mean_config(sums):
        return sum((2 * s - k) / n for (n, k, _, _), s in zip(dists, sums)) / n_strata

    def pooled(sums):
        return sum(2 * s - k for (n, k, _, _), s in zip(dists, sums)) / total_n

    observed = [d[3] for d in dists]
    obs_mean, obs_pooled = mean_config(observed), pooled(observed)
    p_mean = p_pooled = 0.0
    for combo in itertools.product(*[range(len(d[2])) for d in dists]):
        prob = float(np.prod([d[2][s] for d, s in zip(dists, combo)]))
        if prob == 0.0:
            continue
        if abs(mean_config(combo)) >= abs(obs_mean) - 1e-12:
            p_mean += prob
        if abs(pooled(combo)) >= abs(obs_pooled) - 1e-12:
            p_pooled += prob
    events_a = sum(observed)
    events_total = sum(d[1] for d in dists)
    rate_a_mean = sum(s / n for (n, _, _, _), s in zip(dists, observed)) / n_strata
    rate_b_mean = sum((k - s) / n for (n, k, _, _), s in zip(dists, observed)) / n_strata
    return {
        "model_a": a,
        "model_b": b,
        "n_strata": n_strata,
        "events_a": events_a,
        "events_b": events_total - events_a,
        "n_per_model": total_n,
        "mean_config_fpr_a": rate_a_mean,
        "mean_config_fpr_b": rate_b_mean,
        "diff_mean_config_fpr": obs_mean,
        "p_mean_config": min(1.0, p_mean),
        "pooled_fpr_a": events_a / total_n,
        "pooled_fpr_b": (events_total - events_a) / total_n,
        "diff_pooled_fpr": obs_pooled,
        "p_pooled": min(1.0, p_pooled),
    }


def _workload_strata(neg: pd.DataFrame):
    workloads = sorted(neg["workload"].unique())
    if len(workloads) < 2:
        raise RobustnessInputError("workload permutation needs at least 2 workloads")
    sizes: Dict[str, int] = {}
    strata = []
    for _, g in neg.groupby(["model", "threshold"], sort=True):
        counts = []
        events = []
        for w in workloads:
            gw = g[g["workload"] == w]
            if sizes.setdefault(w, len(gw)) != len(gw) or len(gw) == 0:
                raise RobustnessInputError("workload permutation needs a constant, non-zero n per workload across strata")
            counts.append(len(gw))
            events.append(int(gw["is_fp"].sum()))
        strata.append((counts, events))
    return workloads, [sizes[w] for w in workloads], strata


def _mvhypergeom_pmf(sizes: Sequence[int], k: int) -> np.ndarray:
    """Array over (e_1, ..., e_{W-1}); e_W = k - sum is implied."""
    dims = len(sizes) - 1
    arr = np.zeros((k + 1,) * dims)
    denom = math.comb(sum(sizes), k)
    for idx in itertools.product(range(k + 1), repeat=dims):
        last = k - sum(idx)
        if last < 0:
            continue
        num = math.prod(math.comb(n, e) for n, e in zip(sizes, list(idx) + [last]))
        arr[idx] = num / denom
    return arr


def _workload_statistic(totals: np.ndarray, sizes: Sequence[int], n_strata: int) -> np.ndarray:
    """totals: (..., W) event totals per workload -> between-workload SS of
    the per-workload mean per-configuration FPR."""
    rates = totals / (np.asarray(sizes, dtype=float) * n_strata)
    return ((rates - rates.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)


def workload_permutation_exact(neg: pd.DataFrame) -> Dict[str, Any]:
    workloads, sizes, strata = _workload_strata(neg)
    n_strata = len(strata)
    pmf = np.array(1.0).reshape((1,) * (len(sizes) - 1))
    for counts, events in strata:
        pmf = signal.convolve(pmf, _mvhypergeom_pmf(counts, sum(events)), method="direct")
    k_total = sum(sum(e) for _, e in strata)
    observed = np.array([sum(e[i] for _, e in strata) for i in range(len(workloads))], dtype=float)
    t_obs = float(_workload_statistic(observed, sizes, n_strata))
    idx = np.indices(pmf.shape).reshape(len(sizes) - 1, -1).T.astype(float)
    last = k_total - idx.sum(axis=1, keepdims=True)
    totals = np.hstack([idx, last])
    probs = pmf.reshape(-1)
    valid = (last[:, 0] >= 0) & (probs > 0)
    t_all = _workload_statistic(totals[valid], sizes, n_strata)
    p = float(probs[valid][t_all >= t_obs - 1e-15].sum())
    return {
        "workloads": workloads,
        "n_per_config": dict(zip(workloads, sizes)),
        "n_strata": n_strata,
        "events": dict(zip(workloads, [int(x) for x in observed])),
        "mean_config_fpr": dict(zip(workloads, (observed / (np.array(sizes) * n_strata)).tolist())),
        "statistic": t_obs,
        "p_value": min(1.0, p),
    }


def model_permutation_mc(neg: pd.DataFrame, n_perm: int, seed: int) -> Dict[str, Any]:
    """Explicit shuffles of the model label within (workload, threshold)."""
    rng = np.random.default_rng(seed)
    a = sorted(neg["model"].unique())[0]
    n_strata = 0
    stat = np.zeros(n_perm)
    obs = 0.0
    for _, g in neg.groupby(["workload", "threshold"], sort=True):
        n_strata += 1
        y = g["is_fp"].to_numpy()
        is_a = (g["model"] == a).to_numpy()
        n = int(is_a.sum())
        obs += (y[is_a].sum() - y[~is_a].sum()) / n
        shuffled = rng.permuted(np.tile(y, (n_perm, 1)), axis=1)
        stat += (shuffled[:, :n].sum(axis=1) - shuffled[:, n:].sum(axis=1)) / n
    stat /= n_strata
    obs /= n_strata
    hits = int((np.abs(stat) >= abs(obs) - 1e-12).sum())
    return {"n_perm": n_perm, "seed": seed, "statistic": obs, "p_value": (hits + 1) / (n_perm + 1)}


def workload_permutation_mc(neg: pd.DataFrame, n_perm: int, seed: int) -> Dict[str, Any]:
    """Explicit shuffles of the workload label within (model, threshold)."""
    rng = np.random.default_rng(seed)
    workloads, sizes, strata = _workload_strata(neg)
    totals = np.zeros((n_perm, len(workloads)))
    observed = np.zeros(len(workloads))
    for _, g in neg.groupby(["model", "threshold"], sort=True):
        y = g["is_fp"].to_numpy()
        labels = g["workload"].to_numpy()
        shuffled = rng.permuted(np.tile(labels, (n_perm, 1)), axis=1)
        for i, w in enumerate(workloads):
            totals[:, i] += ((shuffled == w) * y).sum(axis=1)
            observed[i] += y[labels == w].sum()
    t_obs = float(_workload_statistic(observed, sizes, len(strata)))
    t_all = _workload_statistic(totals, sizes, len(strata))
    hits = int((t_all >= t_obs - 1e-15).sum())
    return {"n_perm": n_perm, "seed": seed, "statistic": t_obs, "p_value": (hits + 1) / (n_perm + 1)}


def model_cluster_signflip_exact(neg: pd.DataFrame) -> Dict[str, Any]:
    """Model labels swapped per pair, jointly over all of that pair's
    thresholds; statistic = difference of mean per-configuration FPR."""
    a, b = sorted(neg["model"].unique())
    n_strata = neg.groupby(["workload", "threshold"]).ngroups
    n_w = neg[neg["model"] == a].groupby(["workload", "threshold"]).size().groupby("workload").first()
    signed = neg["is_fp"] * np.where(neg["model"] == a, 1, -1) / neg["workload"].map(n_w)
    d = signed.groupby(neg["pair_id"]).sum().to_numpy() / n_strata
    nonzero = d[np.abs(d) > 0]
    obs = float(d.sum())
    if len(nonzero) > 22:
        raise RobustnessInputError(f"{len(nonzero)} discordant pairs: exact sign-flip enumeration too large")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(nonzero)))) if len(nonzero) else np.ones((1, 0))
    t_all = signs @ nonzero
    p = float((np.abs(t_all) >= abs(obs) - 1e-12).mean())
    return {"discordant_pairs": int(len(nonzero)), "n_sign_patterns": int(len(t_all)), "statistic": obs, "p_value": p}


# ---------------------------------------------------------- 2. logistic regression


def aggregate_cells(neg: pd.DataFrame) -> pd.DataFrame:
    return (
        neg.groupby(["model", "workload", "threshold"], sort=True)["is_fp"]
        .agg(n="size", events="sum")
        .reset_index()
    )


def design(formula: str, frame: pd.DataFrame) -> pd.DataFrame:
    return patsy.dmatrix(formula, frame, return_type="dataframe")


def term_columns(X: pd.DataFrame, term: str) -> List[int]:
    info = X.design_info
    sl = info.term_name_slices[term]
    return list(range(sl.start, sl.stop))


def _bern_ll(eta: np.ndarray, events: np.ndarray, n: np.ndarray) -> float:
    return float(np.sum(events * eta - n * np.logaddexp(0.0, eta)))


def firth_fit(
    X: np.ndarray,
    events: np.ndarray,
    n: np.ndarray,
    fixed: Optional[Dict[int, float]] = None,
    start: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Firth-penalised logistic fit on grouped binomial data.

    Penalised log-likelihood = sum_i [y_i eta_i - n_i log(1+e^eta_i)]
    + 0.5 log det(X' W X), W = diag(n p (1-p)). Coefficients listed in
    `fixed` are held at the given value while the penalty is still computed
    from the full X (the convention used for penalised likelihood-ratio tests
    and profile intervals).
    """
    fixed = dict(fixed or {})
    p_dim = X.shape[1]
    free = np.array([j for j in range(p_dim) if j not in fixed], dtype=int)
    beta = np.zeros(p_dim) if start is None else np.array(start, dtype=float)
    for j, v in fixed.items():
        beta[j] = v

    def pen_ll(b):
        eta = X @ b
        pr = special.expit(eta)
        info = X.T @ (X * (n * pr * (1 - pr))[:, None])
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return -np.inf
        return _bern_ll(eta, events, n) + 0.5 * logdet

    current = pen_ll(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        pr = special.expit(X @ beta)
        w = n * pr * (1 - pr)
        info = X.T @ (X * w[:, None])
        info_inv = np.linalg.inv(info)
        h = w * np.einsum("ij,jk,ik->i", X, info_inv, X)
        score = X.T @ (events - n * pr + h * (0.5 - pr))
        if len(free) == 0:
            converged = True
            break
        step = np.zeros(p_dim)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        scale = 1.0
        while True:
            candidate = beta + scale * step
            value = pen_ll(candidate)
            if value >= current - 1e-12 or scale < 1e-8:
                break
            scale /= 2
        beta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            converged = True
            break
    pr = special.expit(X @ beta)
    info = X.T @ (X * (n * pr * (1 - pr))[:, None])
    cov = np.linalg.inv(info)
    return {"beta": beta, "pen_ll": current, "cov": cov, "converged": converged, "iterations": it}


def firth_profile_ci(X, events, n, j, fit, level=0.95) -> Tuple[float, float]:
    crit = stats.chi2.ppf(level, 1)
    b_hat = fit["beta"][j]
    se = math.sqrt(fit["cov"][j, j])

    def f(b):
        try:
            restricted = firth_fit(X, events, n, fixed={j: b}, start=fit["beta"])
        except np.linalg.LinAlgError:
            return math.inf
        return 2 * (fit["pen_ll"] - restricted["pen_ll"]) - crit

    bounds = []
    for direction in (-1, 1):
        step = max(se, 0.1)
        far = b_hat + direction * step
        for _ in range(40):
            if f(far) > 0:
                break
            step *= 1.5
            far = b_hat + direction * step
        else:
            bounds.append(direction * math.inf)
            continue
        bounds.append(optimize.brentq(f, *sorted((b_hat, far)), xtol=1e-8))
    return bounds[0], bounds[1]


def separation_check(X: np.ndarray, events: np.ndarray, n: np.ndarray) -> Dict[str, Any]:
    """Linear-programming check for complete or quasi-complete separation.

    Looks for beta != 0 with x'beta >= 0 on all-event rows, <= 0 on no-event
    rows and = 0 on mixed rows. A strictly positive optimum means some linear
    combination separates, so the ML estimate does not exist.
    """
    pos = events == n
    zero = events == 0
    mixed = ~(pos | zero)
    A_ub = np.vstack([-X[pos], X[zero]]) if (pos.any() or zero.any()) else None
    b_ub = np.zeros(A_ub.shape[0]) if A_ub is not None else None
    A_eq = X[mixed] if mixed.any() else None
    b_eq = np.zeros(A_eq.shape[0]) if A_eq is not None else None
    c = -(X[pos].sum(axis=0) - X[zero].sum(axis=0))
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(-1, 1)] * X.shape[1], method="highs")
    value = float(-res.fun) if res.status == 0 else float("nan")
    return {
        "lp_status": res.message,
        "lp_optimum": value,
        "separation": bool(res.status == 0 and value > 1e-8),
        "all_event_rows": int(pos.sum()),
        "no_event_rows": int(zero.sum()),
    }


def ml_fit(X: pd.DataFrame, cells: pd.DataFrame) -> Dict[str, Any]:
    endog = np.column_stack([cells["events"], cells["n"] - cells["events"]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    return {
        "result": res,
        "converged": bool(res.converged),
        "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
    }


def _decision(p: float, alpha: float) -> str:
    return "reject" if p < alpha else "retain"


def run_logistic(neg: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    cells = aggregate_cells(neg)
    ev = cells["events"].to_numpy(float)
    n = cells["n"].to_numpy(float)
    out: Dict[str, Any] = {"coefficients": [], "tests": [], "fits": {}}
    designs = {f: design(f, cells) for f in (FORMULA_FULL, FORMULA_ADDITIVE)}
    ml = {f: ml_fit(X, cells) for f, X in designs.items()}
    firth = {f: firth_fit(X.to_numpy(), ev, n) for f, X in designs.items()}
    for f, X in designs.items():
        res = ml[f]["result"]
        ci = res.conf_int(alpha)
        sep = separation_check(X.to_numpy(), ev, n)
        fr = firth[f]
        out["fits"][f] = {
            "separation": sep,
            "ml": {
                "converged": ml[f]["converged"],
                "iterations": int(res.fit_history["iteration"]),
                "warnings": ml[f]["warnings"],
                "max_abs_coef": float(np.max(np.abs(res.params))),
                "max_se": float(np.max(res.bse)),
                "llf": float(res.llf),
            },
            "firth": {
                "converged": fr["converged"],
                "iterations": fr["iterations"],
                "penalised_llf": fr["pen_ll"],
            },
        }
        for j, name in enumerate(X.columns):
            out["coefficients"].append(
                _coef_row("logistic_ml", f, name, res.params.iloc[j], res.bse.iloc[j], ci.iloc[j, 0], ci.iloc[j, 1], "wald")
            )
            lo, hi = firth_profile_ci(X.to_numpy(), ev, n, j, fr, 1 - alpha)
            out["coefficients"].append(
                _coef_row("logistic_firth", f, name, fr["beta"][j], math.sqrt(fr["cov"][j, j]), lo, hi, "profile_penalised_likelihood")
            )
    for hyp, (term, f) in TERM_TESTS.items():
        X = designs[f]
        cols = term_columns(X, term)
        # ML: nested likelihood-ratio test.
        if f == FORMULA_FULL:
            llr = 2 * (ml[FORMULA_FULL]["result"].llf - ml[FORMULA_ADDITIVE]["result"].llf)
        else:
            Xr = X.drop(columns=X.columns[cols])
            llr = 2 * (ml[f]["result"].llf - ml_fit(Xr, cells)["result"].llf)
        p_ml = float(stats.chi2.sf(llr, len(cols)))
        out["tests"].append(_test_row("logistic_ml", hyp, term, "likelihood_ratio_chi2", llr, len(cols), p_ml, alpha))
        # Firth: penalised LR test, term coefficients fixed at 0, full-X penalty.
        restricted = firth_fit(X.to_numpy(), ev, n, fixed={j: 0.0 for j in cols}, start=firth[f]["beta"])
        plr = 2 * (firth[f]["pen_ll"] - restricted["pen_ll"])
        p_f = float(stats.chi2.sf(plr, len(cols)))
        out["tests"].append(_test_row("logistic_firth", hyp, term, "penalised_likelihood_ratio_chi2", plr, len(cols), p_f, alpha))
    return out


def _coef_row(method, formula, term, coef, se, lo, hi, ci_method) -> Dict[str, Any]:
    return {
        "method": method,
        "formula": formula,
        "term": term,
        "coef": float(coef),
        "se": float(se),
        "odds_ratio": float(np.exp(coef)),
        "or_ci_low": float(np.exp(lo)),
        "or_ci_high": float(np.exp(hi)),
        "ci_method": ci_method,
    }


def _test_row(method, hyp, term, stat_name, stat, df, p, alpha) -> Dict[str, Any]:
    return {
        "method": method,
        "hypothesis": hyp,
        "term": term,
        "statistic_name": stat_name,
        "statistic": float(stat),
        "df": int(df),
        "p_value": float(p),
        "alpha": alpha,
        "decision": _decision(p, alpha) if hyp.startswith("H0_") else "",
    }


# --------------------------------------------------------- 3. mixed-effects logistic


def _cluster_arrays(neg: pd.DataFrame, X: np.ndarray):
    codes, uniques = pd.factorize(neg["pair_id"], sort=True)
    sizes = np.bincount(codes)
    m = int(sizes.max())
    order = np.argsort(codes, kind="stable")
    pos = np.concatenate([np.arange(s) for s in sizes])
    Xc = np.zeros((len(uniques), m, X.shape[1]))
    Yc = np.zeros((len(uniques), m))
    Mc = np.zeros((len(uniques), m))
    Xc[codes[order], pos] = X[order]
    Yc[codes[order], pos] = neg["is_fp"].to_numpy()[order]
    Mc[codes[order], pos] = 1.0
    return Xc, Yc, Mc


def glmm_loglik(theta: np.ndarray, Xc, Yc, Mc, nodes, weights) -> float:
    """Marginal log-likelihood, random intercept sigma*z, z ~ N(0,1),
    adaptive Gauss-Hermite quadrature centred on each cluster's mode."""
    beta, sigma = theta[:-1], math.exp(theta[-1])
    eta = Xc @ beta
    z = np.zeros(eta.shape[0])
    for _ in range(100):
        pr = special.expit(eta + sigma * z[:, None])
        grad = sigma * ((Yc - pr) * Mc).sum(axis=1) - z
        hess = -(sigma ** 2) * (pr * (1 - pr) * Mc).sum(axis=1) - 1.0
        step = grad / hess
        z = z - step
        if np.max(np.abs(step)) < 1e-10:
            break
    pr = special.expit(eta + sigma * z[:, None])
    curv = (sigma ** 2) * (pr * (1 - pr) * Mc).sum(axis=1) + 1.0
    scale = np.sqrt(2.0 / curv)
    zk = z[:, None] + scale[:, None] * nodes[None, :]
    lin = eta[:, :, None] + sigma * zk[:, None, :]
    cond = ((Yc[:, :, None] * lin - np.logaddexp(0.0, lin)) * Mc[:, :, None]).sum(axis=1)
    g = cond - 0.5 * zk ** 2
    log_int = special.logsumexp(g + nodes[None, :] ** 2 + np.log(weights)[None, :], axis=1)
    return float(np.sum(log_int + np.log(scale) - 0.5 * math.log(2 * math.pi)))


def _numeric_hessian(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h
            ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def glmm_fit(neg: pd.DataFrame, X: pd.DataFrame, start: np.ndarray) -> Dict[str, Any]:
    Xc, Yc, Mc = _cluster_arrays(neg, X.to_numpy())
    nodes, weights = np.polynomial.hermite.hermgauss(GH_NODES)

    def nll(theta):
        return -glmm_loglik(theta, Xc, Yc, Mc, nodes, weights)

    theta0 = np.concatenate([start, [0.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = optimize.minimize(nll, theta0, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
    grad = optimize.approx_fprime(res.x, nll, 1e-6)
    H = _numeric_hessian(nll, res.x)
    eig = np.linalg.eigvalsh(H)
    pd_hessian = bool(eig.min() > 0)
    cov = np.linalg.inv(H) if pd_hessian else np.full(H.shape, np.nan)
    max_grad = float(np.max(np.abs(grad)))
    converged = bool(res.success and pd_hessian and max_grad < 1e-3)
    return {
        "theta": res.x,
        "llf": -float(res.fun),
        "cov": cov,
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "iterations": int(res.nit),
        "max_abs_gradient": max_grad,
        "hessian_min_eigenvalue": float(eig.min()),
        "hessian_max_eigenvalue": float(eig.max()),
        "hessian_positive_definite": pd_hessian,
        "sigma": float(math.exp(res.x[-1])),
        "converged": converged,
        "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
        "columns": list(X.columns),
    }


def run_glmm(neg: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    cells = aggregate_cells(neg)
    fits: Dict[str, Dict[str, Any]] = {}

    designs = {f: design(f, neg) for f in (FORMULA_FULL, FORMULA_ADDITIVE)}
    for f, X in designs.items():
        start = ml_fit(design(f, cells), cells)["result"].params.to_numpy()
        fits[f] = glmm_fit(neg, X, start)
    out: Dict[str, Any] = {"coefficients": [], "tests": [], "fits": {}, "reduced_fits": {}}
    for f, r in fits.items():
        out["fits"][f] = {k: v for k, v in r.items() if k not in ("theta", "cov", "columns")}
        se = np.sqrt(np.diag(r["cov"]))
        z = stats.norm.ppf(1 - alpha / 2)
        for j, name in enumerate(r["columns"]):
            b = r["theta"][j]
            out["coefficients"].append(_coef_row("glmm_aghq", f, name, b, se[j], b - z * se[j], b + z * se[j], "wald"))
        out["coefficients"].append(
            {
                "method": "glmm_aghq",
                "formula": f,
                "term": "sd(pair_id intercept)",
                "coef": r["sigma"],
                "se": float("nan"),
                "odds_ratio": float("nan"),
                "or_ci_low": float("nan"),
                "or_ci_high": float("nan"),
                "ci_method": "",
            }
        )
    for hyp, (term, f) in TERM_TESTS.items():
        X = designs[f]
        cols = term_columns(X, term)
        if f == FORMULA_FULL:
            full, reduced = fits[FORMULA_FULL], fits[FORMULA_ADDITIVE]
        else:
            Xr = X.drop(columns=X.columns[cols])
            start = np.delete(fits[f]["theta"][:-1], cols)
            reduced = glmm_fit(neg, Xr, start)
            full = fits[f]
            out["reduced_fits"][hyp] = {k: v for k, v in reduced.items() if k not in ("theta", "cov", "columns")}
        both = full["converged"] and reduced["converged"]
        llr = 2 * (full["llf"] - reduced["llf"])
        p = float(stats.chi2.sf(llr, len(cols)))
        row = _test_row("glmm_aghq", hyp, term, "likelihood_ratio_chi2", llr, len(cols), p, alpha)
        if not both:
            row["decision"] = "no_verdict_not_converged" if hyp.startswith("H0_") else ""
        out["tests"].append(row)
    return out


# ----------------------------------------------------------- 4. cluster bootstrap


def cluster_bootstrap(neg: pd.DataFrame, n_boot: int, seed: int, alpha: float) -> pd.DataFrame:
    """Pairs resampled with replacement within workload (workload sizes held
    fixed). Rates: pooled FPR per model, mean per-configuration FPR per model,
    pooled FPR per workload, plus the model difference and pairwise workload
    differences. Percentile intervals; the workload differences also carry a
    Bonferroni interval over the pairwise comparisons."""
    rng = np.random.default_rng(seed)
    a, b = sorted(neg["model"].unique())
    workloads = sorted(neg["workload"].unique())
    thresholds_per_pair = neg.groupby(["pair_id", "model"]).size()
    per_pair = neg.pivot_table(index=["workload", "pair_id"], columns="model", values="is_fp", aggfunc="sum").fillna(0)
    obs_per_model = int(thresholds_per_pair.iloc[0])
    if not (thresholds_per_pair == obs_per_model).all():
        raise RobustnessInputError("bootstrap needs every pair observed equally often under each model")
    n_strata = neg.groupby(["workload", "threshold"]).ngroups
    boots: Dict[str, np.ndarray] = {}
    point: Dict[str, float] = {}
    ev_a = {}
    ev_b = {}
    sizes = {}
    for w in workloads:
        block = per_pair.loc[w]
        sizes[w] = len(block)
        idx = rng.integers(0, len(block), size=(n_boot, len(block)))
        ea = block[a].to_numpy()
        eb = block[b].to_numpy()
        ev_a[w] = (ea[idx].sum(axis=1), ea.sum())
        ev_b[w] = (eb[idx].sum(axis=1), eb.sum())
    total = sum(sizes.values()) * obs_per_model
    for k in (0, 1):
        pick = (lambda t: t[0]) if k == 0 else (lambda t: t[1])
        store = boots if k == 0 else point
        store[f"pooled_fpr:{a}"] = sum(pick(ev_a[w]) for w in workloads) / total
        store[f"pooled_fpr:{b}"] = sum(pick(ev_b[w]) for w in workloads) / total
        store[f"mean_config_fpr:{a}"] = sum(pick(ev_a[w]) / sizes[w] for w in workloads) / n_strata
        store[f"mean_config_fpr:{b}"] = sum(pick(ev_b[w]) / sizes[w] for w in workloads) / n_strata
        store[f"diff_pooled_fpr:{a}-{b}"] = store[f"pooled_fpr:{a}"] - store[f"pooled_fpr:{b}"]
        store[f"diff_mean_config_fpr:{a}-{b}"] = store[f"mean_config_fpr:{a}"] - store[f"mean_config_fpr:{b}"]
        for w in workloads:
            store[f"pooled_fpr:{w}"] = (pick(ev_a[w]) + pick(ev_b[w])) / (2 * obs_per_model * sizes[w])
        for w1, w2 in itertools.combinations(workloads, 2):
            store[f"diff_pooled_fpr:{w1}-{w2}"] = store[f"pooled_fpr:{w1}"] - store[f"pooled_fpr:{w2}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            store[f"odds_ratio_pooled:{b}/{a}"] = _odds(store[f"pooled_fpr:{b}"]) / _odds(store[f"pooled_fpr:{a}"])
            for w in workloads[1:]:
                store[f"odds_ratio_pooled:{w}/{workloads[0]}"] = _odds(store[f"pooled_fpr:{w}"]) / _odds(store[f"pooled_fpr:{workloads[0]}"])
    n_pairwise = len(list(itertools.combinations(workloads, 2)))
    rows = []
    for key in boots:
        kind, name = key.split(":", 1)
        dist = boots[key]
        # Odds ratios can be infinite in a resample with no reference-level
        # events, so their percentiles are taken without interpolation.
        # Resamples with no events at either level give 0/0 and are dropped
        # from the percentile (counted in n_undefined).
        q_method = "inverted_cdf" if kind.startswith("odds_ratio") else "linear"
        defined = dist[~np.isnan(dist)]
        lo, hi = np.quantile(defined, [alpha / 2, 1 - alpha / 2], method=q_method)
        row = {
            "quantity": kind,
            "level": name,
            "estimate": float(point[key]),
            "boot_mean": float(dist.mean()) if np.all(np.isfinite(dist)) else float("nan"),
            "n_infinite": int(np.isinf(dist).sum()),
            "n_undefined": int(np.isnan(dist).sum()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_level": 1 - alpha,
            "bonferroni_ci_low": float("nan"),
            "bonferroni_ci_high": float("nan"),
            "bonferroni_ci_level": float("nan"),
        }
        if kind == "diff_pooled_fpr" and name.split("-")[0] in workloads:
            adj = alpha / n_pairwise
            blo, bhi = np.quantile(dist, [adj / 2, 1 - adj / 2])
            row.update(bonferroni_ci_low=float(blo), bonferroni_ci_high=float(bhi), bonferroni_ci_level=1 - adj)
        rows.append(row)
    return pd.DataFrame(rows)


def _odds(rate):
    return rate / (1 - rate)


# ------------------------------------------------ divergence diagnostics (model)


def model_discordance(neg: pd.DataFrame) -> pd.DataFrame:
    """Counts only (no pair ids): per workload and overall, how the two
    models' outcomes on the same pair compare, at (pair, threshold) level and
    at pair level (events summed over thresholds)."""
    a, b = sorted(neg["model"].unique())
    unit = neg.pivot_table(index=["workload", "pair_id", "threshold"], columns="model", values="is_fp", aggfunc="sum")
    pair = neg.pivot_table(index=["workload", "pair_id"], columns="model", values="is_fp", aggfunc="sum")
    rows = []
    for w in sorted(neg["workload"].unique()) + ["ALL"]:
        u = unit if w == "ALL" else unit.loc[[w]]
        pr = pair if w == "ALL" else pair.loc[[w]]
        rows.append(
            {
                "workload": w,
                "units_pair_threshold": int(len(u)),
                "units_both_fp": int(((u[a] == 1) & (u[b] == 1)).sum()),
                f"units_{a}_only_fp": int(((u[a] == 1) & (u[b] == 0)).sum()),
                f"units_{b}_only_fp": int(((u[a] == 0) & (u[b] == 1)).sum()),
                "units_neither_fp": int(((u[a] == 0) & (u[b] == 0)).sum()),
                "pairs": int(len(pr)),
                "pairs_with_any_fp": int(((pr[a] + pr[b]) > 0).sum()),
                f"pairs_more_fp_{a}": int((pr[a] > pr[b]).sum()),
                f"pairs_more_fp_{b}": int((pr[b] > pr[a]).sum()),
                "pairs_equal_nonzero_fp": int(((pr[a] == pr[b]) & (pr[a] > 0)).sum()),
                "phi_between_models_over_units": _phi(u[a].to_numpy(), u[b].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _phi(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def matched_unit_test(neg: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    """Exact conditional (McNemar-type) test and matched-pair odds ratio over
    (pair, threshold) units: OR = (units with only model B FP) / (units with
    only model A FP), exact interval from the Clopper-Pearson interval on the
    discordant proportion. Treats the units as independent, i.e. ignores the
    dependence between one pair's thresholds."""
    a, b = sorted(neg["model"].unique())
    u = neg.pivot_table(index=["pair_id", "threshold"], columns="model", values="is_fp", aggfunc="sum")
    only_a = int(((u[a] == 1) & (u[b] == 0)).sum())
    only_b = int(((u[a] == 0) & (u[b] == 1)).sum())
    m = only_a + only_b
    p = float(stats.binomtest(only_b, m, 0.5).pvalue) if m else 1.0
    if m:
        ci = stats.binomtest(only_b, m, 0.5).proportion_ci(1 - alpha, method="exact")
        lo = ci.low / (1 - ci.low) if ci.low < 1 else math.inf
        hi = ci.high / (1 - ci.high) if ci.high < 1 else math.inf
    else:
        lo, hi = 0.0, math.inf
    or_ = only_b / only_a if only_a else math.inf
    return {"only_a": only_a, "only_b": only_b, "odds_ratio": or_, "or_ci_low": lo, "or_ci_high": hi, "p_value": p, "model_a": a, "model_b": b}


def paired_config_tests(neg: pd.DataFrame) -> Dict[str, Any]:
    """The 15 per-configuration FPR differences (model A - model B) paired by
    (workload, threshold): paired t-test and Wilcoxon signed-rank test, on the
    same per-configuration rates the ANOVA uses."""
    a, b = sorted(neg["model"].unique())
    rates = neg.groupby(["workload", "threshold", "model"])["is_fp"].mean().unstack("model")
    d = (rates[a] - rates[b]).to_numpy()
    t = stats.ttest_1samp(d, 0.0)
    nonzero = d[d != 0]
    w = stats.wilcoxon(nonzero) if len(nonzero) else None
    return {
        "n_configuration_pairs": int(len(d)),
        "n_nonzero_differences": int(len(nonzero)),
        "n_positive_differences": int((d > 0).sum()),
        "n_negative_differences": int((d < 0).sum()),
        "mean_difference": float(d.mean()),
        "t_statistic": float(t.statistic),
        "t_p_value": float(t.pvalue),
        "wilcoxon_statistic": float(w.statistic) if w is not None else float("nan"),
        "wilcoxon_p_value": float(w.pvalue) if w is not None else float("nan"),
    }


#: Per method, how the analysis treats the three candidate drivers of a
#: divergence. Values are properties of the implemented computation.
METHOD_PROPERTIES = {
    "anova_frozen": ("normal", "replicate", "ignored", "per-configuration rate"),
    "paired_config_t": ("normal", "pairing block", "ignored", "per-configuration rate"),
    "paired_config_wilcoxon": ("none (rank)", "pairing block", "ignored", "per-configuration rate"),
    "permutation_exact": ("none (permutation)", "stratum", "ignored", "decision"),
    "permutation_exact_pooled": ("none (permutation)", "stratum", "ignored", "decision"),
    "permutation_monte_carlo": ("none (permutation)", "stratum", "ignored", "decision"),
    "logistic_ml": ("binomial", "fixed factor", "ignored", "decision"),
    "logistic_firth": ("binomial", "fixed factor", "ignored", "decision"),
    "glmm_aghq": ("binomial", "fixed factor", "random intercept", "decision"),
    "matched_unit_exact": ("binomial (conditional)", "matched within pair", "matched per (pair, threshold); thresholds of one pair treated as independent", "decision"),
    "permutation_cluster_signflip": ("none (permutation)", "summed within pair", "resampling unit = pair", "decision"),
    "cluster_bootstrap": ("none (resampling)", "summed within pair", "resampling unit = pair", "decision"),
}

METHOD_ROLES = {
    "anova_frozen": "reference",
    "permutation_exact": "primary",
    "permutation_exact_pooled": "cross_check",
    "permutation_monte_carlo": "cross_check",
    "logistic_ml": "method",
    "logistic_firth": "method",
    "glmm_aghq": "method",
    "cluster_bootstrap": "method",
    "permutation_cluster_signflip": "sensitivity",
    "matched_unit_exact": "diagnostic",
    "paired_config_t": "diagnostic",
    "paired_config_wilcoxon": "diagnostic",
}


# ---------------------------------------------------------------- verdicts + bundle


def read_frozen_anova(anova_csv: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(anova_csv)
    return {r["hypothesis"]: {"p_value": float(r["p_value"]), "decision": r["decision"]} for _, r in df.dropna(subset=["hypothesis"]).iterrows()}


def verdicts(tests: pd.DataFrame, frozen: Dict[str, Dict[str, Any]], boot: pd.DataFrame, models, workloads) -> pd.DataFrame:
    rows = []
    for method in tests["method"].unique():
        t = tests[tests["method"] == method].set_index("hypothesis")
        row = {"method": method}
        for hyp in ("H0_1", "H0_2"):
            if hyp in t.index:
                dec = t.loc[hyp, "decision"]
                row[f"{hyp}_p_value"] = float(t.loc[hyp, "p_value"])
                row[f"{hyp}_decision"] = dec
                row[f"{hyp}_verdict"] = _agree(dec, frozen[hyp]["decision"])
            else:
                row[f"{hyp}_p_value"] = float("nan")
                row[f"{hyp}_decision"] = "not_applicable"
                row[f"{hyp}_verdict"] = "NOT_APPLICABLE"
        rows.append(row)
    a, b = models
    diff = boot[(boot["quantity"] == "diff_mean_config_fpr") & (boot["level"] == f"{a}-{b}")].iloc[0]
    dec1 = "retain" if diff["ci_low"] <= 0 <= diff["ci_high"] else "reject"
    wl = boot[(boot["quantity"] == "diff_pooled_fpr") & boot["level"].str.split("-").str[0].isin(workloads)]
    dec2 = "reject" if ((wl["bonferroni_ci_low"] > 0) | (wl["bonferroni_ci_high"] < 0)).any() else "retain"
    rows.append(
        {
            "method": "cluster_bootstrap",
            "H0_1_p_value": float("nan"),
            "H0_1_decision": dec1,
            "H0_1_verdict": _agree(dec1, frozen["H0_1"]["decision"]),
            "H0_2_p_value": float("nan"),
            "H0_2_decision": dec2,
            "H0_2_verdict": _agree(dec2, frozen["H0_2"]["decision"]),
        }
    )
    out = pd.DataFrame(rows)
    return out


def annotate_methods(verdict: pd.DataFrame) -> pd.DataFrame:
    props = pd.DataFrame(
        [(m, METHOD_ROLES.get(m, ""), *METHOD_PROPERTIES.get(m, ("", "", "", ""))) for m in verdict["method"]],
        columns=["method", "role", "response_family", "threshold_treatment", "pair_structure", "unit_of_analysis"],
    )
    return verdict.merge(props, on="method", how="left")


def _agree(decision: str, frozen_decision: str) -> str:
    if decision not in ("reject", "retain"):
        return "NO_VERDICT"
    return "AGREE" if decision == frozen_decision else "DISAGREE"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_robustness(
    decisions_csv: Path,
    anova_csv: Path,
    out_dir: Path,
    display_paths: Dict[str, str],
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    n_mc: int = DEFAULT_N_MC,
) -> Dict[str, Any]:
    neg = load_negatives(decisions_csv)
    frozen = read_frozen_anova(anova_csv)
    models = sorted(neg["model"].unique())
    workloads = sorted(neg["workload"].unique())

    counts = event_counts(neg)

    perm_model = model_permutation_exact(neg)
    perm_work = workload_permutation_exact(neg)
    mc_model = model_permutation_mc(neg, n_mc, seed)
    mc_work = workload_permutation_mc(neg, n_mc, seed + 1)
    signflip = model_cluster_signflip_exact(neg)
    perm_tests = [
        _test_row("permutation_exact", "H0_1", "model", "diff_mean_config_fpr", perm_model["diff_mean_config_fpr"], 1, perm_model["p_mean_config"], alpha),
        _test_row("permutation_exact", "H0_2", "workload", "between_workload_ss_mean_config_fpr", perm_work["statistic"], len(workloads) - 1, perm_work["p_value"], alpha),
        _test_row("permutation_exact_pooled", "H0_1", "model", "diff_pooled_fpr", perm_model["diff_pooled_fpr"], 1, perm_model["p_pooled"], alpha),
        _test_row("permutation_monte_carlo", "H0_1", "model", "diff_mean_config_fpr", mc_model["statistic"], 1, mc_model["p_value"], alpha),
        _test_row("permutation_monte_carlo", "H0_2", "workload", "between_workload_ss_mean_config_fpr", mc_work["statistic"], len(workloads) - 1, mc_work["p_value"], alpha),
        _test_row("permutation_cluster_signflip", "H0_1", "model", "diff_mean_config_fpr", signflip["statistic"], 1, signflip["p_value"], alpha),
    ]

    matched = matched_unit_test(neg, alpha)
    paired = paired_config_tests(neg)
    discordance = model_discordance(neg)
    perm_tests += [
        _test_row("matched_unit_exact", "H0_1", "model", "discordant_units_model_b_only", matched["only_b"], 1, matched["p_value"], alpha),
        _test_row("paired_config_t", "H0_1", "model", "t", paired["t_statistic"], paired["n_configuration_pairs"] - 1, paired["t_p_value"], alpha),
        _test_row("paired_config_wilcoxon", "H0_1", "model", "wilcoxon_W", paired["wilcoxon_statistic"], paired["n_nonzero_differences"], paired["wilcoxon_p_value"], alpha),
    ]

    logit = run_logistic(neg, alpha)
    glmm = run_glmm(neg, alpha)
    boot = cluster_bootstrap(neg, n_boot, seed, alpha)

    tests = pd.DataFrame(perm_tests + logit["tests"] + glmm["tests"])
    matched_row = _coef_row("matched_unit_exact", "is_fp matched on (pair_id, threshold)", "C(model)[T.%s]" % matched["model_b"], math.log(matched["odds_ratio"]) if 0 < matched["odds_ratio"] < math.inf else float("nan"), float("nan"), float("nan"), float("nan"), "exact_conditional")
    matched_row.update(odds_ratio=matched["odds_ratio"], or_ci_low=matched["or_ci_low"], or_ci_high=matched["or_ci_high"])
    coefs = pd.DataFrame(logit["coefficients"] + glmm["coefficients"] + [matched_row])
    verdict = verdicts(tests, frozen, boot, models, workloads)
    frozen_row = {
        "method": "anova_frozen",
        "H0_1_p_value": frozen["H0_1"]["p_value"],
        "H0_1_decision": frozen["H0_1"]["decision"],
        "H0_1_verdict": "REFERENCE",
        "H0_2_p_value": frozen["H0_2"]["p_value"],
        "H0_2_decision": frozen["H0_2"]["decision"],
        "H0_2_verdict": "REFERENCE",
    }
    verdict = annotate_methods(pd.concat([pd.DataFrame([frozen_row]), verdict], ignore_index=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%.10g"
    counts.to_csv(out_dir / "event_counts.csv", index=False, float_format=fmt)
    tests.to_csv(out_dir / "tests.csv", index=False, float_format=fmt)
    coefs.to_csv(out_dir / "coefficients.csv", index=False, float_format=fmt)
    boot.to_csv(out_dir / "bootstrap.csv", index=False, float_format=fmt)
    verdict.to_csv(out_dir / "verdicts.csv", index=False, float_format=fmt)
    discordance.to_csv(out_dir / "model_discordance.csv", index=False, float_format=fmt)

    meta = {
        "inputs": {
            "decisions_csv": display_paths["decisions_csv"],
            "decisions_csv_sha256": sha256_file(decisions_csv),
            "anova_csv": display_paths["anova_csv"],
            "anova_csv_sha256": sha256_file(anova_csv),
            "n_negative_observations": int(len(neg)),
            "n_events": int(neg["is_fp"].sum()),
            "n_pairs": int(neg["pair_id"].nunique()),
        },
        "response": "is_fp = (outcome == 'FP') over rows with label == 0",
        "alpha": alpha,
        "reference_levels": {"model": models[0], "workload": workloads[0], "threshold": sorted(neg["threshold"].unique())[0]},
        "verdict_rule": {
            "H0_1": "AGREE when the method retains H0_1 (frozen ANOVA: retain)",
            "H0_2": "AGREE when the method rejects H0_2 (frozen ANOVA: reject)",
            "cluster_bootstrap_H0_1": "reject when the percentile CI of diff_mean_config_fpr excludes 0",
            "cluster_bootstrap_H0_2": "reject when any Bonferroni-adjusted pairwise workload diff_pooled_fpr CI excludes 0",
            "roles": "primary = the ticket's method 1; method = the ticket's methods 2-4; cross_check / sensitivity / diagnostic are not substituted for the primary",
            "glmm": "no verdict unless both the full and reduced fits converged (optimizer success, max |gradient| < 1e-3, positive-definite Hessian)",
        },
        "permutation": {
            "model_exact": perm_model,
            "workload_exact": perm_work,
            "model_monte_carlo": mc_model,
            "workload_monte_carlo": mc_work,
            "model_cluster_signflip_exact": signflip,
            "matched_unit_exact": matched,
            "paired_config": paired,
            "model_strata": "workload x threshold",
            "workload_strata": "model x threshold",
        },
        "logistic": {
            "formula_full": FORMULA_FULL,
            "formula_additive": FORMULA_ADDITIVE,
            "term_tests": {k: {"term": v[0], "tested_in": v[1]} for k, v in TERM_TESTS.items()},
            "firth_penalty": "Jeffreys prior, 0.5 * log det(X'WX); tests are penalised LR with term coefficients fixed at 0 and the penalty from the full design",
            "fits": logit["fits"],
        },
        "glmm": {
            "random_effect": "(1 | pair_id)",
            "estimation": f"maximum likelihood, adaptive Gauss-Hermite quadrature, {GH_NODES} nodes, BFGS",
            "fits": glmm["fits"],
            "reduced_fits": glmm["reduced_fits"],
        },
        "bootstrap": {"n_boot": n_boot, "seed": seed, "resampling_unit": "pair_id", "strata": "workload", "interval": "percentile (linear interpolation; inverted CDF for odds ratios)"},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "patsy": patsy.__version__,
        },
    }
    with open(out_dir / "robustness_meta.json", "w") as fh:
        json.dump(_jsonable(meta), fh, indent=2, sort_keys=False)
        fh.write("\n")
    return {"tests": tests, "coefficients": coefs, "bootstrap": boot, "verdicts": verdict, "event_counts": counts, "discordance": discordance, "meta": meta}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj
