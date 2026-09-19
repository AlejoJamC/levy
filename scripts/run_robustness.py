#!/usr/bin/env python
"""
Robustness re-analysis of H0_1/H0_2/H0_3 on the per-decision binary outcome
(LEV-22). Reads a released `decisions.csv` and the frozen `anova.csv`
(read-only), writes event counts, permutation / logistic / mixed-effects test
results, coefficients, cluster-bootstrap intervals, a per-method verdict
table and a metadata sidecar to --out-dir.

Example:
    python scripts/run_robustness.py \\
        --decisions release/d3-results/decisions.csv \\
        --anova release/d3-results/analysis/anova.csv \\
        --out-dir release/d3-results/robustness
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from levy.analysis.robustness import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_N_BOOT,
    DEFAULT_N_MC,
    DEFAULT_SEED,
    RobustnessInputError,
    run_robustness,
)


def _display(path: Path) -> str:
    """Repository-relative path for the metadata sidecar; never absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", type=Path, default=Path("release/d3-results/decisions.csv"))
    parser.add_argument("--anova", type=Path, default=Path("release/d3-results/analysis/anova.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--n-mc", type=int, default=DEFAULT_N_MC, help="Monte Carlo shuffles for the permutation cross-check")
    return parser


def main(argv=None, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = args.out_dir.resolve()
    for protected in (args.decisions.resolve().parent, args.anova.resolve().parent):
        if out_dir == protected:
            output_fn(f"error: --out-dir must not be an input directory ({_display(protected)})")
            return 2
    try:
        result = run_robustness(
            decisions_csv=args.decisions,
            anova_csv=args.anova,
            out_dir=args.out_dir,
            display_paths={"decisions_csv": _display(args.decisions), "anova_csv": _display(args.anova)},
            alpha=args.alpha,
            seed=args.seed,
            n_boot=args.n_boot,
            n_mc=args.n_mc,
        )
    except RobustnessInputError as exc:
        output_fn(f"error: {exc}")
        return 1
    output_fn(result["verdicts"].to_string(index=False))
    output_fn(f"written to {_display(args.out_dir)}{os.sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
