#!/usr/bin/env python
"""
Streamlit results dashboard (LEV-10 / D6): threshold-performance curves, a
hypothesis/kappa summary, and a live query demo -- all read from an analysis
bundle produced by `scripts/run_analysis.py`. Never recomputes a statistic;
this is a viewer over `levy/dashboard/`, the testable, framework-independent
core.

Run (the `--` separator is a Streamlit requirement so args reach this script):

    streamlit run scripts/dashboard.py -- --bundle results/reproduce/analysis

D6 is the frozen plan's lowest-priority, desirable-only deliverable (see
`openspec/changes/add-results-dashboard/proposal.md`): it is not part of
`scripts/reproduce.sh` and not required for any essential deliverable.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from levy.dashboard.bundle import BundleContractError, BundleNotFoundError, load_bundle
from levy.dashboard.curves import available_model_workload_pairs, select_curve
from levy.dashboard.query import build_query_index, evaluate_query
from levy.dataset.io import load_dataset
from levy.embedding_manager import KNOWN_MODEL_NAMES, EmbeddingManager

HIT_RATE_VIABILITY = 0.30
DEFAULT_DATASET = "data/ground_truth.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="Analysis bundle directory (output of scripts/run_analysis.py)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset file for the query demo (default: the bundle's recorded dataset, else data/ground_truth.csv)",
    )
    return parser


@st.cache_resource
def _cached_query_index(dataset_path: str, provider: str, model_name: str):
    """Build once per (dataset, provider, model); a threshold change never invalidates this."""
    pairs = load_dataset(Path(dataset_path))
    manager = EmbeddingManager(model_name=model_name, provider=provider)
    return build_query_index(manager, model_name, pairs)


def _render_provenance(bundle) -> None:
    meta = bundle.meta
    run_meta = meta.get("inputs", {}).get("harness_run_meta", {})
    fixture_only = bundle.kappa.get("provenance", {}).get("fixture_only")
    lines = [
        f"**Bundle:** `{bundle.bundle_dir}`",
        f"**LLM provider:** `{run_meta.get('llm_provider', 'unknown')}` &nbsp;·&nbsp; "
        f"**Embedding provider:** `{run_meta.get('embedding_provider', 'unknown')}` &nbsp;·&nbsp; "
        f"**Configurations:** {meta.get('inputs', {}).get('n_configurations', '?')}",
    ]
    if fixture_only:
        lines.append(
            "**FIXTURE ONLY** -- this bundle is derived from the synthetic placeholder "
            "dataset, not the dissertation's real 900-pair results."
        )
    st.info("  \n".join(lines))


def _render_curve_panel(bundle) -> None:
    st.header("Threshold-performance explorer")
    pairs = available_model_workload_pairs(bundle.hit_rate)
    if not pairs:
        st.warning("No (model, workload) pairs found in this bundle.")
        return

    labels = [f"{model} / {workload}" for model, workload in pairs]
    choice = st.selectbox("Model / workload", labels)
    model, workload = pairs[labels.index(choice)]

    hit_points = select_curve(bundle.hit_rate, model, workload)
    precision_points = select_curve(bundle.precision, model, workload)

    fig, (ax_hit, ax_prec) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, points, title, viability in (
        (ax_hit, hit_points, "Hit rate", HIT_RATE_VIABILITY),
        (ax_prec, precision_points, "Precision", None),
    ):
        measured = [p for p in points if not p.zero_div]
        degenerate = [p for p in points if p.zero_div]
        if measured:
            ax.plot(
                [p.threshold for p in measured],
                [p.value for p in measured],
                marker="o",
                linewidth=1.6,
                label="measured",
            )
        if degenerate:
            ax.scatter(
                [p.threshold for p in degenerate],
                [p.value for p in degenerate],
                marker="x",
                color="red",
                s=70,
                label="degenerate (zero-division)",
                zorder=5,
            )
        if viability is not None:
            ax.axhline(viability, color="black", linestyle="--", linewidth=1.0, label=f"viability ({viability:.0%})")
        ax.set_xlabel("Similarity threshold (1/(1+L2) scale)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} -- {model} / {workload}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Points marked with a red X carry the harness's zero-division flag: the value shown "
        "(0.0) is a reporting convention, not a measured rate."
    )


def _render_hypothesis_panel(bundle) -> None:
    st.header("Hypothesis and agreement summary")

    st.subheader("Two-way ANOVA on false positive rate")
    display_cols = ["hypothesis", "effect", "df", "sum_sq", "F", "p_value", "decision"]
    anova_display = bundle.anova.loc[bundle.anova["hypothesis"] != "", display_cols]
    st.dataframe(anova_display, hide_index=True)

    st.subheader("Tukey HSD status")
    st.dataframe(bundle.tukey_status, hide_index=True)
    if not bundle.tukey.empty:
        st.subheader("Tukey HSD pairwise comparisons")
        st.dataframe(bundle.tukey, hide_index=True)

    st.subheader("Cohen's kappa (annotation validity)")
    kappa = bundle.kappa
    if kappa.get("status") == "computed":
        overall = kappa["overall"]
        note = kappa.get("provenance", {}).get("note", "")
        st.write(f"kappa = **{overall['kappa']}** (threshold {kappa['threshold']}) -- {note}")
    else:
        st.write(f"kappa unavailable: {kappa.get('reason', 'unknown')}")


def _render_query_panel(bundle, dataset_path: str) -> None:
    st.header("Live query demo")
    st.caption(
        "Embeds your text with the selected model, searches an in-process index built from the "
        "dataset's queries, and reports the nearest match under the production 1/(1+L2) semantics."
    )

    provider = st.radio(
        "Embedding provider",
        options=["mock", "sentence-transformers"],
        horizontal=True,
        help=(
            "mock is fully offline and deterministic; sentence-transformers downloads model "
            "weights on first use and needs network access."
        ),
    )
    model_name = st.selectbox("Embedding model", KNOWN_MODEL_NAMES)
    threshold = st.slider("Similarity threshold (1/(1+L2) scale)", 0.0, 1.0, 0.85, 0.01)
    text = st.text_input("Query text")

    if not text:
        return

    with st.spinner("Building index (cached per dataset + provider + model)..."):
        cache = _cached_query_index(dataset_path, provider, model_name)
    decision = evaluate_query(cache, text, threshold)

    if decision.similarity is None:
        st.warning("Index is empty -- no queries to compare against.")
        return

    if decision.hit:
        st.success(f"HIT -- similarity {decision.similarity:.4f} >= threshold {threshold:.2f}")
    else:
        st.error(f"MISS -- nearest similarity {decision.similarity:.4f} < threshold {threshold:.2f}")
    st.write(f"Nearest match: {decision.matched_text!r}")


def main() -> None:
    args = build_arg_parser().parse_args()
    st.set_page_config(page_title="Levy results dashboard", layout="wide")
    st.title("Levy -- results dashboard")
    st.caption("D6 (desirable, LEV-10): a viewer over an analysis bundle. Never recomputes a statistic.")

    try:
        bundle = load_bundle(args.bundle)
    except (BundleNotFoundError, BundleContractError) as exc:
        st.error(str(exc))
        st.stop()

    _render_provenance(bundle)
    _render_curve_panel(bundle)
    _render_hypothesis_panel(bundle)

    dataset_path = args.dataset or bundle.meta.get("inputs", {}).get("dataset_path_for_kappa") or DEFAULT_DATASET
    if not Path(dataset_path).is_file():
        st.warning(f"Query demo needs a dataset file; not found: {dataset_path}. Pass --dataset explicitly.")
    else:
        _render_query_panel(bundle, dataset_path)


main()
