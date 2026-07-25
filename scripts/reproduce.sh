#!/usr/bin/env bash
#
# Levy — full evaluation pipeline, in one command.
#
# This script is the SINGLE definition of the evaluation pipeline: the
# reproduction guide (docs/REPRODUCTION.md) shows these same commands and the
# container image runs this script as its CMD. If a script's flags change, this
# file breaks loudly rather than leaving the documentation quietly stale.
#
# Stages:
#   1. scripts/run_experiments.py    — the frozen grid sweep (harness outputs)
#   2. scripts/run_analysis.py       — the statistical analysis bundle
#   3. scripts/check_replication.py  — the +/-5% replication criterion
#
# Defaults are offline: the committed fixture dataset and mock embeddings, so no
# API key, no model download, and no network access are required.
#
# Usage:
#   scripts/reproduce.sh [DATASET] [OUT_DIR]
#
# Environment overrides (flags take precedence over these only via the
# positional arguments above; everything else is an env var):
#   LEVY_DATASET             dataset file, .csv or .json   (default data/ground_truth.csv)
#   LEVY_OUT_DIR             output directory               (default results/reproduce)
#   LEVY_EMBEDDING_PROVIDER  mock | sentence-transformers | ollama (default mock)
#   LEVY_MODELS              comma-separated model subset   (default: full frozen grid)
#   LEVY_WORKLOADS           comma-separated workload subset (default: faq,code,chat)
#   LEVY_THRESHOLDS          comma-separated threshold subset (default: 0.70..0.90)
#
# Non-mock embedding providers download model weights on first use and require
# network access; the Anthropic backend is never used by this pipeline (the
# harness replays through the mock LLM by design).

set -euo pipefail

DATASET="${1:-${LEVY_DATASET:-data/ground_truth.csv}}"
OUT_DIR="${2:-${LEVY_OUT_DIR:-results/reproduce}}"
EMBEDDING_PROVIDER="${LEVY_EMBEDDING_PROVIDER:-mock}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ANALYSIS_DIR="${OUT_DIR}/analysis"

# Optional grid subsets — omitted entirely when unset, so the frozen
# 2 models x 3 workloads x 5 thresholds = 30 configurations run by default.
GRID_ARGS=()
[ -n "${LEVY_MODELS:-}" ] && GRID_ARGS+=(--models "$LEVY_MODELS")
[ -n "${LEVY_WORKLOADS:-}" ] && GRID_ARGS+=(--workloads "$LEVY_WORKLOADS")
[ -n "${LEVY_THRESHOLDS:-}" ] && GRID_ARGS+=(--thresholds "$LEVY_THRESHOLDS")

echo "=============================================================="
echo "Levy evaluation pipeline"
echo "  dataset            : ${DATASET}"
echo "  output directory   : ${OUT_DIR}"
echo "  analysis bundle    : ${ANALYSIS_DIR}"
echo "  embedding provider : ${EMBEDDING_PROVIDER}"
if [ ${#GRID_ARGS[@]} -gt 0 ]; then
  echo "  grid subset        : ${GRID_ARGS[*]}"
else
  echo "  grid               : full frozen grid (30 configurations)"
fi
echo "=============================================================="

echo
echo "[1/3] Experiment sweep -> ${OUT_DIR}"
python scripts/run_experiments.py \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR" \
  --embedding-provider "$EMBEDDING_PROVIDER" \
  ${GRID_ARGS[@]+"${GRID_ARGS[@]}"}

echo
echo "[2/3] Statistical analysis -> ${ANALYSIS_DIR}"
python scripts/run_analysis.py \
  --results-dir "$OUT_DIR" \
  --out-dir "$ANALYSIS_DIR" \
  --dataset "$DATASET"

echo
echo "[3/3] Replication check against ${OUT_DIR}/results.csv"
python scripts/check_replication.py \
  --reference "${OUT_DIR}/results.csv" \
  --dataset "$DATASET"

echo
echo "=============================================================="
echo "Pipeline complete."
echo "  harness outputs : ${OUT_DIR}/results.csv, ${OUT_DIR}/decisions.csv, ${OUT_DIR}/run_meta.json"
echo "  analysis bundle : ${ANALYSIS_DIR}/ (tables, figures/, analysis_meta.json)"
echo "=============================================================="
