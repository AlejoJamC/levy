# Provenance

This folder holds the final, validated benchmark results for Levy — the D3
evaluation grid and the latency measurement. It is git-tracked (not
gitignored) so it ships with the repository. There is exactly one version:
when the results are regenerated, these files are overwritten in place. This
file is never renamed, dated, or suffixed, and neither is anything else here.

- **Producing commit:** `fc1b5c2e8b092f774b7426601ae8951ca6d7af4b`
- **Source data:** `results/run-003/` (D3 grid: 30-configuration sweep,
  statistical analysis) and `results/latency-faq/` (latency measurement),
  both gitignored working-tree output, copied here after review.
  `d3-results/replication/cross-environment.*` is not copied from anywhere —
  it is an independently generated re-run of the same grid inside the D7
  container (LEV-16), kept alongside the original as replication evidence.
- **Compliance check performed before copying:** every file was checked for
  third-party corpus text, secrets/API keys, emails, IP addresses, and PII.
  None found. No file here contains original query text from any source
  corpus (Quora QQP, SODD, Twitter PIT-2015) — `decisions.csv` carries only
  the internal `pair_id` (e.g. `faq-0001`), never the source corpus id or
  text; `responses.jsonl` carries only a hash of each prompt plus the model's
  own generated answer, never the prompt itself.
- **Licence:** Apache 2.0, same as the rest of this repository.

## Contents

Organised by deliverable, per `docs/Specification_and_Design_Report.md:282-296`
(D3 = evaluation results/analysis; the latency pilot is not itself one of the
named D1-D7 deliverables, so it is kept in its own folder rather than under a
D-number it doesn't have).

| File | Description |
|---|---|
| `d3-results/results.csv` | 30 rows, one per configuration: TP/FP/TN/FN, precision, recall, F0.5, false positive rate, hit rate |
| `d3-results/decisions.csv` | 9,000 rows, per-pair cache decisions underlying `results.csv` |
| `d3-results/run_meta.json` | Dataset path, providers, resolved model checkpoints, grid definition |
| `d3-results/replication/determinism.replication.json` | Same-host re-run: every diff expected to be exactly 0.0 (byte-deterministic harness) |
| `d3-results/replication/cross-environment.replication.json` | Independent re-run inside the D7 container (Linux, real `sentence-transformers` embeddings) against the frozen ±5% Success Criterion 3; measured max abs deviation 0.0 |
| `d3-results/replication/cross-environment.results.csv` | The container run's own `results.csv`, generated independently, not copied — byte-identical to `d3-results/results.csv` |
| `d3-results/analysis/anova.csv`, `.../tukey.csv`, `.../tukey_status.csv` | Two-way ANOVA and Tukey HSD results for H0(1-3) |
| `d3-results/analysis/curves_precision.csv`, `.../curves_hit_rate.csv` | Threshold-vs-metric tables per (model, workload) |
| `d3-results/analysis/kappa.json` | Cohen's kappa (annotation agreement) |
| `d3-results/analysis/analysis_meta.json` | Analysis run metadata |
| `d3-results/analysis/figures/*.png`, `.../figures/*.pdf` | Threshold-vs-precision and threshold-vs-hit-rate figures |
| `latency/latency.csv` | Per-configuration lookup-overhead percentiles (embedding, index search, exact-cache lookup, total) |
| `latency/latency_meta.json` | Host spec, library versions, protocol, latency-vs-savings figures |
| `latency/llm_calls.json` | Observed cost/token/latency totals for the 600 real provider calls |
| `latency/responses.jsonl` | 600 records: `sha256(prompt)`, model, token counts, latency, and the model's generated response text — never the prompt |

## Integrity

`checksums.sha256` in this folder is a SHA-256 manifest of every other file
listed above, generated at the same time this folder was populated. It
verifies this folder's own contents — it does not reference `results/`.
