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
- **Compliance check performed before copying:** every file was checked for
  third-party corpus text, secrets/API keys, emails, IP addresses, and PII.
  None found. No file here contains original query text from any source
  corpus (Quora QQP, SODD, Twitter PIT-2015) — `decisions.csv` carries only
  the internal `pair_id` (e.g. `faq-0001`), never the source corpus id or
  text; `responses.jsonl` carries only a hash of each prompt plus the model's
  own generated answer, never the prompt itself.
- **Licence:** Apache 2.0, same as the rest of this repository.

## Contents

| File | Description |
|---|---|
| `results.csv` | 30 rows, one per configuration: TP/FP/TN/FN, precision, recall, F0.5, false positive rate, hit rate |
| `decisions.csv` | 9,000 rows, per-pair cache decisions underlying `results.csv` |
| `run_meta.json` | Dataset path, providers, resolved model checkpoints, grid definition |
| `replication.json` | ±5% replication check against the frozen Success Criterion 3 |
| `analysis/anova.csv`, `analysis/tukey.csv`, `analysis/tukey_status.csv` | Two-way ANOVA and Tukey HSD results for H0(1-3) |
| `analysis/curves_precision.csv`, `analysis/curves_hit_rate.csv` | Threshold-vs-metric tables per (model, workload) |
| `analysis/kappa.json` | Cohen's kappa (annotation agreement) |
| `analysis/analysis_meta.json` | Analysis run metadata |
| `analysis/figures/*.png`, `analysis/figures/*.pdf` | Threshold-vs-precision and threshold-vs-hit-rate figures |
| `latency.csv` | Per-configuration lookup-overhead percentiles (embedding, index search, exact-cache lookup, total) |
| `latency_meta.json` | Host spec, library versions, protocol, latency-vs-savings figures |
| `llm_calls.json` | Observed cost/token/latency totals for the 600 real provider calls |
| `responses.jsonl` | 600 records: `sha256(prompt)`, model, token counts, latency, and the model's generated response text — never the prompt |

## Integrity

`checksums.sha256` in this folder is a SHA-256 manifest of every other file
listed above, generated at the same time this folder was populated. It
verifies this folder's own contents — it does not reference `results/`.
