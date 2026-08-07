"""
Latency measurement for the cache lookup path (LEV-14 / D1 economic viability).

The frozen Project Proposal defines economic viability by hit rate **and**
"latency measurements (cache lookup overhead vs LLM call savings)". This
package supplies the second half: a decomposed measurement of what a lookup
costs, and the artefact contract that reports it.

Two measurements live here, and they have different reproducibility:

- **Lookup overhead** (`TimingCollector`, `benchmark.py`) is computed entirely
  offline and replicates on other hardware, modulo the host specification
  recorded in the metadata sidecar.
- **Provider latency** (`corpus.py`, driven by `scripts/populate_responses.py`)
  is provider-, region- and time-dependent and does **not** replicate. Every
  figure derived from it carries the resolved model identifier that produced it.

No module in this package imports a network library at module scope; the
offline suite imports all of it. The billed path lives in a script and is
guarded by an AST test, exactly as `scripts/fetch_corpora.py` is.
"""

from levy.latency.timing import TimingCollector, SEGMENT_EMBED, SEGMENT_EXACT_LOOKUP, SEGMENT_INDEX_SEARCH, SEGMENT_TOTAL_LOOKUP

__all__ = [
    "TimingCollector",
    "SEGMENT_EMBED",
    "SEGMENT_EXACT_LOOKUP",
    "SEGMENT_INDEX_SEARCH",
    "SEGMENT_TOTAL_LOOKUP",
]
