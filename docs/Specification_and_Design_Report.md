# Specification and Design Report

## A.	The Specification

### Project Context 

Large Language Model (LLM) APIs are increasingly central to production applications, but their operational costs and latency remain significant barriers. Semantic caching, storing and reusing responses for semantically similar queries, promises substantial reductions in both dimensions. Early open‑source tools (GPTCache, Upstash Semantic Cache) have revealed persistent quality problems: false positives (incorrect cache hits) are described as “inevitable and unacceptable in production” (Bang, 2023), yet no systematic guidance exists on embedding model selection or workload‑specific threshold tuning.

The research context is therefore a gap between industrial deployment (semantic caches handling hundreds of millions of daily queries) and academic evidence. Practitioners currently rely on trial‑and‑error; the literature lacks comparative evaluations of embedding strategies under controlled conditions. Levy addresses this gap through rigorous comparative evaluation of general‑purpose versus domain‑optimized embedding models across FAQ, code generation, and conversational chat workloads, employing design science research methodology to build a working prototype and applied systems evaluation to measure performance through controlled experiments.

### The Problem Statement

Semantic caching systems suffer from false positives that degrade output quality, and current tools provide no empirical guidance on embedding model selection or similarity threshold configuration for different workload types. This forces practitioners to either accept suboptimal cache quality or invest expensive experimentation that produces site-specific findings rather than transferable guidance.

The problem connects directly to the context: industrial deployment is widespread, the quality issue is publicly acknowledged by the most-cited tools in the space, yet no comparative study exists in the literature. The absence of evidence is itself the problem.

### The Dissertation Question(s)

Primary Research Question:
Does embedding model selection meaningfully impact false positive rates in semantic caching across production LLM workloads?

### Sub‑questions:

1.	What are empirical semantic duplication rates in FAQ, code, and conversational workloads?
2.	Does ModernBERT (optimised for longer sequences and modern pre-training) outperform all-MiniLM (widely deployed general-purpose baseline) on cache precision metrics?
3.	What similarity threshold per workload minimises false positives while preserving economic viability (hit rate >30%)?

### Ethical Considerations

Ethics approval is not required for this project.

•	No human participants are directly recruited.

•	All datasets are publicly available human‑annotated corpora (e.g., Quora Question Pairs, Stack Overflow duplicate questions, ConvAI2) with existing human labels.

•	The author performs only blind re‑annotation of already public data; no new human subjects are involved.

•	The LLM backend (Anthropic API) processes only synthetic and public queries; no personal or sensitive data is transmitted.

•	All code and data will be released under Apache 2.0 licence with anonymised content.

### The Anticipated Outcomes

The research will produce three scholarly contributions:

1.	First systematic duplication rate measurement across three workload types using a dual‑annotation methodology (900 query pairs sampled from public corpora, author’s blind re‑annotation compared against original human labels, Cohen’s kappa >0.7).
2.	First rigorous false positive comparison of two embedding model architectures for semantic caching, using precision, recall, F0.5, and FP rate.
3.	Reproducible benchmarking methodology with open‑source code, dataset, and analysis scripts.

IT Artefact – Levy Semantic Caching Engine

A working Python prototype consisting of:

•	FastAPI request router (intercepts LLM queries, returns cached responses or forwards to LLM).

•	Semantic cache module (ModernBERT/all‑MiniLM embeddings + Faiss HNSW vector index).

•	LLM backend connector (Anthropic API integration).

The artefact will be evaluated via offline replay experiments using the annotated dataset, comparing 30 configurations (2 models × 3 workloads × 5 thresholds). Success is defined as measurable precision differences between models while maintaining hit rates >30%.

### Major Modifications

No major modifications from the original proposal (submitted Week 8). The timeline and scope remain unchanged. The annotation methodology has been refined (using existing human‑labelled corpora + blind re‑annotation by author) but this does not alter the project’s aims, objectives, or deliverables.

### Literature Survey:

1.	Bang, F. (2023) GPTCache: An open‑source semantic cache for LLM applications. In NLP‑OSS 2023, pp. 212‑218.
      Annotation: This paper introduces the first open‑source semantic cache for LLMs. It explicitly reports cache hit rates below 90% and “inevitable false positives unacceptable in production”, but offers no domain adaptation or threshold guidance. Levy uses this acknowledged limitation as the primary motivation for systematic embedding model comparison.

2.	Gill, W. et al. (2025) MeanCache: User‑centric semantic caching for LLM web services. IPDPS, pp. 1298‑1310.
      Annotation: MeanCache correctly identifies precision and recall as the appropriate metrics for semantic caching and proposes F‑score with β=0.5 to prioritise precision. However, it is limited to a single production system. Levy adopts their evaluation framework (precision, recall, F0.5) and extends it across multiple workloads and two embedding models.

3.	Wang et al. (2025) Category‑aware semantic caching for heterogeneous workloads. arXiv:2510.26835.
      Annotation: This paper validates that different workloads (code vs. chat) require different similarity thresholds (0.9 vs. 0.75) based on one production deployment. It lacks a reproducible methodology. Levy builds on this by systematically testing five thresholds (0.70–0.90) across three workloads with controlled experiments.

4.	Warner, B. et al. (2024) ModernBERT: A modern approach to encoder‑only transformers. arXiv:2412.13663.
      Annotation: ModernBERT is a 149M parameter encoder with 8K context length, achieving state‑of‑the‑art results among compact models. Levy uses ModernBERT as the primary embedding model under evaluation, comparing its longer‑context architecture against all‑MiniLM to determine whether architectural differences translate to measurable cache quality improvements.

5.	Zheng, L. et al. (2024) LMSYS‑Chat‑1M: A large‑scale real‑world LLM conversation dataset. ICLR.
      Annotation: The largest public LLM conversation dataset (1M conversations). It explicitly states that the dataset “contains repeated data” but leaves “quantification for future work”. Levy directly fills this gap by measuring empirical duplication rates across workload types (Objective O1).

6.	Hevner, A.R. et al. (2004) Design science in information systems research. MIS Quarterly, 28(1), pp. 75‑105.
      Annotation: This paper defines the Design Science Research (DSR) methodology. Levy follows DSR by constructing a purposeful IT artefact (the caching engine) and evaluating it rigorously against defined criteria.

7.	Upstash (2025) Upstash Semantic Cache (GitHub).
      Annotation: A managed semantic cache providing a fuzzy key‑value store API. It offers no threshold selection guidance, false positive analysis, or model comparison data. Levy’s threshold selection guidelines (O3) aim to fill this commercial gap.

8.	Reimers, N. & Gurevych, I. (2019) Sentence‑BERT: Sentence embeddings using Siamese BERT‑networks. EMNLP, pp. 3982‑3992.
      Annotation: Introduces the sentence‑transformers library and the all‑MiniLM‑L6‑v2 model, a de facto baseline for semantic similarity. Levy uses all‑MiniLM as the general‑purpose embedding baseline.

9.	Johnson, J., Douze, M. & Jégou, H. (2021) Billion‑scale similarity search with GPUs. IEEE TPAMI, 43(2), pp. 421‑435.
      Annotation: Describes Faiss, the library Levy uses for vector indexing. Explains HNSW (Hierarchical Navigable Small World) graphs. Levy employs Faiss with HNSW for fast approximate nearest neighbour search.

10.	Vaswani, A. et al. (2017) Attention is all you need. NeurIPS, pp. 5998‑6008.
       Annotation: The Transformer architecture underlies both embedding models used. This foundational paper provides context for why encoder‑only models produce rich semantic representations suitable for similarity comparisons.

### Conduct of the Project:

#### •	Background on the research

LLM API costs and latency drive practitioner interest in semantic caching, while the technique's reported quality problems (false positives, workload sensitivity) drive academic interest in its evaluation. The literature confirms the problem and partially characterises it but has not produced controlled comparative evidence on embedding architecture choice. Levy is positioned to fill that evidential gap with an MSc-scoped empirical study.

#### •	What information will be used

o	Quantitative: precision, recall, F0.5, false positive rates, hit rates.

o	Qualitative: workload characteristics (average query length, duplication patterns).

#### •	Research methods

Design Science Research (Hevner et al., 2004) guides prototype construction. Applied Systems Evaluation guides empirical testing: controlled experiments with 2 × 3 × 5 = 30 conditions.

#### •	Data required

o	900 query pairs (300 per workload) sampled from public human‑annotated corpora.

o	Binary similarity labels from original annotators, plus author’s blind re‑annotations for validation.

o	No personal or sensitive data.

#### •	New skills and acquisition

o	Faiss HNSW index building and serialisation – Faiss tutorial (2 hours).

o	Sentence‑transformers embedding caching and batching, official examples (1 hour).

o	FastAPI async routing, FastAPI documentation (3 hours).

o	Statistical analysis (ANOVA, Cohen’s kappa), revision (2 hours).

#### •	Software to be used

o	FastAPI (web framework)

o	sentence‑transformers (embeddings)

o	Faiss (vector index)

o	Anthropic SDK (LLM)

o	scipy, numpy, pandas, scikit‑learn (statistics)

o	pytest (testing)

All open‑source, installed via pip.

## B.	The Design

### How the production of the IT artefact links to the dissertation question(s)

| Dissertation Sub-question | Artefact Component | Evaluation Metric |
|---|---|---|
| What are empirical duplication rates per workload? | Annotation pipeline (stratified sampling + blind re-annotation) | Per-workload duplicate prevalence, Cohen's kappa (author vs. original labels) |
| Does ModernBERT outperform all-MiniLM on cache precision? | `EmbeddingManager` with runtime model switch | Precision, Recall, F-score (β=0.5) per (model, workload) |
| Does embedding choice impact false positives? | Offline experiment harness (`run_experiment`) | False Positive Rate = FP / (FP + TN) per (model, workload) |
| What threshold per workload is optimal? | Threshold sweep loop (0.70–0.90, step 0.05) | Precision-vs-hit-rate curves, optimal threshold per (model, workload) |

### Hypotheses to be tested

The project tests three null hypotheses, derived directly from the sub-questions:
•	H0₁: Mean false positive rate of ModernBERT equals that of all-MiniLM across all workloads (no main effect of embedding model).
•	H0₂: Mean false positive rate is equal across the three workloads, holding embedding model constant (no main effect of workload).
•	H0₃: No interaction effect exists between embedding model and workload on false positive rate.
These are tested with two-way ANOVA on the false-positive-rate measurements over the 30 configurations. Significant rejection of any H0 supports the substantive claim that embedding choice and/or workload type meaningfully affect cache quality.


How the IT artefact will be evaluated to address the dissertation question(s)

Test environment. All experiments run locally on a MacBook Pro (Apple M3 Max, 64 GB RAM). Embedding models execute on-device via sentence-transformers without GPU acceleration. The Faiss HNSW index is built in-memory. The Anthropic API is invoked only on simulated cache misses to populate the cache realistically; once populated, replay experiments are deterministic and reproducible from the released dataset and configuration files.

Experimental procedure. For each of the 30 configurations (embedding model × workload × threshold):

1.	Initialise an empty Levy cache instance with the configuration's embedding model and threshold.
2.	For each query pair in the workload's 300 pairs of the ground truth: a. Submit query1 to the cache. By construction this is a cache miss; the cache stores the embedding-response pair. b. Submit query2 to the cache. The cache decides to hit / miss by comparing query2's embedding against the index using the configuration's threshold. c. Compare the cache decision against the ground-truth label: increment TP, FP, TN, or FN accordingly.
3.	After all, 300 pairs: compute precision, recall, F0.5, false positive rate, and hit rate.

### Statistical analysis.

•	Two-way ANOVA on false positive rate with factors (embedding model, workload).
•	Tukey HSD post-hoc tests where interactions are significant.
•	Cohen's kappa between the author's blind re-annotation and the original human labels, computed over the full 900 pairs, with a success threshold of κ > 0.7.
•	Replication check: re-running the released experiment harness on the released dataset must reproduce headline precision and recall values within ±5%.

Anticipated components and their organisation

### Component responsibilities:
•	FastAPI Router: exposes /v1/chat/completions (main caching endpoint), /admin/cache/stats, /admin/cache/clear.

•	Cache Orchestrator: coordinates lookup flow, hit/miss decision, structured logging for experimental analysis.

•	Embedding Manager: loads the configured model (all-MiniLM-L6-v2 or ModernBERT), generates embeddings, caches them in memory to avoid redundant recomputation during replay experiments.

•	Vector Store (Faiss): HNSW index over embedding vectors (L2 distance), with a separate metadata dictionary mapping internal IDs to (query_text, response, embedding_model).

•	LLM Connector: asynchronous wrapper around the Anthropic SDK with retry and token-accounting logic.

```
[Client] → [FastAPI Router (main.py)]
               ↓
         [Cache Orchestrator]
         ↙         ↘
[Embedding Manager]  [Vector Store (Faiss)]
    ↓                      ↓
[Model A / Model B]   [HNSW index + metadata]
         ↘         ↙
      [Cache Hit?]
       ↙       ↘
   [Hit]      [Miss]
     ↓           ↓
[Return cached] [LLM Connector → Anthropic API]
                  ↓
            [Store embedding + response]
```

Figure 1: Levy semantic caching engine component architecture


## Algorithms (pseudo‑code for key methods)

### 1.	Cache lookup algorithm

```python
    def cache_lookup(query: str, threshold: float, model: str) -> Optional[str]:
    emb = embedding_manager.get_embedding(query, model)        # shape (dim,)
    distances, indices = faiss_index.search(emb.reshape(1, -1), k=1)
    best_distance = distances[0][0]
    similarity = 1.0 / (1.0 + best_distance)                   # L2 → similarity
    if similarity >= threshold:
        entry = metadata[indices[0][0]]
        return entry.response
    return None
```

### 2.	Offline experiment harness

```python
def run_experiment(config: ExperimentConfig, ground_truth: List[QueryPair]) -> EvaluationResult:
    cache = LevyCache(embedding_model=config.embedding_model, threshold=config.threshold)
    tp = fp = tn = fn = 0
    for pair in ground_truth:
        # First query: always miss
        resp1 = cache.get(pair.query1)
        # Second query: may hit
        resp2 = cache.get(pair.query2)
        if pair.is_duplicate:  # original human label
            if resp2 is not None:
                tp += 1
            else:
                fn += 1
        else:
            if resp2 is not None:
                fp += 1
            else:
                tn += 1
    return compute_metrics(tp, fp, tn, fn)
```

Intended interface (API contract)

Endpoint: POST /v1/chat/completions
Request body:
```json
{
  "messages": [{"role": "user", "content": "What is semantic caching?"}],
  "model": "claude-3-sonnet-20240229",
  "cache_config": {
    "threshold": 0.85,
    "embedding_model": "modernbert"
  }
}
```

Response (cache hit): headers X-Cache-Status: HIT, X-Cache-Similarity: 0.92; body matches the Anthropic response format. Response (cache miss): header X-Cache-Status: MISS; body is the fresh Anthropic response.

Admin endpoint: GET /admin/cache/stats returns hit rate, index size, per-model breakdown.

## C.	Statement of Deliverables

### Essential (required for project success)

D1: Levy Semantic Caching Engine Prototype. Working Python implementation with the three components (FastAPI router, embedding manager + Faiss HNSW index, Anthropic connector), runtime model switching via configuration, unit tests with pytest (target coverage >80%), released on GitHub under Apache 2.0 licence with installation, usage, and reproduction instructions.

D2: Annotated Ground Truth Dataset. 900 query pairs (300 per workload) sampled from publicly available human-annotated corpora. Each pair retains the original human label and the author's blind re-annotation. Cohen's kappa between the author and the original labels is computed over the full set with a target of κ > 0.7. Released publicly in CSV and JSON formats with a datasheet documenting source corpora, sampling protocol, annotation guidelines, and known limitations.

D3: Evaluation Results and Analysis. Complete results across the 30 experimental configurations: precision, recall, F0.5, FP rate, and hit rate per configuration; ANOVA tables and post-hoc results; threshold-vs-hit-rate curves per (model, workload). Released as machine-readable tables (CSV) and as figures.

D4: Capstone Dissertation. Academic thesis (12,000–18,000 words excluding references and appendices) covering methodology, results, threshold-selection guidelines, limitations, and future work, formatted to University of Liverpool requirements.

D5: Project Documentation. README, architecture documentation, reproduction guide, and annotated dataset datasheet, all included in the public repository.

### Desirable (if time permits)

D6: Interactive Dashboard. Streamlit or Gradio application visualising threshold-performance curves and allowing user-supplied queries to be tested against pre-computed configurations.

D7: Dockerised Deployment. Dockerfile and docker-compose.yml enabling one-command reproduction of the full evaluation pipeline.

## D.	Project Plan 

| Week(s) | Phase | Activities | Deliverable                                                   |
|---|---|---|---------------------------------------------------------------|
| 12–15 | Phase 1: Literature & Data | Extend literature review; select public corpora per workload; sample 900 pairs (stratified); blind re-annotation by author; compute Cohen's kappa | Annotated dataset (Week 15)                                   |
| 16–20 | Phase 2: Design & Implementation | Finalise component design; implement FastAPI router, embedding manager, Faiss index, Anthropic connector; unit tests | **S&D Report (Week 16, external)**; working prototype (Week 20) |
| 21–27 | Phase 3: Experimental Evaluation | Run 30 configurations, collect metrics, generate tables/plots | Benchmark results, model comparison                           |
| 28 | Phase 4: Poster | Prepare research poster | Poster Submission (Week 28)                                   |
| 29–36 | Phase 5: Dissertation Draft | Write results, analysis, guidelines; share draft with advisor | Draft dissertation (Week 36)                    |
| 37–40 | Phase 6: Finalisation | Revise, package code/data, record video, submit | Final dissertation, artefact, video                      |

### Critical path dependencies

•	Dataset preparation (Weeks 12‑15) → must complete before experimentation (Week 21).
•	Prototype implementation (Week 20) → must complete before experimentation.
•	Experiments (Week 27) → must complete before dissertation writing (Week 29)

### Estimated effort

•	Phase 1: 40 hours
•	Phase 2: 50 hours
•	Phase 3: 40 hours
•	Phase 4: 10 hours
•	Phase 5: 140 hours
•	Total: 280 hours (≈10 hours/week over 28 active weeks)

## Reference

Bang, F. (2023) 'GPTCache: An open-source semantic cache for LLM applications enabling faster answers and cost savings', in Tan, L., Milajevs, D., Chauhan, G., Gwinnup, J. and Rippeth, E. (eds.) Proceedings of the 3rd Workshop for Natural Language Processing Open-Source Software (NLP-OSS 2023). Singapore: Association for Computational Linguistics, pp. 212–218. doi: 10.18653/v1/2023.nlposs-1.24.


Gill, W. et al. (2025) 'MeanCache: User-Centric Semantic Caching for LLM Web Services', in Proceedings - IEEE International Parallel and Distributed Processing Symposium. IEEE, pp. 1298–1310. Available at: https://doi.org/10.1109/IPDPS64566.2025.00117.

Hevner, A.R. et al. (2004) 'Design Science in Information Systems Research', MIS Quarterly, 28(1), pp. 75–105. Available at: https://doi.org/10.2307/25148625.

Johnson, J., Douze, M. and Jégou, H. (2021) 'Billion-Scale Similarity Search with GPUs', IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(2), pp. 421–435.

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks', in Proceedings of EMNLP. Association for Computational Linguistics, pp. 3982–3992.

Upstash (2025) Upstash Semantic Cache [GitHub repository]. Available at: https://github.com/upstash/semantic-cache.

Vaswani, A. et al. (2017) 'Attention Is All You Need', in Advances in Neural Information Processing Systems (NeurIPS), pp. 5998–6008.

Wang et al. (2025) 'Category-Aware Semantic Caching for Heterogeneous Workloads', arXiv:2510.26835.

Warner, B. et al. (2024) 'ModernBERT: A Modern Approach to Encoder-Only Transformers', arXiv:2412.13663.

Zheng, L. et al. (2024) 'LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset', in International Conference on Learning Representations (ICLR).
