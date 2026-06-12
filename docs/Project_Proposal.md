# PROJECT

### The Project Aims and Objectives:

Levy aims to systematically benchmark semantic caching effectiveness for large language model (LLM) APIs, specifically investigating whether embedding model selection meaningfully impacts false positive rates across different workloads. Current production tools like GPTCache (Bang, 2023) and Upstash Semantic Cache report persistent quality problems, GPTCache explicitly acknowledges “<90% hit rates with inevitable false positives unacceptable in production scenarios” yet provide no model selection guidance or workload-specific deployment guidelines. While semantic caching is deployed at scale (production systems handling hundreds of millions of daily queries), practitioners lack empirical evidence comparing embedding strategies, quantified false positive analyses, or threshold selection guidance, leaving organizations to conduct expensive trial-and-error experimentation. Levy addresses this gap through rigorous comparative evaluation of general-purpose versus domain-optimized embedding models across FAQ, code generation, and conversational chat workloads, employing design science research methodology to build a working prototype and applied systems evaluation to measure performance through controlled experiments.

The project delivers four concrete objectives:

O1: Workload Benchmarking measures empirical semantic duplication rates across three production-relevant workload types, creating 900 manually annotated query pairs (300 per workload) as ground truth data.

O2: Embedding Strategy Comparison evaluates two pre-trained embedding models, all-MiniLM (general-purpose baseline) versus ModernBERT (architecture optimised for longer sequences and modern pre-training), measuring precision, recall, F-score, and false positive rates across the annotated dataset to determine whether model selection produces meaningful performance differences for semantic caching.

O3: Threshold Optimization tests five similarity thresholds (0.70-0.90) across both embedding strategies to identify optimal values per workload that minimize false positives while maintaining economic viability, producing evidence-based threshold selection guidelines.

O4: Prototype Development builds the Levy semantic caching engine comprising three components, FastAPI request router, ModernBERT+Faiss semantic cache module, and Anthropic LLM backend connector, releasing fully documented code, annotated dataset, and benchmark results on GitHub under Apache 2.0 licence.

The contribution is both scholarly and practical. Academically, Levy provides the first systematic embedding model comparison for semantic caching with reproducible methodology, addressing gaps where existing literature defers duplication quantification and false positive mitigation to future work. Commercially, practitioners gain evidence-based guidelines on which embedding architecture suits their workload, replacing vendor claims with quantified data. The scope fits an MSc capstone: empirical evaluation of existing techniques using established tools (FastAPI, Faiss, sentence-transformers) across three systematically annotated workloads.


In the table below, please state your dissertation question(s); the research methods you will use to guide the development of your IT artefact; the kind of IT artefact you will produce; and how you will evaluate the IT artefact in the light of the dissertation question(s) 

|Step|	Short Description|
|---|---|
|Dissertation Question(s)|	Primary Research Question: Does embedding model selection meaningfully impact false positive rates in semantic caching across production LLM workloads? sub-questions: - What are empirical semantic duplication rates in FAQ, code, and conversational workloads? - Does ModernBERT outperform all-MiniLM on cache precision metrics? - What similarity threshold per workload minimises false positives while preserving economic value?
| Research Methods	| The project employs Design Science Research (Hevner et al., 2004) combined with Applied Systems Evaluation. Design science guides the construction of the Levy prototype as a solution to the identified problem (false positives in semantic caching), while applied systems evaluation provides the framework for rigorous empirical testing. The experimental methodology follows controlled comparison: testing 2 pre-trained embedding models (all-MiniLM vs ModernBERT) across 3 workloads (FAQ, code, chat) with 5 similarity thresholds (0.70–0.90), yielding 30 experimental conditions. Cohen's kappa (>0.7) ensures inter-rater reliability for manual annotations. This quantitative, reproducible approach enables systematic evaluation of whether model selection produces meaningful performance differences for semantic caching.|
| IT Artefact	| The Levy Semantic Caching Engine consists of three integrated software components built using established open-source libraries. The Request Router (FastAPI) intercepts incoming LLM queries and routes them through the caching pipeline while logging all interactions for experimental analysis. The Semantic Cache Module uses ModernBERT embeddings and Faiss HNSW vector index to identify semantically similar queries and retrieve cached responses, supporting both embedding models via runtime configuration. The LLM Backend Connector integrates with the Anthropic API to generate fresh responses on cache misses and store them with embeddings for future reuse. Deliverables include the working Python prototype (Apache 2.0 licence), 900 annotated query pairs as ground truth data, and comprehensive documentation enabling reproducibility.|
| Evaluation	| The artefact will be evaluated through offline replay experiments using the annotated dataset. Each of the 30 experimental configurations (2 embedding models × 3 workloads × 5 thresholds) will process all 900 query pairs, measuring precision (proportion of cache hits that are correct), recall (proportion of true duplicates detected), F-score with β=0.5 (prioritising precision over recall), and false positive rate (incorrect cache hits). Economic viability will be assessed through hit rate analysis (percentage of queries served from cache) and latency measurements (cache lookup overhead vs LLM call savings). Success is defined as identifying measurable precision differences between embedding models across workloads while maintaining hit rates >30%.


## Project Outline

Levy follows a four-phase execution plan. Phase 1 (Weeks 12–15) establishes the research foundation through comprehensive literature review on LLM serving architectures, semantic caching strategies, and embedding model selection, while creating the annotated ground truth dataset of 900 query pairs (300 per workload: FAQ, code generation, conversational chat). Phase 2 (Weeks 16–20) implements the Levy prototype comprising three components: FastAPI request router, ModernBERT+Faiss semantic cache module, and Anthropic LLM backend connector, culminating in the Specification and Design Report submission (Week 16). Phase 3 (Weeks 21–27) conducts controlled experiments testing 30 configurations (2 embedding models × 3 workloads × 5 similarity thresholds 0.70–0.90), measuring precision, recall, F-score, and false positive rates through offline replay experiments. Phase 4 (Weeks 28–40) performs results analysis, generates evidence-based threshold selection guidelines, and completes dissertation writing. The research methods combine Design Science Research (Hevner et al., 2004) for prototype construction with Applied Systems Evaluation for empirical testing. The IT artefact is a working semantic caching engine released on GitHub under Apache 2.0 licence with full documentation and annotated dataset enabling reproducibility.



### Literature Survey / Resources’ List:

Core Papers (Semantic Caching)
‘GPTCache: An open-source semantic cache for LLM applications enabling faster answers and cost savings’ (Bang, F. (2023))
Foundational open-source semantic caching framework for LLM applications. Reports cache hit rates below 90% with "inevitable false positives unacceptable in production scenarios," yet provides no domain adaptation methodology or workload-specific threshold guidance. Levy uses GPTCache's acknowledged limitations as the primary motivation for systematic embedding model comparison.

“MeanCache: User-Centric Semantic Caching for LLM Web Services” (Gill, W. et al. (2025))
Introduces precision and recall as the correct evaluation metrics for semantic caching and proposes F-score with β=0.5 to prioritise precision over recall. Limited to a single production system. Levy adopts their evaluation framework (precision, recall, F-score β=0.5) and extends the comparison across multiple workload types and embedding models.

'Category-Aware Semantic Caching for Heterogeneous Workloads' (Wang, et al. (2025))
Proposes per-category similarity thresholds (code: 0.9, chat: 0.75) based on a single production deployment. Validates the intuition that different workloads require different thresholds but lacks reproducible methodology. Levy builds on this insight by systematically testing five thresholds across three workloads with controlled experiments.

Embedding Models
'ModernBERT: A Modern Approach to Encoder-Only Transformers' (Warner, B. et al. (2024))
149M parameter encoder with 8K context length, achieving state-of-the-art results among compact models. Levy uses ModernBERT as the primary embedding model under evaluation, comparing its longer-context architecture against the widely deployed all-MiniLM baseline to determine whether architectural differences translate to measurable cache quality improvements.

Datasets & Benchmarking
'LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset' (Zheng, L., et al. (2024))
The largest public LLM conversation dataset (1M conversations). Explicitly acknowledges that the dataset "contains repeated data" but states "this quantification is left for future work." This gap directly motivates Levy's O1 objective: measuring empirical duplication rates across workload types to provide the baselines the field currently lacks.

Research Methodology
“Design Science in Information Systems Research” (Hevner, A.R. et al. (2004))
Defines the Design Science Research methodology framework used to guide Levy's prototype construction. Establishes the dual requirement of building a purposeful IT artefact (the caching engine) and evaluating it rigorously against defined criteria, which structures Levy's approach of combining prototype development with controlled empirical evaluation.

Commercial Baselines
Upstash Semantic Cache (2025) GitHub: github.com/upstash/semantic-cache.
Managed semantic cache providing a fuzzy key-value store API but no threshold selection guidance, false positive analysis, or model comparison data. Represents the commercial state-of-practice where practitioners deploy caching without empirical evidence for configuration decisions, the gap Levy's threshold selection guidelines (O3) aim to fill.


### Scholarly Contributions of the Project

Levy makes three distinct scholarly contributions addressing empirical gaps in LLM optimization literature. First, it provides the first systematic duplication rate measurement across multiple LLM workload types with rigorous annotation methodology (900 manually labeled query pairs with Cohen's kappa >0.7 inter-rater reliability). LMSYS-Chat-1M (Zheng et al., 2024), the largest public LLM conversation dataset, explicitly states in its limitations that "dataset contains repeated data... this quantification is left for future work," leaving researchers without empirical baselines for estimating cache effectiveness across different application domains. Levy fills this gap through controlled measurement across FAQ, code generation, and conversational chat workloads, enabling future researchers to predict cache viability pre-deployment.

Second, Levy delivers the first rigorous false positive comparison of embedding model architectures for semantic caching. While GPTCache (Bang, 2023) acknowledges "inevitable false positives," MeanCache (Gill et al., 2025) identifies the problem as critical, and InfoQ Banking Study (2025) documents 99% baseline FP rates, no comparative analysis exists evaluating how model selection affects cache quality across workload types. Levy systematically compares all-MiniLM (general-purpose) against ModernBERT (optimised for longer sequences) across three workloads using precision, recall, F-score (β=0.5), and false positive rate metrics, establishing whether embedding model choice produces meaningful performance differences and under what conditions.

Third, the project produces reproducible benchmarking methodology filling the gap between vendor black-box implementations (GPTCache, Upstash provide APIs but no validation data) and academic toy prototypes (MeanCache, Category-Aware code not publicly available). Levy releases open-source code, annotated datasets, complete experimental procedures, and analysis scripts under Apache 2.0 licence, enabling future researchers to validate findings, extend to additional workloads, or compare alternative embedding models using standardised evaluation procedures.


### Description of the Deliverables:

D1: Levy Semantic Caching Engine Prototype: Working Python implementation (3 components: FastAPI router, ModernBERT+Faiss cache module, Anthropic LLM backend connector) supporting both embedding models via runtime configuration. Released on GitHub under Apache 2.0 licence with complete documentation, installation instructions, and usage examples enabling practitioners to deploy or adapt the system.

D2: Annotated Ground Truth Dataset: 900 manually labelled query pairs (300 per workload: FAQ, code, conversational) with binary similarity judgments, achieving Cohen's kappa >0.7 inter-rater reliability. Released publicly to enable reproducibility and serve as benchmark for future research.

D3: Capstone Dissertation: Complete academic thesis documenting methodology, experimental results across 30 configurations (2 models × 3 workloads × 5 thresholds), threshold selection guidelines, and practical implications, formatted according to University of Liverpool requirements.
Evaluation Criteria:

Success will be measured through three quantitative criteria aligned with the dissertation question. Criterion 1: Model Differentiation requires that the two embedding models produce measurably different precision and false positive rates across workloads, establishing that model selection is a meaningful design decision for semantic caching. Criterion 2: Economic Viability requires maintaining hit rates >30% across workloads to ensure caching provides cost reduction, validating that precision-optimised thresholds don't eliminate economic benefit. Criterion 3: Reproducibility Validation requires that released code and data can replicate core findings (precision/recall within 5% of reported values), ensuring scholarly contribution is verifiable.

Assessment methodology involves running all 30 experimental configurations, computing precision/recall/F-score/FP-rate for each, and documenting results in standardised tables. Differences in model performance across workloads will be analysed and discussed to inform practical model selection guidance.


### Resource Plan:

Hardware/Infrastructure:
MacBook Pro (Apple M3 Max, 64GB RAM) sufficient for development, embedding generation, and experimentation; no GPU required as both embedding models run efficiently on Apple Silicon. Cloud compute for LLM API calls via Anthropic API. No specialised hardware acquisition needed.

Software/Libraries (All Open-Source):
FastAPI (web framework), sentence-transformers (embeddings), Faiss (vector index), Anthropic SDK (LLM access), scipy/numpy (statistics), pytest (testing). All available via pip at zero cost.



Data Resources:
Quora Question Pairs (public dataset), Stack Overflow questions (public via API), ConvAI2 (academic dataset, freely available). Manual annotation labour: researcher's own time, 100 pairs/day × 9 days = 90 hours total.

Financial Costs:
LLM API usage via Anthropic Claude Sonnet estimated at $50 (10,000 queries with experimental overhead). Embedding models run locally at zero cost. Total budget: $200 allocated, ~$50 expected actual cost. No travel or equipment purchases required.

Personnel:
Solo researcher project; no collaborators or paid assistants. Dissertation advisor provides feedback but not implementation support.


## Project Plan and Timing

Phase 1: Literature Review & Data Preparation (Weeks 12–15)
Comprehensive literature review covering semantic caching strategies, embedding model architectures, and evaluation methodologies. Dataset selection and annotation pipeline established, followed by manual annotation of 900 query pairs (300 per workload) with Cohen's kappa >0.7 inter-rater reliability. Deliverables: validated annotated dataset and literature review summary.

Phase 2: Specification, Design & Implementation (Weeks 16–20)
Prototype architecture defined and documented for the Specification and Design Report (due Week 16). Implementation of all three components: FastAPI router, ModernBERT+Faiss cache module with runtime model switching, and Anthropic LLM backend connector. Deliverables: Specification and Design Report, working Levy prototype with unit tests and configuration system.
University Milestone: Specification and Design Report — Week 16 (18 May 2026).

Phase 3: Experimental Evaluation (Weeks 21–27)
Experimental harness deployed, executing all 30 configurations (2 models × 3 workloads × 5 thresholds). Results analysis, visualisation, and model comparison complete. Deliverables: benchmark results tables, threshold-performance curves, model comparison data.

Phase 4: Poster & Preliminary Writing (Week 28)
Research poster summarising methodology, key findings, and contributions prepared for submission. Deliverables: research poster.
University Milestone: Poster — Week 28 (10 August 2026).

Phase 5: Dissertation Writing & Finalisation (Weeks 29–40)
Full dissertation writing: methodology, results analysis, threshold selection guidelines, limitations, and future work. Draft submitted for advisor review (Week 36). Final revisions, IT artefact packaging (code, dataset, documentation on GitHub), and video demonstration prepared. Deliverables: dissertation draft (Week 36), final dissertation, IT artefact release, video demonstration.
University Milestones: Dissertation draft — Week 36 (5 October 2026).
Final submission (Dissertation, IT Artefact, Video Demonstration) — Week 40 (2 November 2026).

Critical Path Dependencies: Annotation (Weeks 12–15) must complete before experimentation (Weeks 21–27); implementation (Weeks 16–20) must complete before experimentation; all experiments must complete before dissertation writing (Weeks 29–40). Literature review and documentation can run in parallel with their respective phases.

Estimated Effort: ~10 hours/week across 28 active weeks (~280 hours total). Phase 1: 40h; Phase 2: 50h; Phase 3: 40h; Phase 4: 10h; Phase 5: 140h.


## Risk Assessment:

Risk 1: Annotation Delays:  Manual labelling of 900 query pairs may take longer than estimated.
Contingency: Use existing labelled datasets (Quora duplicate pairs, PAWS paraphrase corpus) as fallback; reduces novelty but maintains experimental validity.

Risk 2: No Meaningful Model Differentiation: ModernBERT and all-MiniLM may produce similar results across workloads.
Contingency: A null result is still a scholarly contribution; documenting that model selection does not significantly affect cache quality is valuable guidance for practitioners who would otherwise invest effort evaluating models.

Risk 3: LLM API Cost Overruns:  Experimentation may consume more API calls than budgeted.
Contingency: $200 budget provides 4x safety margin over ~$50 estimate; can reduce configurations from 30 to 18 (2 models × 3 workloads × 3 thresholds instead of 5).

Risk 4: Dataset Access Restrictions: ConvAI2 or Stack Overflow may change licensing or access policies.
Contingency: Multiple fallback datasets identified (MS MARCO for FAQ, CodeSearchNet for code, DailyDialog for chat).

Risk 5: Implementation Bugs: Prototype may have subtle bugs affecting experimental validity.
Contingency: Unit testing (pytest), manual validation of cache hit/miss decisions on sample queries, statistical sanity checks (hit rates within expected ranges).


## Quality Assurance:

Progress will be monitored through weekly self-assessment against milestones documented in the project plan, with formal checkpoints at phase boundaries (end of Weeks 15, 20, 27, 28, 36, 40). Each phase concludes with a deliverable that enables validation: Phase 1 produces annotated dataset verifiable via inter-rater reliability (Cohen's kappa >0.7); Phase 2 produces working prototype verifiable via unit tests and manual query testing; Phase 3 produces experimental results verifiable via analysis and visualisation; Phase 5 produces dissertation verifiable via advisor feedback.

Internal Quality Controls: Unit tests for all prototype components using pytest. Manual validation with known duplicate/non-duplicate query pairs to verify cache behaviour. Statistical sanity checks ensuring hit rates, precision, and recall values fall within theoretically plausible ranges. Code review via self-documented pull requests explaining implementation decisions.

Success Indicators per Phase: Phase 1: Cohen's kappa >0.7, 900 pairs annotated, literature review identifies clear research gap. Phase 2: Prototype passes unit tests, manually verified cache hits/misses correct, runtime model switching works. Phase 3: All 30 configurations execute without errors, results tables complete. Phase 4: Poster submitted. Phase 5: Dissertation meets word count (12,000–18,000 excluding references and appendices), advisor approves final draft, code and data released publicly.

## Reference

Bang, F. (2023) ‘GPTCache: An open-source semantic cache for LLM applications enabling faster answers and cost savings’, en Tan, L., Milajevs, D., Chauhan, G., Gwinnup, J. y Rippeth, E. (eds.) Proceedings of the 3rd Workshop for Natural Language Processing Open-Source Software (NLP-OSS 2023). Singapur, Diciembre. Association for Computational Linguistics, pp. 212–218. doi: 10.18653/v1/2023.nlposs-1.24.
Gill, W. et al. (2025) “MeanCache: User-Centric Semantic Caching for LLM Web Services,” in Proceedings - IEEE International Parallel and Distributed Processing Symposium. IEEE, pp. 1298–1310. Available at: https://doi.org/10.1109/IPDPS64566.2025.00117.
Wang, et al. (2025) 'Category-Aware Semantic Caching for Heterogeneous Workloads', arXiv:2510.26835.
Warner, B. et al. (2024) 'ModernBERT: A Modern Approach to Encoder-Only Transformers', arXiv:2412.13663.
Zheng, L., et al. (2024) 'LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset', International Conference on Learning Representations (ICLR).
Hevner, A.R. et al. (2004) “Design Science in Information Systems Research,” MIS quarterly, 28(1), pp. 75–105. Available at: https://doi.org/10.2307/25148625.

