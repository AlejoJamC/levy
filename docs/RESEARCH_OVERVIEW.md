# Levy: Semantic Caching for LLM API Cost Optimization
## Research Overview & Academic Context

**Author:** John Alejandro Mantilla Celis  
**Institution:** University of Liverpool  
**Program:** MSc Artificial Intelligence  
**Module:** CSCK508 Research Methods in Computer Science  
**Project Type:** MSc Capstone Research Project

---

## 1. Introduction & Personal Background

I am a Senior Backend Engineer with over 10 years of experience in distributed systems, currently contracted to one of the largest banks in the Netherlands. My professional background centers on event-driven architectures, particularly Apache Kafka, which I leveraged extensively at one of the fastest growing and most influential apps in Latin America where I scaled systems handling over 500,000 daily orders.

I am completing my MSc in Artificial Intelligence at the University of Liverpool, with expected completion in 2025. This research project, Levy, bridges my distributed systems expertise with emerging AI infrastructure challenges, specifically addressing the cost optimization problem in large language model (LLM) API usage.

---

## 2. Industry Context & Commercial Motivation

### 2.1 The Explosive Growth of LLM-Powered Applications

Large language models have rapidly become integral components of modern applications across industries: customer support chatbots, coding copilots, content generation tools, and knowledge assistants. These systems are being deployed at scale from healthcare to finance to e-commerce.

However, each interaction with an LLM involves billions of floating-point operations. These models contain billions of parameters and require substantial computational infrastructure to operate effectively. In my work at the bank, I have witnessed firsthand how LLM API bills can escalate to tens or even hundreds of thousands of euros per month.

### 2.2 The Fundamental Cost Problem

Organizations pay per token and per request. Every single API call adds multiple dimensions of cost:

- **Financial cost**: Direct API pricing
- **Latency overhead**: Often hundreds of milliseconds per request
- **Reliability risks**: Potential failures requiring retry logic
- **Performance throttling**: Rate limits that constrain application throughput
- **External dependencies**: Operational risks from third-party service reliance

To illustrate the scale: if a financial services application processes 50,000 customer queries daily at 5 cents per query through an API provider, that's €2,500 per day—over €900,000 annually—just for LLM inference. This is a conservative estimate for a mid-sized deployment.

### 2.3 The Redundancy Insight from Academic Research

Prior studies on web search engines—including work by Lempel and Moran (2003), Markatos (2001), and Xie and O'Hallaron (2002)—found that approximately 33% of queries submitted to search engines are repeated or near-duplicates.

Recent research from Virginia Tech and Redis (Gill et al., 2025; arXiv:2504.02268v1) confirms this same phenomenon occurs with LLM-based services. Consider typical use cases:

- **FAQ systems**: Users ask "How do I reset my password?" versus "What's the password reset procedure?"—different wording, identical semantic intent, same required answer
- **Customer support summarization**: Repeated issue patterns with linguistic variation
- **Internal documentation tools**: Common queries phrased differently by different users
- **Code assistance**: Similar programming questions with contextual differences

In specialized domains like banking, duplication rates can be even higher due to regulatory compliance queries, transaction dispute explanations, and account management questions following predictable patterns.

### 2.4 The Technical Gap in Current Practice

Current organizational approaches to this problem are inadequate:

**Exact-match caching**: Traditional key-value cache approach stores the exact prompt string and retrieves if seen again. The problem: it completely misses semantic duplicates. "Reset my password" and "forgot login credentials" won't match, even though they require essentially the same response.

**Ad-hoc manual reuse**: Some teams manually identify common queries and hard-code responses. This doesn't scale, requires continuous maintenance, and breaks when query patterns evolve.

**Existing observability tools**: Platforms like Helicone and LangFuse focus on monitoring—they'll show you the problem ("look, you have duplicate queries") but don't optimize it.

**GPTCache**: Exists as a Python library for semantic caching but lacks production-grade optimization for real workloads, doesn't address domain-specific adaptation, and provides limited empirical evidence on actual cost savings versus quality trade-offs.

### 2.5 The Research Gap

This brings us to the core research gap that Levy addresses. We lack empirical evidence on semantic caching under realistic production conditions across three critical dimensions:

1. **Actual cost savings**: How much can organizations really save? Is it 10%, 50%, or negligible?
2. **Quality impact**: When we return a cached answer for a semantically similar query, how often is it actually the right answer? What's the false positive rate?
3. **Latency overhead**: Computing embeddings and searching vector databases adds time. Is it 20 milliseconds or 200 milliseconds? At what point does caching overhead negate the benefits?

Additionally, we don't know which workload patterns benefit most from semantic caching. Is it universally applicable, or only valuable for specific domains like customer support versus creative content generation?

### 2.6 Project Management Approach

I am managing Levy as a structured 12-week research project with four clear phases:

1. **Phase One**: Scoping and comprehensive literature review—understanding the state of semantic caching, embedding models, and production LLM optimization
2. **Phase Two**: Design and implementation of the Levy caching engine prototype—building a working system with exact-match and semantic caching layers
3. **Phase Three**: Experimental evaluation on realistic workloads—testing across FAQ, conversational support, and retrieval-augmented generation scenarios
4. **Phase Four**: Analysis, commercial interpretation, and dissertation writing

The timeline is tight but achievable given my ten years of distributed systems experience, particularly with event-driven architectures and caching strategies at scale.

### 2.7 Commercial Value Proposition

This commercially relevant gap—companies lacking evidence and systematic tools to reuse semantically similar answers—is precisely what Levy addresses. By validating semantic caching with rigorous empirical benchmarks, Levy will provide practitioners with decision frameworks: when to deploy semantic caching, which embedding strategies work best, and what cost-quality-latency trade-offs to expect.

The potential impact is substantial. If semantic caching can reduce LLM API costs by 30-50% while maintaining acceptable quality, that represents:

- Hundreds of thousands in annual savings for organizations with high query volumes
- Faster response times for end users
- Reduced dependency on external API providers

---

## 3. Problem Statement, Hypothesis & Research Questions

### 3.1 The Problem

Building on the context described above, the core problem is clear: LLM systems in production incur substantial usage-based costs, and studies show 30-40% of queries are semantically redundant. Yet current practice remains inadequate. Exact-match caching misses semantic duplicates entirely. Existing semantic caching libraries like GPTCache exist, but they lack rigorous empirical benchmarks on real production workloads.

This leaves us without critical evidence. We don't know the actual cost savings achievable in practice. We don't understand the quality trade-off—how often cached answers mismatch user intent, leading to wrong responses. And we haven't quantified the latency overhead from embedding generation and vector search operations.

Without this empirical foundation, organizations cannot make informed decisions about deploying semantic caching infrastructure.

### 3.2 Working Hypothesis

This research gap leads directly to my working hypothesis:

> **Semantic caching can reduce LLM API costs by a meaningful margin—specifically, 30-60%—while maintaining answer quality with false positive rates below 5% and keeping latency overhead below 100 milliseconds at the 99th percentile.**

This hypothesis is testable, measurable, and commercially significant.

### 3.3 Research Questions

To validate this hypothesis systematically, I have structured the investigation around three research questions:

**RQ1: What cache hit rates and cost savings can semantic caching achieve across typical LLM workloads?**

This quantifies the economic benefit by measuring performance across:
- FAQ systems (high repetition expected)
- Conversational support scenarios (moderate overlap)
- Retrieval-augmented generation applications (variable duplication)

**RQ2: How do similarity thresholds and cache policies affect the three-way trade-off between cost, latency, and answer quality?**

For instance, a similarity threshold of 0.95 might deliver high precision but low cache hit rates, while a threshold of 0.75 might achieve high hit rates but introduce too many false positives. This question explores the parameter space to identify optimal configurations.

**RQ3: Under which workload patterns is semantic caching most beneficial, and when does it provide little value?**

Creative content generation likely exhibits low query duplication, while technical support systems likely show high repetition. This question will deliver practical deployment guidelines.

Together, these three questions are designed to produce actionable insights for both academic researchers advancing LLM optimization techniques and industry practitioners evaluating technology investments.

---

## 4. Research Methodology & The Levy Artifact

### 4.1 Methodological Approach

My methodological approach combines design science with applied systems evaluation and quantitative analysis. Rather than purely theoretical modeling, I am building and evaluating a working semantic caching module for LLM APIs—the Levy prototype. The focus is deliberately on clarity and reproducibility rather than production-scale optimization, which is appropriate for a capstone research project while still generating commercially meaningful insights.

### 4.2 The Levy Caching Engine Prototype

The Levy caching engine prototype comprises five key technical components that work together as an integrated system:

**Component 1: Client/Proxy Layer**  
A simple client or proxy that receives LLM requests from an application. This acts as the interception point where we can analyze queries before they reach the actual LLM provider.

**Component 2: Exact-Match Cache**  
A traditional cache keyed by the full prompt string. This serves as our baseline—traditional caching that only matches identical queries character-for-character. By comparing against this baseline, we can quantify the additional value that semantic caching provides.

**Component 3: Semantic Cache (Novel Contribution)**  
The core innovation uses embeddings and vector similarity search to identify semantically similar queries even when wording differs. For instance, recognizing that "reset my password" and "forgot login credentials" should retrieve the same cached response.

**Component 4: Vector Store**  
A vector database (likely FAISS or similar) which efficiently stores and retrieves high-dimensional embeddings. This is critical for maintaining acceptable latency as the cache grows.

**Component 5: LLM Backend Connector**  
A backend connector to an LLM provider. This establishes baseline behavior without any caching, giving us ground truth for comparison against both exact-match and semantic caching strategies.

These five components together form a complete experimental system for rigorously testing semantic caching effectiveness.

### 4.3 Project Execution Phases

The research execution is structured across four project phases spanning twelve weeks:

**Phase One: Scoping and Literature Review (Weeks 1-3)**  
Understanding the current state of the art in LLM serving architectures, existing caching strategies, and cost optimization techniques in production systems.

**Phase Two: Design and Implementation (Weeks 4-6)**  
Building the working Levy system with all five components integrated and functioning together.

**Phase Three: Experimental Evaluation (Weeks 7-9)**  
The empirical core of the research—testing across different query patterns to answer the research questions systematically.

**Phase Four: Analysis and Writing (Weeks 10-12)**  
Translating raw experimental results into actionable insights and scholarly contributions through comprehensive analysis and dissertation writing.

### 4.4 Risks and Ethics

**Technical Risks**:  
The main technical risk is API costs escalating during experimentation and potential dataset availability constraints. I am mitigating this by using carefully limited workloads with cost controls in place.

**Legal and Ethical Considerations**:  
I am committed to using only anonymized text with no personally identifiable information, strictly adhering to dataset licenses and API provider terms of service. Should real user data become necessary, I would seek formal ethical approval, but the current plan relies exclusively on public datasets and synthetically generated data.

---

## 5. Data Collection Strategy & Analytical Framework

### 5.1 Data Strategy Overview

The data strategy relies on publicly available and synthetically generated datasets that accurately mimic common LLM workloads while avoiding any privacy or licensing complications. I will be working with three distinct workload types, each representing different real-world usage patterns.

### 5.2 Workload Types

**Workload Type 1: FAQ-Style Question Answering**

Represents scenarios like customer support systems where users repeatedly ask questions about a service or product. Think questions like "How do I reset my password?" or "What are your business hours?" This workload type naturally exhibits high semantic duplication, making it an ideal candidate for semantic caching.

**Data sources**:
- Quora Question Pairs dataset: Over 400,000 question pairs with duplicate labels
- MS MARCO: Additional FAQ-style queries

**Workload Type 2: Short Conversational Interactions**

Simulates support chat scenarios with moderate semantic overlap. These conversations show partial repetition—opening greetings, common troubleshooting steps, closing statements—but with enough variation to test the boundaries of semantic similarity matching. I will construct these from publicly available conversational datasets.

**Workload Type 3: Retrieval-Augmented Generation (RAG)**

Queries posed over documents or knowledge bases. This exhibits variable duplication depending on document popularity and query diversity. Some documents get queried repeatedly with semantic variations, while niche documents see unique queries.

**Data source**:
- SQuAD dataset for document-based query patterns

### 5.3 Workload Construction Methodology

To ensure realism, I will construct these workloads by replaying actual prompt sequences rather than random sampling. The experimental process follows a structured progression:

1. **Baseline runs**: Collect runs without any caching to establish ground truth costs and latency
2. **Exact-match caching**: Introduce traditional caching to quantify conventional benefits
3. **Semantic caching**: Deploy semantic caching with systematically varied similarity thresholds (0.70 to 0.95) and different time-to-live (TTL) policies

### 5.4 Analytical Framework: Four Metric Dimensions

**Dimension 1: Cost Metrics**

- Total tokens consumed per configuration
- Cumulative cost per configuration
- Relative savings versus baseline

This directly answers the economic viability question from Research Question 1.

**Dimension 2: Performance Metrics**

- Average latency
- Percentile latencies: P50, P95, P99 (to understand full distribution)
- Overhead decomposition: Separate caching logic from embedding generation

This ensures we understand exactly where latency is introduced and whether it remains within acceptable bounds.

**Dimension 3: Cache Behavior Metrics**

- Hit rates and miss rates
- Distribution of similarity scores (understanding matching patterns)
- Effect of time-to-live settings on cache freshness and utility

These metrics reveal the underlying mechanics of how semantic matching performs in practice.

**Dimension 4: Quality Metrics**

- Comparison of cached answers versus fresh LLM responses on representative samples
- Evaluation methodology: Either reference answers from labeled datasets or LLM-as-judge techniques where a separate model evaluates answer equivalence

This is absolutely critical because a cache hit that returns an incorrect or outdated answer is worse than a cache miss—it degrades user experience while appearing to optimize cost.

### 5.5 Goal: Identifying Configuration "Sweet Spots"

The overarching goal of this multi-dimensional analysis is to identify configuration "sweet spots" where cost savings are substantial but quality and latency remain within acceptable operational thresholds.

For example, I might discover that a similarity threshold of 0.85 delivers:
- 40% cost savings
- Less than 5% quality degradation
- Latency overhead below 100 milliseconds

Such a finding would constitute a commercially viable configuration with clear deployment value.

### 5.6 Privacy and Ethics Note

Throughout this data collection and analysis, no personally identifiable information will be stored or processed. The research focus is entirely on aggregate query patterns and semantic redundancy characteristics, not on individual user data or behavior.

---

## 6. Expected Findings & Contributions

### 6.1 Expected Findings

Based on existing literature and preliminary analysis, I anticipate the following results:

**For RQ1: Cache Hit Rates by Workload**

- **FAQ-style workloads**: 50-60% hit rates due to inherently repetitive nature—users asking the same fundamental questions with slight wording variations
- **Conversational support scenarios**: 30-40% hit rates, reflecting partial repetition mixed with contextual variation
- **RAG workloads**: 20-50% variability depending on document popularity and query diversity

**Economic translation**: A 40% cache hit rate at current LLM API pricing could translate to thousands of dollars in monthly savings for a mid-sized application processing tens of thousands of queries daily.

**For RQ2: Similarity Thresholds and Trade-offs**

I expect to identify an optimal point around a cosine similarity threshold of **0.85**:

- **Restrictive enough**: Minimizes false positives (preventing semantically different queries from matching incorrectly)
- **Flexible enough**: Captures genuine semantic equivalence across diverse phrasings

**Below 0.80**: Quality degradation likely as the system begins matching queries requiring genuinely different responses

**Above 0.90**: Hit rate likely drops significantly, undermining cost optimization benefit

**Latency expectations**: If embedding generation is properly optimized using compact models like ModernBERT (149 million parameters), overhead should remain below 100 milliseconds, which is acceptable for most interactive applications.

**For RQ3: Deployment Contexts**

The findings should reveal that semantic caching delivers maximum value for workloads exhibiting natural query repetition:

- ✅ FAQ systems
- ✅ Customer support chatbots
- ✅ Document summarization tasks

Conversely, minimal benefit for:

- ❌ Highly creative or generative tasks where each query is inherently unique
- ❌ Novel story generation
- ❌ Creative writing assistance

These expected patterns will inform practical deployment guidelines, helping organizations make evidence-based decisions about when semantic caching infrastructure investment is justified.

### 6.2 Research Contributions

This work will deliver four distinct contributions that advance both scientific understanding and practical application of semantic caching for LLM APIs.

**Contribution 1: Comprehensive Empirical Evidence**

The first contribution is systematic, reproducible data measuring actual cost savings, quality trade-offs, and latency overhead across diverse real-world usage patterns. Existing research in this area is either purely theoretical or conducted at small scale with limited applicability. Levy will provide the empirical foundation currently missing from the literature.

**Contribution 2: Reusable Research Prototype**

The Levy caching engine will be open-sourced, enabling other researchers to build upon this foundation and allowing practitioners to adapt the implementation for their specific production environments. This artifact contribution extends the research impact beyond the dissertation itself.

**Contribution 3: Practical Deployment Guidelines**

Rather than simply proving "semantic caching works," this research will deliver decision frameworks specifying:
- When semantic caching is worth deploying
- Which workload characteristics justify infrastructure investment
- What configuration parameters deliver optimal results for different use cases

**Contribution 4: Informing Broader AI Infrastructure Design**

Levy represents one building block of what I call "event-driven AI infrastructure"—systems that combine intelligent batching, comprehensive observability, and semantic caching to optimize LLM operations at scale. This connects to my distributed systems background and positions semantic caching within a larger architectural vision.

### 6.3 Commercial Impact

From a commercial perspective, this research directly addresses a concrete pain point that organizations experience every month when LLM API bills arrive. By rigorously quantifying potential savings and providing evidence-based deployment guidelines, this work will accelerate the adoption of cost optimization techniques in production LLM systems, delivering tangible business value alongside academic contributions.

---

## 7. Next Steps & Timeline

### 7.1 Immediate Next Steps

**Step 1: Deepen Literature Review (Weeks 1-2)**  
Focusing specifically on LLM serving architectures, existing caching strategies in production systems, and published cost optimization techniques. This ensures the research builds appropriately on prior work.

**Step 2: Finalize Technical Architecture (Week 3)**  
Making concrete decisions on vector store implementation, embedding model selection, and similarity metric calculations.

**Step 3: Implement Levy Prototype (Weeks 4-5)**  
Building the complete system and conducting initial pilot experiments to validate the technical approach and identify any implementation issues early.

**Step 4: Execute Full Experimental Evaluation (Weeks 6-9)**  
Running comprehensive experiments across all workload types with systematic parameter variation.

**Step 5: Analysis and Dissertation Writing (Weeks 10-12)**  
Performing statistical analysis, synthesizing findings, and completing the final capstone dissertation.

### 7.2 Success Criteria

The project will be considered successful if it delivers:

1. **Technical rigor**: A working prototype with reproducible experimental results
2. **Commercial meaning**: Clear quantification of cost savings and deployment guidelines
3. **Academic contribution**: Novel insights publishable in ML systems or distributed systems conferences
4. **Practical feasibility**: Completion within the 12-week timeline

---

## 8. Summary

The goal is to deliver a technically rigorous, commercially meaningful, and realistically feasible capstone project. Levy bridges my ten years of distributed systems expertise with emerging AI infrastructure research, producing results that will be valuable to both academic researchers and industry practitioners.

By systematically validating semantic caching for LLM APIs across multiple dimensions—cost, quality, latency, and workload patterns—this research will either establish semantic caching as a production-ready optimization technique or identify the precise conditions under which it fails, which is equally valuable scientifically.

The combination of empirical rigor, practical applicability, and connection to broader infrastructure concerns positions Levy as a meaningful contribution to the rapidly evolving field of LLM operations and cost optimization.

---

## References

1. Lempel, R., & Moran, S. (2003). Predictive caching and prefetching of query results in search engines. *Proceedings of the 12th International Conference on World Wide Web*, 19-28.

2. Markatos, E. P. (2001). On caching search engine query results. *Computer Communications*, 24(2), 137-143.

3. Xie, Y., & O'Hallaron, D. (2002). Locality in search engine queries and its implications for caching. *Proceedings of the Twenty-First Annual Joint Conference of the IEEE Computer and Communications Societies*, 3, 1238-1247.

4. Gill, W., et al. (2025). MeanCache: User-Centric Semantic Caching for LLM Web Services. *IEEE International Parallel and Distributed Processing Symposium (IPDPS)*. arXiv:2403.02694

5. Gill, W., et al. (2025). Advancing Semantic Caching for LLMs with Domain-Specific Embeddings and Synthetic Data. arXiv:2504.02268v1

6. Bang, F. (2023). GPTCache: An open-source semantic cache for LLM applications enabling faster answers and cost savings. *Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)*, 212-218.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: [Your university email if appropriate]
