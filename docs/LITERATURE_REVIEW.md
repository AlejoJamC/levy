# Literature Review: Semantic Caching for LLMs

> **Status: working skeleton.** This is the author's in-progress reading list and
> research-gaps matrix, not a finished review; entries marked
> `[summary + gaps]` are still to be written. The literature positioning that
> supports the study is in the frozen
> [Project_Proposal.md](Project_Proposal.md) and
> [Specification_and_Design_Report.md](Specification_and_Design_Report.md).

## Core Papers (Must-Read)
1. GPTCache (Bang, 2023) - [summary + gaps]
2. MeanCache (Gill et al., 2025) - [summary + gaps]
3. Category-Aware Caching (Wang et al., 2025) - [summary + gaps]
4. vCache (2025) - [summary + gaps]
5. LMSYS-Chat-1M (Zheng et al., 2024) - [summary + gaps]
6. TTL Approximation (Mazziane et al.) - [summary + gaps]

## Embeddings & Fine-tuning
7. ModernBERT (Warner et al., 2024)
8. SimCSE (Gao et al., 2021)
9. Qwen3-Embedding (2025)
10. Domain-Specific Embeddings Legal (Herrewijnen & Craandijk)

## Vector Search
11. HNSW Tutorial (Redis)
12. Product Quantization (Qdrant)

## Negative Results
13. Test-Time Memory for LLM Agents (OpenReview)
14. Timing Attacks on Prompt Caching (Stanford CS191)
15. Catchpoint Real-World Monitoring

## Research Gaps Matrix
| **Gap** | **Paper that identifies it** | **Opportunity for Levy** |
|---------|----------------------------|--------------------------|
| **No duplication measurement** | LMSYS-Chat-1M | RQ1: Quantify across workloads |
| **False positive epidemic** | GPTCache, MeanCache | RQ2: Threshold optimization |
| **Single threshold fails** | Category-Aware Caching | RQ3: Workload-specific policies |
| **No temporal drift study** | NINGUNO | Novel contribution |
| **Domain adaptation unclear** | Legal embeddings paper | Domain fine-tuning validation |
| **TTL theory unvalidated** | TTL Approximation paper | Empirical TTL validation |
