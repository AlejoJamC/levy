import sys
import os
import logging
import time

# Ensure we can import levy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from levy import LevyEngine, LevyConfig

def run_experiment(prompts: list[str], config: LevyConfig, name: str):
    print(f"\n--- Running Experiment: {name} ---")
    print(f"Config: Exact={config.enable_exact_cache}, Semantic={config.enable_semantic_cache}, Thresh={config.similarity_threshold}")
    
    engine = LevyEngine(config)
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Query: '{prompt}'")
        result = engine.generate(prompt)
        print(f"   -> Source: {result.source} | Latency: {result.latency_ms:.2f}ms")
        if result.similarity_score:
            print(f"   -> Similarity: {result.similarity_score:.4f}")
        # print(f"   -> Answer: {result.answer}")
        print("-" * 40)
        
    print("\nMetrics Summary:")
    print(engine.get_metrics_summary())

def main():
    logging.basicConfig(level=logging.WARNING)
    
    # Sample dataset: Contains exact duplicates and semantically similar queries
    prompts = [
        "What is the capital of France?",
        "What is the capital of France?",  # Exact duplicate
        "Tell me the capital city of France", # Semantic duplicate
        "How do I cook pasta?",
        "Recipe for cooking pasta", # Semantic duplicate
        "What is the capital of Germany?",
        "Who is the president of USA?",
        "What is the capital of France?", # Another exact duplicate
    ]
    
    # 1. Baseline: No Cache
    config_none = LevyConfig(
        enable_exact_cache=False, 
        enable_semantic_cache=False,
        llm_provider="mock",
        embedding_provider="mock"
    )
    run_experiment(prompts, config_none, "Baseline (No Cache)")

    # 2. Exact Cache Only
    config_exact = LevyConfig(
        enable_exact_cache=True, 
        enable_semantic_cache=False,
        llm_provider="mock",
        embedding_provider="mock"
    )
    run_experiment(prompts, config_exact, "Exact Cache Only")

    # 3. Semantic Cache
    # We use 'mock' embeddings which are random-seeded by text in my implementation,
    # so semantic similarity might be low or random unless we used a real model.
    # If the user installs 'sentence-transformers', they can switch provider to "sentence-transformers".
    
    # I'll check if sentence-transformers is installed, otherwise warn about mock semantic limitations used in scaffold
    try:
        import sentence_transformers
        emb_provider = "sentence-transformers"
        print("\n(Using real SentenceTransformers for semantic cache)")
    except ImportError:
        emb_provider = "mock"
        print("\n(Using MOCK embeddings - semantic matching will be random/hash-based)")

    config_semantic = LevyConfig(
        enable_exact_cache=True, 
        enable_semantic_cache=True,
        similarity_threshold=0.7, # Lower threshold for demo
        llm_provider="mock",
        embedding_provider=emb_provider
    )
    run_experiment(prompts, config_semantic, "Exact + Semantic Cache")

if __name__ == "__main__":
    main()
