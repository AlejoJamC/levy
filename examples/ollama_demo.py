import sys
import os
import logging
# Ensure we can import levy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from levy import LevyEngine, LevyConfig

def main():
    logging.basicConfig(level=logging.INFO)
    print("--- Levy with Local Ollama ---")

    # Configuration for Ollama. The models are not written here: set them in .env
    # (LEVY_OLLAMA_MODEL_<n>, LEVY_OLLAMA_EMBEDDING_MODEL_<n>) to models you have pulled
    # and are serving with `ollama serve`.
    defaults = LevyConfig()
    if not defaults.ollama_models or not defaults.ollama_embedding_models:
        print("Set LEVY_OLLAMA_MODEL_1 and LEVY_OLLAMA_EMBEDDING_MODEL_1 in .env first.")
        return
    config = LevyConfig(
        llm_provider="ollama",
        model_name=defaults.ollama_models[0],
        enable_semantic_cache=True,
        embedding_provider="ollama",
        embedding_model=defaults.ollama_embedding_models[0],
        similarity_threshold=0.8
    )

    try:
        engine = LevyEngine(config)
    except Exception as e:
        print(f"Failed to initialize engine (is Ollama running?): {e}")
        return

    prompts = [
        "Why is the sky blue?",
        "Why is the sky blue?", # Exact match
        "Explain the color of the sky", # Semantic match hopefull
    ]

    for p in prompts:
        print(f"\nQuery: {p}")
        try:
            res = engine.generate(p)
            print(f"Answer: {res.answer[:100]}...")
            print(f"Source: {res.source}")
            if res.similarity_score:
                print(f"Score:  {res.similarity_score}")
        except Exception as e:
            print(f"Error processing request: {e}")

    print("\nMetrics:")
    print(engine.get_metrics_summary())

if __name__ == "__main__":
    main()
