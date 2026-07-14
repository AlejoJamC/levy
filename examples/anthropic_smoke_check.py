"""
One-shot, real-API smoke check for the Anthropic connector (LEV-6).

Not part of the offline test suite: it lives in examples/ (not tests/, and
not named test_*.py), so pytest never discovers or runs it, and it makes a
real network call billed against the real ANTHROPIC_API_KEY in `.env`.

Run once before relying on the connector for real cache-population/experiment
runs, to confirm the key, model, and budget config are wired correctly:

    python examples/anthropic_smoke_check.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from levy import LevyEngine, LevyConfig


def main():
    config = LevyConfig(
        llm_provider="anthropic",
        embedding_provider="mock",
        enable_semantic_cache=False,
    )
    if not config.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set (check .env) -- aborting smoke check.")
        sys.exit(1)

    engine = LevyEngine(config)
    result = engine.generate("Reply with exactly the word: pong")

    print(f"source: {result.source}")
    print(f"answer: {result.answer}")
    print(f"model: {result.original_response.model if result.original_response else 'n/a'}")
    print(f"token_usage: {result.original_response.token_usage if result.original_response else 'n/a'}")
    print(f"request_count: {engine.llm_client.request_count}")
    print(f"estimated_cost_usd: {engine.llm_client.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    main()
