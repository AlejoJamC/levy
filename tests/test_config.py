"""
Tests for reading models and prices from the environment (LEV-23, design D14).

No model id or price is written in Levy's code: `LevyConfig` reads them from
LEVY_ANTHROPIC_MODELS, LEVY_OLLAMA_MODELS and LEVY_OLLAMA_EMBEDDING_MODELS.
Every test sets or clears those variables itself, so a developer's own `.env`
cannot change the outcome.
"""

import os
import unittest
from unittest import mock

from levy.config import LevyConfig

VARS = ("LEVY_ANTHROPIC_MODELS", "LEVY_OLLAMA_MODELS", "LEVY_OLLAMA_EMBEDDING_MODELS")


def _env(**values):
    cleared = {name: "" for name in VARS}
    cleared.update(values)
    return mock.patch.dict(os.environ, cleared)


class TestModelsFromEnvironment(unittest.TestCase):

    def test_unset_means_no_model_and_no_price(self):
        with _env():
            config = LevyConfig()
        self.assertEqual(config.anthropic_models, ())
        self.assertIsNone(config.anthropic_model)
        self.assertIsNone(config.anthropic_input_price_per_mtok)
        self.assertIsNone(config.anthropic_output_price_per_mtok)
        self.assertEqual(config.ollama_models, ())
        self.assertIsNone(config.model_name)
        self.assertEqual(config.ollama_embedding_models, ())

    def test_first_anthropic_entry_is_the_default_triple(self):
        with _env(LEVY_ANTHROPIC_MODELS=" model-a:1:5 , model-b:3.5:15 "):
            config = LevyConfig()
        self.assertEqual(config.anthropic_models, (("model-a", 1.0, 5.0), ("model-b", 3.5, 15.0)))
        self.assertEqual(
            (config.anthropic_model, config.anthropic_input_price_per_mtok, config.anthropic_output_price_per_mtok),
            ("model-a", 1.0, 5.0),
        )
        self.assertEqual(config.anthropic_prices("model-b"), (3.5, 15.0))
        self.assertIsNone(config.anthropic_prices("not-configured"))

    def test_ollama_lists(self):
        with _env(LEVY_OLLAMA_MODELS="local-a:latest,local-b", LEVY_OLLAMA_EMBEDDING_MODELS="embed-a"):
            config = LevyConfig()
        self.assertEqual(config.ollama_models, ("local-a:latest", "local-b"))
        self.assertEqual(config.model_name, "local-a:latest")
        self.assertEqual(config.ollama_embedding_models, ("embed-a",))

    def test_malformed_anthropic_entry_names_the_variable(self):
        for bad in ("model-only", "model:1", "model:one:5", ":1:5", "model:1:5:9"):
            with self.subTest(bad=bad), _env(LEVY_ANTHROPIC_MODELS=bad):
                with self.assertRaises(ValueError) as ctx:
                    LevyConfig()
                self.assertIn("LEVY_ANTHROPIC_MODELS", str(ctx.exception))

    def test_explicit_values_ignore_the_environment(self):
        with _env(LEVY_ANTHROPIC_MODELS="model-a:1:5"):
            config = LevyConfig(anthropic_model="explicit", anthropic_input_price_per_mtok=2.0,
                                anthropic_output_price_per_mtok=8.0)
        self.assertEqual(config.anthropic_model, "explicit")
        self.assertEqual(config.anthropic_input_price_per_mtok, 2.0)


if __name__ == "__main__":
    unittest.main()
