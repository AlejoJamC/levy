"""
Tests for reading models and prices from `.env`.

No model id or price is written in Levy's code. `.env` names them one fact per
line -- LEVY_ANTHROPIC_MODEL_<n> with its _INPUT_USD_PER_MTOK and
_OUTPUT_USD_PER_MTOK, LEVY_OLLAMA_MODEL_<n>, LEVY_OLLAMA_EMBEDDING_MODEL_<n> --
and is read again on use, so an edit needs no restart. The parser is tested on
plain mappings, and the re-read on a temporary `.env`, so a developer's own
`.env` cannot change the outcome.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import levy.config as config_module
from levy.config import LevyConfig, ModelCatalog, read_model_catalog


def _anthropic(n, model, input_usd, output_usd):
    return {
        f"LEVY_ANTHROPIC_MODEL_{n}": model,
        f"LEVY_ANTHROPIC_MODEL_{n}_INPUT_USD_PER_MTOK": input_usd,
        f"LEVY_ANTHROPIC_MODEL_{n}_OUTPUT_USD_PER_MTOK": output_usd,
    }


class TestReadModelCatalog(unittest.TestCase):

    def test_nothing_configured(self):
        self.assertEqual(read_model_catalog({"UNRELATED": "x"}), ModelCatalog())

    def test_models_are_ordered_by_number_and_numbers_need_not_be_consecutive(self):
        environ = {
            **_anthropic(3, "model-c", "15.00", "75"),
            **_anthropic(1, "model-a", "1.00", "5.00"),
            **_anthropic(2, "model-b", "0.80", "4"),
            "LEVY_OLLAMA_MODEL_8": "local-h",
            "LEVY_OLLAMA_MODEL_2": "local-b:latest",
            "LEVY_OLLAMA_EMBEDDING_MODEL_1": "embed-a",
        }
        catalog = read_model_catalog(environ)
        self.assertEqual(catalog.anthropic, (("model-a", 1.0, 5.0), ("model-b", 0.8, 4.0), ("model-c", 15.0, 75.0)))
        self.assertEqual(catalog.ollama, ("local-b:latest", "local-h"))
        self.assertEqual(catalog.ollama_embedding, ("embed-a",))

    def test_prices_are_plain_numbers_with_a_dot(self):
        for bad in ("$1", "1usd", "3eur", "1,25", "USD 1", "1.", ".5", "-1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    read_model_catalog(_anthropic(1, "model-a", bad, "5"))
                self.assertIn("LEVY_ANTHROPIC_MODEL_1_INPUT_USD_PER_MTOK", str(ctx.exception))
                self.assertIn("1.25", str(ctx.exception))

    def test_a_model_without_its_prices_names_what_is_missing(self):
        with self.assertRaises(ValueError) as ctx:
            read_model_catalog({"LEVY_ANTHROPIC_MODEL_4": "model-d", "LEVY_ANTHROPIC_MODEL_4_INPUT_USD_PER_MTOK": "1"})
        self.assertIn("LEVY_ANTHROPIC_MODEL_4_OUTPUT_USD_PER_MTOK", str(ctx.exception))

    def test_a_price_without_its_model_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            read_model_catalog({"LEVY_ANTHROPIC_MODEL_5_INPUT_USD_PER_MTOK": "1"})
        self.assertIn("LEVY_ANTHROPIC_MODEL_5 is not", str(ctx.exception))

    def test_blank_template_lines_are_not_configured(self):
        blank = {**_anthropic(1, "", "", ""), "LEVY_OLLAMA_MODEL_1": "  ", "LEVY_OLLAMA_EMBEDDING_MODEL_1": ""}
        self.assertEqual(read_model_catalog(blank), ModelCatalog())


class TestConfigFromCatalog(unittest.TestCase):

    def test_lowest_number_is_the_default_triple(self):
        catalog = read_model_catalog({**_anthropic(2, "model-b", "3", "15"), **_anthropic(1, "model-a", "1", "5"),
                                      "LEVY_OLLAMA_MODEL_1": "local-a"})
        config = LevyConfig(llm_provider="mock").with_models(catalog)
        self.assertEqual(
            (config.anthropic_model, config.anthropic_input_price_per_mtok, config.anthropic_output_price_per_mtok),
            ("model-a", 1.0, 5.0),
        )
        self.assertEqual(config.anthropic_prices("model-b"), (3.0, 15.0))
        self.assertIsNone(config.anthropic_prices("not-configured"))
        self.assertEqual(config.model_name, "local-a")

    def test_empty_catalog_means_no_model_and_no_price(self):
        config = LevyConfig(llm_provider="mock").with_models(ModelCatalog())
        self.assertIsNone(config.anthropic_model)
        self.assertIsNone(config.anthropic_input_price_per_mtok)
        self.assertIsNone(config.model_name)
        self.assertEqual(config.ollama_embedding_models, ())


class TestDotenvIsReadAgain(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.dir.name) / ".env"
        clean = {k: v for k, v in os.environ.items() if not k.startswith("LEVY_")}
        self.patches = [
            mock.patch.object(config_module, "_DOTENV_PATH", str(self.env_file)),
            mock.patch.object(config_module, "_LOADED_FROM_DOTENV", set()),
            mock.patch.dict(os.environ, clean, clear=True),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.dir.cleanup()

    def test_an_edited_env_file_is_seen_without_a_restart(self):
        self.env_file.write_text("LEVY_OLLAMA_MODEL_1=local-a\n")
        self.assertEqual(config_module.current_model_catalog().ollama, ("local-a",))
        self.env_file.write_text("LEVY_OLLAMA_MODEL_1=local-a\nLEVY_OLLAMA_MODEL_8=local-h\n")
        self.assertEqual(config_module.current_model_catalog().ollama, ("local-a", "local-h"))
        self.assertEqual(LevyConfig(llm_provider="mock").ollama_models, ("local-a", "local-h"))

    def test_a_variable_set_in_the_process_wins_over_env(self):
        self.env_file.write_text("LEVY_OLLAMA_MODEL_1=from-file\n")
        with mock.patch.dict(os.environ, {"LEVY_OLLAMA_MODEL_1": "from-process"}):
            self.assertEqual(config_module.current_model_catalog().ollama, ("from-process",))

    def test_no_env_file(self):
        with mock.patch.object(config_module, "_DOTENV_PATH", ""):
            self.assertEqual(config_module.current_model_catalog(), ModelCatalog())


if __name__ == "__main__":
    unittest.main()
