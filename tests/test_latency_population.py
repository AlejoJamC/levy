"""
Tests for response-corpus population (LEV-14 / 5.5).

The billed entry point, `scripts/populate_responses.py`, is never invoked here
— an AST guard in `tests/test_corpus_acquisition.py` enforces that. What is
exercised is the library function it is a CLI over, driven through a real
`AnthropicLLMClient` whose transport is an `httpx.MockTransport`, the same
injection `tests/test_anthropic_client.py` uses. No socket is opened and no
`ANTHROPIC_API_KEY` is needed.

Three behaviours carry money or data integrity and are pinned here:

- **Resume without double spend.** A prompt already in the corpus is not called
  again — the property that makes an interrupted 600-call run safe to restart.
- **The budget guard stops cleanly.** Everything recorded before the halt is on
  disk and readable; the guard raises before sending, so nothing is paid for
  and lost.
- **A refusal is not recorded.** There is no response text to serve, and an
  empty one would be replayed into every configuration.
"""

import json
import tempfile
import unittest
from pathlib import Path

import anthropic
import httpx

from levy.latency.corpus import load_corpus, response_key
from levy.latency.population import populate_corpus
from levy.llm_client import AnthropicLLMClient


def _message_response(text="an answer", model="claude-haiku-4-5-20251001", stop_reason="end_turn",
                      input_tokens=10, output_tokens=5) -> dict:
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _client(handler, **kwargs) -> AnthropicLLMClient:
    http_client = anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("model", "claude-haiku-4-5-20251001")
    kwargs.setdefault("max_retries", 0)
    return AnthropicLLMClient(api_key="sk-test", http_client=http_client, **kwargs)


class _CountingHandler:
    """Answers every request, recording how many arrived."""

    def __init__(self, **response_kwargs):
        self.calls = 0
        self.response_kwargs = response_kwargs

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, json=_message_response(**self.response_kwargs))


class TestCorpusWriter(unittest.TestCase):

    def test_one_record_per_prompt_with_its_measurement_fields(self):
        handler = _CountingHandler(text="cached answer", input_tokens=42, output_tokens=17)
        prompts = ["first question", "second question"]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            outcome = populate_corpus(prompts, path, client=_client(handler))
            records = load_corpus(path)

        self.assertEqual(handler.calls, 2)
        self.assertEqual((outcome.requested, outcome.called, outcome.skipped), (2, 2, 0))
        self.assertIsNone(outcome.halted)
        self.assertEqual(sorted(records), sorted(response_key(prompt) for prompt in prompts))

        record = records[response_key("first question")]
        self.assertEqual(record.response_text, "cached answer")
        self.assertEqual(record.input_tokens, 42)
        self.assertEqual(record.output_tokens, 17)
        self.assertEqual(record.model, "claude-haiku-4-5-20251001")
        self.assertGreaterEqual(record.latency_ms, 0.0)
        self.assertTrue(record.timestamp_utc.endswith("+00:00"))
        self.assertEqual(record.stop_reason, "end_turn")

    def test_progress_callback_is_invoked_once_per_recorded_call(self):
        handler = _CountingHandler()
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            populate_corpus(
                ["a", "b"],
                Path(tmp) / "responses.jsonl",
                client=_client(handler),
                on_progress=lambda index, total: seen.append((index, total)),
            )
        self.assertEqual(seen, [(1, 2), (2, 2)])

    def test_max_calls_bounds_a_single_invocation(self):
        handler = _CountingHandler()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            outcome = populate_corpus(["a", "b", "c", "d"], path, client=_client(handler), max_calls=2)

        self.assertEqual(handler.calls, 2)
        self.assertEqual(outcome.called, 2)


class TestResumeSkipsWhatWasAlreadyPaidFor(unittest.TestCase):

    def test_a_prompt_already_in_the_corpus_is_not_called_again(self):
        prompts = ["alpha", "beta", "gamma"]
        first_handler = _CountingHandler()
        second_handler = _CountingHandler()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"

            first = populate_corpus(prompts[:2], path, client=_client(first_handler))
            second = populate_corpus(prompts, path, client=_client(second_handler))
            records = load_corpus(path)

        self.assertEqual(first.called, 2)
        self.assertEqual(first_handler.calls, 2)

        # Only the third prompt was outstanding on the second pass.
        self.assertEqual(second_handler.calls, 1)
        self.assertEqual((second.requested, second.skipped, second.called), (3, 2, 1))
        self.assertEqual(len(records), 3)

    def test_a_complete_corpus_makes_no_calls_at_all(self):
        prompts = ["alpha", "beta"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            populate_corpus(prompts, path, client=_client(_CountingHandler()))

            handler = _CountingHandler()
            outcome = populate_corpus(prompts, path, client=_client(handler))

        self.assertEqual(handler.calls, 0)
        self.assertEqual((outcome.called, outcome.skipped), (0, 2))


class TestBudgetStop(unittest.TestCase):

    def test_the_guard_halts_the_run_and_leaves_a_valid_partial_corpus(self):
        # 1000 input + 1000 output tokens per call at $1/$5 per MTok = $0.006 a
        # call, so a $0.01 cap is reached after the second call and the third is
        # never sent.
        handler = _CountingHandler(input_tokens=1000, output_tokens=1000)
        client = _client(
            handler,
            budget_cap_usd=0.01,
            input_price_per_mtok=1.0,
            output_price_per_mtok=5.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            outcome = populate_corpus(["a", "b", "c", "d"], path, client=client)

            body = path.read_text(encoding="utf-8")
            records = load_corpus(path)

        self.assertTrue(outcome.stopped_early)
        self.assertIn("budget cap", outcome.halted)
        self.assertEqual(outcome.called, 2)
        self.assertEqual(handler.calls, 2)

        # The partial corpus is valid, readable JSONL -- every line parses.
        lines = [line for line in body.splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            json.loads(line)
        self.assertEqual(len(records), 2)

    def test_a_halted_run_resumes_where_it_stopped(self):
        handler = _CountingHandler(input_tokens=1000, output_tokens=1000)
        capped = _client(handler, budget_cap_usd=0.01, input_price_per_mtok=1.0, output_price_per_mtok=5.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            populate_corpus(["a", "b", "c"], path, client=capped)

            resumed_handler = _CountingHandler()
            resumed = populate_corpus(["a", "b", "c"], path, client=_client(resumed_handler))

        self.assertEqual(resumed.skipped, 2)
        self.assertEqual(resumed_handler.calls, 1)


class TestRefusals(unittest.TestCase):

    def test_a_refusal_is_counted_and_nothing_is_recorded_for_it(self):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if payload["messages"][0]["content"] == "refused prompt":
                return httpx.Response(200, json=_message_response(text="", stop_reason="refusal"))
            return httpx.Response(200, json=_message_response())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            outcome = populate_corpus(
                ["fine prompt", "refused prompt", "another fine prompt"],
                path,
                client=_client(handler),
            )
            records = load_corpus(path)

        self.assertEqual(outcome.refusals, 1)
        self.assertEqual(outcome.called, 2)
        self.assertNotIn(response_key("refused prompt"), records)
        # A refusal does not stop the run.
        self.assertIn(response_key("another fine prompt"), records)


if __name__ == "__main__":
    unittest.main()
