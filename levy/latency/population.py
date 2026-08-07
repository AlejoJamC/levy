"""
Population of the response corpus (LEV-14 / 5.1-5.3).

The loop lives here, not in the script, for one reason: it has to be tested.
The spec requires both that the corpus writer, the resume-skip and the
budget-stop are covered offline, and that no test ever invokes the billed
entry point. Those are only compatible if the logic is a library function
taking an injected `LLMClient` — the tests drive it with an
`AnthropicLLMClient` wrapping `httpx.MockTransport`, exactly as
`tests/test_anthropic_client.py` does, while `scripts/populate_responses.py`
remains the only thing that ever constructs a real one.

Nothing here opens a socket, and nothing here imports a network library.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Union

from levy.latency.corpus import ResponseRecord, append_record, load_corpus, response_key
from levy.llm_client import AnthropicRefusalError, BudgetExceededError, LLMClient
from levy.models import LLMRequest

PathLike = Union[str, Path]


@dataclass
class PopulationOutcome:
    """What one population invocation did, and why it stopped."""

    requested: int          # prompts in scope
    skipped: int            # already in the corpus, so already paid for
    called: int             # provider calls that returned a usable response
    refusals: int = 0       # refused responses: not recorded, nothing to serve
    halted: Optional[str] = None   # budget-guard message, if the run stopped early
    keys: List[str] = field(default_factory=list)

    @property
    def stopped_early(self) -> bool:
        return self.halted is not None


def populate_corpus(
    prompts: List[str],
    corpus_path: PathLike,
    client: LLMClient,
    max_tokens: int = 256,
    max_calls: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> PopulationOutcome:
    """
    Call `client` once per prompt not already in the corpus, appending as it goes.

    Skips, in this order: prompts whose `sha256` is already recorded (an
    interrupted run resumes without paying twice), then anything beyond
    `max_calls`.

    A `BudgetExceededError` stops the loop rather than propagating: the guard
    raises *before* sending, so everything recorded is already on disk and the
    partial corpus is valid, complete JSONL. A refusal is counted and skipped —
    there is no response text to serve, and recording an empty one would put a
    non-answer into every configuration that replays that prompt.
    """
    corpus_path = Path(corpus_path)
    existing = load_corpus(corpus_path)

    pending = [prompt for prompt in prompts if response_key(prompt) not in existing]
    skipped = len(prompts) - len(pending)
    if max_calls is not None:
        pending = pending[:max_calls]

    outcome = PopulationOutcome(requested=len(prompts), skipped=skipped, called=0)

    for index, prompt in enumerate(pending, start=1):
        request = LLMRequest(prompt=prompt, max_tokens=max_tokens)
        started = time.perf_counter()
        try:
            response = client.generate(request)
        except BudgetExceededError as exc:
            outcome.halted = str(exc)
            break
        except AnthropicRefusalError:
            outcome.refusals += 1
            continue
        latency_ms = (time.perf_counter() - started) * 1000.0

        key = response_key(prompt)
        append_record(
            corpus_path,
            ResponseRecord(
                key=key,
                response_text=response.text,
                latency_ms=latency_ms,
                input_tokens=int(response.metadata.get("input_tokens", 0)),
                output_tokens=int(response.metadata.get("output_tokens", 0)),
                model=response.model,
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                stop_reason=response.metadata.get("stop_reason"),
            ),
        )
        outcome.called += 1
        outcome.keys.append(key)
        if on_progress is not None:
            on_progress(index, len(pending))

    return outcome
