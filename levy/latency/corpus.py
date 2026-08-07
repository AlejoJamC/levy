"""
Response corpus: real provider responses, keyed by prompt hash (LEV-14 / 5.2).

One billed population run serves every configuration of a workload, because
`ExactCache` already keys on `sha256(prompt)` and so does this file. Records are
appended one JSON object per line as each call returns, so an interrupted run
leaves a valid, readable corpus and a resumed run skips everything already paid
for.

**The prompt text is not stored.** The key is its SHA-256, which is all the
lookup needs, and the corpus is built over Quora QQP text that carries no
redistribution right — the same reasoning that keeps `data/ground_truth.full.*`
out of the tree. The response text *is* stored: it is model output, and the
replay needs something to serve.

This module reads and writes files and computes summaries. It never calls a
provider — that is `scripts/populate_responses.py`, the billed entry point,
which is excluded from the test suite by the same AST guard that excludes
`scripts/fetch_corpora.py`.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from levy.llm_client import LLMClient
from levy.models import LLMRequest, LLMResponse

PathLike = Union[str, Path]


def response_key(prompt: str) -> str:
    """SHA-256 of the prompt — the corpus key, matching `ExactCache`'s key."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass
class ResponseRecord:
    """One real provider call: its response and its measurement fields."""

    key: str
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    timestamp_utc: str
    stop_reason: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict) -> "ResponseRecord":
        return cls(
            key=payload["key"],
            response_text=payload["response_text"],
            latency_ms=float(payload["latency_ms"]),
            input_tokens=int(payload["input_tokens"]),
            output_tokens=int(payload["output_tokens"]),
            model=payload["model"],
            timestamp_utc=payload["timestamp_utc"],
            stop_reason=payload.get("stop_reason"),
        )


class ResponseCorpusError(Exception):
    """Raised when a corpus line cannot be read as a record."""


# ----------------------------------------------------------------------
# Reading / writing
# ----------------------------------------------------------------------


def load_corpus(path: PathLike) -> Dict[str, ResponseRecord]:
    """
    Read a corpus into `{key: record}`. A missing file is an empty corpus —
    the population run's first invocation and a replay with nothing recorded
    yet are both legitimate.

    A later record for a key wins over an earlier one, so a re-populated prompt
    (a refusal retried, say) does not need the file rewritten.
    """
    path = Path(path)
    if not path.exists():
        return {}

    records: Dict[str, ResponseRecord] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = ResponseRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ResponseCorpusError(f"{path}:{lineno}: unreadable corpus record ({exc})") from exc
            records[record.key] = record
    return records


def append_record(path: PathLike, record: ResponseRecord) -> None:
    """Append one record and flush, so an interrupt loses at most the call in flight."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")
        fh.flush()


# ----------------------------------------------------------------------
# Serving a recorded corpus back to the engine
# ----------------------------------------------------------------------


class CorpusLLMClient(LLMClient):
    """
    Serves recorded responses by prompt hash; delegates anything unrecorded.

    The timed replay uses this so the measured run makes no provider call while
    still storing response text of realistic length. `served` and `delegated`
    count which happened, so a run can report how much of it was real.
    """

    def __init__(self, records: Dict[str, ResponseRecord], fallback: LLMClient):
        self._records = records
        self._fallback = fallback
        self.served = 0
        self.delegated = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        record = self._records.get(response_key(request.prompt))
        if record is None:
            self.delegated += 1
            return self._fallback.generate(request)

        self.served += 1
        return LLMResponse(
            text=record.response_text,
            token_usage=record.input_tokens + record.output_tokens,
            model=record.model,
            metadata={
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "model": record.model,
                "stop_reason": record.stop_reason,
                "source": "response_corpus",
            },
        )
