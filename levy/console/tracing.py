"""
Per-request traces for the console, held in a bounded in-memory buffer.

`TraceRecorder` owns the buffer, a global version that every create/update
advances (the page polls with `after=<version>`), an epoch replaced on clear,
and the hit-to-origin map that attributes a hit's savings to the miss that
stored its entry. `LiveTimingCollector` is the engine's ordinary
`TimingCollector`, plus a report to the recorder when each segment starts and
ends -- which is what makes a pending provider call visible while it is still
running. Nothing here changes what the engine decides; it only watches.

Traces are never written to disk.
"""

import copy
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from levy.config import LevyConfig
from levy.console.options import ConsoleSettings
from levy.latency.timing import SEGMENT_LLM_CALL, TimingCollector
from levy.models import LLMResponse, SemanticLookupObservation

DEFAULT_TRACE_BUFFER_SIZE = 500
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


class TraceRecorder:
    def __init__(self, max_traces: int = DEFAULT_TRACE_BUFFER_SIZE):
        self.max_traces = max_traces
        self._lock = threading.Lock()
        self._traces: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._version = 0
        self._seq = 0
        self._epoch = uuid.uuid4().hex[:8]
        self._origins: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # -- buffer -------------------------------------------------------------

    def _bump(self, trace: Dict[str, Any]) -> None:
        """Caller holds the lock."""
        self._version += 1
        trace["version"] = self._version

    def open(self, fields: Dict[str, Any]) -> str:
        with self._lock:
            self._seq += 1
            trace_id = str(uuid.uuid4())
            trace = {
                "id": trace_id,
                "seq": self._seq,
                "status": "in_flight",
                "outcome": None,
                "current_stage": "router",
                "spans": [],
                **fields,
            }
            self._bump(trace)
            self._traces[trace_id] = trace
            while len(self._traces) > self.max_traces:
                self._traces.popitem(last=False)
            return trace_id

    def update(self, trace_id: str, **fields: Any) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:  # evicted while still running
                return
            trace.update(fields)
            self._bump(trace)

    def stage_started(self, trace_id: str, name: str) -> None:
        self.update(trace_id, current_stage=name)

    def span_finished(self, trace_id: str, name: str, start_ms: float, duration_ms: float) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace["spans"].append({"name": name, "start_ms": start_ms, "duration_ms": duration_ms})
            self._bump(trace)

    def current_stage(self, trace_id: str) -> Optional[str]:
        with self._lock:
            trace = self._traces.get(trace_id)
            return trace["current_stage"] if trace is not None else None

    def finish(self, trace_id: str, **fields: Any) -> None:
        self.update(trace_id, status="done", current_stage=None, **fields)

    def since(self, after: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(t) for t in self._traces.values() if t["version"] > after]

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            trace = self._traces.get(trace_id)
            return copy.deepcopy(trace) if trace is not None else None

    @property
    def version(self) -> int:
        return self._version

    @property
    def epoch(self) -> str:
        return self._epoch

    def clear(self) -> str:
        """Empty the history and start a new epoch. Origins belong to the caches, not the history."""
        with self._lock:
            self._traces.clear()
            self._epoch = uuid.uuid4().hex[:8]
            return self._epoch

    # -- hit-to-origin attribution -----------------------------------------

    def remember_origin(self, key: Tuple[str, str], origin: Dict[str, Any]) -> None:
        with self._lock:
            self._origins[key] = dict(origin)

    def origin(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        with self._lock:
            origin = self._origins.get(key)
            return dict(origin) if origin is not None else None

    def clear_origins(self) -> None:
        with self._lock:
            self._origins.clear()


class LiveTimingCollector(TimingCollector):
    """
    A `TimingCollector` that also tells the recorder when each segment starts
    and ends, with its start offset from the request's arrival. Durations are
    recorded exactly as the base class records them.
    """

    def __init__(
        self,
        recorder: TraceRecorder,
        trace_id: str,
        t0: float,
        observation: Optional[SemanticLookupObservation] = None,
    ):
        super().__init__()
        self._recorder = recorder
        self._trace_id = trace_id
        self._t0 = t0
        self._observation = observation

    @contextmanager
    def measure(self, segment: str) -> Iterator[None]:
        start = time.perf_counter()
        start_ms = (start - self._t0) * 1000.0
        if segment == SEGMENT_LLM_CALL and self._observation is not None and self._observation.reached:
            # The lookup is over by now: publish its decision while the provider
            # call is still running, so a pending miss already shows its nearest neighbour.
            self._recorder.update(self._trace_id, semantic=observation_dict(self._observation))
        self._recorder.stage_started(self._trace_id, segment)
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.record(segment, duration_ms)
            self._recorder.span_finished(self._trace_id, segment, start_ms, duration_ms)


# -- trace fields -------------------------------------------------------------


def observation_dict(observation: SemanticLookupObservation) -> Dict[str, Any]:
    """The semantic block; `cosine = 1 - d^2/2` holds because every vector is L2-normalised.

    `threshold_cosine` is the cosine the threshold enforces: `d = 1/τ - 1` on the
    `1/(1+d)` scale.
    """
    cosine = None
    if observation.distance is not None:
        cosine = 1.0 - observation.distance ** 2 / 2.0
    threshold_cosine = None
    if observation.threshold:
        tau_distance = 1.0 / observation.threshold - 1.0
        threshold_cosine = 1.0 - tau_distance ** 2 / 2.0
    return {
        "reached": observation.reached,
        "index_size": observation.index_size,
        "threshold": observation.threshold,
        "threshold_cosine": threshold_cosine,
        "nearest_prompt": observation.nearest_prompt,
        "distance": observation.distance,
        "similarity": observation.similarity,
        "cosine": cosine,
        "hit": observation.hit,
    }


def provider_endpoint(settings: ConsoleSettings, base: LevyConfig, client: Any) -> Tuple[str, str]:
    """(endpoint URL, `local` | `remote`) of the provider a request was sent to."""
    if settings.llm_provider == "anthropic":
        sdk = getattr(client, "_client", None)
        url = str(getattr(sdk, "base_url", ANTHROPIC_DEFAULT_BASE_URL))
    else:
        url = base.ollama_base_url
    host = urlparse(url).hostname or url
    location = "local" if host in LOOPBACK_HOSTS else "remote"
    return url.rstrip("/"), location


def token_usage(response: LLMResponse) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """(input tokens, output tokens, stop reason) from whichever provider produced `response`."""
    meta = response.metadata or {}
    if "input_tokens" in meta:  # Anthropic
        return meta.get("input_tokens"), meta.get("output_tokens"), meta.get("stop_reason")
    if "prompt_eval_count" in meta or "eval_count" in meta:  # Ollama
        return meta.get("prompt_eval_count"), meta.get("eval_count"), meta.get("done_reason")
    return None, None, None


def call_cost_usd(
    prices: Optional[Tuple[float, float]], input_tokens: Optional[int], output_tokens: Optional[int]
) -> Optional[float]:
    """`_BudgetGuard.record`'s formula at the model's configured prices; None when none are configured."""
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = prices
    return input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price


def llm_block(
    settings: ConsoleSettings, base: LevyConfig, client: Any, response: LLMResponse, duration_ms: float
) -> Dict[str, Any]:
    endpoint, location = provider_endpoint(settings, base, client)
    input_tokens, output_tokens, stop_reason = token_usage(response)
    return {
        "provider": settings.llm_provider,
        "endpoint": endpoint,
        "location": location,
        "model": response.model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": response.token_usage,
        "stop_reason": stop_reason,
        "cost_usd": call_cost_usd(base.anthropic_prices(settings.llm_model), input_tokens, output_tokens),
        "duration_ms": duration_ms,
    }
