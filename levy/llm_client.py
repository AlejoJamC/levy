from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import httpx
import anthropic
from levy.models import LLMRequest, LLMResponse

class LLMClient(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        pass

class MockLLMClient(LLMClient):
    """A mock client that echoes the prompt (reversed) for testing."""
    def __init__(self, latency_seconds: float = 0.5):
        self.latency_seconds = latency_seconds

    def generate(self, request: LLMRequest) -> LLMResponse:
        # Simulate network latency
        if self.latency_seconds:
            time.sleep(self.latency_seconds)
        response_text = f"Computed response for: {request.prompt[::-1]}" # Reverse string as 'computation'
        # Simple token estimation
        tokens = len(response_text.split())
        return LLMResponse(
            text=response_text,
            token_usage=tokens,
            model="mock-v1"
        )

class BudgetExceededError(Exception):
    """Raised when accumulated estimated Anthropic spend has reached the configured cap."""
    def __init__(self, cap_usd: float, estimated_cost_usd: float):
        self.cap_usd = cap_usd
        self.estimated_cost_usd = estimated_cost_usd
        super().__init__(
            f"Anthropic budget cap of ${cap_usd:.2f} reached (estimated spend: "
            f"${estimated_cost_usd:.2f}); no request was sent."
        )

class AnthropicRefusalError(Exception):
    """Raised when the Anthropic API returns a refusal stop reason, so the caller never
    receives (and the engine never caches) empty or partial refusal content."""
    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason
        super().__init__(f"Anthropic response refused (stop_reason={stop_reason!r}); not caching.")

class _BudgetGuard:
    """Per-instance request counter and estimated-cost accumulator (tokens x per-MTok prices)."""
    def __init__(self, cap_usd: float, input_price_per_mtok: float, output_price_per_mtok: float):
        self.cap_usd = cap_usd
        self.input_price_per_mtok = input_price_per_mtok
        self.output_price_per_mtok = output_price_per_mtok
        self.request_count = 0
        self.estimated_cost_usd = 0.0

    def check(self) -> None:
        if self.estimated_cost_usd >= self.cap_usd:
            raise BudgetExceededError(self.cap_usd, self.estimated_cost_usd)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.request_count += 1
        self.estimated_cost_usd += (
            input_tokens / 1_000_000 * self.input_price_per_mtok
            + output_tokens / 1_000_000 * self.output_price_per_mtok
        )

class AnthropicLLMClient(LLMClient):
    """Client backed by the official Anthropic SDK.

    Retry (connection errors, 408/409/429/5xx) is the SDK's own exponential backoff,
    configured via `max_retries` rather than reimplemented. A per-instance budget guard
    halts further requests once estimated spend reaches `budget_cap_usd`.
    """
    def __init__(
        self,
        api_key: Optional[str],
        model: Optional[str] = None,
        max_retries: int = 2,
        budget_cap_usd: float = 200.0,
        input_price_per_mtok: Optional[float] = None,
        output_price_per_mtok: Optional[float] = None,
        http_client: Optional[Any] = None,
    ):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the 'anthropic' provider")
        # No model or price is written in the code: they change every few months and
        # come from LEVY_ANTHROPIC_MODEL_<n> in .env.
        if not model or input_price_per_mtok is None or output_price_per_mtok is None:
            raise ValueError(
                "An Anthropic model and its input/output prices per MTok are required; "
                "configure them in .env as LEVY_ANTHROPIC_MODEL_<n>, LEVY_ANTHROPIC_MODEL_<n>_INPUT_USD_PER_MTOK "
                "and LEVY_ANTHROPIC_MODEL_<n>_OUTPUT_USD_PER_MTOK"
            )
        self.model = model
        self.budget = _BudgetGuard(budget_cap_usd, input_price_per_mtok, output_price_per_mtok)

        client_kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = anthropic.Anthropic(**client_kwargs)

    @property
    def request_count(self) -> int:
        return self.budget.request_count

    @property
    def estimated_cost_usd(self) -> float:
        return self.budget.estimated_cost_usd

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.budget.check()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            messages=[{"role": "user", "content": request.prompt}],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.budget.record(input_tokens, output_tokens)

        if response.stop_reason == "refusal":
            raise AnthropicRefusalError(response.stop_reason)

        text = next((block.text for block in response.content if block.type == "text"), "")

        return LLMResponse(
            text=text,
            token_usage=input_tokens + output_tokens,
            model=response.model,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
        )

class OllamaLLMClient(LLMClient):
    """Client for local Ollama instances."""
    def __init__(self, base_url: str = "http://localhost:11434", model: Optional[str] = None):
        if not model:
            raise ValueError("An Ollama model is required; configure it in .env as LEVY_OLLAMA_MODEL_<n>")
        self.base_url = base_url
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover -- requires a running Ollama server
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                # "num_predict": request.max_tokens # Ollama uses different param names sometimes, but num_predict is roughly max_tokens
            }
        }
        # Merge extra params if needed
        if "max_tokens" in request.extra_params:
             payload["options"]["num_predict"] = request.extra_params["max_tokens"]
        
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            content = data["message"]["content"]
            # Ollama returns explicit token counts in 'eval_count' (output) + 'prompt_eval_count' (input)
            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
            
            return LLMResponse(
                text=content,
                token_usage=tokens,
                model=self.model,
                metadata=data
            )
